///! Wall Tracker — Independent Density Screener
///! =============================================
///! Runs as a standalone `tokio::task`, scanning OrderBookStore every 500ms.
///! Tracks large walls (limit order clusters) with:
///!   - Age tracking (anti-spoofing: walls < 3h are unreliable)
///!   - Dynamic thresholds (wall_min = avg_1h_volume × 5%)
///!   - Cascade detection (clusters of walls < 0.5% apart)
///!   - Refresh counting (how many times MM refilled the wall)
///!
///! Also includes a depth-only WebSocket scanner (`run_depth_scanner`)
///! that subscribes to @depth@100ms for ALL 100 top instruments,
///! feeding into the shared OrderBookStore.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error};

use crate::live::order_book::{self, OrderBookStore, Level};

// ── Configuration ───────────────────────────────────────────────────────────

/// Minimum age (seconds) for a wall to be considered "reliable" (not spoofed)
const WALL_RELIABLE_AGE_SECS: u64 = 3 * 3600; // 3 hours

/// Scan interval
const SCAN_INTERVAL_MS: u64 = 500;

/// How close walls must be to form a cascade (0.5% = 0.005)
const CASCADE_GAP_PCT: f64 = 0.005;

/// Default wall threshold as fraction of 1h volume (5%)
const WALL_VOLUME_FRACTION: f64 = 0.05;

/// Minimum absolute wall size in USD (fallback when no volume data)
const WALL_MIN_USD_FALLBACK: f64 = 50_000.0;

/// Number of scans to track for stability calculation (500ms × 20 = 10 seconds)
const STABILITY_HISTORY_LEN: usize = 20;

/// Stability threshold below which a wall is considered a spoofer
const SPOOF_STABILITY_THRESHOLD: f64 = 0.6;

/// Minimum refill_count to flag as iceberg
const ICEBERG_REFILL_THRESHOLD: u32 = 3;

/// Buffer size for Dynamic Targets: we exit 0.05% before the wall
const DYNAMIC_TARGET_BUFFER_PCT: f64 = 0.0005;

/// Price tolerance for matching walls across scans (0.01%)
const _PRICE_MATCH_TOLERANCE: f64 = 0.0001;

/// How often to save wall state to disk
const PERSIST_INTERVAL_SECS: u64 = 300; // 5 minutes

/// Path for persisted wall state
const PERSIST_PATH: &str = "data/wall_state.json";

// ── Data Types ──────────────────────────────────────────────────────────────

/// Which side of the book the wall is on
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WallSide {
    Bid,  // Support wall (buying pressure)
    Ask,  // Resistance wall (selling pressure)
}

/// Serializable wall info for disk persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedWall {
    symbol: String,
    price: f64,
    side: WallSide,
    first_seen_unix: u64,  // Unix timestamp
    max_size_usd: f64,
    refresh_count: u32,
    touch_count: u32,
}

/// A tracked wall (large limit order at a price level)
#[derive(Debug, Clone)]
pub struct WallInfo {
    pub price: f64,
    pub side: WallSide,
    pub first_seen: Instant,
    pub max_size_usd: f64,
    pub current_size_usd: f64,
    pub refresh_count: u32,
    pub touch_count: u32,       // how many times price approached this wall
    pub last_touch: Option<Instant>,
    // ── Phase 29C+2: Stability & Iceberg ──
    /// Rolling presence history for stability calculation (true = present in scan)
    pub presence_history: VecDeque<bool>,
    /// Stability score: fraction of recent scans where wall was present (0.0 = ghost, 1.0 = rock)
    pub stability: f64,
    /// Previous scan size (for iceberg refill detection)
    pub prev_size_usd: f64,
    /// True if this wall is suspected to be an iceberg (refill_count >= threshold)
    pub is_iceberg: bool,
    /// True if this wall is suspected to be a spoofer (stability < threshold)
    pub is_spoof: bool,
}

impl WallInfo {
    /// Wall age in seconds
    pub fn age_secs(&self) -> u64 {
        Instant::now()
            .checked_duration_since(self.first_seen)
            .unwrap_or_default()
            .as_secs()
    }

    /// Wall age in hours (float)
    pub fn age_hours(&self) -> f64 {
        Instant::now()
            .checked_duration_since(self.first_seen)
            .unwrap_or_default()
            .as_secs_f64() / 3600.0
    }

    /// Is this wall old enough to be trusted?
    pub fn is_reliable(&self, tracker_start: Instant) -> bool {
        // Tracker must have been running for at least WALL_RELIABLE_AGE_SECS
        let tracker_age = Instant::now()
            .checked_duration_since(tracker_start)
            .unwrap_or_default()
            .as_secs();
        if tracker_age < WALL_RELIABLE_AGE_SECS {
            return false; // Still warming up
        }
        self.age_secs() >= WALL_RELIABLE_AGE_SECS
    }

    /// How much of the wall has been eaten (0.0 = full, 1.0 = fully eaten)
    pub fn eaten_pct(&self) -> f64 {
        if self.max_size_usd <= 0.0 { return 0.0; }
        1.0 - (self.current_size_usd / self.max_size_usd)
    }
}

/// A cluster of walls close together (cascade)
#[derive(Debug, Clone)]
pub struct CascadeCluster {
    pub walls: Vec<WallInfo>,
    pub total_size_usd: f64,
    pub thickness: usize,       // number of levels
    pub bottom_price: f64,
    pub top_price: f64,
    pub side: WallSide,
}

impl CascadeCluster {
    /// Average eaten percentage across all walls in the cascade
    pub fn avg_eaten_pct(&self) -> f64 {
        if self.walls.is_empty() { return 0.0; }
        let sum: f64 = self.walls.iter().map(|w| w.eaten_pct()).sum();
        sum / self.walls.len() as f64
    }
}

/// Public snapshot of all walls for one symbol
#[derive(Debug, Clone)]
pub struct WallSnapshot {
    pub walls: Vec<WallInfo>,
    pub cascades: Vec<CascadeCluster>,
    pub is_warming_up: bool,
    pub wall_threshold_usd: f64,  // current dynamic threshold
}

impl WallSnapshot {
    /// Get reliable walls only
    pub fn reliable_walls(&self, tracker_start: Instant) -> Vec<&WallInfo> {
        self.walls.iter().filter(|w| w.is_reliable(tracker_start)).collect()
    }

    /// Get ask walls (resistance) sorted by price ascending
    pub fn ask_walls(&self) -> Vec<&WallInfo> {
        let mut walls: Vec<&WallInfo> = self.walls.iter()
            .filter(|w| w.side == WallSide::Ask)
            .collect();
        walls.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
        walls
    }

    /// Get bid walls (support) sorted by price descending
    pub fn bid_walls(&self) -> Vec<&WallInfo> {
        let mut walls: Vec<&WallInfo> = self.walls.iter()
            .filter(|w| w.side == WallSide::Bid)
            .collect();
        walls.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
        walls
    }

    /// Get all walls within a price range
    pub fn nearby_walls(&self, price: f64, tolerance_pct: f64) -> Vec<&WallInfo> {
        self.walls.iter()
            .filter(|w| {
                let dist = (w.price - price).abs() / price;
                dist <= tolerance_pct
            })
            .collect()
    }

    /// Phase 29C+2: Find the nearest wall AHEAD of current price in trade direction.
    /// For LONG: find nearest Ask wall above price (resistance we're heading toward).
    /// For SHORT: find nearest Bid wall below price (support we're heading toward).
    /// Skips spoofed walls. Returns None if no valid wall in [min_pct, max_pct] range.
    pub fn find_wall_ahead(&self, price: f64, is_long: bool, min_pct: f64, max_pct: f64) -> Option<&WallInfo> {
        let candidates: Vec<&WallInfo> = if is_long {
            // LONG: look for Ask walls above us
            self.walls.iter()
                .filter(|w| w.side == WallSide::Ask && w.price > price && !w.is_spoof)
                .collect()
        } else {
            // SHORT: look for Bid walls below us
            self.walls.iter()
                .filter(|w| w.side == WallSide::Bid && w.price < price && !w.is_spoof)
                .collect()
        };

        // Find nearest wall in the valid distance range
        candidates.iter()
            .filter(|w| {
                let dist_pct = (w.price - price).abs() / price;
                dist_pct >= min_pct && dist_pct <= max_pct
            })
            .min_by(|a, b| {
                let da = (a.price - price).abs();
                let db = (b.price - price).abs();
                da.partial_cmp(&db).unwrap()
            })
            .copied()
    }

    /// Phase 29C+2: Find iceberg walls (refill_count >= threshold)
    pub fn iceberg_walls(&self) -> Vec<&WallInfo> {
        self.walls.iter().filter(|w| w.is_iceberg).collect()
    }
}

/// Shared wall tracker store (lock-free reads from strategies)
pub type WallStore = Arc<DashMap<String, WallSnapshot>>;

/// Create a new empty wall store
pub fn new_store() -> WallStore {
    Arc::new(DashMap::new())
}

// ── Internal State ──────────────────────────────────────────────────────────

/// per-symbol tracker for known walls
struct SymbolTracker {
    walls: HashMap<u64, WallInfo>,  // key = price_key (price * 100 as u64)
    hourly_volumes: Vec<(Instant, f64)>,  // rolling 1h volume samples
}

impl SymbolTracker {
    fn new() -> Self {
        Self {
            walls: HashMap::new(),
            hourly_volumes: Vec::new(),
        }
    }

    fn price_key(price: f64) -> u64 {
        (price * 100.0) as u64
    }

    /// Estimate the dynamic wall threshold based on recent volume
    fn wall_threshold(&self) -> f64 {
        if self.hourly_volumes.is_empty() {
            return WALL_MIN_USD_FALLBACK;
        }
        let cutoff = Instant::now().checked_sub(Duration::from_secs(3600)).unwrap_or_else(Instant::now);
        
        let mut count: f64 = 0.0;
        let sum_vol: f64 = self.hourly_volumes.iter()
            .filter(|(t, _)| *t > cutoff)
            .map(|(_, v)| { count += 1.0; *v })
            .sum();
            
        let avg_vol = sum_vol / count.max(1.0);
        let threshold = avg_vol * WALL_VOLUME_FRACTION;
        
        threshold.max(WALL_MIN_USD_FALLBACK)
    }
}

// ── Main Loop ───────────────────────────────────────────────────────────────

/// Run the wall tracker as an independent background task.
/// Scans OrderBookStore every 500ms and updates WallStore.
pub async fn run(ob_store: OrderBookStore, wall_store: WallStore) {
    let tracker_start = Instant::now();
    let mut trackers: HashMap<String, SymbolTracker> = HashMap::new();
    let mut scan_count: u64 = 0;
    let mut last_log = Instant::now();
    let mut last_persist = Instant::now();

    // Load persisted wall state from disk (survives restarts)
    let loaded = load_persisted_walls(&mut trackers);
    if loaded > 0 {
        info!("🧱 WallTracker: Restored {} walls from disk (data preserved across restart)", loaded);
    }

    info!("🧱 WallTracker started. Warming up for {}h before signals are reliable.",
        WALL_RELIABLE_AGE_SECS / 3600);

    loop {
        tokio::time::sleep(Duration::from_millis(SCAN_INTERVAL_MS)).await;
        scan_count += 1;

        let is_warming_up = Instant::now()
            .checked_duration_since(tracker_start)
            .unwrap_or_default()
            .as_secs() < WALL_RELIABLE_AGE_SECS;

        // Scan each symbol's order book
        for entry in ob_store.iter() {
            let symbol = entry.key().clone();
            let book = entry.value();

            let tracker = trackers.entry(symbol.clone())
                .or_insert_with(SymbolTracker::new);

            let threshold = tracker.wall_threshold();

            // Record volume sample (from top levels as approximation)
            let top_bid_vol: f64 = book.bids.iter().take(20)
                .map(|l| l.price * l.quantity).sum();
            let top_ask_vol: f64 = book.asks.iter().take(20)
                .map(|l| l.price * l.quantity).sum();
            tracker.hourly_volumes.push((Instant::now(), top_bid_vol + top_ask_vol));
            // Trim old volume samples (keep 1h)
            let vol_cutoff = Instant::now().checked_sub(Duration::from_secs(3600)).unwrap_or_else(Instant::now);
            tracker.hourly_volumes.retain(|(t, _)| *t > vol_cutoff);

            // Mark all existing walls as "not seen this scan"
            let mut seen_keys: Vec<u64> = Vec::new();

            // Scan bid walls (support)
            scan_levels(&book.bids, WallSide::Bid, threshold, tracker, &mut seen_keys);

            // Scan ask walls (resistance)
            scan_levels(&book.asks, WallSide::Ask, threshold, tracker, &mut seen_keys);

            // Remove walls that disappeared (spoofed or filled)
            tracker.walls.retain(|k, _| seen_keys.contains(k));

            // Build cascades
            let cascades = build_cascades(&tracker.walls);

            // Publish snapshot
            let snapshot = WallSnapshot {
                walls: tracker.walls.values().cloned().collect(),
                cascades,
                is_warming_up,
                wall_threshold_usd: threshold,
            };

            wall_store.insert(symbol.clone(), snapshot);
        }

        // Periodic logging (every 60s)
        if Instant::now().checked_duration_since(last_log).unwrap_or_default() > Duration::from_secs(60) {
            last_log = Instant::now();
            let warmup_str = if is_warming_up {
                let tracker_age = Instant::now().checked_duration_since(tracker_start).unwrap_or_default().as_secs();
                let remaining = WALL_RELIABLE_AGE_SECS.saturating_sub(tracker_age.min(WALL_RELIABLE_AGE_SECS));
                format!("WARMING UP ({:.0}m left)", remaining as f64 / 60.0)
            } else {
                "RELIABLE".to_string()
            };

            let total_walls: usize = wall_store.iter()
                .map(|e| e.value().walls.len()).sum();
            let total_cascades: usize = wall_store.iter()
                .map(|e| e.value().cascades.len()).sum();

            info!("🧱 WallTracker: {} walls, {} cascades across {} symbols | {} | scans: {}",
                total_walls, total_cascades, wall_store.len(), warmup_str, scan_count);

            // Log top walls per symbol (only if there are walls)
            for entry in wall_store.iter() {
                let sym = entry.key();
                let snap = entry.value();
                if !snap.walls.is_empty() {
                    let best_wall = snap.walls.iter()
                        .max_by(|a, b| a.current_size_usd.partial_cmp(&b.current_size_usd).unwrap());
                    if let Some(w) = best_wall {
                        tracing::debug!("   🧱 [{}] biggest: {:?}@{:.2} ${:.0}k age:{:.1}h refresh:{}x eaten:{:.0}%",
                            sym, w.side, w.price, w.current_size_usd / 1000.0,
                            w.age_hours(), w.refresh_count, w.eaten_pct() * 100.0);
                    }
                }
            }
        }

        // Persist to disk every 5 minutes
        if Instant::now().checked_duration_since(last_persist).unwrap_or_default() > Duration::from_secs(PERSIST_INTERVAL_SECS) {
            last_persist = Instant::now();
            save_persisted_walls(&trackers);
        }
    }
}

/// Scan a set of price levels and update/create walls
fn scan_levels(
    levels: &[Level],
    side: WallSide,
    threshold: f64,
    tracker: &mut SymbolTracker,
    seen_keys: &mut Vec<u64>,
) {
    for level in levels {
        let size_usd = level.price * level.quantity;
        if size_usd < threshold {
            continue;
        }

        let key = SymbolTracker::price_key(level.price);
        seen_keys.push(key);

        if let Some(wall) = tracker.walls.get_mut(&key) {
            // Existing wall — update
            wall.prev_size_usd = wall.current_size_usd;
            wall.current_size_usd = size_usd;

            if size_usd > wall.max_size_usd {
                wall.max_size_usd = size_usd;
            }

            // Detect refresh: size increased after being eaten >30%
            if wall.prev_size_usd > 0.0 {
                let was_eaten = wall.prev_size_usd < wall.max_size_usd * 0.7;
                let recovered = size_usd > wall.prev_size_usd * 1.2;
                if was_eaten && recovered {
                    wall.refresh_count += 1;
                }
            }

            // Phase 29C+2: Update stability (this wall was seen in this scan)
            wall.presence_history.push_back(true);
            if wall.presence_history.len() > STABILITY_HISTORY_LEN {
                wall.presence_history.pop_front();
            }
            let present_count = wall.presence_history.iter().filter(|&&p| p).count();
            wall.stability = present_count as f64 / wall.presence_history.len() as f64;

            // Update spoof/iceberg flags
            wall.is_spoof = wall.stability < SPOOF_STABILITY_THRESHOLD;
            wall.is_iceberg = wall.refresh_count >= ICEBERG_REFILL_THRESHOLD;
        } else {
            // New wall discovered
            let mut presence = VecDeque::with_capacity(STABILITY_HISTORY_LEN);
            presence.push_back(true);
            tracker.walls.insert(key, WallInfo {
                price: level.price,
                side,
                first_seen: Instant::now(),
                max_size_usd: size_usd,
                current_size_usd: size_usd,
                refresh_count: 0,
                touch_count: 0,
                last_touch: None,
                presence_history: presence,
                stability: 1.0,
                prev_size_usd: 0.0,
                is_iceberg: false,
                is_spoof: false,
            });
        }
    }
}

/// Group nearby walls into cascade clusters
fn build_cascades(walls: &HashMap<u64, WallInfo>) -> Vec<CascadeCluster> {
    if walls.is_empty() {
        return Vec::new();
    }

    // Separate by side
    let mut ask_walls: Vec<&WallInfo> = walls.values()
        .filter(|w| w.side == WallSide::Ask).collect();
    let mut bid_walls: Vec<&WallInfo> = walls.values()
        .filter(|w| w.side == WallSide::Bid).collect();

    ask_walls.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
    bid_walls.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());

    let mut cascades = Vec::new();

    // Build cascades for each side
    for (sorted_walls, side) in [(&ask_walls, WallSide::Ask), (&bid_walls, WallSide::Bid)] {
        if sorted_walls.is_empty() { continue; }

        let mut cluster: Vec<WallInfo> = vec![(*sorted_walls[0]).clone()];

        for i in 1..sorted_walls.len() {
            let prev_price = sorted_walls[i - 1].price;
            let curr_price = sorted_walls[i].price;
            let gap = (curr_price - prev_price).abs() / prev_price;

            if gap <= CASCADE_GAP_PCT {
                // Close enough — add to current cluster
                cluster.push((*sorted_walls[i]).clone());
            } else {
                // Gap too large — finalize current cluster if it has 2+ walls
                if cluster.len() >= 2 {
                    cascades.push(finalize_cluster(cluster, side));
                }
                cluster = vec![(*sorted_walls[i]).clone()];
            }
        }

        // Don't forget the last cluster
        if cluster.len() >= 2 {
            cascades.push(finalize_cluster(cluster, side));
        }
    }

    cascades
}

fn finalize_cluster(walls: Vec<WallInfo>, side: WallSide) -> CascadeCluster {
    let total_size_usd: f64 = walls.iter().map(|w| w.current_size_usd).sum();
    let bottom = walls.iter().map(|w| w.price).fold(f64::MAX, f64::min);
    let top = walls.iter().map(|w| w.price).fold(f64::MIN, f64::max);
    let thickness = walls.len();

    CascadeCluster {
        walls,
        total_size_usd,
        thickness,
        bottom_price: bottom,
        top_price: top,
        side,
    }
}

// ── Persistence ─────────────────────────────────────────────────────────────

/// Convert Instant to approximate Unix timestamp
fn instant_to_unix(instant: Instant) -> u64 {
    let elapsed = Instant::now()
        .checked_duration_since(instant)
        .unwrap_or_default();
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    now_unix.saturating_sub(elapsed.as_secs())
}

/// Convert Unix timestamp to Instant (approximate, relative to now)
fn unix_to_instant(unix_ts: u64) -> Instant {
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    // Protect against future timestamps or overflow
    if unix_ts >= now_unix {
        return Instant::now();
    }
    
    let age_secs = now_unix.saturating_sub(unix_ts);
    Instant::now().checked_sub(Duration::from_secs(age_secs)).unwrap_or_else(Instant::now)
}

/// Save all tracked walls to disk
fn save_persisted_walls(trackers: &HashMap<String, SymbolTracker>) {
    let mut persisted: Vec<PersistedWall> = Vec::new();

    for (symbol, tracker) in trackers {
        for wall in tracker.walls.values() {
            persisted.push(PersistedWall {
                symbol: symbol.clone(),
                price: wall.price,
                side: wall.side,
                first_seen_unix: instant_to_unix(wall.first_seen),
                max_size_usd: wall.max_size_usd,
                refresh_count: wall.refresh_count,
                touch_count: wall.touch_count,
            });
        }
    }

    match serde_json::to_string(&persisted) {
        Ok(json) => {
            if let Err(e) = std::fs::write(PERSIST_PATH, &json) {
                warn!("🧱 Failed to save wall state: {}", e);
            } else {
                info!("🧱 Persisted {} walls to {}", persisted.len(), PERSIST_PATH);
            }
        }
        Err(e) => warn!("🧱 Failed to serialize wall state: {}", e),
    }
}

/// Load persisted walls from disk and restore into trackers
fn load_persisted_walls(trackers: &mut HashMap<String, SymbolTracker>) -> usize {
    let path = std::path::Path::new(PERSIST_PATH);
    if !path.exists() {
        return 0;
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            warn!("🧱 Failed to read wall state: {}", e);
            return 0;
        }
    };

    let persisted: Vec<PersistedWall> = match serde_json::from_str(&content) {
        Ok(p) => p,
        Err(e) => {
            warn!("🧱 Failed to parse wall state: {}", e);
            return 0;
        }
    };

    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut count = 0;
    for pw in &persisted {
        let tracker = trackers.entry(pw.symbol.clone())
            .or_insert_with(SymbolTracker::new);

        let key = SymbolTracker::price_key(pw.price);
        tracker.walls.insert(key, WallInfo {
            price: pw.price,
            side: pw.side,
            first_seen: unix_to_instant(pw.first_seen_unix),
            max_size_usd: pw.max_size_usd,
            current_size_usd: 0.0,  // will be updated on next scan
            refresh_count: pw.refresh_count,
            touch_count: pw.touch_count,
            last_touch: None,
            presence_history: {
                let mut h = std::collections::VecDeque::with_capacity(20);
                h.push_back(true);
                h
            },
            stability: 1.0,
            prev_size_usd: 0.0,
            is_iceberg: pw.refresh_count >= 3,
            is_spoof: false,
        });
        count += 1;
    }

    count
}

// ── Depth Scanner WebSocket ─────────────────────────────────────────────────

const WS_BASE_URL: &str = "wss://fstream.binance.com/stream?streams=";
const MAX_STREAMS_PER_WS: usize = 20; // Lower chunk size so URLs don't exceed 2048 bytes
const DEPTH_RECONNECT_BASE_SECS: u64 = 5;
const DEPTH_MAX_RECONNECT_SECS: u64 = 120;
const DEPTH_MAX_CONN_LIFE_SECS: u64 = 23 * 3600; // 23h forced reconnect
const DEPTH_HEALTH_TIMEOUT_SECS: u64 = 300;

#[derive(Deserialize)]
struct DepthWsWrapper {
    stream: String,
    data: serde_json::Value,
}

/// Run a dedicated depth-only WebSocket for all scanner symbols.
/// Reads symbol list from `data/top_symbols.json` and subscribes to @depth@100ms.
/// Feeds depth updates into the shared OrderBookStore so WallTracker can scan them.
pub async fn run_depth_scanner(ob_store: OrderBookStore) {
    // Load all 100 symbols from top_symbols.json
    let symbols = load_scanner_symbols();
    if symbols.is_empty() {
        warn!("🧱 DepthScanner: No symbols loaded from data/top_symbols.json. Disabled.");
        // Do not return, as that would cancel tokio::select! in main.rs
        loop { tokio::time::sleep(Duration::from_secs(3600)).await; }
    }

    info!("🧱 DepthScanner: Loaded {} symbols for full-market depth scanning", symbols.len());

    // Split into chunks if needed (Binance max 200 streams per ws)
    let chunks: Vec<Vec<String>> = symbols
        .chunks(MAX_STREAMS_PER_WS)
        .map(|c| c.to_vec())
        .collect();

    // Spawn one WebSocket per chunk
    let mut handles = Vec::new();
    for (i, chunk) in chunks.into_iter().enumerate() {
        let ob = ob_store.clone();
        let handle = tokio::spawn(async move {
            run_depth_ws(chunk, ob, i).await;
        });
        handles.push(handle);
    }

    // Wait for all (runs forever)
    for h in handles {
        let _ = h.await;
    }
}

/// Single depth WebSocket connection with reconnection logic
async fn run_depth_ws(symbols: Vec<String>, ob_store: OrderBookStore, ws_id: usize) {
    let mut attempt: u32 = 0;

    loop {
        if attempt > 0 {
            let base = DEPTH_RECONNECT_BASE_SECS * 2u64.pow(attempt.min(6) - 1);
            let jitter = rand::random::<u64>() % 3;
            let delay = base.min(DEPTH_MAX_RECONNECT_SECS) + jitter;
            info!("🧱 DepthScanner[{}] reconnecting in {}s (attempt #{})...", ws_id, delay, attempt);
            tokio::time::sleep(Duration::from_secs(delay)).await;
        }

        // Build URL: symbol1@depth@100ms/symbol2@depth@100ms/...
        let streams: Vec<String> = symbols.iter()
            .map(|s| format!("{}@depth@100ms", s.to_lowercase().replace("/", "").replace("_", "")))
            .collect();
        let url = format!("{}{}", WS_BASE_URL, streams.join("/"));

        info!("🧱 DepthScanner[{}] connecting ({} streams)...", ws_id, symbols.len());

        match connect_async(&url).await {
            Ok((ws_stream, _)) => {
                attempt = 0;
                info!("🧱 DepthScanner[{}] connected! Streaming depth for {} symbols.", ws_id, symbols.len());

                let conn_start = Instant::now();
                let mut last_msg = Instant::now();
                let (mut _write, mut read) = ws_stream.split();

                loop {
                    let silence = Instant::now().checked_duration_since(last_msg).unwrap_or_default().as_secs();
                    let age = Instant::now().checked_duration_since(conn_start).unwrap_or_default().as_secs();

                    if silence > DEPTH_HEALTH_TIMEOUT_SECS {
                        warn!("🧱 DepthScanner[{}] STALE: no data for {}s", ws_id, silence);
                        break;
                    }
                    if age > DEPTH_MAX_CONN_LIFE_SECS {
                        info!("🧱 DepthScanner[{}] 23h reconnect", ws_id);
                        break;
                    }

                    let msg = tokio::time::timeout(
                        Duration::from_secs(30),
                        read.next(),
                    ).await;

                    match msg {
                        Ok(Some(Ok(Message::Text(text)))) => {
                            last_msg = Instant::now();
                            // Parse and apply depth update
                            if let Ok(wrapper) = serde_json::from_str::<DepthWsWrapper>(&text) {
                                if wrapper.stream.contains("depth") {
                                    order_book::apply_depth_update(&ob_store, &wrapper.data);
                                }
                            }
                        }
                        Ok(Some(Ok(Message::Ping(data)))) => {
                            last_msg = Instant::now();
                            if let Err(e) = _write.send(Message::Pong(data)).await {
                                warn!("🧱 DepthScanner[{}] pong failed: {}", ws_id, e);
                                break;
                            }
                        }
                        Ok(Some(Ok(Message::Close(_)))) => {
                            info!("🧱 DepthScanner[{}] closed by server", ws_id);
                            break;
                        }
                        Ok(Some(Err(e))) => {
                            warn!("🧱 DepthScanner[{}] error: {}", ws_id, e);
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => continue, // Timeout, keep going
                        _ => continue,
                    }
                }
            }
            Err(e) => {
                error!("🧱 DepthScanner[{}] connection failed: {}", ws_id, e);
            }
        }

        attempt += 1;
    }
}

/// Load scanner symbols from data/top_symbols.json
fn load_scanner_symbols() -> Vec<String> {
    let path = std::path::Path::new("data/top_symbols.json");
    if !path.exists() {
        warn!("🧱 data/top_symbols.json not found. Run download_historical.py first.");
        return Vec::new();
    }
    match std::fs::read_to_string(path) {
        Ok(content) => {
            match serde_json::from_str::<Vec<String>>(&content) {
                Ok(symbols) => symbols,
                Err(e) => {
                    error!("🧱 Failed to parse top_symbols.json: {}", e);
                    Vec::new()
                }
            }
        }
        Err(e) => {
            error!("🧱 Failed to read top_symbols.json: {}", e);
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wall_info_age() {
        let wall = WallInfo {
            price: 100.0,
            side: WallSide::Ask,
            first_seen: Instant::now().checked_sub(Duration::from_secs(7200)).unwrap_or_else(Instant::now), // 2 hours ago
            max_size_usd: 100_000.0,
            current_size_usd: 80_000.0,
            refresh_count: 3,
            touch_count: 0,
            last_touch: None,
        };
        assert!(wall.age_secs() >= 7199);
        assert!((wall.eaten_pct() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_wall_reliable() {
        let old_start = Instant::now().checked_sub(Duration::from_secs(4 * 3600)).unwrap_or_else(Instant::now); // 4h ago
        let new_start = Instant::now().checked_sub(Duration::from_secs(1 * 3600)).unwrap_or_else(Instant::now); // 1h ago

        let wall = WallInfo {
            price: 100.0,
            side: WallSide::Bid,
            first_seen: Instant::now().checked_sub(Duration::from_secs(4 * 3600)).unwrap_or_else(Instant::now),
            max_size_usd: 50_000.0,
            current_size_usd: 50_000.0,
            refresh_count: 0,
            touch_count: 0,
            last_touch: None,
        };

        assert!(wall.is_reliable(old_start), "Old wall + old tracker = reliable");
        assert!(!wall.is_reliable(new_start), "Old wall + new tracker = unreliable (warming up)");
    }

    #[test]
    fn test_cascade_grouping() {
        let mut walls = HashMap::new();

        // 3 walls within 0.5% = cascade
        walls.insert(10000, WallInfo {
            price: 100.00, side: WallSide::Ask,
            first_seen: Instant::now(), max_size_usd: 60_000.0,
            current_size_usd: 60_000.0, refresh_count: 0,
            touch_count: 0, last_touch: None,
        });
        walls.insert(10010, WallInfo {
            price: 100.10, side: WallSide::Ask,
            first_seen: Instant::now(), max_size_usd: 80_000.0,
            current_size_usd: 80_000.0, refresh_count: 0,
            touch_count: 0, last_touch: None,
        });
        walls.insert(10040, WallInfo {
            price: 100.40, side: WallSide::Ask,
            first_seen: Instant::now(), max_size_usd: 70_000.0,
            current_size_usd: 70_000.0, refresh_count: 0,
            touch_count: 0, last_touch: None,
        });

        // 1 wall far away = NOT in cascade
        walls.insert(11400, WallInfo {
            price: 114.00, side: WallSide::Ask,
            first_seen: Instant::now(), max_size_usd: 200_000.0,
            current_size_usd: 200_000.0, refresh_count: 0,
            touch_count: 0, last_touch: None,
        });

        let cascades = build_cascades(&walls);
        assert_eq!(cascades.len(), 1, "Should have 1 cascade");
        assert_eq!(cascades[0].thickness, 3, "Cascade should have 3 walls");
        assert!((cascades[0].total_size_usd - 210_000.0).abs() < 1.0);
    }
}
