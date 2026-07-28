///! Liquidation Radar — @forceOrder WebSocket Feed + Velocity Aggregator
///! =====================================================================
///! Phase 13: Connects to Binance `!forceOrder@arr` stream to detect
///! liquidation cascades in real-time. Provides:
///!   - Per-symbol and aggregate liquidation tracking
///!   - Velocity Z-Score for cascade detection (>3σ = cascade)
///!   - Side ratio analysis (long_liq vs short_liq)
///!   - Systemic vs isolated cascade classification
///!   - Two-level WARN output for Orchestrator:
///!     → Trending strategies (SMC/ORB/ScalpMTF): 5min WARN cooldown
///!     → Reversal strategies (KnifeCatcher): PREPARE bonus when cascade ENDS

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use futures_util::StreamExt;
use serde::Deserialize;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};

// ── Configuration ───────────────────────────────────────────────────────────

/// Z-Score threshold for cascade detection
const CASCADE_ZSCORE_THRESHOLD: f64 = 3.0;

/// Rolling window for velocity calculation (seconds)
const VELOCITY_WINDOW_SECS: u64 = 5;

/// Rolling history for Z-Score baseline calculation (keep 24h of 5s buckets)
const HISTORY_BUCKETS: usize = 24 * 3600 / 5; // 17,280 buckets

/// WARN cooldown for trending strategies after cascade ends (seconds)
const CASCADE_COOLDOWN_SECS: u64 = 300; // 5 minutes

/// Minimum cascade size in USD to trigger alerts
const MIN_CASCADE_USD: f64 = 500_000.0;

/// Number of distinct symbols needed for "systemic" classification
const SYSTEMIC_MIN_SYMBOLS: usize = 3;

// ── Data Types ──────────────────────────────────────────────────────────────

/// A single liquidation event from Binance
#[derive(Debug, Clone)]
pub struct LiquidationEvent {
    pub symbol: String,
    pub side: LiqSide,      // Which side got liquidated
    pub price: f64,
    pub quantity_usd: f64,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiqSide {
    Long,   // Long position was liquidated (bearish signal)
    Short,  // Short position was liquidated (bullish signal)
}

/// Aggregate state visible to the Orchestrator
#[derive(Debug, Clone)]
pub struct CascadeState {
    /// Is a cascade currently active?
    pub is_cascade: bool,
    /// Current velocity Z-Score
    pub velocity_zscore: f64,
    /// Total USD liquidated in current window
    pub window_total_usd: f64,
    /// Ratio of long liquidations to short (>1 = longs getting rekt)
    pub long_short_ratio: f64,
    /// Is this a systemic cascade (BTC + multiple alts)?
    pub is_systemic: bool,
    /// Symbols involved in current cascade
    pub cascade_symbols: Vec<String>,
    /// When the last cascade ended (for WARN cooldown)
    pub last_cascade_ended: Option<Instant>,
    /// Is WARN cooldown active for trending strategies?
    pub trending_warn_active: bool,
    /// Is PREPARE mode active for KnifeCatcher?
    pub knife_prepare_active: bool,
}

impl Default for CascadeState {
    fn default() -> Self {
        Self {
            is_cascade: false,
            velocity_zscore: 0.0,
            window_total_usd: 0.0,
            long_short_ratio: 1.0,
            is_systemic: false,
            cascade_symbols: Vec::new(),
            last_cascade_ended: None,
            trending_warn_active: false,
            knife_prepare_active: false,
        }
    }
}

/// Thread-safe store for cascade state
pub type LiqStore = Arc<DashMap<String, CascadeState>>;

pub fn new_store() -> LiqStore {
    let store = Arc::new(DashMap::new());
    store.insert("__global__".to_string(), CascadeState::default());
    store
}

// ── Aggregator ──────────────────────────────────────────────────────────────

struct LiqAggregator {
    /// Recent events in the velocity window
    recent_events: VecDeque<LiquidationEvent>,
    /// Historical 5s bucket totals for Z-Score
    history_buckets: VecDeque<f64>,
    /// Running sum/count for mean/std
    history_sum: f64,
    history_sq_sum: f64,
    history_count: usize,
    /// Current 5s bucket accumulator
    current_bucket_usd: f64,
    current_bucket_start: Instant,
    /// Cascade tracking
    cascade_active: bool,
    cascade_start: Option<Instant>,
    cascade_symbols: std::collections::HashSet<String>,
    cascade_long_usd: f64,
    cascade_short_usd: f64,
    /// Output store
    store: LiqStore,
}

impl LiqAggregator {
    fn new(store: LiqStore) -> Self {
        Self {
            recent_events: VecDeque::with_capacity(1000),
            history_buckets: VecDeque::with_capacity(HISTORY_BUCKETS),
            history_sum: 0.0,
            history_sq_sum: 0.0,
            history_count: 0,
            current_bucket_usd: 0.0,
            current_bucket_start: Instant::now(),
            cascade_active: false,
            cascade_start: None,
            cascade_symbols: std::collections::HashSet::new(),
            cascade_long_usd: 0.0,
            cascade_short_usd: 0.0,
            store,
        }
    }

    fn process_event(&mut self, event: LiquidationEvent) {
        let now = Instant::now();

        // Add to recent events
        self.recent_events.push_back(event.clone());

        // Accumulate current bucket
        self.current_bucket_usd += event.quantity_usd;

        // Track side
        match event.side {
            LiqSide::Long => self.cascade_long_usd += event.quantity_usd,
            LiqSide::Short => self.cascade_short_usd += event.quantity_usd,
        }

        // Track symbol
        self.cascade_symbols.insert(event.symbol.clone());

        // Flush bucket every 5 seconds
        if now.duration_since(self.current_bucket_start).as_secs() >= VELOCITY_WINDOW_SECS {
            self.flush_bucket();
        }

        // Prune old events (keep last 30s)
        while let Some(front) = self.recent_events.front() {
            let age_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64 - front.timestamp_ms;
            if age_ms > 30_000 {
                self.recent_events.pop_front();
            } else {
                break;
            }
        }

        // Calculate velocity (USD per second in current window)
        let window_total: f64 = self.recent_events.iter()
            .filter(|e| {
                let age = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64 - e.timestamp_ms;
                age <= (VELOCITY_WINDOW_SECS * 1000)
            })
            .map(|e| e.quantity_usd)
            .sum();

        // Calculate Z-Score
        let zscore = self.calculate_zscore(window_total);

        // Cascade detection
        let was_cascade = self.cascade_active;
        self.cascade_active = zscore > CASCADE_ZSCORE_THRESHOLD && window_total > MIN_CASCADE_USD;

        // Cascade started
        if self.cascade_active && !was_cascade {
            self.cascade_start = Some(now);
            info!("🌊 LIQUIDATION CASCADE STARTED! Z={:.1} total=${:.0}k",
                zscore, window_total / 1000.0);
        }

        // Cascade ended
        if !self.cascade_active && was_cascade {
            let duration = self.cascade_start.map(|s| now.duration_since(s).as_secs()).unwrap_or(0);
            let ratio = if self.cascade_short_usd > 0.0 {
                self.cascade_long_usd / self.cascade_short_usd
            } else {
                f64::MAX
            };
            let is_systemic = self.cascade_symbols.len() >= SYSTEMIC_MIN_SYMBOLS
                && self.cascade_symbols.contains("BTC/USDT");

            info!("🌊 CASCADE ENDED: duration={}s, L/S ratio={:.1}, systemic={}, symbols={:?}",
                duration, ratio, is_systemic, self.cascade_symbols);

            // Update store with cascade-ended state
            if let Some(mut state) = self.store.get_mut("__global__") {
                state.last_cascade_ended = Some(now);
                state.trending_warn_active = true;
                state.knife_prepare_active = true;
            }

            // Reset cascade counters
            self.cascade_symbols.clear();
            self.cascade_long_usd = 0.0;
            self.cascade_short_usd = 0.0;
        }

        // Update global store
        let long_short_ratio = if self.cascade_short_usd > 0.0 {
            self.cascade_long_usd / self.cascade_short_usd
        } else if self.cascade_long_usd > 0.0 {
            f64::MAX
        } else {
            1.0
        };

        if let Some(mut state) = self.store.get_mut("__global__") {
            state.is_cascade = self.cascade_active;
            state.velocity_zscore = zscore;
            state.window_total_usd = window_total;
            state.long_short_ratio = long_short_ratio;
            state.is_systemic = self.cascade_symbols.len() >= SYSTEMIC_MIN_SYMBOLS
                && self.cascade_symbols.contains("BTC/USDT");
            state.cascade_symbols = self.cascade_symbols.iter().cloned().collect();

            // Check if WARN cooldown expired
            if let Some(ended) = state.last_cascade_ended {
                let since_ended = now.duration_since(ended).as_secs();
                if since_ended > CASCADE_COOLDOWN_SECS {
                    state.trending_warn_active = false;
                }
                // KnifeCatcher PREPARE lasts only 60s after cascade end
                if since_ended > 60 {
                    state.knife_prepare_active = false;
                }
            }
        }

        // Also update per-symbol store
        if let Some(mut state) = self.store.get_mut(&event.symbol) {
            state.velocity_zscore = zscore;
            state.window_total_usd = event.quantity_usd;
        } else {
            self.store.insert(event.symbol.clone(), CascadeState {
                velocity_zscore: zscore,
                window_total_usd: event.quantity_usd,
                ..Default::default()
            });
        }
    }

    fn flush_bucket(&mut self) {
        let bucket_val = self.current_bucket_usd;

        // Add to history
        self.history_buckets.push_back(bucket_val);
        self.history_sum += bucket_val;
        self.history_sq_sum += bucket_val * bucket_val;
        self.history_count += 1;

        // Trim to max history size
        if self.history_buckets.len() > HISTORY_BUCKETS {
            if let Some(removed) = self.history_buckets.pop_front() {
                self.history_sum -= removed;
                self.history_sq_sum -= removed * removed;
                self.history_count -= 1;
            }
        }

        // Reset bucket
        self.current_bucket_usd = 0.0;
        self.current_bucket_start = Instant::now();
    }

    fn calculate_zscore(&self, current_value: f64) -> f64 {
        if self.history_count < 10 {
            return 0.0; // Not enough history for meaningful Z-Score
        }

        let mean = self.history_sum / self.history_count as f64;
        let variance = (self.history_sq_sum / self.history_count as f64) - mean * mean;
        let std_dev = variance.max(0.0).sqrt();

        if std_dev < 1.0 {
            return 0.0; // Avoid division by near-zero
        }

        (current_value - mean) / std_dev
    }
}

// ── Binance Message Parsing ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct ForceOrderWrapper {
    o: ForceOrderData,
}

#[derive(Deserialize)]
struct ForceOrderData {
    s: String,       // Symbol (e.g., "BTCUSDT")
    #[serde(rename = "S")]
    side: String,    // "SELL" = long liquidated, "BUY" = short liquidated
    p: String,       // Price
    q: String,       // Quantity
    #[serde(rename = "T")]
    trade_time: u64, // Trade time in ms
}

// ── Public API: Run the feed ────────────────────────────────────────────────

/// Start the liquidation feed WebSocket and aggregator
pub async fn run_liquidation_feed(store: LiqStore) {
    let mut aggregator = LiqAggregator::new(store);
    let mut attempt: u32 = 0;

    loop {
        if attempt > 0 {
            let delay = (5u64 * 2u64.pow(attempt.min(6) - 1)).min(120) + rand::random::<u64>() % 3;
            info!("🌊 LiqFeed reconnecting in {}s (attempt #{})...", delay, attempt);
            tokio::time::sleep(Duration::from_secs(delay)).await;
        }

        let url = "wss://fstream.binance.com/ws/!forceOrder@arr";
        info!("🌊 Connecting to Binance Liquidation Feed...");

        match connect_async(url).await {
            Ok((ws_stream, _)) => {
                attempt = 0;
                info!("🌊 ✅ Liquidation Feed connected!");

                let connection_start = Instant::now();
                let mut last_msg = Instant::now();
                let (mut _write, mut read) = ws_stream.split();

                loop {
                    // Health checks
                    if last_msg.elapsed().as_secs() > 600 {
                        warn!("🌊 STALE: No liquidations for 10min. Reconnecting...");
                        break;
                    }
                    if connection_start.elapsed().as_secs() > 23 * 3600 {
                        info!("🌊 23h limit. Reconnecting...");
                        break;
                    }

                    let msg = tokio::time::timeout(Duration::from_secs(60), read.next()).await;

                    match msg {
                        Ok(Some(Ok(Message::Text(text)))) => {
                            last_msg = Instant::now();

                            // Parse force order
                            let parsed: Result<ForceOrderWrapper, _> = serde_json::from_str(&text);
                            if let Ok(fo) = parsed {
                                let price: f64 = fo.o.p.parse().unwrap_or(0.0);
                                let qty: f64 = fo.o.q.parse().unwrap_or(0.0);
                                let side = if fo.o.side == "SELL" {
                                    LiqSide::Long  // Sell order = long got liquidated
                                } else {
                                    LiqSide::Short // Buy order = short got liquidated
                                };

                                let symbol = super::ws_feed::format_symbol(&fo.o.s);
                                let quantity_usd = price * qty;

                                aggregator.process_event(LiquidationEvent {
                                    symbol,
                                    side,
                                    price,
                                    quantity_usd,
                                    timestamp_ms: fo.o.trade_time,
                                });
                            }
                        }
                        Ok(Some(Ok(Message::Ping(data)))) => {
                            last_msg = Instant::now();
                            if let Err(e) = futures_util::SinkExt::send(&mut _write, Message::Pong(data)).await {
                                warn!("🌊 Pong failed: {}", e);
                                break;
                            }
                        }
                        Ok(Some(Ok(Message::Close(_)))) => {
                            info!("🌊 LiqFeed closed by server.");
                            break;
                        }
                        Ok(Some(Err(e))) => {
                            warn!("🌊 LiqFeed WS error: {}", e);
                            break;
                        }
                        Ok(None) => {
                            info!("🌊 LiqFeed stream ended.");
                            break;
                        }
                        Err(_) => continue, // Timeout (normal — liq events are sporadic)
                        _ => continue,
                    }
                }
            }
            Err(e) => {
                error!("🌊 LiqFeed connection failed: {}", e);
            }
        }

        attempt += 1;
    }
}
