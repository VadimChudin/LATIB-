///! HFT Absorber — Сверхчастотный Ловец Дна
///! ============================================
///! Микросервис, который включается по наводке макро-стратегий (knife.rs).
///! Отслеживает Ленту (CVD, Tape Speed, Whale Prints) и Стакан (Walls)
///! в течение 30-60 секунд, ожидая истощения сквиза.
///!
///! Балльная система:
///!   Лента (макс 70): CVD Reversal +30, Speed Drop +20, Whale Prints +20
///!   Стакан (макс 30): Fresh Wall +20, No Refresh +10
///!   Порог: score >= 50 → FIRE
///!
///! Ограничения: максимум 2 одновременных потока.
///! Приоритет: монета с наибольшей |дельтой| (самый жёсткий сквиз).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::{info, warn};

use super::position_manager::Direction;
use super::tape_reader::TapeStore;
use super::wall_tracker::{WallStore, WallSnapshot};

// ── Config ──────────────────────────────────────────────────────────────────

/// Maximum concurrent absorber tracking tasks
const MAX_CONCURRENT_TRACKS: usize = 8;

/// Polling interval for the micro-loop
const POLL_INTERVAL_MS: u64 = 50;

/// Maximum time to wait for absorption (seconds) — base value, can be extended
const TRACK_TIMEOUT_SECS: u64 = 60;

/// Maximum total tracking time (hard cap to prevent infinite loops)
const TRACK_MAX_TOTAL_SECS: u64 = 600; // 10 minutes hard cap

/// Delta threshold for reject (aggressive counter-move)
const REJECT_DELTA_THRESHOLD: f64 = 0.25;

/// Minimum confidence score to trigger entry
const FIRE_THRESHOLD: i32 = 50;

// ── Types ───────────────────────────────────────────────────────────────────

/// A target handed off by the macro strategy (e.g. knife.rs)
#[derive(Debug, Clone)]
pub struct HftTarget {
    pub symbol: String,
    pub direction: Direction,
    pub created_at: Instant,
    /// Absolute CVD delta at the moment of the signal (for prioritization)
    pub initial_delta_abs: f64,
    /// Strategy type: "knife" for reversal, "breakout" for density breakout
    pub strategy_type: String,
    /// Wall Magnet Price (Phase 29C+1)
    pub target_wall_price: Option<f64>,
}

/// Result of the absorber tracking loop
#[derive(Debug, Clone)]
pub enum AbsorberResult {
    /// Absorption detected — ready to fire
    Fired {
        symbol: String,
        direction: Direction,
        confidence: i32,
        entry_price: f64,
        target_wall_price: Option<f64>,
        is_wall_backed: bool,
    },
    /// Timeout — squeeze still going, abort
    Timeout {
        symbol: String,
    },
    /// Rejected — max concurrent slots full
    Rejected {
        symbol: String,
    },
}

/// Snapshot of tape metrics captured at the start of tracking
#[derive(Debug, Clone)]
pub struct TapeBaseline {
    pub peak_speed: f64,
    pub initial_cvd: f64,
    pub initial_delta: f64,
}

// ── Absorber Tracker ────────────────────────────────────────────────────────

/// Thread-safe absorber state shared across the engine
pub struct AbsorberTracker {
    /// Currently active tracking symbols (symbol → spawn instant)
    active_tracks: Arc<Mutex<HashMap<String, Instant>>>,
}

impl AbsorberTracker {
    pub fn new() -> Self {
        Self {
            active_tracks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Try to accept a new HftTarget. Returns false if rejected (max slots or duplicate).
    pub async fn try_accept(&self, target: &HftTarget) -> bool {
        let mut tracks = self.active_tracks.lock().await;

        // Clean up expired tracks
        tracks.retain(|_, started| started.elapsed() < Duration::from_secs(TRACK_MAX_TOTAL_SECS + 5));

        // Check duplicate
        if tracks.contains_key(&target.symbol) {
            warn!("🔪 Absorber: {} already being tracked — skipping", target.symbol);
            return false;
        }

        // Check capacity
        if tracks.len() >= MAX_CONCURRENT_TRACKS {
            warn!("🔪 Absorber: max {} concurrent tracks reached — rejecting {}",
                MAX_CONCURRENT_TRACKS, target.symbol);
            return false;
        }

        tracks.insert(target.symbol.clone(), Instant::now());
        true
    }

    /// Remove a symbol from active tracking (called when tracking finishes)
    pub async fn release(&self, symbol: &str) {
        let mut tracks = self.active_tracks.lock().await;
        tracks.remove(symbol);
    }

    /// How many slots are currently active
    pub async fn active_count(&self) -> usize {
        let tracks = self.active_tracks.lock().await;
        tracks.len()
    }
}

/// Prioritize targets: sort by |delta| descending, return top MAX_CONCURRENT_TRACKS
pub fn prioritize_targets(mut targets: Vec<HftTarget>) -> Vec<HftTarget> {
    // Sort by absolute delta (strongest squeeze first)
    targets.sort_by(|a, b| {
        b.initial_delta_abs
            .partial_cmp(&a.initial_delta_abs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    targets.truncate(MAX_CONCURRENT_TRACKS);
    targets
}

// ── Core Micro-Loop ─────────────────────────────────────────────────────────

/// The main HFT tracking function. Runs as an independent tokio::task.
/// Polls tape + walls every 50ms for up to 60 seconds.
///
/// Returns `Fired` if absorption is detected, `Timeout` if squeeze continues.
pub async fn track_absorption(
    symbol: String,
    direction: Direction,
    strategy_type: String,
    wall_store: WallStore,
    tape_store: TapeStore,
    tracker: Arc<AbsorberTracker>,
) -> AbsorberResult {
    let start = std::time::Instant::now();
    let base_timeout = std::time::Duration::from_secs(TRACK_TIMEOUT_SECS);
    let hard_cap = std::time::Duration::from_secs(TRACK_MAX_TOTAL_SECS);
    let poll = std::time::Duration::from_millis(POLL_INTERVAL_MS);
    let mut current_deadline = start + base_timeout;

    let event_id = super::hft_logger::now_ms().to_string();
    super::hft_logger::log_event_meta(&event_id, &symbol, &format!("START_{}", strategy_type));

    info!("🔪 Absorber ACTIVATED on {} ({:?}, {}) — tracking for {}s (max {}s) [ID={}]",
        symbol, direction, strategy_type, TRACK_TIMEOUT_SECS, TRACK_MAX_TOTAL_SECS, event_id);

    // Capture baseline tape metrics
    let baseline = {
        let tape = tape_store.get(&symbol);
        match tape {
            Some(state) => TapeBaseline {
                peak_speed: state.tape_speed(),
                initial_cvd: state.cvd,
                initial_delta: state.normalized_delta(),
            },
            None => {
                warn!("🔪 Absorber: no tape data for {} — aborting", symbol);
                super::hft_logger::log_event_meta(&event_id, &symbol, "ERROR_NO_TAPE");
                tracker.release(&symbol).await;
                return AbsorberResult::Timeout { symbol };
            }
        }
    };

    info!("🔪 Baseline [{}]: speed={:.1}t/s cvd={:.1} delta={:.3}",
        symbol, baseline.peak_speed, baseline.initial_cvd, baseline.initial_delta);

    let mut best_score: i32 = 0;
    let mut peak_speed_seen = baseline.peak_speed;
    let mut extensions: u32 = 0;

    // ── Micro-Loop ──────────────────────────────────────────────────────
    loop {
        // Hard cap check
        if start.elapsed() > hard_cap {
            info!("🔪 Absorber HARD CAP on {} — {} extensions, best_score={}",
                symbol, extensions, best_score);
            super::hft_logger::log_event_meta(&event_id, &symbol, "TIMEOUT_HARDCAP");
            tracker.release(&symbol).await;
            return AbsorberResult::Timeout { symbol };
        }

        // Deadline check with Alive Extension
        if Instant::now() > current_deadline {
            // Check if we should extend
            let should_extend = if strategy_type == "breakout" {
                // For breakout: extend if price is still near the wall (delta not aggressively negative)
                match tape_store.get(&symbol) {
                    Some(state) => {
                        let delta = state.normalized_delta();
                        match direction {
                            Direction::Long => delta > -0.1,  // not aggressively selling
                            Direction::Short => delta < 0.1,  // not aggressively buying
                        }
                    }
                    None => false,
                }
            } else {
                // Knife: extend if speed is still dropping (squeeze ongoing)
                match tape_store.get(&symbol) {
                    Some(state) => state.tape_speed() < peak_speed_seen * 0.7,
                    None => false,
                }
            };

            if should_extend {
                extensions += 1;
                current_deadline = Instant::now() + base_timeout;
                info!("🔪 Absorber EXTENDED on {} (ext #{}) — price still consolidating near target",
                    symbol, extensions);
            } else {
                info!("🔪 Absorber TIMEOUT on {} — {} extensions, best_score={}",
                    symbol, extensions, best_score);
                super::hft_logger::log_event_meta(&event_id, &symbol, "TIMEOUT_DEADLINE");
                tracker.release(&symbol).await;
                return AbsorberResult::Timeout { symbol };
            }
        }

        tokio::time::sleep(poll).await;

        // ── Read Tape ───────────────────────────────────────────────────
        let (current_speed, current_delta, current_cvd, _cvd_trend, 
             whale_buys, whale_sells, last_price) = {
            match tape_store.get(&symbol) {
                Some(state) => (
                    state.tape_speed(),
                    state.normalized_delta(),
                    state.cvd,
                    state.cvd_trend(),
                    state.large_prints().0,
                    state.large_prints().1,
                    state.last_price,
                ),
                None => continue,
            }
        };

        // Track peak speed for drop calculation
        if current_speed > peak_speed_seen {
            peak_speed_seen = current_speed;
        }

        // ── REJECT CHECK: aggressive counter-delta ───────────────────────
        let rejected = match direction {
            Direction::Long => current_delta < -REJECT_DELTA_THRESHOLD,
            Direction::Short => current_delta > REJECT_DELTA_THRESHOLD,
        };
        if rejected {
            info!("🔪 ❌ Absorber REJECTED {} ({:?}) — aggressive counter-delta {:.3}",
                symbol, direction, current_delta);
            super::hft_logger::log_event_meta(&event_id, &symbol, "REJECT_DELTA");
            tracker.release(&symbol).await;
            return AbsorberResult::Timeout { symbol };
        }

        // ── Calculate Confidence Score ───────────────────────────────────
        let (score, wall_metrics) = calculate_confidence_score(
            &strategy_type,
            direction,
            &baseline,
            current_speed,
            current_delta,
            whale_buys as usize,
            whale_sells as usize,
            last_price,
            wall_store.get(&symbol).as_deref(), // Using deref to get &WallSnapshot
            peak_speed_seen,
        );

        // HFT SNAPSHOT CAPTURE (Phase 15)
        let snapshot = super::hft_logger::HftSnapshot {
            event_id: event_id.clone(),
            ts_ms: super::hft_logger::now_ms(),
            symbol: symbol.clone(),
            price: last_price,
            direction: format!("{:?}", direction),
            strategy: strategy_type.clone(),
            tape_speed: current_speed,
            cvd: current_cvd,
            delta: current_delta,
            whale_buys: whale_buys as usize,
            whale_sells: whale_sells as usize,
            wall_dist_pct: wall_metrics.0,
            wall_size_usd: wall_metrics.1,
            wall_eaten_pct: wall_metrics.2,
            score,
            extensions,
            status: "Tracking".to_string(),
        };
        super::hft_logger::append_snapshot(&snapshot);

        // Track best score for logging
        if score > best_score {
            best_score = score;
            info!("🔪 Absorber [{}][{}] new best score: {} (speed={:.1} delta={:.3} whales_buy={} whales_sell={})",
                symbol, strategy_type, score, current_speed, current_delta, whale_buys, whale_sells);
        }

        // ── FIRE! ───────────────────────────────────────────────────────
        if score >= FIRE_THRESHOLD {
            info!("🔪🔥 Absorber FIRED on {} ({:?}, {}) | score={} | price={:.4} | elapsed={:.1}s | extensions={}",
                symbol, direction, strategy_type, score, last_price, start.elapsed().as_secs_f64(), extensions);
            super::hft_logger::log_event_meta(&event_id, &symbol, "FIRED");
            tracker.release(&symbol).await;
            return AbsorberResult::Fired {
                symbol,
                direction,
                confidence: score,
                entry_price: last_price,
                target_wall_price: None,
                is_wall_backed: false,
            };
        }
    }
}

/// Pure function for calculating HFT confidence score.
/// Externalized for easier unit testing and synthetic benchmarking.
pub fn calculate_confidence_score(
    strategy_type: &str,
    direction: Direction,
    baseline: &TapeBaseline,
    current_speed: f64,
    current_delta: f64,
    whale_buys: usize,
    whale_sells: usize,
    last_price: f64,
    wall_snap: Option<&WallSnapshot>,
    peak_speed_seen: f64,
) -> (i32, (f64, f64, f64)) {
    let mut score: i32 = 0;
    let mut wall_metrics = (0.0, 0.0, 0.0); // (dist_pct, size_usd, eaten_pct)

    if strategy_type == "breakout" {
        // === BREAKOUT SCORING (momentum continuation) ===
        
        // 1. Speed Acceleration (+30): tape getting faster = momentum building
        if current_speed > baseline.peak_speed * 1.5 {
            score += 30;  // Massive acceleration
        } else if current_speed > baseline.peak_speed * 1.2 {
            score += 15;  // Moderate acceleration
        }

        // 2. Delta Spike in direction (+30): strong directional flow
        match direction {
            Direction::Long => {
                let delta_improvement = current_delta - baseline.initial_delta;
                if delta_improvement > 0.2 {
                    score += 30;  // Massive buyer surge
                } else if delta_improvement > 0.1 {
                    score += 15;  // Moderate buyer pressure
                }
            }
            Direction::Short => {
                let delta_drop = baseline.initial_delta - current_delta;
                if delta_drop > 0.2 {
                    score += 30;
                } else if delta_drop > 0.1 {
                    score += 15;
                }
            }
        }

        // 3. Whale Prints in breakout direction (+20)
        match direction {
            Direction::Long => {
                if whale_buys >= 2 { score += 20; }
                else if whale_buys >= 1 { score += 10; }
            }
            Direction::Short => {
                if whale_sells >= 2 { score += 20; }
                else if whale_sells >= 1 { score += 10; }
            }
        }

        // 4. Wall being eaten (resistance crumbling) (+20)
        if let Some(snap) = wall_snap {
            if !snap.is_warming_up {
                match direction {
                    Direction::Long => {
                        for wall in snap.ask_walls().iter() {
                            let dist = (wall.price - last_price) / last_price;
                            if dist < 0.01 && dist > 0.0 {
                                wall_metrics = (dist * 100.0, wall.current_size_usd, wall.eaten_pct());
                                if wall.eaten_pct() > 0.40 {
                                    score += 20;  // Resistance wall being devoured
                                    break;
                                }
                            }
                        }
                        // Spoofer support bonus: fresh bid walls below
                        for wall in snap.bid_walls().iter() {
                            let dist = (last_price - wall.price) / last_price;
                            if dist < 0.01 && dist > 0.0 && wall.age_secs() < 300 {
                                score += 10;  // Fresh algorithmic support
                                break;
                            }
                        }
                    }
                    Direction::Short => {
                        for wall in snap.bid_walls().iter() {
                            let dist = (last_price - wall.price) / last_price;
                            if dist < 0.01 && dist > 0.0 {
                                wall_metrics = (dist * 100.0, wall.current_size_usd, wall.eaten_pct());
                                if wall.eaten_pct() > 0.40 {
                                    score += 20;
                                    break;
                                }
                            }
                        }
                        for wall in snap.ask_walls().iter() {
                            let dist = (wall.price - last_price) / last_price;
                            if dist < 0.01 && dist > 0.0 && wall.age_secs() < 300 {
                                score += 10;
                                break;
                            }
                        }
                    }
                }
            }
        }

    } else {
        // === KNIFE SCORING (reversal / exhaustion) ===

        // 1. CVD Reversal (+30): delta was extreme, now stabilizing
        match direction {
            Direction::Long => {
                let delta_improvement = current_delta - baseline.initial_delta;
                if baseline.initial_delta < -0.1 && delta_improvement > 0.15 {
                    score += 30;
                } else if baseline.initial_delta < -0.1 && delta_improvement > 0.08 {
                    score += 15;
                }
            }
            Direction::Short => {
                let delta_drop = baseline.initial_delta - current_delta;
                if baseline.initial_delta > 0.1 && delta_drop > 0.15 {
                    score += 30;
                } else if baseline.initial_delta > 0.1 && delta_drop > 0.08 {
                    score += 15;
                }
            }
        }

        // 2. Tape Speed Drop (+20): sellers/buyers drying up
        if peak_speed_seen > 5.0 {
            let speed_ratio = current_speed / peak_speed_seen;
            if speed_ratio < 0.3 {
                score += 20;
            } else if speed_ratio < 0.5 {
                score += 10;
            }
        }

        // 3. Whale Prints in reversal direction (+20)
        match direction {
            Direction::Long => {
                if whale_buys >= 2 { score += 20; }
                else if whale_buys >= 1 { score += 10; }
            }
            Direction::Short => {
                if whale_sells >= 2 { score += 20; }
                else if whale_sells >= 1 { score += 10; }
            }
        }

        // 4. Wall signals (fresh support/resistance wall)
        if let Some(snap) = wall_snap {
            if !snap.is_warming_up {
                match direction {
                    Direction::Long => {
                        let bid_walls = snap.bid_walls();
                        if let Some(wall) = bid_walls.first() {
                            let dist_pct = (last_price - wall.price) / last_price * 100.0;
                            if dist_pct < 1.0 && dist_pct > 0.0 {
                                wall_metrics = (dist_pct, wall.current_size_usd, wall.eaten_pct());
                                let eaten = wall.eaten_pct();
                                if eaten < 0.20 {
                                    score += 20;
                                    if wall.refresh_count == 0 { score += 10; }
                                } else if eaten < 0.40 {
                                    score += 10;
                                }
                            }
                        }
                    }
                    Direction::Short => {
                        let ask_walls = snap.ask_walls();
                        if let Some(wall) = ask_walls.first() {
                            let dist_pct = (wall.price - last_price) / last_price * 100.0;
                            if dist_pct < 1.0 && dist_pct > 0.0 {
                                wall_metrics = (dist_pct, wall.current_size_usd, wall.eaten_pct());
                                let eaten = wall.eaten_pct();
                                if eaten < 0.20 {
                                    score += 20;
                                    if wall.refresh_count == 0 { score += 10; }
                                } else if eaten < 0.40 {
                                    score += 10;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (score, wall_metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prioritize_targets() {
        let targets = vec![
            HftTarget {
                symbol: "BTC".into(),
                direction: Direction::Long,
                created_at: Instant::now(),
                initial_delta_abs: 0.3,
                strategy_type: "knife".into(),
            },
            HftTarget {
                symbol: "ETH".into(),
                direction: Direction::Long,
                created_at: Instant::now(),
                initial_delta_abs: 0.8,
                strategy_type: "knife".into(),
            },
            HftTarget {
                symbol: "SOL".into(),
                direction: Direction::Short,
                created_at: Instant::now(),
                initial_delta_abs: 0.5,
                strategy_type: "knife".into(),
            },
        ];

        let result = prioritize_targets(targets);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].symbol, "ETH");  // Highest delta
        assert_eq!(result[1].symbol, "SOL");  // Second highest
    }
}

/// Phase 31 v4: Exact replica of backtester knife_tick.rs 4-step entry checklist
/// ================================================================================
/// The orchestrator has already detected a squeeze (macro signal from candle backtest).
/// This absorber now waits for the micro-structure confirmation:
///
///   1. CVD DIVERGENCE: CVD improving vs squeeze start (buyers returning)
///   2. ABSORPTION: high volume but price stopped falling (vol/δprice ratio)
///   3. SPEED: tape speed dropped below baseline × max_speed_mult
///   4. RECLAIM: price bounced from local extreme by min_reclaim_pct
///
/// params[0]: window_ms          params[1]: min_zscore (unused here, already filtered by macro)
/// params[2]: min_vol_spike      params[3]: (unused_tp)
/// params[4]: sl_buffer_pct      params[5]: be_trigger_pct
/// params[6]: trail_pct          params[7]: micro_window_ms
/// params[8]: min_absorption     params[9]: min_reclaim_pct
/// params[10]: max_speed_mult
pub async fn track_knife_tick_v3(
    symbol: String,
    direction: Direction,
    strategy_type: String,
    _wall_store: WallStore,
    tape_store: TapeStore,
    tracker: Arc<AbsorberTracker>,
    params: Vec<f64>,
    target_wall_price: Option<f64>,
) -> AbsorberResult {
    // ── Parse params (same order as config_loader.rs / knife_tick.rs) ──
    let win_ms = params.get(0).copied().unwrap_or(2000.0) as u64;
    let micro_window_ms = params.get(7).copied().unwrap_or(1000.0) as i64;
    let min_absorption = params.get(8).copied().unwrap_or(2.0) as f64;
    let min_reclaim_pct = params.get(9).copied().unwrap_or(0.001) as f64;
    let max_speed_mult = params.get(10).copied().unwrap_or(3.0) as f64;
    
    // NEW Phase 29C DE Params
    let baseline_window_sec = params.get(11).copied().unwrap_or(30.0) as u64;
    let max_absorber_sec = params.get(12).copied().unwrap_or(30.0) as u64;
    // let rewake_cooldown_sec = params.get(13).copied().unwrap_or(60.0) as u64; // managed by macro trigger

    let start = tokio::time::Instant::now();
    let _timeout = std::time::Duration::from_millis(win_ms.max(5000)); // at least 5s to find bottom
    let hard_cap = std::time::Duration::from_secs(max_absorber_sec); // Using DE param [12]
    let poll = std::time::Duration::from_millis(50); // 50ms poll (20 checks/sec)

    let event_id = super::hft_logger::now_ms().to_string();
    tracing::info!(
        "🔪 [V4] track_knife_tick ACTIVATED on {} ({:?}) — micro_win={} abs={:.1} recl={:.4} spd={:.1} base={}s dur={}s",
        symbol, direction, micro_window_ms, min_absorption, min_reclaim_pct, max_speed_mult, baseline_window_sec, max_absorber_sec
    );

    // ── Capture baseline (Using DE param [11]) ──
    let (baseline_tps, baseline_absorption, squeeze_cvd, initial_price) = match tape_store.get(&symbol) {
        Some(state) => {
            let (bl_tps_raw, _bl_avg_size, _bl_flow, _bl_abs) = state.get_baseline_metrics(baseline_window_sec as i64, micro_window_ms);
            let bl_tps = bl_tps_raw.max(100.0 / 60000.0 * micro_window_ms as f64); // safety min

            // Baseline absorption = total_quote_vol / price_range
            let (_, bl_quote_vol, _, bl_high, bl_low, _bl_count, cvd) =
                state.get_micro_absorption_metrics((baseline_window_sec * 1000) as i64); // DE param [11]
            let bl_range_pct = if state.last_price > 0.0 && bl_high > bl_low {
                (bl_high - bl_low) / state.last_price
            } else {
                0.00005 // minimum
            };
            let bl_absorption = if bl_range_pct > 0.00001 {
                bl_quote_vol as f64 / bl_range_pct
            } else {
                1_000_000.0
            };

            (bl_tps, bl_absorption, cvd, state.last_price)
        }
        None => {
            tracing::warn!("🔪 [V4] no tape data for {} — aborting", symbol);
            tracker.release(&symbol).await;
            return AbsorberResult::Timeout { symbol };
        }
    };

    tracing::info!(
        "🔪 [V4] Baseline [{}]: tps={:.1}/micro_win cvd={:.1} absorption_base={:.0} price={:.4}",
        symbol, baseline_tps, squeeze_cvd, baseline_absorption, initial_price
    );

    // ── Track local extreme (lowest/highest price since activation) ──
    let mut local_extreme = initial_price;
    let mut local_extreme_time = tokio::time::Instant::now();
    let mut checks: u32 = 0;
    let mut best_score = 0_i32;

    loop {
        // Hard cap: 30 seconds from local extreme (matching backtester line 164)
        if tokio::time::Instant::now().duration_since(local_extreme_time).as_secs() > 30 {
            tracing::info!("🔪 [V4] TIMEOUT (30s from local extreme) on {} — {} checks, best_score={}", symbol, checks, best_score);
            tracker.release(&symbol).await;
            return AbsorberResult::Timeout { symbol };
        }

        // Also overall hard cap
        if start.elapsed() > hard_cap {
            tracing::info!("🔪 [V4] HARD CAP on {} — {} checks", symbol, checks);
            tracker.release(&symbol).await;
            return AbsorberResult::Timeout { symbol };
        }

        tokio::time::sleep(poll).await;
        checks += 1;

        // ── Get current tape state ──
        let (micro_tps, micro_quote_vol, micro_delta, micro_high, micro_low, micro_count, current_cvd, last_price) =
            match tape_store.get(&symbol) {
                Some(state) => {
                    let (tps, qv, delta, high, low, count, cvd) =
                        state.get_micro_absorption_metrics(micro_window_ms);
                    (tps, qv, delta, high, low, count, cvd, state.last_price)
                }
                None => continue,
            };

        // ── Update local extreme ──
        match direction {
            Direction::Long => {
                if last_price < local_extreme {
                    local_extreme = last_price;
                    local_extreme_time = tokio::time::Instant::now();
                }
            }
            Direction::Short => {
                if last_price > local_extreme {
                    local_extreme = last_price;
                    local_extreme_time = tokio::time::Instant::now();
                }
            }
        }

        // Must wait at least micro_window_ms after local extreme before checking
        let ms_since_extreme = tokio::time::Instant::now()
            .duration_since(local_extreme_time)
            .as_millis() as i64;
        if ms_since_extreme < micro_window_ms {
            continue;
        }

        // Need minimum trades in micro window
        if micro_count < 3 {
            continue;
        }

        // ═══ CHECK 1: CVD DIVERGENCE ═══
        // CVD improving vs squeeze start (buyers returning for LONG)
        let cond_cvd = match direction {
            Direction::Long => current_cvd > squeeze_cvd,
            Direction::Short => current_cvd < squeeze_cvd,
        };
        if !cond_cvd { continue; }

        // ═══ CHECK 2: ABSORPTION ═══
        // High volume but price stopped falling = absorption
        let micro_range = if last_price > 0.0 && micro_high > micro_low {
            (micro_high - micro_low) / last_price
        } else {
            0.0
        };
        let absorption_ratio = if micro_range > 0.000001 {
            micro_quote_vol / (micro_range * last_price)
        } else {
            micro_quote_vol * 100.0 // very flat = infinite absorption
        };
        let cond_absorption = absorption_ratio >= baseline_absorption * min_absorption;
        if !cond_absorption { continue; }

        // ═══ CHECK 3: SPEED ═══
        // Tape speed dropped (exhaustion) — micro_tps <= baseline_tps * max_speed_mult
        let cond_speed = micro_tps <= baseline_tps * max_speed_mult;
        if !cond_speed { continue; }

        // ═══ CHECK 4: RECLAIM ═══
        // Price bounced from local extreme by min_reclaim_pct
        let reclaim = match direction {
            Direction::Long => {
                if local_extreme > 0.0 { (last_price - local_extreme) / local_extreme } else { 0.0 }
            }
            Direction::Short => {
                if local_extreme > 0.0 { (local_extreme - last_price) / local_extreme } else { 0.0 }
            }
        };
        let cond_reclaim = reclaim >= min_reclaim_pct;
        if !cond_reclaim { continue; }

        // ═══ ALL 4 CHECKS PASSED → FIRE! ═══
        let score = 100; // all conditions met
        tracing::info!(
            "🔪🔥 [V4] ALL 4 CHECKS PASSED on {} ({:?}) | price={:.6} extreme={:.6} | \
             CVD: {:.1}→{:.1} ✅ | Absorption: {:.0} vs {:.0}×{:.1} ✅ | Speed: {:.1} vs {:.1}×{:.1} ✅ | Reclaim: {:.4}% vs {:.4}% ✅ | elapsed={:.0}ms",
            symbol, direction, last_price, local_extreme,
            squeeze_cvd, current_cvd,
            absorption_ratio, baseline_absorption, min_absorption,
            micro_tps, baseline_tps, max_speed_mult,
            reclaim * 100.0, min_reclaim_pct * 100.0,
            start.elapsed().as_millis()
        );

        // HFT LOGGING
        let snapshot = super::hft_logger::HftSnapshot {
            event_id: event_id.clone(),
            ts_ms: super::hft_logger::now_ms(),
            symbol: symbol.clone(),
            price: last_price,
            direction: format!("{:?}", direction),
            strategy: strategy_type.clone(),
            tape_speed: micro_tps,
            cvd: current_cvd,
            delta: micro_delta,
            whale_buys: micro_count as usize,
            whale_sells: 0,
            wall_dist_pct: reclaim,
            wall_size_usd: absorption_ratio,
            wall_eaten_pct: baseline_absorption,
            score,
            extensions: checks,
            status: "FIRED_V4".to_string(),
        };
        super::hft_logger::append_snapshot(&snapshot);

        tracker.release(&symbol).await;
        return AbsorberResult::Fired {
            symbol,
            direction,
            confidence: score,
            entry_price: last_price,
            target_wall_price,
            is_wall_backed: target_wall_price.is_some(),
        };
    }
}
