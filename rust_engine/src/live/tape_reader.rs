///! Trade Tape Reader + Order Flow Analyzer
///! ==========================================
///! Consumes aggTrade events and calculates:
///! - Trade Delta (per tick: +qty if buyer aggressor, -qty if seller)
///! - CVD (Cumulative Volume Delta) over rolling window
///! - Imbalance Ratio (aggressive buy vol / sell vol)
///! - Tape Speed (ticks/sec + acceleration)
///! - Large Print Detection (whale trades)

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;

/// Minimum volume ratio to consider "strong imbalance"
const IMBALANCE_STRONG: f64 = 3.0;

/// Large print threshold multiplier (vs average trade size)
const LARGE_PRINT_MULT: f64 = 10.0;

/// Short window for tape speed calculation (seconds)
const SPEED_WINDOW_SECS: i64 = 10;

/// Window for sweep detection (seconds)
const SWEEP_WINDOW_SECS: i64 = 5;

/// Minimum sweep score to consider a sweep bot active
const SWEEP_SCORE_THRESHOLD: f64 = 15.0;

/// Minimum single print size in USD to flag as whale
const WHALE_PRINT_USD_THRESHOLD: f64 = 50_000.0;

/// Number of 1-second buckets to keep for iceberg detection
const ICEBERG_HISTORY_BUCKETS: usize = 300; // 5 minutes of history

/// Z-Score threshold for anomalous volume
const ICEBERG_ZSCORE_THRESHOLD: f64 = 2.5;

/// A single recorded trade for analysis
#[derive(Debug, Clone)]
struct TradeRecord {
    delta: f64,          // +qty (buy) or -qty (sell)
    abs_volume: f64,     // absolute volume in contracts
    timestamp_ms: i64,
    is_buy: bool,
    price: f64,          // trade price (for absorption / reclaim calculations)
    quote_volume: f64,   // price * qty (for absorption ratio)
}

/// Unified order flow signal for strategies
#[derive(Debug, Clone)]
pub struct OrderFlowSignal {
    /// Normalized delta: -1.0 (all sells) to +1.0 (all buys)
    pub delta: f64,
    /// CVD (cumulative volume delta)
    pub cvd: f64,
    /// CVD trend (positive = buying pressure increasing)
    pub cvd_trend: f64,
    /// Imbalance ratio: buy_vol / sell_vol (>3 = strong buyer, <0.33 = strong seller)
    pub imbalance_ratio: f64,
    /// Ticks per second (current speed)
    pub tape_speed: f64,
    /// Speed acceleration: >1.0 = speeding up, <1.0 = slowing down
    pub speed_acceleration: f64,
    /// Number of large prints (whale trades) in window
    pub large_prints_buy: u32,
    /// Number of large sell prints
    pub large_prints_sell: u32,
    /// Total trades in window
    pub trade_count: usize,
    /// Iceberg buy pressure: 0.0..1.0 (hidden algorithmic buying via split orders)
    pub iceberg_buy_pressure: f64,
    /// Iceberg sell pressure: 0.0..1.0 (hidden algorithmic selling via split orders)
    pub iceberg_sell_pressure: f64,
    /// Volume Z-Score: how many stddevs current 1s volume is from the mean
    pub volume_zscore: f64,
    // ── Phase 29C+2: Sweep & Whale Detection ──
    /// Sweep score: consecutive same-direction prints in last 5 seconds
    pub sweep_score: f64,
    /// Direction of the sweep (true = buying sweep, false = selling sweep)
    pub sweep_direction_is_buy: bool,
    /// Largest single trade (in USD) in the window
    pub max_single_print_usd: f64,
}

impl Default for OrderFlowSignal {
    fn default() -> Self {
        Self {
            delta: 0.0,
            cvd: 0.0,
            cvd_trend: 0.0,
            imbalance_ratio: 1.0,
            tape_speed: 0.0,
            speed_acceleration: 1.0,
            large_prints_buy: 0,
            large_prints_sell: 0,
            trade_count: 0,
            iceberg_buy_pressure: 0.0,
            iceberg_sell_pressure: 0.0,
            volume_zscore: 0.0,
            sweep_score: 0.0,
            sweep_direction_is_buy: true,
            max_single_print_usd: 0.0,
        }
    }
}

impl OrderFlowSignal {
    /// Is the buyer dominant?
    pub fn is_buyer_dominant(&self) -> bool {
        self.delta > 0.3 && self.imbalance_ratio > IMBALANCE_STRONG
    }

    /// Is the seller dominant?
    pub fn is_seller_dominant(&self) -> bool {
        self.delta < -0.3 && self.imbalance_ratio < (1.0 / IMBALANCE_STRONG)
    }

    /// Is the tape accelerating?
    pub fn is_accelerating(&self) -> bool {
        self.speed_acceleration > 1.5
    }
}

/// One-second volume bucket for iceberg detection
#[derive(Debug, Clone, Default)]
struct SecondBucket {
    buy_vol: f64,
    sell_vol: f64,
    trade_count: u32,
    timestamp_sec: i64,
}

/// Rolling CVD + Order Flow state for one symbol
#[derive(Debug, Clone)]
pub struct TapeState {
    /// Recent trades with full info
    trades: VecDeque<TradeRecord>,
    /// Last traded price
    pub last_price: f64,
    /// Current CVD (sum of deltas in window)
    pub cvd: f64,
    /// Running buy volume in window
    buy_volume: f64,
    /// Running sell volume in window
    sell_volume: f64,
    /// Window duration for calculation
    window_ms: i64,
    /// Previous tape speed (for acceleration calc)
    prev_speed: f64,
    /// Average trade size (running estimate)
    avg_trade_size: f64,
    /// EMA smoothing for avg trade size
    trade_count_total: u64,
    // ── Iceberg Detection ──
    /// Ring buffer of per-second volume buckets
    second_buckets: VecDeque<SecondBucket>,
    /// Current (active) second bucket
    current_bucket: SecondBucket,
    /// Running mean of 1s volume (EMA)
    avg_1s_volume: f64,
    /// Running variance for Z-Score (Welford's algorithm)
    vol_m2: f64,
    /// Count of completed buckets (for Welford)
    bucket_count: u64,
}

impl TapeState {
    pub fn new(window: Duration) -> Self {
        Self {
            trades: VecDeque::with_capacity(50_000),
            last_price: 0.0,
            cvd: 0.0,
            buy_volume: 0.0,
            sell_volume: 0.0,
            window_ms: window.as_millis() as i64,
            prev_speed: 0.0,
            avg_trade_size: 0.0,
            trade_count_total: 0,
            second_buckets: VecDeque::with_capacity(ICEBERG_HISTORY_BUCKETS + 10),
            current_bucket: SecondBucket::default(),
            avg_1s_volume: 0.0,
            vol_m2: 0.0,
            bucket_count: 0,
        }
    }

    /// Add a new trade and update all metrics
    pub fn add_trade(&mut self, quantity: f64, price: f64, is_buyer_maker: bool, timestamp_ms: i64) {
        // is_buyer_maker = true means the SELLER was the aggressor (market sell)
        let is_buy = !is_buyer_maker;
        let delta = if is_buy { quantity } else { -quantity };
        let volume_usd = quantity * price;
        
        self.last_price = price;

        // Update running volumes
        if is_buy {
            self.buy_volume += quantity;
        } else {
            self.sell_volume += quantity;
        }

        self.trades.push_back(TradeRecord {
            delta,
            abs_volume: quantity,
            timestamp_ms,
            is_buy,
            price,
            quote_volume: volume_usd,
        });
        self.cvd += delta;

        // ── Iceberg: aggregate into 1-second buckets ──
        let current_sec = timestamp_ms / 1000;
        if self.current_bucket.timestamp_sec == 0 {
            // First trade ever
            self.current_bucket.timestamp_sec = current_sec;
        }
        if current_sec == self.current_bucket.timestamp_sec {
            // Same second — accumulate
            if is_buy {
                self.current_bucket.buy_vol += volume_usd;
            } else {
                self.current_bucket.sell_vol += volume_usd;
            }
            self.current_bucket.trade_count += 1;
        } else {
            // New second — finalize previous bucket
            self.finalize_bucket();
            // Start new bucket
            self.current_bucket = SecondBucket {
                buy_vol: if is_buy { volume_usd } else { 0.0 },
                sell_vol: if !is_buy { volume_usd } else { 0.0 },
                trade_count: 1,
                timestamp_sec: current_sec,
            };
        }

        // Update average trade size (EMA)
        self.trade_count_total += 1;
        let alpha = 2.0 / (1000.0_f64.min(self.trade_count_total as f64) + 1.0);
        self.avg_trade_size = quantity * alpha + self.avg_trade_size * (1.0 - alpha);

        // Trim old entries outside the window
        let cutoff = timestamp_ms - self.window_ms;
        while let Some(old) = self.trades.front() {
            if old.timestamp_ms < cutoff {
                self.cvd -= old.delta;
                if old.is_buy {
                    self.buy_volume -= old.abs_volume;
                } else {
                    self.sell_volume -= old.abs_volume;
                }
                self.trades.pop_front();
            } else {
                break;
            }
        }
    }

    /// Calculate baseline metrics for knife_tick_v3 over the last N seconds.
    /// Returns: (baseline_tps, baseline_avg_size, baseline_flow_per_ms)
    pub fn get_baseline_metrics(&self, baseline_secs: i64, micro_win_ms: i64) -> (f64, f64, f64, f64) {
        let now_ms = crate::live::hft_logger::now_ms() as i64;
        let cutoff = now_ms - (baseline_secs * 1000);
        let mut bl_ticks = 0;
        let mut bl_vol = 0.0;
        let mut bl_delta = 0.0;
        let mut high = f64::MIN;
        let mut low = f64::MAX;
        
        for t in self.trades.iter().rev() {
            if t.timestamp_ms >= cutoff {
                bl_ticks += 1;
                bl_vol += t.quote_volume;
                bl_delta += t.delta;
                if t.price > high { high = t.price; }
                if t.price < low { low = t.price; }
            } else {
                break;
            }
        }
        
        let baseline_tps = (bl_ticks as f64) / (baseline_secs as f64);
        let baseline_avg_size = if bl_ticks > 0 { bl_vol / (bl_ticks as f64) } else { 0.0 };
        let baseline_flow = bl_delta / (baseline_secs as f64 * 1000.0) * (micro_win_ms as f64);

        let baseline_range_pct = if self.last_price > 0.0 && high != f64::MIN && low != f64::MAX {
            (high - low) / self.last_price
        } else {
            0.0
        };
        let baseline_absorption = if baseline_range_pct > 0.00001 {
            bl_vol / baseline_range_pct
        } else {
            1_000_000.0
        };

        (baseline_tps, baseline_avg_size, baseline_flow, baseline_absorption.max(1.0))
    }

    /// Calculate micro metrics over the last `micro_win_ms`.
    /// Returns: (tps, avg_size, delta)
    pub fn get_micro_metrics(&self, micro_win_ms: i64) -> (f64, f64, f64) {
        let now_ms = crate::live::hft_logger::now_ms() as i64;
        let cutoff = now_ms - micro_win_ms;
        let mut m_ticks = 0;
        let mut m_vol = 0.0;
        let mut m_delta = 0.0;
        
        for t in self.trades.iter().rev() {
            if t.timestamp_ms >= cutoff {
                m_ticks += 1;
                m_vol += t.abs_volume;
                m_delta += t.delta;
            } else {
                break;
            }
        }
        
        let m_tps = m_ticks as f64;
        let m_avg_size = if m_ticks > 0 { m_vol / (m_ticks as f64) } else { 0.0 };
        
        (m_tps, m_avg_size, m_delta)
    }

    /// Get detailed micro metrics for the absorber's 4-step checklist.
    /// Returns: (micro_tps, micro_quote_vol, micro_delta, micro_high, micro_low, micro_trade_count, cvd)
    pub fn get_micro_absorption_metrics(&self, micro_win_ms: i64) -> (f64, f64, f64, f64, f64, u32, f64) {
        let now_ms = crate::live::hft_logger::now_ms() as i64;
        let cutoff = now_ms - micro_win_ms;
        let mut m_ticks: u32 = 0;
        let mut m_quote_vol: f64 = 0.0;
        let mut m_delta: f64 = 0.0;
        let mut m_high: f64 = f64::MIN;
        let mut m_low: f64 = f64::MAX;

        for t in self.trades.iter().rev() {
            if t.timestamp_ms >= cutoff {
                m_ticks += 1;
                m_quote_vol += t.quote_volume;
                m_delta += t.delta * t.price; // quote-weighted delta like backtester
                if t.price > m_high { m_high = t.price; }
                if t.price < m_low { m_low = t.price; }
            } else {
                break;
            }
        }

        if m_ticks == 0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0, self.cvd);
        }

        let micro_seconds = micro_win_ms as f64 / 1000.0;
        let micro_tps = m_ticks as f64 / micro_seconds.max(0.001);

        (micro_tps, m_quote_vol, m_delta, m_high, m_low, m_ticks, self.cvd)
    }

    /// Normalized trade delta: CVD / total volume in window
    /// Range: -1.0 (all sells) to +1.0 (all buys)
    pub fn normalized_delta(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total <= 0.0 { return 0.0; }
        self.cvd / total
    }

    /// Number of trades in the current window
    pub fn trade_count(&self) -> usize {
        self.trades.len()
    }

    /// CVD trend: positive slope = buying pressure increasing
    pub fn cvd_trend(&self) -> f64 {
        let n = self.trades.len();
        if n < 20 { return 0.0; }
        let quarter = n / 4;
        let early_sum: f64 = self.trades.iter().take(quarter).map(|t| t.delta).sum();
        let late_sum: f64 = self.trades.iter().rev().take(quarter).map(|t| t.delta).sum();
        late_sum - early_sum
    }

    /// Imbalance ratio: buy_volume / sell_volume
    /// >3.0 = strong buyer, <0.33 = strong seller, ~1.0 = balanced
    pub fn imbalance_ratio(&self) -> f64 {
        if self.sell_volume <= 0.0 {
            return if self.buy_volume > 0.0 { 99.0 } else { 1.0 };
        }
        self.buy_volume / self.sell_volume
    }

    /// Tape speed: trades per second (over last SPEED_WINDOW_SECS)
    pub fn tape_speed(&self) -> f64 {
        if self.trades.is_empty() { return 0.0; }
        let latest_ts = self.trades.back().map(|t| t.timestamp_ms).unwrap_or(0);
        let speed_cutoff = latest_ts - (SPEED_WINDOW_SECS * 1000);
        let recent_count = self.trades.iter().rev()
            .take_while(|t| t.timestamp_ms >= speed_cutoff)
            .count();
        recent_count as f64 / SPEED_WINDOW_SECS as f64
    }

    /// Speed acceleration: current_speed / previous_speed
    /// >1.5 = accelerating, <0.7 = decelerating
    pub fn speed_acceleration(&self) -> f64 {
        let current = self.tape_speed();
        if self.prev_speed <= 0.0 { return 1.0; }
        current / self.prev_speed
    }

    /// Count large prints (whale trades) in window
    pub fn large_prints(&self) -> (u32, u32) {
        if self.avg_trade_size <= 0.0 { return (0, 0); }
        let threshold = self.avg_trade_size * LARGE_PRINT_MULT;
        let mut buy_prints = 0u32;
        let mut sell_prints = 0u32;
        for t in &self.trades {
            if t.abs_volume >= threshold {
                if t.is_buy { buy_prints += 1; } else { sell_prints += 1; }
            }
        }
        (buy_prints, sell_prints)
    }

    // ── Iceberg Detection Methods ────────────────────────────────────

    /// Finalize the current 1-second bucket and push it to history
    fn finalize_bucket(&mut self) {
        let total_vol = self.current_bucket.buy_vol + self.current_bucket.sell_vol;

        // Update Welford's online mean + variance
        self.bucket_count += 1;
        let n = self.bucket_count as f64;
        let delta_w = total_vol - self.avg_1s_volume;
        self.avg_1s_volume += delta_w / n;
        let delta_w2 = total_vol - self.avg_1s_volume;
        self.vol_m2 += delta_w * delta_w2;

        // Push to ring buffer
        self.second_buckets.push_back(self.current_bucket.clone());
        if self.second_buckets.len() > ICEBERG_HISTORY_BUCKETS {
            self.second_buckets.pop_front();
        }
    }

    /// Volume Z-Score for the current second's bucket
    /// How many standard deviations is this second's volume from the mean?
    pub fn volume_zscore(&self) -> f64 {
        if self.bucket_count < 30 { return 0.0; } // Need history
        let variance = self.vol_m2 / self.bucket_count as f64;
        let stddev = variance.sqrt();
        if stddev <= 0.0 { return 0.0; }
        let current_vol = self.current_bucket.buy_vol + self.current_bucket.sell_vol;
        (current_vol - self.avg_1s_volume) / stddev
    }

    /// Iceberg pressure: normalized 0..1
    /// Measures hidden algorithmic buying/selling via aggregated small orders
    pub fn iceberg_pressure(&self) -> (f64, f64) {
        if self.bucket_count < 30 { return (0.0, 0.0); }
        let zscore = self.volume_zscore();
        if zscore < ICEBERG_ZSCORE_THRESHOLD { return (0.0, 0.0); }

        let total = self.current_bucket.buy_vol + self.current_bucket.sell_vol;
        if total <= 0.0 { return (0.0, 0.0); }

        // Normalize the anomaly to 0..1 (zscore 2.5 → 0.0, zscore 7+ → 1.0)
        let pressure = ((zscore - ICEBERG_ZSCORE_THRESHOLD) / 4.5).min(1.0);

        let buy_ratio = self.current_bucket.buy_vol / total;
        let sell_ratio = self.current_bucket.sell_vol / total;

        (pressure * buy_ratio, pressure * sell_ratio)
    }

    /// Get a unified order flow signal snapshot
    pub fn order_flow_signal(&mut self) -> OrderFlowSignal {
        let speed = self.tape_speed();
        let accel = self.speed_acceleration();
        let (lp_buy, lp_sell) = self.large_prints();
        let (ice_buy, ice_sell) = self.iceberg_pressure();
        let vol_z = self.volume_zscore();
        let (sweep, sweep_is_buy) = self.sweep_score();
        let max_print = self.max_print_usd();

        // Update prev_speed for next acceleration calc
        self.prev_speed = speed;

        OrderFlowSignal {
            delta: self.normalized_delta(),
            cvd: self.cvd,
            cvd_trend: self.cvd_trend(),
            imbalance_ratio: self.imbalance_ratio(),
            tape_speed: speed,
            speed_acceleration: accel,
            large_prints_buy: lp_buy,
            large_prints_sell: lp_sell,
            trade_count: self.trades.len(),
            iceberg_buy_pressure: ice_buy,
            iceberg_sell_pressure: ice_sell,
            volume_zscore: vol_z,
            sweep_score: sweep,
            sweep_direction_is_buy: sweep_is_buy,
            max_single_print_usd: max_print,
        }
    }

    /// Phase 29C+2: Sweep score — count consecutive same-direction prints in last N seconds.
    /// Returns (score, is_buy_direction).
    /// High sweep_score (>15) = aggressive sweep bot or stop-hunt in progress.
    pub fn sweep_score(&self) -> (f64, bool) {
        if self.trades.is_empty() { return (0.0, true); }

        let latest_ts = self.trades.back().map(|t| t.timestamp_ms).unwrap_or(0);
        let cutoff = latest_ts - (SWEEP_WINDOW_SECS * 1000);

        // Count longest consecutive streak of same-direction trades
        let mut max_streak: u32 = 0;
        let mut current_streak: u32 = 0;
        let mut streak_is_buy = true;
        let mut best_is_buy = true;
        let mut prev_is_buy: Option<bool> = None;

        for t in self.trades.iter().rev() {
            if t.timestamp_ms < cutoff { break; }

            match prev_is_buy {
                None => {
                    current_streak = 1;
                    streak_is_buy = t.is_buy;
                    prev_is_buy = Some(t.is_buy);
                }
                Some(prev) if prev == t.is_buy => {
                    current_streak += 1;
                }
                _ => {
                    if current_streak > max_streak {
                        max_streak = current_streak;
                        best_is_buy = streak_is_buy;
                    }
                    current_streak = 1;
                    streak_is_buy = t.is_buy;
                    prev_is_buy = Some(t.is_buy);
                }
            }
        }
        // Final streak
        if current_streak > max_streak {
            max_streak = current_streak;
            best_is_buy = streak_is_buy;
        }

        (max_streak as f64, best_is_buy)
    }

    /// Phase 29C+2: Largest single trade in USD within the window
    pub fn max_print_usd(&self) -> f64 {
        self.trades.iter()
            .map(|t| t.quote_volume)
            .fold(0.0_f64, f64::max)
    }
}

/// Shared tape state store
pub type TapeStore = Arc<DashMap<String, TapeState>>;

/// Create a new tape store for given symbols
pub fn new_store(symbols: &[String], window: Duration) -> TapeStore {
    let store = Arc::new(DashMap::new());
    for sym in symbols {
        store.insert(sym.clone(), TapeState::new(window));
    }
    store
}

/// Record a trade into the tape store
pub fn record_trade(store: &TapeStore, symbol: &str, price: f64, quantity: f64, is_buyer_maker: bool, timestamp_ms: i64) {
    if let Some(mut state) = store.get_mut(symbol) {
        state.add_trade(quantity, price, is_buyer_maker, timestamp_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvd_basic() {
        let mut state = TapeState::new(Duration::from_secs(60));

        // 3 buys, 1 sell
        state.add_trade(1.0, 0.0, false, 1000); // Buy +1
        state.add_trade(2.0, 0.0, false, 2000); // Buy +2
        state.add_trade(0.5, 0.0, true, 3000);  // Sell -0.5
        state.add_trade(1.0, 0.0, false, 4000); // Buy +1

        assert!((state.cvd - 3.5).abs() < 0.01);
        assert!(state.normalized_delta() > 0.0);
    }

    #[test]
    fn test_window_expiry() {
        let mut state = TapeState::new(Duration::from_secs(5));

        state.add_trade(10.0, 0.0, false, 1000);  // Buy at t=1s
        state.add_trade(1.0, 0.0, true, 7000);    // Sell at t=7s (old buy should be expired)

        assert!((state.cvd - (-1.0)).abs() < 0.01, "Old buy should be evicted");
    }

    #[test]
    fn test_imbalance_ratio() {
        let mut state = TapeState::new(Duration::from_secs(60));

        // 3 buys of 10, 1 sell of 2 → ratio = 30/2 = 15.0
        state.add_trade(10.0, 0.0, false, 1000);
        state.add_trade(10.0, 0.0, false, 2000);
        state.add_trade(10.0, 0.0, false, 3000);
        state.add_trade(2.0, 0.0, true, 4000);

        let ratio = state.imbalance_ratio();
        assert!(ratio > 10.0, "Should be strongly buyer-dominated: {}", ratio);
    }

    #[test]
    fn test_order_flow_signal() {
        let mut state = TapeState::new(Duration::from_secs(60));

        // Simulate rapid buying
        for i in 0..50 {
            state.add_trade(1.0, 0.0, false, 1000 + i * 100); // Buy every 100ms
        }
        state.add_trade(0.1, 0.0, true, 6000); // Tiny sell

        let signal = state.order_flow_signal();
        assert!(signal.is_buyer_dominant(), "Delta={}, Imbalance={}", signal.delta, signal.imbalance_ratio);
        assert!(signal.tape_speed > 0.0, "Should have nonzero tape speed");
    }
}
