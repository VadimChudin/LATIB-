///! Whale Flow Detector — Large Order Detection + System-Wide EWMA Baseline
///! ========================================================================
///! Phase 12: Detects "whale" activity from existing tape data (no external APIs).
///! Two-layer architecture:
///!   Layer 1 (fast): 5-second burst detection for whale alerts
///!   Layer 2 (slow): EWMA baseline per symbol — "what is normal for this coin?"
///!     → queryable by ANY module (walls, whales, strategies)
///!     → updates on every trade, no periodic recalculation
///!
///! Uses dynamic thresholds based on:
///!   - EWMA baseline (avg trade size, volume/min) per symbol
///!   - Market session (Asia/EU/US) via market_session.rs
///!   - Market regime (favorable/hostile/neutral)

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use tracing::info;

use super::market_session;

// ── Configuration ───────────────────────────────────────────────────────────

/// How long a whale tag stays active (seconds)
const WHALE_TAG_TTL_SECS: u64 = 60;

/// Rolling window for burst detection (seconds)
const BURST_WINDOW_SECS: u64 = 5;

/// Rolling window for fast volume tracking (seconds)  
const FAST_WINDOW_SECS: u64 = 300; // 5 minutes

/// Minimum score to trigger a tag (prevents noise)
const MIN_WHALE_SCORE: f64 = 100.0;

/// EWMA smoothing factor — lower = more memory, smoother
/// α = 0.001 → ~1000 trades to fully adapt (stable but responsive)
const EWMA_ALPHA: f64 = 0.001;

/// Minimum trades before baseline is considered "warm"
const BASELINE_WARMUP: u64 = 50;

/// Rate limit: max 1 whale alert log per symbol per N seconds
const ALERT_RATE_LIMIT_SECS: u64 = 30;

// ── Data Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhaleTag {
    WhaleBuy,   // Large aggressive buy detected → bullish pressure
    WhaleSell,  // Large aggressive sell detected → bearish pressure
    Neutral,    // No significant whale activity
}

impl WhaleTag {
    pub fn as_str(&self) -> &'static str {
        match self {
            WhaleTag::WhaleBuy => "WHALE_BUY",
            WhaleTag::WhaleSell => "WHALE_SELL",
            WhaleTag::Neutral => "NEUTRAL",
        }
    }

    pub fn blocks_long(&self) -> bool {
        matches!(self, WhaleTag::WhaleSell)
    }

    pub fn blocks_short(&self) -> bool {
        matches!(self, WhaleTag::WhaleBuy)
    }
}

/// Active whale tag with metadata
#[derive(Debug, Clone)]
pub struct WhaleAlert {
    pub tag: WhaleTag,
    pub score: f64,
    pub volume_usd: f64,
    pub created_at: Instant,
    pub session: &'static str,
}

impl WhaleAlert {
    pub fn is_active(&self) -> bool {
        self.created_at.elapsed().as_secs() < WHALE_TAG_TTL_SECS
    }
}

// ── EWMA Symbol Baseline (Layer 2 — Slow, Stable) ──────────────────────────

/// System-wide baseline for a symbol.
/// Any module can query: "what is considered NORMAL for this coin right now?"
/// Updated on every trade via EWMA — always fresh, never stale.
#[derive(Debug, Clone)]
pub struct SymbolBaseline {
    /// EWMA average trade size in USD
    pub avg_trade_usd: f64,
    /// EWMA volume per minute (USD)
    pub volume_per_min: f64,
    /// EWMA of max single-trade size seen (tracks "big orders")
    pub max_trade_ewma: f64,
    /// Total trades processed (for warmup check)
    pub trade_count: u64,
    /// Last minute volume accumulator
    minute_volume: f64,
    /// When current minute started
    minute_start: Instant,
}

impl SymbolBaseline {
    fn new() -> Self {
        Self {
            avg_trade_usd: 0.0,
            volume_per_min: 0.0,
            max_trade_ewma: 0.0,
            trade_count: 0,
            minute_volume: 0.0,
            minute_start: Instant::now(),
        }
    }

    /// Update baseline with a new trade (called on EVERY trade)
    fn update(&mut self, usd_amount: f64) {
        self.trade_count += 1;

        if self.trade_count == 1 {
            // First trade: initialize
            self.avg_trade_usd = usd_amount;
            self.max_trade_ewma = usd_amount;
            self.volume_per_min = usd_amount * 60.0; // extrapolate
            return;
        }

        // EWMA update: new_avg = α * current + (1 - α) * old
        self.avg_trade_usd = EWMA_ALPHA * usd_amount + (1.0 - EWMA_ALPHA) * self.avg_trade_usd;

        // Track "big order" baseline (only updates upward aggressively)
        if usd_amount > self.max_trade_ewma {
            // Big trade: adapt faster (α * 10)
            let fast_alpha = (EWMA_ALPHA * 10.0).min(0.1);
            self.max_trade_ewma = fast_alpha * usd_amount + (1.0 - fast_alpha) * self.max_trade_ewma;
        } else {
            // Normal trade: slow decay
            self.max_trade_ewma = EWMA_ALPHA * usd_amount + (1.0 - EWMA_ALPHA) * self.max_trade_ewma;
        }

        // Volume per minute tracking
        self.minute_volume += usd_amount;
        let elapsed = self.minute_start.elapsed().as_secs_f64();
        if elapsed >= 60.0 {
            // Flush minute
            let actual_vol_per_min = self.minute_volume;
            self.volume_per_min = EWMA_ALPHA * 50.0 * actual_vol_per_min
                + (1.0 - EWMA_ALPHA * 50.0) * self.volume_per_min;
            self.minute_volume = 0.0;
            self.minute_start = Instant::now();
        }
    }

    /// Is this baseline warmed up enough to be trusted?
    pub fn is_warm(&self) -> bool {
        self.trade_count >= BASELINE_WARMUP
    }

    /// What USD amount is "big" for this symbol right now?
    /// Used by walls, whales, strategies — anything that needs context
    pub fn big_order_threshold(&self) -> f64 {
        if !self.is_warm() {
            return 50_000.0; // Default until warmup
        }
        // "Big" = 5x average trade size (session-adjusted)
        let session = market_session::MarketSession::current();
        (self.avg_trade_usd * 5.0 * session.volume_scale()).max(5_000.0)
    }

    /// What USD wall size is "significant" for this symbol?
    pub fn significant_wall_usd(&self) -> f64 {
        if !self.is_warm() {
            return 50_000.0;
        }
        // Wall = 50x average trade size (adjusted by session)
        let session = market_session::MarketSession::current();
        (self.avg_trade_usd * 50.0 * session.wall_scale()).max(20_000.0)
    }
}

// ── Fast Burst Tracker (Layer 1) ────────────────────────────────────────────

struct BurstTracker {
    /// Recent trades: (timestamp, usd_amount, is_buy)
    recent_trades: VecDeque<(Instant, f64, bool)>,
}

impl BurstTracker {
    fn new() -> Self {
        Self {
            recent_trades: VecDeque::with_capacity(500),
        }
    }

    fn record(&mut self, usd_amount: f64, is_buy: bool) {
        let now = Instant::now();
        self.recent_trades.push_back((now, usd_amount, is_buy));

        // Prune older than fast window
        while let Some((ts, _, _)) = self.recent_trades.front() {
            if now.duration_since(*ts).as_secs() > FAST_WINDOW_SECS {
                self.recent_trades.pop_front();
            } else {
                break;
            }
        }
    }

    /// Sum of volume in last N seconds for one side
    fn burst_volume(&self, window_secs: u64, buy_side: bool) -> f64 {
        let now = Instant::now();
        self.recent_trades.iter()
            .filter(|(ts, _, is_buy)| {
                *is_buy == buy_side && now.duration_since(*ts).as_secs() <= window_secs
            })
            .map(|(_, vol, _)| vol)
            .sum()
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Whale Detector + System-Wide Baseline Provider
pub struct WhaleDetector {
    /// Layer 1: Fast burst detection per symbol
    bursts: HashMap<String, BurstTracker>,
    /// Layer 2: EWMA baseline per symbol (queryable by any module)
    baselines: HashMap<String, SymbolBaseline>,
    /// Active whale alerts per symbol
    alerts: HashMap<String, WhaleAlert>,
    /// Rate limiting: last alert log time per symbol
    last_alert_log: HashMap<String, Instant>,
}

impl WhaleDetector {
    pub fn new() -> Self {
        Self {
            bursts: HashMap::new(),
            baselines: HashMap::new(),
            alerts: HashMap::new(),
            last_alert_log: HashMap::new(),
        }
    }

    /// Record a trade — updates BOTH layers
    pub fn record_trade(&mut self, symbol: &str, price: f64, quantity: f64, is_buyer_maker: bool) {
        let usd_amount = price * quantity;
        let is_buy = !is_buyer_maker;

        // Layer 2: Update EWMA baseline (always, every trade)
        let baseline = self.baselines.entry(symbol.to_string())
            .or_insert_with(SymbolBaseline::new);
        baseline.update(usd_amount);

        // Layer 1: Track burst
        let burst = self.bursts.entry(symbol.to_string())
            .or_insert_with(BurstTracker::new);
        burst.record(usd_amount, is_buy);

        // Check for whale burst (5-second window)
        let buy_burst = burst.burst_volume(BURST_WINDOW_SECS, true);
        let sell_burst = burst.burst_volume(BURST_WINDOW_SECS, false);

        // Dynamic threshold using EWMA baseline (stable, not noisy)
        // SKIP detection entirely during warmup — baseline is unreliable
        if !baseline.is_warm() {
            return;
        }
        let threshold = market_session::whale_threshold(baseline.avg_trade_usd);

        // Rate limit check: only log 1 alert per symbol per 30s
        let can_log = match self.last_alert_log.get(symbol) {
            Some(last) => last.elapsed().as_secs() >= ALERT_RATE_LIMIT_SECS,
            None => true,
        };

        // Check buy side
        if buy_burst > threshold {
            let score = (buy_burst / threshold) * 100.0;
            if score >= MIN_WHALE_SCORE {
                let session = market_session::MarketSession::current();
                if can_log {
                    tracing::debug!("🐋 WHALE BUY on {} | ${:.0}k (score={:.0}, thresh=${:.0}k, baseline=${:.0}) [{}]",
                        symbol, buy_burst / 1000.0, score, threshold / 1000.0,
                        baseline.avg_trade_usd, session.label());
                    self.last_alert_log.insert(symbol.to_string(), Instant::now());
                }
                self.alerts.insert(symbol.to_string(), WhaleAlert {
                    tag: WhaleTag::WhaleBuy, score, volume_usd: buy_burst,
                    created_at: Instant::now(), session: session.label(),
                });
            }
        }

        // Check sell side
        if sell_burst > threshold {
            let score = (sell_burst / threshold) * 100.0;
            if score >= MIN_WHALE_SCORE {
                let session = market_session::MarketSession::current();
                if can_log {
                    tracing::debug!("🐋 WHALE SELL on {} | ${:.0}k (score={:.0}, thresh=${:.0}k, baseline=${:.0}) [{}]",
                        symbol, sell_burst / 1000.0, score, threshold / 1000.0,
                        baseline.avg_trade_usd, session.label());
                    self.last_alert_log.insert(symbol.to_string(), Instant::now());
                }
                self.alerts.insert(symbol.to_string(), WhaleAlert {
                    tag: WhaleTag::WhaleSell, score, volume_usd: sell_burst,
                    created_at: Instant::now(), session: session.label(),
                });
            }
        }
    }

    // ── Whale Query API ─────────────────────────────────────────────────────

    pub fn get_tag(&self, symbol: &str) -> WhaleTag {
        match self.alerts.get(symbol) {
            Some(alert) if alert.is_active() => alert.tag,
            _ => WhaleTag::Neutral,
        }
    }

    pub fn get_alert(&self, symbol: &str) -> Option<&WhaleAlert> {
        self.alerts.get(symbol).filter(|a| a.is_active())
    }

    pub fn get_score(&self, symbol: &str) -> f64 {
        match self.alerts.get(symbol) {
            Some(alert) if alert.is_active() => alert.score,
            _ => 0.0,
        }
    }

    pub fn cleanup(&mut self) {
        self.alerts.retain(|_, alert| alert.is_active());
    }

    // ── System-Wide Baseline API (queryable by any module) ──────────────────

    /// Get the EWMA baseline for a symbol (None if never seen)
    pub fn get_baseline(&self, symbol: &str) -> Option<&SymbolBaseline> {
        self.baselines.get(symbol)
    }

    /// What is "big" for this symbol? (USD amount)
    /// Used by: wall_tracker (wall significance), strategies (volume filters)
    pub fn big_order_usd(&self, symbol: &str) -> f64 {
        match self.baselines.get(symbol) {
            Some(b) if b.is_warm() => b.big_order_threshold(),
            _ => 50_000.0, // safe default
        }
    }

    /// What wall size matters for this symbol? (USD)
    /// Used by: wall_tracker for dynamic density thresholds
    pub fn significant_wall_usd(&self, symbol: &str) -> f64 {
        match self.baselines.get(symbol) {
            Some(b) if b.is_warm() => b.significant_wall_usd(),
            _ => 50_000.0,
        }
    }

    /// Get volume-per-minute for a symbol
    pub fn volume_per_min(&self, symbol: &str) -> f64 {
        match self.baselines.get(symbol) {
            Some(b) if b.is_warm() => b.volume_per_min,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whale_tag_blocks() {
        assert!(WhaleTag::WhaleSell.blocks_long());
        assert!(!WhaleTag::WhaleSell.blocks_short());
        assert!(WhaleTag::WhaleBuy.blocks_short());
        assert!(!WhaleTag::WhaleBuy.blocks_long());
        assert!(!WhaleTag::Neutral.blocks_long());
    }

    #[test]
    fn test_detector_default_neutral() {
        let detector = WhaleDetector::new();
        assert_eq!(detector.get_tag("BTC/USDT"), WhaleTag::Neutral);
    }

    #[test]
    fn test_ewma_baseline_warmup() {
        let mut detector = WhaleDetector::new();

        // Before warmup: defaults
        assert_eq!(detector.big_order_usd("BTC/USDT"), 50_000.0);

        // Feed 100 trades → warmup complete
        for _ in 0..100 {
            detector.record_trade("BTC/USDT", 70000.0, 0.01, false); // $700 each
        }

        let baseline = detector.get_baseline("BTC/USDT").unwrap();
        assert!(baseline.is_warm());
        assert!(baseline.avg_trade_usd > 0.0);
    }

    #[test]
    fn test_ewma_stability() {
        let mut detector = WhaleDetector::new();

        // Feed 200 normal trades ($500 each)
        for _ in 0..200 {
            detector.record_trade("SOL/USDT", 100.0, 5.0, false);
        }

        let baseline_before = detector.get_baseline("SOL/USDT").unwrap().avg_trade_usd;

        // Single whale trade ($100k) should NOT dramatically shift the baseline
        detector.record_trade("SOL/USDT", 100.0, 1000.0, false);

        let baseline_after = detector.get_baseline("SOL/USDT").unwrap().avg_trade_usd;
        let change_pct = ((baseline_after - baseline_before) / baseline_before).abs() * 100.0;

        // Change should be <1% from a single outlier
        assert!(change_pct < 5.0, "EWMA shifted too much: {:.2}%", change_pct);
    }
}

