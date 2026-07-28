///! Market Session Detector — Time-Based Dynamic Scaling
///! =====================================================
///! Detects current trading session and provides dynamic multipliers
///! that scale thresholds across the entire system:
///!   - Wall significance thresholds (WallTracker)
///!   - Whale detection thresholds (WhaleDetector)
///!   - Order flow sensitivity (TapeReader)
///!
///! Sessions (UTC):
///!   🌙 ASIA   00:00-08:00  — lowest liquidity, thin books
///!   🌍 EUROPE 08:00-16:00  — medium liquidity, steady flow
///!   🇺🇸 US     16:00-00:00  — peak liquidity, largest moves
///!
///! Usage:
///!   let session = MarketSession::current();
///!   let wall_threshold = base_threshold * session.volume_scale();

use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

// ── Session Types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketSession {
    Asia,    // 00:00-08:00 UTC — thin order books, low volume
    Europe,  // 08:00-16:00 UTC — medium activity
    Us,      // 16:00-00:00 UTC — peak volume, most volatility
}

impl MarketSession {
    /// Detect the current session from UTC time
    pub fn current() -> Self {
        let hour = current_utc_hour();
        match hour {
            0..=7   => MarketSession::Asia,
            8..=15  => MarketSession::Europe,
            16..=23 => MarketSession::Us,
            _       => MarketSession::Europe, // fallback
        }
    }

    /// Volume scaling factor — how much thinner the market is vs US peak
    /// Used to scale DOWN thresholds in thin sessions (makes system more sensitive)
    ///
    /// US = 1.0 (baseline)
    /// EU = 0.7 (walls/whales need to be 70% of US size to be significant)
    /// Asia = 0.4 (walls/whales at 40% = already significant)
    pub fn volume_scale(&self) -> f64 {
        match self {
            MarketSession::Us     => 1.0,
            MarketSession::Europe => 0.7,
            MarketSession::Asia   => 0.4,
        }
    }

    /// Whale threshold multiplier — applied to per-symbol rolling avg
    /// Lower in thin sessions = catch smaller whales
    pub fn whale_multiplier(&self) -> f64 {
        match self {
            MarketSession::Us     => 8.0,  // Need 8x avg to be a whale
            MarketSession::Europe => 6.0,  // 6x during EU
            MarketSession::Asia   => 4.0,  // 4x during Asia (thin market)
        }
    }

    /// Wall significance multiplier — applied to wall_tracker thresholds
    /// Smaller walls matter more in thin sessions
    pub fn wall_scale(&self) -> f64 {
        match self {
            MarketSession::Us     => 1.0,   // $50k+ walls in US
            MarketSession::Europe => 0.7,   // $35k+ walls in EU
            MarketSession::Asia   => 0.4,   // $20k+ walls in Asia
        }
    }

    /// Human-readable emoji label
    pub fn label(&self) -> &'static str {
        match self {
            MarketSession::Asia   => "🌙 ASIA",
            MarketSession::Europe => "🌍 EU",
            MarketSession::Us     => "🇺🇸 US",
        }
    }
}

// ── Market Regime (from aggregate_journal) ──────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketRegime {
    Favorable,  // Recent trades profitable, market "easy"
    Neutral,    // Default / unclear
    Hostile,    // Recent trades losing, dangerous conditions
}

impl MarketRegime {
    /// Load current regime from journal stats file
    pub fn load() -> Self {
        let path = std::path::Path::new("data/journal_stats.json");
        if !path.exists() {
            return MarketRegime::Neutral;
        }

        match std::fs::read_to_string(path) {
            Ok(content) => {
                if content.contains("\"favorable\"") {
                    MarketRegime::Favorable
                } else if content.contains("\"hostile\"") {
                    MarketRegime::Hostile
                } else {
                    MarketRegime::Neutral
                }
            }
            Err(_) => MarketRegime::Neutral,
        }
    }

    /// Regime multiplier for whale detection
    /// In hostile regime, be MORE sensitive (lower threshold)
    pub fn whale_adjust(&self) -> f64 {
        match self {
            MarketRegime::Favorable => 1.2,  // Slightly higher bar
            MarketRegime::Neutral   => 1.0,  // Standard
            MarketRegime::Hostile   => 0.7,  // Lower bar = catch more whales
        }
    }
}

// ── Combined Dynamic Threshold ──────────────────────────────────────────────

/// Calculate the whale threshold for a symbol given its avg order size
pub fn whale_threshold(avg_order_usd: f64) -> f64 {
    let session = MarketSession::current();
    let regime = MarketRegime::load();

    let threshold = avg_order_usd
        * session.whale_multiplier()
        * regime.whale_adjust();

    // Minimum floor: $10k (anything smaller is noise regardless)
    threshold.max(10_000.0)
}

/// Calculate the effective wall significance threshold (USD)
pub fn wall_threshold(base_threshold_usd: f64) -> f64 {
    let session = MarketSession::current();
    base_threshold_usd * session.wall_scale()
}

/// Log current market context (called periodically)
pub fn log_context() {
    let session = MarketSession::current();
    let regime = MarketRegime::load();
    info!("🕐 Session: {} | Regime: {:?} | Vol-scale: {:.1}x | Whale-mult: {:.0}x",
        session.label(), regime, session.volume_scale(), session.whale_multiplier());
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn current_utc_hour() -> u32 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    ((secs % 86400) / 3600) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_scale() {
        assert_eq!(MarketSession::Us.volume_scale(), 1.0);
        assert!(MarketSession::Asia.volume_scale() < MarketSession::Europe.volume_scale());
    }

    #[test]
    fn test_whale_threshold_floor() {
        // Even with tiny avg order, threshold should be at least $10k
        let t = whale_threshold(100.0);
        assert!(t >= 10_000.0);
    }

    #[test]
    fn test_regime_load_default() {
        // No file = Neutral
        assert_eq!(MarketRegime::load(), MarketRegime::Neutral);
    }
}
