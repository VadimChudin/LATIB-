///! Smart Trailing Engine
///! ======================
///! Combines three data sources for intelligent position exit:
///!
///! 1. **OB Imbalance** (30% weight) — Passive order pressure
///! 2. **Trade Delta**  (50% weight) — Aggressive order flow
///! 3. **CVD Trend**    (20% weight) — Cumulative direction bias
///!
///! Composite Score range: [-1.0, +1.0]
///! - Score < -0.5 (LONG) or > +0.5 (SHORT) → Force close
///! - Score aligned with position → Tighten trail (momentum)

use std::time::{Duration, Instant};

use tracing::info;

use super::order_book::OrderBookStore;
use super::position_manager::{Direction, ExitReason, Position};
use super::tape_reader::TapeStore;

/// Configuration for the Smart Trailer
#[derive(Debug, Clone)]
pub struct TrailerConfig {
    /// Weight for Order Book imbalance in composite
    pub ob_weight: f64,
    /// Weight for Trade Delta (tape) in composite
    pub tape_weight: f64,
    /// Weight for CVD trend in composite
    pub cvd_weight: f64,
    /// Threshold for force-closing against position
    pub force_exit_threshold: f64,
    /// Threshold for tightening trail with momentum
    pub momentum_threshold: f64,
    /// Minimum profit in R before smart exit is allowed
    pub min_profit_r: f64,
    /// How often to evaluate (avoid spamming)
    pub eval_interval: Duration,
}

impl Default for TrailerConfig {
    fn default() -> Self {
        Self {
            ob_weight: 0.3,
            tape_weight: 0.5,
            cvd_weight: 0.2,
            force_exit_threshold: 0.5,
            momentum_threshold: 0.3,
            min_profit_r: 0.5,
            eval_interval: Duration::from_millis(500),
        }
    }
}

/// Smart Trailer tracks composite scores per symbol
pub struct SmartTrailer {
    config: TrailerConfig,
    last_eval: std::collections::HashMap<String, Instant>,
}

impl SmartTrailer {
    pub fn new(config: TrailerConfig) -> Self {
        Self {
            config,
            last_eval: std::collections::HashMap::new(),
        }
    }

    /// Evaluate a single position and decide exit action
    pub fn evaluate(
        &mut self,
        position: &Position,
        current_price: f64,
        ob_store: &OrderBookStore,
        tape_store: &TapeStore,
    ) -> TrailAction {
        // Rate limit evaluations
        let now = Instant::now();
        if let Some(last) = self.last_eval.get(&position.symbol) {
            if now.duration_since(*last) < self.config.eval_interval {
                return TrailAction::Hold;
            }
        }
        self.last_eval.insert(position.symbol.clone(), now);

        // Calculate current profit in R
        let profit_r = match position.direction {
            Direction::Long => {
                if position.risk_dist > 0.0 {
                    (current_price - position.entry_price) / position.risk_dist
                } else {
                    0.0
                }
            }
            Direction::Short => {
                if position.risk_dist > 0.0 {
                    (position.entry_price - current_price) / position.risk_dist
                } else {
                    0.0
                }
            }
        };

        // Don't smart-exit if in loss
        if profit_r < self.config.min_profit_r {
            return TrailAction::Hold;
        }

        // Calculate composite score
        let composite = self.calculate_composite(&position.symbol, ob_store, tape_store);

        // Decision matrix
        match position.direction {
            Direction::Long => {
                if composite < -self.config.force_exit_threshold {
                    info!(
                        "⚡ SMART EXIT {} LONG @ {:.4} | Composite={:.3} | Profit={:.1}R",
                        position.symbol, current_price, composite, profit_r
                    );
                    TrailAction::ForceClose(ExitReason::SmartExit)
                } else if composite < 0.0 && !position.partially_exited && profit_r >= self.config.min_profit_r {
                    info!(
                        "📉 MOMENTUM FADED {} LONG @ {:.4} | Composite={:.3} | Dropping 50% & moving to BE",
                        position.symbol, current_price, composite
                    );
                    TrailAction::PartialExit
                } else if composite > self.config.momentum_threshold {
                    // Strong buying momentum — tighten trail to lock profits
                    TrailAction::TightenTrail { factor: 0.7 }
                } else {
                    TrailAction::Hold
                }
            }
            Direction::Short => {
                if composite > self.config.force_exit_threshold {
                    info!(
                        "⚡ SMART EXIT {} SHORT @ {:.4} | Composite={:.3} | Profit={:.1}R",
                        position.symbol, current_price, composite, profit_r
                    );
                    TrailAction::ForceClose(ExitReason::SmartExit)
                } else if composite > 0.0 && !position.partially_exited && profit_r >= self.config.min_profit_r {
                    info!(
                        "📉 MOMENTUM FADED {} SHORT @ {:.4} | Composite={:.3} | Dropping 50% & moving to BE",
                        position.symbol, current_price, composite
                    );
                    TrailAction::PartialExit
                } else if composite < -self.config.momentum_threshold {
                    TrailAction::TightenTrail { factor: 0.7 }
                } else {
                    TrailAction::Hold
                }
            }
        }
    }

    /// Calculate the composite score from all three data sources
    fn calculate_composite(
        &self,
        symbol: &str,
        ob_store: &OrderBookStore,
        tape_store: &TapeStore,
    ) -> f64 {
        // 1. Order Book Imbalance [-1.0, +1.0]
        let ob_imb = ob_store
            .get(symbol)
            .map(|book| book.imbalance())
            .unwrap_or(0.0);

        // 2. Trade Delta [-1.0, +1.0] (normalized)
        let trade_delta = tape_store
            .get(symbol)
            .map(|state| state.normalized_delta())
            .unwrap_or(0.0);

        // 3. CVD Trend (normalized to [-1.0, +1.0])
        let cvd_raw = tape_store
            .get(symbol)
            .map(|state| state.cvd_trend())
            .unwrap_or(0.0);
        let cvd_norm = cvd_raw.signum() * cvd_raw.abs().min(1.0);

        // Weighted composite
        let composite = self.config.ob_weight * ob_imb
            + self.config.tape_weight * trade_delta
            + self.config.cvd_weight * cvd_norm;

        // Clamp to [-1.0, +1.0]
        composite.clamp(-1.0, 1.0)
    }

    /// Get stats for monitoring/logging
    pub fn get_composite_score(
        &self,
        symbol: &str,
        ob_store: &OrderBookStore,
        tape_store: &TapeStore,
    ) -> CompositeScore {
        let ob_imb = ob_store
            .get(symbol)
            .map(|book| book.imbalance())
            .unwrap_or(0.0);

        let trade_delta = tape_store
            .get(symbol)
            .map(|state| state.normalized_delta())
            .unwrap_or(0.0);

        let cvd_trend = tape_store
            .get(symbol)
            .map(|state| state.cvd_trend())
            .unwrap_or(0.0);

        let composite = self.calculate_composite(symbol, ob_store, tape_store);

        CompositeScore {
            ob_imbalance: ob_imb,
            trade_delta,
            cvd_trend,
            composite,
        }
    }
}

/// Action to take after smart evaluation
#[derive(Debug, Clone)]
pub enum TrailAction {
    /// No change — keep current trail
    Hold,
    /// Force close the position immediately
    ForceClose(ExitReason),
    /// Drop 50% of the position and move SL to breakeven
    PartialExit,
    /// Tighten the trailing stop (multiply trail distance by factor < 1.0)
    TightenTrail { factor: f64 },
}

/// Diagnostic breakdown of the composite score
#[derive(Debug, Clone)]
pub struct CompositeScore {
    pub ob_imbalance: f64,
    pub trade_delta: f64,
    pub cvd_trend: f64,
    pub composite: f64,
}

impl std::fmt::Display for CompositeScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OB={:.2} Tape={:.2} CVD={:.2} → Composite={:.3}",
            self.ob_imbalance, self.trade_delta, self.cvd_trend, self.composite
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_default_config() {
        let config = TrailerConfig::default();
        assert!((config.ob_weight + config.tape_weight + config.cvd_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hold_when_in_loss() {
        let ob_store = Arc::new(dashmap::DashMap::new());
        let tape_store = Arc::new(dashmap::DashMap::new());

        let mut trailer = SmartTrailer::new(TrailerConfig::default());
        let pos = Position {
            symbol: "BTC/USDT".into(),
            direction: Direction::Long,
            entry_price: 70000.0,
            size: 0.1,
            sl_price: 69000.0,
            tp_price: Some(73000.0),
            strategy: "test".into(),
            risk_dist: 1000.0,
            trail_activate_r: 1.0,
            trail_atr_mult: 0.5,
            be_trigger_pct: 0.003,
            trail_pct: 0.0028,
            is_breakeven: false,
            trailing: super::super::position_manager::TrailingState {
                active: false,
                best_price: 70000.0,
                current_sl: 69000.0,
                last_candle_update: Instant::now(),
            },
            partially_exited: false,
            open_time: Instant::now(),
            observation: super::super::position_manager::ObservationWindow::new(),
        };

        // Price below entry = loss → should Hold
        let action = trailer.evaluate(&pos, 69500.0, &ob_store, &tape_store);
        assert!(matches!(action, TrailAction::Hold));
    }
}
