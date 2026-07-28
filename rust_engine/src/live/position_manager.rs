///! Position Manager
///! =================
///! Tracks open positions, monitors SL/TP hits on every tick,
///! and manages the Smart Trailing logic using OB + Tape + CVD composite score.
///! Phase 30: Adaptive Smart Trail — adjusts trail width based on live order flow.
///! Phase 30B: Observation Window — distinguishes pullback corrections from true reversals.
///! Phase 29C+2: Probabilistic Exit Engine — Decision Matrix with tag-based management.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tracing::info;

use super::order_book::OrderBookStore;
use super::tape_reader::{TapeStore, OrderFlowSignal};
use super::wall_tracker::{WallStore, WallSnapshot, WallSide};
use super::decision_matrix::{TradeTag, TradeAction, decide, format_tags};

/// Minimum TP distance as fraction of entry price.
/// TP must cover round-trip commission (0.05% × 2 = 0.10%) plus margin.
/// 0.15% ensures every TP hit is a real profit.
const MIN_TP_PCT: f64 = 0.0015;

/// Duration of the observation window before making cut/hold decision
const OBSERVATION_WINDOW_SECS: u64 = 60; // Phase 31C: 60s for knife_tick grid fills

/// Direction of the position
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Long,
    Short,
}

/// State of trailing stop
#[derive(Debug, Clone)]
pub struct TrailingState {
    pub active: bool,
    pub best_price: f64,
    pub current_sl: f64,
    pub last_candle_update: Instant,
}

/// Phase 30B: Observation Window for distinguishing corrections from reversals.
/// When price goes against us, instead of cutting instantly, we observe for N seconds.
#[derive(Debug, Clone)]
pub struct ObservationWindow {
    pub active: bool,
    pub started_at: Instant,
    /// Tape speed at the moment we entered observation
    pub entry_tape_speed: f64,
    /// How many ticks delta was against our direction
    pub delta_against_ticks: u32,
    /// Total ticks observed
    pub total_ticks: u32,
    /// Cooldown: after a decision, don't re-trigger for N seconds
    pub last_decision_at: Option<Instant>,
}

impl ObservationWindow {
    pub fn new() -> Self {
        Self {
            active: false,
            started_at: Instant::now(),
            entry_tape_speed: 0.0,
            delta_against_ticks: 0,
            total_ticks: 0,
            last_decision_at: None,
        }
    }

    pub fn start(&mut self, tape_speed: f64) {
        self.active = true;
        self.started_at = Instant::now();
        self.entry_tape_speed = tape_speed;
        self.delta_against_ticks = 0;
        self.total_ticks = 0;
    }

    pub fn elapsed_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    pub fn in_cooldown(&self) -> bool {
        if let Some(last) = self.last_decision_at {
            last.elapsed().as_secs() < 30 // 30s cooldown after each decision
        } else {
            false
        }
    }
}

/// A single open position
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub direction: Direction,
    pub entry_price: f64,
    pub size: f64,
    pub target_size: f64, // Phase 29C+1: Initial full intended grid size
    pub sl_price: f64,
    pub tp_price: Option<f64>,
    pub strategy: String,
    pub risk_dist: f64,           // |entry - original_sl|
    pub trail_activate_r: f64,    // Activation threshold in R (candle-based trail)
    pub trail_atr_mult: f64,      // Trail distance multiplier (candle-based trail)
    pub be_trigger_pct: f64,      // Breakeven trigger % (from GA, e.g. 0.003 = 0.3%)
    pub trail_pct: f64,           // Trailing stop % (from GA, e.g. 0.0028 = 0.28%)
    pub is_breakeven: bool,       // Has BE been triggered?
    pub trailing: TrailingState,
    pub partially_exited: bool,   // Has dropped 50% of size?
    pub open_time: Instant,
    pub observation: ObservationWindow, // Phase 30B: correction vs reversal detector
    pub pending_grid: Vec<(f64, f64)>,  // Phase 29C+1: Grid tracking (price, size)
    pub is_wall_backed: bool,     // Phase 31C: true if grid was placed on a real wall/cascade
    // Phase 36: RL Integration Fields
    pub squeeze_cvd: f64,
    pub local_extreme: f64,
}

impl Position {
    /// Phase 29C+1: Internal tracking of Limit Grid fills
    pub fn check_grid_fills(&mut self, current_price: f64) -> bool {
        if self.pending_grid.is_empty() {
            return false;
        }

        let mut newly_filled_size = 0.0;
        let mut value_filled = 0.0;

        // Separate filled vs still pending
        let mut remaining = Vec::new();

        for (price, q) in &self.pending_grid {
            let filled = match self.direction {
                Direction::Long => current_price <= *price,
                Direction::Short => current_price >= *price,
            };

            if filled {
                newly_filled_size += q;
                value_filled += price * q;
            } else {
                remaining.push((*price, *q));
            }
        }

        if newly_filled_size > 0.0 {
            let old_size = self.size;
            let old_entry = self.entry_price;
            
            // Calculate new weighted avg entry price
            let total_size = old_size + newly_filled_size;
            let new_entry = (old_entry * old_size + value_filled) / total_size;

            self.size = total_size;
            self.entry_price = new_entry;
            self.pending_grid = remaining;

            info!("📊 GRID FILL: {} {} {} added @ {:.4} (New Avg: {:.4}, Size: {:.4}, Pending: {})",
                if self.direction == Direction::Long { "LONG" } else { "SHORT" },
                self.symbol, newly_filled_size, value_filled / newly_filled_size, new_entry, total_size, self.pending_grid.len());
            
            return true;
        }
        false
    }

    /// Scale out a percentage of the current position size.
    /// Returns the absolute size amount that was reduced.
    pub fn scale_out(&mut self, pct: f64) -> f64 {
        let reduce_amount = self.size * pct.clamp(0.0, 1.0);
        self.size -= reduce_amount;
        
        if pct >= 0.5 {
            self.partially_exited = true;
        }
        
        info!("✂️ SCALED OUT: {} {} by {:.1}% ({} size). Remaining: {:.4}",
            if self.direction == Direction::Long { "LONG" } else { "SHORT" },
            self.symbol, pct * 100.0, reduce_amount, self.size);
            
        reduce_amount
    }

    /// Check if SL or TP is hit by the current candle
    /// FIX 4: 30-second SL immunity — SL cannot trigger in the first 30s after entry
    pub fn check_exit(&self, high: f64, low: f64) -> Option<ExitReason> {
        // FIX 4: SL immunity window — give trade 30s to breathe after entry
        // REMOVED for knife_tick — HFT scalps must cut losses instantly!
        let sl_immune = if self.strategy == "knifetick" || self.strategy == "knife_tick" {
            false
        } else {
            self.open_time.elapsed().as_secs() < 30 && !self.trailing.active
        };

        match self.direction {
            Direction::Long => {
                if low <= self.sl_price && !sl_immune {
                    return Some(if self.trailing.active {
                        ExitReason::Trail
                    } else {
                        ExitReason::StopLoss
                    });
                }
                if let Some(tp) = self.tp_price {
                    if high >= tp {
                        return Some(ExitReason::TakeProfit);
                    }
                }
            }
            Direction::Short => {
                if high >= self.sl_price && !sl_immune {
                    return Some(if self.trailing.active {
                        ExitReason::Trail
                    } else {
                        ExitReason::StopLoss
                    });
                }
                if let Some(tp) = self.tp_price {
                    if low <= tp {
                        return Some(ExitReason::TakeProfit);
                    }
                }
            }
        }
        None
    }

    /// Update trailing stop on every tick — matches backtest knife_tick.rs exactly.
    /// BE: when price moves >= be_trigger_pct from entry, SL moves to entry_price.
    /// Trail: best_price * (1 - trail_pct) for LONG, best_price * (1 + trail_pct) for SHORT.
    /// Returns true if SL was updated.
    pub fn update_trail_tick(&mut self, price: f64) -> bool {
        let mut updated = false;

        // Phase 29C+1: Process simulated Grid Limit fills FIRST
        if self.check_grid_fills(price) {
            updated = true;
        }

        if self.trail_pct <= 0.0 {
            return updated; // No tick-level trailing configured, but grid might have filled
        }


        match self.direction {
            Direction::Long => {
                // Update best price
                if price > self.trailing.best_price {
                    self.trailing.best_price = price;
                }

                // BE trigger: price moved up by be_trigger_pct from entry
                if !self.is_breakeven && self.be_trigger_pct > 0.0 {
                    let be_price = self.entry_price * (1.0 + self.be_trigger_pct);
                    if price >= be_price {
                        self.sl_price = self.entry_price;
                        self.trailing.current_sl = self.entry_price;
                        self.is_breakeven = true;
                        updated = true;
                        info!("🔄 {} BE triggered @ {:.4} (entry={:.4})",
                            self.symbol, price, self.entry_price);
                    }
                }

                // Trailing: only AFTER BE is triggered. Before BE, GA SL stands.
                if self.is_breakeven {
                    let trail_sl = self.trailing.best_price * (1.0 - self.trail_pct);
                    // SL can only move UP for LONG (take the higher of current SL and trail)
                    let effective_sl = self.sl_price.max(trail_sl);
                    if effective_sl > self.sl_price {
                        self.sl_price = effective_sl;
                        self.trailing.current_sl = effective_sl;
                        self.trailing.active = true;
                        updated = true;
                    }
                }
            }
            Direction::Short => {
                // Update best price (lowest for SHORT)
                if price < self.trailing.best_price {
                    self.trailing.best_price = price;
                }

                // BE trigger: price moved down by be_trigger_pct from entry
                if !self.is_breakeven && self.be_trigger_pct > 0.0 {
                    let be_price = self.entry_price * (1.0 - self.be_trigger_pct);
                    if price <= be_price {
                        self.sl_price = self.entry_price;
                        self.trailing.current_sl = self.entry_price;
                        self.is_breakeven = true;
                        updated = true;
                        info!("🔄 {} BE triggered @ {:.4} (entry={:.4})",
                            self.symbol, price, self.entry_price);
                    }
                }

                // Trailing: only AFTER BE is triggered. Before BE, GA SL stands.
                if self.is_breakeven {
                    let trail_sl = self.trailing.best_price * (1.0 + self.trail_pct);
                    // SL can only move DOWN for SHORT (take the lower of current SL and trail)
                    let effective_sl = self.sl_price.min(trail_sl);
                    if effective_sl < self.sl_price {
                        self.sl_price = effective_sl;
                        self.trailing.current_sl = effective_sl;
                        self.trailing.active = true;
                        updated = true;
                    }
                }
            }
        }

        updated
    }

    /// Phase 30B: Smart Trail V2 — Observation Window approach.
    /// Instead of instant cut on negative delta, we observe for 10 seconds
    /// and measure 3 metrics to distinguish corrections from reversals:
    ///   1. delta_persistence: what % of ticks had delta against us?
    ///   2. volume_ratio: is tape speed increasing (reversal) or decreasing (correction)?
    ///   3. speed_acceleration: is the move accelerating or fading?
    /// Returns (trail_updated, should_early_cut)
    pub fn update_trail_tick_smart(&mut self, price: f64, flow: &OrderFlowSignal) -> (bool, bool) {
        // HFT scalps and other strategies use the Observation Window lag
        if self.trail_pct <= 0.0 {
            return (self.update_trail_tick(price), false);
        }

        let delta = flow.delta;

        // Calculate current PnL in R
        let pnl_r = if self.risk_dist > 0.0 {
            match self.direction {
                Direction::Long => (price - self.entry_price) / self.risk_dist,
                Direction::Short => (self.entry_price - price) / self.risk_dist,
            }
        } else {
            0.0
        };

        // Is delta supporting our direction?
        let delta_supports = match self.direction {
            Direction::Long => delta > 0.05,
            Direction::Short => delta < -0.05,
        };
        let delta_against = match self.direction {
            Direction::Long => delta < -0.05,
            Direction::Short => delta > 0.05,
        };
        let lacks_support = match self.direction {
            Direction::Long => delta <= 0.0,
            Direction::Short => delta >= 0.0,
        };

        // === PROFIT EXHAUSTION DETECTOR (Phase 32) ===
        // When we're in good profit, monitor the tape for momentum death.
        // If delta flips against us AND price confirms retreat from best → instant exit.
        // This catches the TOP of the bounce instead of giving it all back to trail.
        if pnl_r > 0.5 {
            let delta_flipped = match self.direction {
                Direction::Long  => delta < -0.05,  // buyers exhausted, sellers taking over
                Direction::Short => delta > 0.05,   // sellers exhausted, buyers taking over
            };
            let price_retreating = match self.direction {
                Direction::Long  => price < self.trailing.best_price * 0.999,  // 0.1% off high
                Direction::Short => price > self.trailing.best_price * 1.001,  // 0.1% off low
            };

            if delta_flipped && price_retreating {
                let retreat_pct = match self.direction {
                    Direction::Long  => (1.0 - price / self.trailing.best_price) * 100.0,
                    Direction::Short => (price / self.trailing.best_price - 1.0) * 100.0,
                };
                info!("🔪💰 {} PROFIT EXHAUSTION: pnl={:.2}R, delta={:.3}, retreat={:.3}% from best={:.6} → INSTANT EXIT",
                    self.symbol, pnl_r, delta, retreat_pct, self.trailing.best_price);
                return (false, true); // cut with profit locked
            }
        }

        // === TOXIC PANIC EJECT (Phase 33) ===
        // If we are in negative PnL and the tape speed is violently high combined with delta against us,
        // it means an HFT/Iceberg is smashing our wall. Do not wait 60 seconds! Cut instantly.
        if pnl_r < -0.2 && flow.tape_speed >= 4.5 && delta_against {
            info!("🚨💥 {} TOXIC SPEED EJECT: pnl={:.2}R, speed={:.1} t/s, delta={:.3}. ICEBERG DETECTED! INSTANT CLOSE.",
                self.symbol, pnl_r, flow.tape_speed, delta);
            return (false, true); // instant early cut
        }

        // === OBSERVATION WINDOW (Phase 30B) ===
        // Trigger: PnL is negative, delta lacks support, not in BE, not in cooldown
        if pnl_r < -0.3 && lacks_support && !self.is_breakeven {
            if !self.observation.active && !self.observation.in_cooldown() {
                // PHASE 1: Start observation — DON'T cut yet!
                self.observation.start(flow.tape_speed);
                info!("🔍 {} OBSERVATION START: pnl={:.2}R, delta={:.3}, speed={:.1}",
                    self.symbol, pnl_r, delta, flow.tape_speed);
                return (false, false);
            }

            if self.observation.active {
                // PHASE 2: Collecting data
                self.observation.total_ticks += 1;
                if delta_against {
                    self.observation.delta_against_ticks += 1;
                }

                if self.observation.elapsed_secs() < OBSERVATION_WINDOW_SECS {
                    // Still observing, don't decide yet
                    return (false, false);
                }

                // PHASE 3: Decision time!
                self.observation.active = false;
                self.observation.last_decision_at = Some(Instant::now());

                let total = self.observation.total_ticks.max(1) as f64;

                // Metric 1: Delta persistence (0.0–1.0)
                // High = delta was consistently against us = reversal
                let delta_persistence = self.observation.delta_against_ticks as f64 / total;

                // Metric 2: Volume ratio — is tape getting louder?
                // current speed / speed when observation started
                // > 1.0 = louder (reversal), < 1.0 = quieter (correction)
                let volume_ratio = if self.observation.entry_tape_speed > 0.1 {
                    (flow.tape_speed / self.observation.entry_tape_speed).min(3.0) / 3.0
                } else {
                    0.5 // unknown baseline
                };

                // Metric 3: Speed acceleration
                // > 1.0 = accelerating (reversal), < 1.0 = decelerating (correction)
                let speed_factor = (flow.speed_acceleration.min(3.0) / 3.0).max(0.0);

                // Reversal Score: weighted composite
                let reversal_score = 0.40 * delta_persistence
                                   + 0.35 * volume_ratio
                                   + 0.25 * speed_factor;

                info!("🔍 {} DECISION: score={:.2} [delta_p={:.2} vol_r={:.2} speed_f={:.2}] pnl={:.2}R",
                    self.symbol, reversal_score, delta_persistence, volume_ratio, speed_factor, pnl_r);

                if reversal_score > 0.6 {
                    info!("🔪🔴 {} REVERSAL CONFIRMED (score={:.2}) — cutting loss",
                        self.symbol, reversal_score);
                    return (false, true); // TRUE reversal → cut
                } else {
                    info!("💎🟢 {} CORRECTION DETECTED (score={:.2}) — holding position",
                        self.symbol, reversal_score);
                    // It's a correction, don't cut. Let SL handle worst case.
                    return (false, false);
                }
            }
        } else if self.observation.active && !lacks_support {
            // Delta recovered while we were observing → cancel observation
            info!("💎 {} observation cancelled — delta recovered (delta={:.3})",
                self.symbol, delta);
            self.observation.active = false;
        }

        // === ADAPTIVE TRAIL WIDTH ===
        let adaptive_trail = if delta_supports {
            self.trail_pct * 2.5
        } else if lacks_support {
            self.trail_pct * 0.8
        } else {
            self.trail_pct
        };

        let mut updated = false;

        match self.direction {
            Direction::Long => {
                if price > self.trailing.best_price {
                    self.trailing.best_price = price;
                }

                if !self.is_breakeven && self.be_trigger_pct > 0.0 {
                    let be_price = self.entry_price * (1.0 + self.be_trigger_pct);
                    if price >= be_price {
                        self.sl_price = self.entry_price;
                        self.trailing.current_sl = self.entry_price;
                        self.is_breakeven = true;
                        updated = true;
                        info!("🔄 {} BE triggered @ {:.4} (entry={:.4})",
                            self.symbol, price, self.entry_price);
                    }
                }

                if self.is_breakeven {
                    let trail_sl = self.trailing.best_price * (1.0 - adaptive_trail);
                    let effective_sl = self.sl_price.max(trail_sl);
                    if effective_sl > self.sl_price {
                        self.sl_price = effective_sl;
                        self.trailing.current_sl = effective_sl;
                        self.trailing.active = true;
                        updated = true;
                    }
                }
            }
            Direction::Short => {
                if price < self.trailing.best_price {
                    self.trailing.best_price = price;
                }

                if !self.is_breakeven && self.be_trigger_pct > 0.0 {
                    let be_price = self.entry_price * (1.0 - self.be_trigger_pct);
                    if price <= be_price {
                        self.sl_price = self.entry_price;
                        self.trailing.current_sl = self.entry_price;
                        self.is_breakeven = true;
                        updated = true;
                        info!("🔄 {} BE triggered @ {:.4} (entry={:.4})",
                            self.symbol, price, self.entry_price);
                    }
                }

                if self.is_breakeven {
                    let trail_sl = self.trailing.best_price * (1.0 + adaptive_trail);
                    let effective_sl = self.sl_price.min(trail_sl);
                    if effective_sl < self.sl_price {
                        self.sl_price = effective_sl;
                        self.trailing.current_sl = effective_sl;
                        self.trailing.active = true;
                        updated = true;
                    }
                }
            }
        }

        (updated, false)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 29C+2: PROBABILISTIC EXIT ENGINE (Decision Matrix)
    // ═══════════════════════════════════════════════════════════════════════

    /// Full-brain update: collect tags from all pillars, run Decision Matrix,
    /// then delegate to appropriate trail behavior.
    ///
    /// Arguments:
    ///   - price: current market price
    ///   - flow: tape/order flow signal
    ///   - wall_snap: optional wall snapshot for this symbol
    ///   - btc_momentum: BTC 5m momentum (from candle buffers), None if unavailable
    ///
    /// Returns: (trail_updated: bool, should_exit: bool)
    pub fn update_with_brain(
        &mut self,
        price: f64,
        flow: &OrderFlowSignal,
        wall_snap: Option<&WallSnapshot>,
        btc_momentum: Option<f64>,
    ) -> (bool, bool) {
        let mut tags: Vec<TradeTag> = Vec::with_capacity(8);
        let is_long = self.direction == Direction::Long;
        let elapsed = self.open_time.elapsed().as_secs();

        // Calculate current PnL in R
        let pnl_r = if self.risk_dist > 0.0 {
            match self.direction {
                Direction::Long => (price - self.entry_price) / self.risk_dist,
                Direction::Short => (self.entry_price - price) / self.risk_dist,
            }
        } else {
            0.0
        };

        // ── PILLAR 1: Time Decay ──────────────────────────────────────────
        if elapsed > 60 && pnl_r < 0.1 {
            tags.push(TradeTag::StaleTrade);
        }
        if elapsed > 90 && pnl_r < 0.0 {
            tags.push(TradeTag::StaleLosing);
        }

        // ── PILLAR 2: Order Book (Walls) ──────────────────────────────────
        if let Some(snap) = wall_snap {
            // Dynamic Target: find nearest wall ahead
            if let Some(wall_ahead) = snap.find_wall_ahead(price, is_long, 0.003, 0.020) {
                if wall_ahead.is_spoof {
                    tags.push(TradeTag::SpoofWallAhead);
                } else {
                    tags.push(TradeTag::DynamicTargetSet);
                    // Calculate dynamic TP: stop 0.05% before the wall
                    let buffer = wall_ahead.price * 0.0005;
                    let dynamic_tp = if is_long {
                        wall_ahead.price - buffer
                    } else {
                        wall_ahead.price + buffer
                    };
                    // Only set dynamic TP if it gives better R than current TP
                    let dynamic_r = if self.risk_dist > 0.0 {
                        (dynamic_tp - self.entry_price).abs() / self.risk_dist
                    } else { 0.0 };
                    if dynamic_r > 0.3 {
                        // Override TP with dynamic target
                        let old_tp = self.tp_price;
                        self.tp_price = Some(dynamic_tp);
                        info!("🎯 {} DYNAMIC TP: wall@{:.6} stab={:.2} → TP={:.6} ({:.1}R) [was {:?}]",
                            self.symbol, wall_ahead.price, wall_ahead.stability, dynamic_tp, dynamic_r, old_tp);
                    }
                }
            }

            // Iceberg detection: check walls near our position
            for w in snap.iceberg_walls() {
                let dist_pct = (w.price - price).abs() / price;
                if dist_pct > 0.02 { continue; } // Only care about <2% away

                let wall_is_behind_us = match self.direction {
                    Direction::Long => w.side == WallSide::Bid && w.price < price,
                    Direction::Short => w.side == WallSide::Ask && w.price > price,
                };
                let wall_is_ahead_of_us = match self.direction {
                    Direction::Long => w.side == WallSide::Ask && w.price > price,
                    Direction::Short => w.side == WallSide::Bid && w.price < price,
                };

                if wall_is_behind_us {
                    tags.push(TradeTag::IcebergShield);
                    info!("🧊🛡️ {} ICEBERG SHIELD: {:?}@{:.6} refills={} behind our {:?}",
                        self.symbol, w.side, w.price, w.refresh_count, self.direction);
                }
                if wall_is_ahead_of_us {
                    tags.push(TradeTag::IcebergAgainst);
                    info!("🧊⚠️ {} ICEBERG AGAINST: {:?}@{:.6} refills={} blocking our exit",
                        self.symbol, w.side, w.price, w.refresh_count);
                }
            }
        }

        // ── PILLAR 3: Tape (Sweep & Whale Detection) ─────────────────────
        if flow.sweep_score > 15.0 {
            let sweep_supports = match self.direction {
                Direction::Long => flow.sweep_direction_is_buy,
                Direction::Short => !flow.sweep_direction_is_buy,
            };
            if sweep_supports {
                tags.push(TradeTag::SweepForUs);
                info!("🌊✅ {} SWEEP FOR US: score={:.0} direction={}",
                    self.symbol, flow.sweep_score, if flow.sweep_direction_is_buy { "BUY" } else { "SELL" });
            } else {
                tags.push(TradeTag::SweepAgainst);
                info!("🌊⚠️ {} SWEEP AGAINST: score={:.0} direction={}",
                    self.symbol, flow.sweep_score, if flow.sweep_direction_is_buy { "BUY" } else { "SELL" });
            }
        }

        // Whale print detection
        if flow.max_single_print_usd > 50_000.0 {
            // Check if latest whale print direction matches our trade
            let whale_is_buy = flow.large_prints_buy > flow.large_prints_sell;
            let whale_supports = match self.direction {
                Direction::Long => whale_is_buy,
                Direction::Short => !whale_is_buy,
            };
            if whale_supports {
                tags.push(TradeTag::WhalePrintFor);
            } else {
                tags.push(TradeTag::WhalePrintAgainst);
            }
        }

        // ── PILLAR 4: BTC Macro ──────────────────────────────────────────
        if let Some(btc_mom) = btc_momentum {
            let wind_for = match self.direction {
                Direction::Long => btc_mom > 0.0015,   // BTC up >0.15% on 5m
                Direction::Short => btc_mom < -0.0020,  // BTC down >0.20% on 5m
            };
            let wind_against = match self.direction {
                Direction::Long => btc_mom < -0.0020,
                Direction::Short => btc_mom > 0.0015,
            };
            if wind_for {
                tags.push(TradeTag::BtcWindFor);
            }
            if wind_against {
                tags.push(TradeTag::BtcWindAgainst);
            }
        }

        // ── DECISION ─────────────────────────────────────────────────────
        let action = decide(&tags);

        // Log tags + action (only when something interesting is happening)
        if !tags.is_empty() {
            info!("🧠 {} BRAIN: {} → {} | pnl={:.2}R elapsed={}s",
                self.symbol, format_tags(&tags), action, pnl_r, elapsed);
        }

        // ── APPLY ACTION ─────────────────────────────────────────────────
        match action {
            TradeAction::PanicEject => {
                info!("🚨 {} PANIC EJECT by Decision Matrix! tags={}",
                    self.symbol, format_tags(&tags));
                return (false, true); // instant exit
            }
            TradeAction::TightenTrail => {
                // Override trail_pct temporarily to 0.15% for this tick
                let saved = self.trail_pct;
                self.trail_pct = 0.0015; // Very tight — exit on first micro-pullback
                let result = self.update_trail_tick_smart(price, flow);
                self.trail_pct = saved;
                return result;
            }
            TradeAction::WidenTrail => {
                // Widen trail to 0.8% — let the momentum run
                let saved = self.trail_pct;
                self.trail_pct = 0.008;
                let result = self.update_trail_tick_smart(price, flow);
                self.trail_pct = saved;
                return result;
            }
            TradeAction::RideSweep => {
                // God Mode: 1.0% trail, skip Profit Exhaustion check
                let saved = self.trail_pct;
                self.trail_pct = 0.010;
                // Call basic trail (skip smart trail's profit exhaustion)
                let updated = self.update_trail_tick(price);
                self.trail_pct = saved;
                return (updated, false);
            }
            TradeAction::Hold => {
                // Standard behavior — delegate normally
                return self.update_trail_tick_smart(price, flow);
            }
        }
    }

    /// Enforce minimum TP distance so commission never eats the profit.
    /// Call after constructing Position.
    pub fn enforce_min_tp(&mut self) {
        if let Some(tp) = self.tp_price {
            let min_dist = self.entry_price * MIN_TP_PCT;
            let actual_dist = (tp - self.entry_price).abs();
            if actual_dist < min_dist {
                let new_tp = match self.direction {
                    Direction::Long => self.entry_price + min_dist,
                    Direction::Short => self.entry_price - min_dist,
                };
                info!("⚠️ {} TP too close ({:.6} < {:.6}), enforcing min TP → {:.4}",
                    self.symbol, actual_dist, min_dist, new_tp);
                self.tp_price = Some(new_tp);
            }
        }
    }

    /// Update trailing stop based on candle close (5-min intervals)
    /// Returns true if trail was updated
    pub fn update_trail_candle(&mut self, high: f64, low: f64) -> bool {
        if self.risk_dist <= 0.0 {
            return false;
        }

        // Only update every 5 minutes (candle-based, not tick-based)
        if self.trailing.last_candle_update.elapsed() < Duration::from_secs(295) {
            return false;
        }
        self.trailing.last_candle_update = Instant::now();

        // Update best price
        match self.direction {
            Direction::Long => {
                self.trailing.best_price = self.trailing.best_price.max(high);
                let profit_r = (self.trailing.best_price - self.entry_price) / self.risk_dist;

                if profit_r >= self.trail_activate_r {
                    let new_sl = self.trailing.best_price - self.risk_dist * self.trail_atr_mult;
                    // Profit floor: never trail below entry + 0.3R
                    let min_sl = self.entry_price + self.risk_dist * 0.3;
                    let new_sl = new_sl.max(min_sl);

                    if !self.trailing.active {
                        info!("🔄 {} trailing activated @ {:.4} ({:.1}R)",
                            self.symbol, self.trailing.best_price, profit_r);
                        self.trailing.active = true;
                    }

                    if new_sl > self.sl_price {
                        self.sl_price = new_sl;
                        self.trailing.current_sl = new_sl;
                        return true;
                    }
                }
            }
            Direction::Short => {
                self.trailing.best_price = self.trailing.best_price.min(low);
                let profit_r = (self.entry_price - self.trailing.best_price) / self.risk_dist;

                if profit_r >= self.trail_activate_r {
                    let new_sl = self.trailing.best_price + self.risk_dist * self.trail_atr_mult;
                    let min_sl = self.entry_price - self.risk_dist * 0.3;
                    let new_sl = new_sl.min(min_sl);

                    if !self.trailing.active {
                        info!("🔄 {} trailing activated @ {:.4} ({:.1}R)",
                            self.symbol, self.trailing.best_price, profit_r);
                        self.trailing.active = true;
                    }

                    if new_sl < self.sl_price {
                        self.sl_price = new_sl;
                        self.trailing.current_sl = new_sl;
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Smart exit check using OB Imbalance + Trade Delta + CVD
    /// Returns true if position should be force-closed
    pub fn should_smart_exit(
        &self,
        ob_store: &OrderBookStore,
        tape_store: &TapeStore,
    ) -> bool {
        let ob_imb = ob_store
            .get(&self.symbol)
            .map(|book| book.imbalance())
            .unwrap_or(0.0);

        let (trade_delta, cvd_trend) = tape_store
            .get(&self.symbol)
            .map(|state| (state.normalized_delta(), state.cvd_trend()))
            .unwrap_or((0.0, 0.0));

        // Normalize CVD trend to [-1, 1] range
        let cvd_norm = cvd_trend.signum() * (cvd_trend.abs().min(1.0));

        // Composite score: weighted combination
        let composite = 0.3 * ob_imb + 0.5 * trade_delta + 0.2 * cvd_norm;

        match self.direction {
            Direction::Long => composite < -0.5,  // Heavy selling pressure
            Direction::Short => composite > 0.5,   // Heavy buying pressure
        }
    }
}

/// Reason for closing a position
#[derive(Debug, Clone, Copy)]
pub enum ExitReason {
    StopLoss,
    TakeProfit,
    Trail,
    SmartExit, // OB + Tape forced exit
}

impl ExitReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExitReason::StopLoss => "SL",
            ExitReason::TakeProfit => "TP",
            ExitReason::Trail => "TRAIL",
            ExitReason::SmartExit => "SMART_EXIT",
        }
    }
}

/// Manages all open positions
pub struct PositionManager {
    pub positions: HashMap<String, Position>, // symbol → position
}

impl PositionManager {
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
        }
    }

    pub fn open(&mut self, position: Position) {
        info!("📄 OPEN {} {} @ {:.4} | SL={:.4} | size={:.6}",
            position.direction_str(), position.symbol,
            position.entry_price, position.sl_price, position.size);
        self.positions.insert(position.symbol.clone(), position);
    }

    pub fn close(&mut self, symbol: &str, reason: ExitReason) -> Option<Position> {
        if let Some(pos) = self.positions.remove(symbol) {
            info!("📄 CLOSED {} {} | Reason: {}",
                pos.direction_str(), symbol, reason.as_str());
            Some(pos)
        } else {
            None
        }
    }

    pub fn has_position(&self, symbol: &str) -> bool {
        self.positions.contains_key(symbol)
    }

    pub fn long_count(&self) -> usize {
        self.positions.values().filter(|p| p.direction == Direction::Long).count()
    }

    pub fn short_count(&self) -> usize {
        self.positions.values().filter(|p| p.direction == Direction::Short).count()
    }
}

impl Position {
    pub fn direction_str(&self) -> &'static str {
        match self.direction {
            Direction::Long => "LONG",
            Direction::Short => "SHORT",
        }
    }
}
