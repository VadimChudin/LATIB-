///! Density Radar — Phase 30.5: Real-time S/R proximity monitor
///! ==============================================================
///! Monitors price distance to nearest S/R level for each symbol.
///! State machine per symbol:
///!
///!   Idle ──► Yellow (<1.5%) ──► Red (<0.5% + wall + age>2h) ──► Active (Absorber tracking)
///!
///! When Red triggers, creates an HftTarget{strategy_type: "breakout"} and hands
///! it to the AbsorberTracker for 50ms micro-loop confirmation.
///!
///! Dependencies: LevelStore (S/R levels), WallStore (wall validation), TapeStore (delta/speed)

use std::collections::HashMap;
use std::time::Instant;

use tracing::{info, debug};

use super::absorber::HftTarget;
use super::level_tracker::LevelStore;
use super::position_manager::Direction;
use super::wall_tracker::WallStore;
use super::tape_reader::TapeStore;

// ── Config ──────────────────────────────────────────────────────────────────

/// Distance thresholds (fraction of price)
const YELLOW_ZONE_PCT: f64 = 0.015;    // 1.5% from level → start watching
const RED_ZONE_PCT: f64 = 0.005;       // 0.5% from level → escalate if conditions met

/// Wall validation
const MIN_WALL_AGE_SECS: u64 = 7200;   // Wall must be ≥ 2 hours old (anti-spoof)
const MIN_WALL_SIZE_USD: f64 = 30_000.0; // Minimum wall size to qualify

/// Cooldowns
const FIRE_COOLDOWN_SECS: u64 = 300;   // 5 min cooldown after firing on a level
const RED_HOLD_SECS: u64 = 10;         // Must stay in Red zone for 10s before firing

// ── State Machine ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RadarState {
    /// Price is far from any level — no action
    Idle,
    /// Price within 1.5% of a level — logging, preparing
    Yellow {
        level_price: f64,
        level_touches: u32,
        entered_at: Instant,
    },
    /// Price within 0.5% + wall confirmed — ready to fire
    Red {
        level_price: f64,
        level_touches: u32,
        wall_price: f64,
        wall_size_usd: f64,
        entered_at: Instant,
        side: LevelSide,
    },
    /// Already fired — cooling down
    Active {
        level_price: f64,
        fired_at: Instant,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LevelSide {
    /// Price approaching from below (testing resistance)
    Resistance,
    /// Price approaching from above (testing support)
    Support,
}

/// Per-symbol radar tracking state
#[derive(Debug, Clone)]
struct SymbolRadar {
    state: RadarState,
    last_check: Instant,
}

// ── Density Radar ───────────────────────────────────────────────────────────

pub struct DensityRadar {
    /// Per-symbol state machines
    states: HashMap<String, SymbolRadar>,
}

impl DensityRadar {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    /// Main check — called every 5s from orchestrator on_candle_close or periodic timer.
    /// Returns list of HftTargets to send to the AbsorberTracker.
    pub fn check(
        &mut self,
        symbol: &str,
        current_price: f64,
        level_store: &LevelStore,
        wall_store: &WallStore,
        tape_store: &TapeStore,
    ) -> Option<HftTarget> {
        // Get or create symbol state
        let radar = self.states.entry(symbol.to_string()).or_insert(SymbolRadar {
            state: RadarState::Idle,
            last_check: Instant::now(),
        });
        radar.last_check = Instant::now();

        // Find nearest S/R level
        let level_snap = level_store.get(symbol)?;
        let nearest = level_snap.nearest_level(current_price, 3.0)?; // Within 3 ATR

        let dist_pct = (current_price - nearest.price).abs() / current_price;
        let side = if current_price < nearest.price {
            LevelSide::Resistance
        } else {
            LevelSide::Support
        };

        // State machine transitions
        match &radar.state {
            RadarState::Idle => {
                if dist_pct < YELLOW_ZONE_PCT {
                    info!("🟡 DensityRadar [{}] → YELLOW: price {:.6} near level {:.6} ({:.2}%)",
                        symbol, current_price, nearest.price, dist_pct * 100.0);
                    radar.state = RadarState::Yellow {
                        level_price: nearest.price,
                        level_touches: nearest.touches,
                        entered_at: Instant::now(),
                    };
                }
                None
            }

            RadarState::Yellow { level_price, level_touches, .. } => {
                let lp = *level_price;
                let lt = *level_touches;

                // Check if we left the zone
                if dist_pct > YELLOW_ZONE_PCT * 1.2 {
                    debug!("🟡→⚫ DensityRadar [{}] left Yellow zone", symbol);
                    radar.state = RadarState::Idle;
                    return None;
                }

                // Check for Red conditions
                if dist_pct < RED_ZONE_PCT {
                    // Validate wall
                    if let Some(wall_info) = validate_wall(symbol, &side, wall_store) {
                        info!("🔴 DensityRadar [{}] → RED: {:.6} within {:.3}% of level {:.6} | wall ${:.0}k age={:.0}h",
                            symbol, current_price, dist_pct * 100.0, lp,
                            wall_info.0 / 1000.0, wall_info.1 / 3600.0);
                        radar.state = RadarState::Red {
                            level_price: lp,
                            level_touches: lt,
                            wall_price: wall_info.2,
                            wall_size_usd: wall_info.0,
                            entered_at: Instant::now(),
                            side: side.clone(),
                        };
                    }
                }
                None
            }

            RadarState::Red { level_price, level_touches, wall_price, wall_size_usd, entered_at, side: red_side } => {
                let lp = *level_price;
                let lt = *level_touches;
                let wp = *wall_price;
                let ws = *wall_size_usd;
                let entered = *entered_at;
                let rs = red_side.clone();

                // Left the zone?
                if dist_pct > RED_ZONE_PCT * 2.0 {
                    info!("🔴→🟡 DensityRadar [{}] left Red zone", symbol);
                    radar.state = RadarState::Yellow {
                        level_price: lp,
                        level_touches: lt,
                        entered_at: Instant::now(),
                    };
                    return None;
                }

                // Must hold Red position for RED_HOLD_SECS
                if entered.elapsed().as_secs() < RED_HOLD_SECS {
                    return None;
                }

                // Check tape: need some activity (not dead market)
                let has_flow = tape_store.get(symbol)
                    .map(|t| t.tape_speed() > 1.0)
                    .unwrap_or(false);

                if !has_flow {
                    return None;
                }

                // ═══ FIRE → Create HftTarget ═══
                let direction = match rs {
                    LevelSide::Resistance => Direction::Long,  // Breakout up through resistance
                    LevelSide::Support => Direction::Short,     // Breakout down through support
                };

                let delta_abs = tape_store.get(symbol)
                    .map(|t| t.normalized_delta().abs())
                    .unwrap_or(0.0);

                info!("🔴🔥 DensityRadar FIRE [{}] → {:?} breakout near {:.6} | wall={:.6} ${:.0}k | touches={} | delta={:.3}",
                    symbol, direction, lp, wp, ws / 1000.0, lt, delta_abs);

                radar.state = RadarState::Active {
                    level_price: lp,
                    fired_at: Instant::now(),
                };

                Some(HftTarget {
                    symbol: symbol.to_string(),
                    direction,
                    created_at: Instant::now(),
                    initial_delta_abs: delta_abs,
                    strategy_type: "breakout".to_string(),
                    target_wall_price: Some(wp),
                })
            }

            RadarState::Active { level_price, fired_at } => {
                let lp = *level_price;

                // Cooldown expired?
                if fired_at.elapsed().as_secs() > FIRE_COOLDOWN_SECS {
                    debug!("🔴→⚫ DensityRadar [{}] cooldown expired", symbol);
                    radar.state = RadarState::Idle;
                }

                // Check if price broke away from level significantly
                if dist_pct > YELLOW_ZONE_PCT * 2.0 {
                    debug!("🔴→⚫ DensityRadar [{}] price moved away after fire", symbol);
                    radar.state = RadarState::Idle;
                }

                None
            }
        }
    }



    /// Get current state for a symbol (for telemetry/logging)
    pub fn get_state(&self, symbol: &str) -> Option<&RadarState> {
        self.states.get(symbol).map(|r| &r.state)
    }

    /// Count how many symbols are in each state
    pub fn state_summary(&self) -> (usize, usize, usize, usize) {
        let mut idle = 0;
        let mut yellow = 0;
        let mut red = 0;
        let mut active = 0;
        for r in self.states.values() {
            match r.state {
                RadarState::Idle => idle += 1,
                RadarState::Yellow { .. } => yellow += 1,
                RadarState::Red { .. } => red += 1,
                RadarState::Active { .. } => active += 1,
            }
        }
        (idle, yellow, red, active)
    }
}

// ── Free Functions ──────────────────────────────────────────────────────────

/// Validate that a real wall exists on the correct side (free fn to avoid borrow issues)
fn validate_wall(
    symbol: &str,
    side: &LevelSide,
    wall_store: &WallStore,
) -> Option<(f64, f64, f64)> {
    // Returns: (wall_size_usd, wall_age_secs, wall_price)
    let snap = wall_store.get(symbol)?;
    if snap.is_warming_up {
        return None;
    }

    let walls = match side {
        LevelSide::Resistance => snap.ask_walls(),
        LevelSide::Support => snap.bid_walls(),
    };

    for wall in &walls {
        if wall.current_size_usd >= MIN_WALL_SIZE_USD {
            let age = wall.age_secs();
            if age >= MIN_WALL_AGE_SECS {
                let eaten = wall.eaten_pct();
                if eaten < 0.8 {
                    return Some((wall.current_size_usd, age as f64, wall.price));
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_machine_transitions() {
        let mut radar = DensityRadar::new();
        // Just verify construction works
        assert_eq!(radar.state_summary(), (0, 0, 0, 0));
    }
}
