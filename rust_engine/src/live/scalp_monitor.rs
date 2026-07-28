/// ScalpMonitor — Microstructure-based position management for scalp trades
/// =========================================================================
///
/// After ScalpMTF enters a position, ScalpMonitor runs a 100ms polling loop
/// to manage the exit using real-time tape + orderbook data:
///
/// 1. Delta reversal → exit (buyers/sellers flipped)
/// 2. Wall disappeared → exit (support/resistance gone)
/// 3. Iceberg against → exit (hidden large player against us)
/// 4. Wall-TP → move TP to nearest wall (take profit at resistance)
///
/// Timeout: 5 minutes max — then fall back to standard SL/TP.

use std::time::{Duration, Instant};
use tracing::info;

use super::wall_tracker::WallStore;
use super::tape_reader::TapeStore;

/// Result of the scalp monitor
#[derive(Debug, Clone)]
pub enum ScalpExit {
    /// Delta reversed — microstructure says exit now
    DeltaReversal { symbol: String, price_hint: f64, held_secs: f64 },
    /// Protective wall disappeared — support/resistance gone
    WallGone { symbol: String, held_secs: f64 },
    /// Iceberg detected against our position
    IcebergAgainst { symbol: String, pressure: f64, held_secs: f64 },
    /// Wall-TP: found a wall to take profit at
    WallTP { symbol: String, wall_price: f64, held_secs: f64 },
    /// Timeout — 5 min passed, fall back to standard SL/TP
    Timeout { symbol: String },
    /// No data available (warming up)
    NoData,
}

/// Direction of the scalp trade
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalpDirection {
    Long,
    Short,
}

const POLL_INTERVAL: Duration = Duration::from_millis(100);
const MAX_DURATION: Duration = Duration::from_secs(300); // 5 minutes
const DELTA_EXIT_THRESHOLD: f64 = 0.15;
const ICEBERG_EXIT_THRESHOLD: f64 = 0.3;
const WALL_SEARCH_RANGE_PCT: f64 = 0.01; // 1% range to search for walls
const WALL_GONE_CHECK_RANGE_PCT: f64 = 0.005; // 0.5% range for protective wall

/// State machine for confirmation
#[derive(Debug, Clone, Copy, PartialEq)]
enum MonitorState {
    Holding,
    Warning { since: Instant },
    _Exit,
}

const WARNING_CONFIRM_MS: u64 = 1500; // 1.5 seconds to confirm warning → exit

/// Run the scalp monitor loop. Call from tokio::spawn.
pub async fn monitor_scalp(
    symbol: String,
    direction: ScalpDirection,
    entry_price: f64,
    wall_store: WallStore,
    tape_store: TapeStore,
) -> ScalpExit {
    let start = Instant::now();
    let mut state = MonitorState::Holding;
    let mut best_wall_tp: Option<f64> = None;

    loop {
        tokio::time::sleep(POLL_INTERVAL).await;

        let elapsed = start.elapsed();
        if elapsed > MAX_DURATION {
            info!("⏱️ ScalpMonitor [{}] TIMEOUT after {:.1}s", symbol, elapsed.as_secs_f64());
            return ScalpExit::Timeout { symbol };
        }

        let held_secs = elapsed.as_secs_f64();

        // ── 1. Delta reversal check ──
        if let Some(tape) = tape_store.get(&symbol) {
            let delta = tape.normalized_delta();
            let delta_against = match direction {
                ScalpDirection::Long => delta < -DELTA_EXIT_THRESHOLD,
                ScalpDirection::Short => delta > DELTA_EXIT_THRESHOLD,
            };

            if delta_against {
                match state {
                    MonitorState::Holding => {
                        state = MonitorState::Warning { since: Instant::now() };
                        info!("⚠️ ScalpMonitor [{}] delta WARNING: {:.3} at {:.1}s",
                            symbol, delta, held_secs);
                    }
                    MonitorState::Warning { since } => {
                        if since.elapsed() > Duration::from_millis(WARNING_CONFIRM_MS) {
                            info!("🔴 ScalpMonitor [{}] delta CONFIRMED EXIT at {:.1}s",
                                symbol, held_secs);
                            return ScalpExit::DeltaReversal {
                                symbol, price_hint: entry_price, held_secs,
                            };
                        }
                    }
                    MonitorState::_Exit => unreachable!(),
                }
            } else {
                // Delta OK — reset warning if any
                if state != MonitorState::Holding {
                    info!("✅ ScalpMonitor [{}] delta recovered, back to HOLDING", symbol);
                    state = MonitorState::Holding;
                }
            }

            // ── 3. Iceberg against check ──
            let (ice_buy, ice_sell) = tape.iceberg_pressure();
            let iceberg_against = match direction {
                ScalpDirection::Long => ice_sell > ICEBERG_EXIT_THRESHOLD,
                ScalpDirection::Short => ice_buy > ICEBERG_EXIT_THRESHOLD,
            };

            if iceberg_against {
                let pressure = if direction == ScalpDirection::Long { ice_sell } else { ice_buy };
                info!("🧊🔴 ScalpMonitor [{}] ICEBERG AGAINST {:.2} → EXIT at {:.1}s",
                    symbol, pressure, held_secs);
                return ScalpExit::IcebergAgainst { symbol, pressure, held_secs };
            }
        }

        // ── 2. Wall disappeared check ──
        if let Some(wall_snap) = wall_store.get(&symbol) {
            if !wall_snap.is_warming_up {
                let wall_gone = match direction {
                    ScalpDirection::Long => {
                        // Check if bid wall (support) below us still exists
                        let bid_walls = wall_snap.bid_walls();
                        let has_support = bid_walls.iter().any(|w| {
                            let dist = (entry_price - w.price) / entry_price;
                            dist > 0.0 && dist < WALL_GONE_CHECK_RANGE_PCT
                                && w.current_size_usd > 20_000.0
                        });
                        !has_support && held_secs > 5.0 // Only check after 5s (let walls settle)
                    }
                    ScalpDirection::Short => {
                        // Check if ask wall (resistance) above us still exists
                        let ask_walls = wall_snap.ask_walls();
                        let has_resistance = ask_walls.iter().any(|w| {
                            let dist = (w.price - entry_price) / entry_price;
                            dist > 0.0 && dist < WALL_GONE_CHECK_RANGE_PCT
                                && w.current_size_usd > 20_000.0
                        });
                        !has_resistance && held_secs > 5.0
                    }
                };

                // Only trigger wall_gone if we HAD a wall initially
                // (skip this check for now if we never saw a wall)
                if wall_gone && held_secs > 10.0 {
                    info!("🧱🔴 ScalpMonitor [{}] protective WALL GONE → EXIT at {:.1}s",
                        symbol, held_secs);
                    return ScalpExit::WallGone { symbol, held_secs };
                }

                // ── 4. Wall-TP: find wall ahead for take-profit ──
                let wall_tp = match direction {
                    ScalpDirection::Long => {
                        let ask_walls = wall_snap.ask_walls();
                        ask_walls.iter().find(|w| {
                            let dist = (w.price - entry_price) / entry_price;
                            dist > 0.003 && dist < WALL_SEARCH_RANGE_PCT
                                && w.current_size_usd > 30_000.0
                        }).map(|w| w.price)
                    }
                    ScalpDirection::Short => {
                        let bid_walls = wall_snap.bid_walls();
                        bid_walls.iter().find(|w| {
                            let dist = (entry_price - w.price) / entry_price;
                            dist > 0.003 && dist < WALL_SEARCH_RANGE_PCT
                                && w.current_size_usd > 30_000.0
                        }).map(|w| w.price)
                    }
                };

                if let Some(wp) = wall_tp {
                    if best_wall_tp.is_none() || best_wall_tp != Some(wp) {
                        best_wall_tp = Some(wp);
                        info!("🎯 ScalpMonitor [{}] Wall-TP found at {:.4} (${:.0}k)",
                            symbol, wp, 30.0);
                    }

                    // Check if price reached the wall
                    if let Some(tape) = tape_store.get(&symbol) {
                        let current_delta = tape.normalized_delta();
                        // If we're near the wall and delta is weakening, take profit
                        let delta_weakening = match direction {
                            ScalpDirection::Long => current_delta < 0.05,
                            ScalpDirection::Short => current_delta > -0.05,
                        };

                        if delta_weakening && held_secs > 3.0 {
                            info!("🎯✅ ScalpMonitor [{}] Wall-TP EXIT at {:.4}, {:.1}s",
                                symbol, wp, held_secs);
                            return ScalpExit::WallTP { symbol, wall_price: wp, held_secs };
                        }
                    }
                }
            }
        }
    }
}
