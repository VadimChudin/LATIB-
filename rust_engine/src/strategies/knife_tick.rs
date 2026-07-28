use crate::backtest::Trade;
use crate::tick_backtest::Epicenter;
use std::collections::VecDeque;

/// Phase 31 v6: v3 logic (CVD + Absorption + Reclaim) + v5 risk management
/// =========================================================================
/// Entry checklist (ALL must pass after squeeze):
///   1. CVD DIVERGENCE: CVD improving vs squeeze start (buyers returning)
///   2. ABSORPTION: high volume but price stopped falling (vol/δprice)
///   3. RECLAIM: price bounced from local low by min_reclaim_pct
///   + speed filter
///
/// Risk management:
///   - Grid entry: avg price over 5 ticks
///   - Dynamic TP: FIXED 80% recovery of dump size
///   - SL: below local low + buffer
///   - Trail ONLY after BE
///
/// params:
///   0: window_ms         - dump detection window
///   1: min_zscore         - Z-score threshold
///   2: min_vol_spike      - volume >= baseline × this
///   3: (unused)           - tp_recovery hardcoded at 0.8
///   4: sl_buffer_pct      - SL buffer below local low
///   5: be_trigger_pct     - move SL to BE when profit reaches this
///   6: trail_pct          - trailing stop (only after BE)
///   7: micro_window_ms    - window for absorption/speed checks
///   8: min_absorption     - absorption ratio threshold
///   9: min_reclaim_pct    - price must bounce this % from low
///  10: max_speed_mult     - tape speed <= baseline × this
pub fn evaluate_epicenter(epicenter: &Epicenter, params: &[f64]) -> Option<Trade> {
    if params.len() < 5 { return None; }
    let direction = &epicenter.direction;
    
    let window_ms = params[0] as u64;
    let min_zscore = params[1] as f32;
    let min_vol_spike = if params.len() > 2 { params[2] as f32 } else { 1.5 };
    let tp_recovery_pct = 0.8;  // FIXED: 80% recovery
    let sl_buffer_pct = params[4];
    let be_trigger_pct = if params.len() > 5 { params[5] } else { 0.003 };
    let trail_pct = if params.len() > 6 { params[6] } else { 0.002 };
    let micro_window_ms = if params.len() > 7 { params[7] as u64 } else { 500 };
    let min_absorption = if params.len() > 8 { params[8] as f32 } else { 2.0 };
    let min_reclaim_pct = if params.len() > 9 { params[9] as f32 } else { 0.001 };
    let max_speed_mult = if params.len() > 10 { params[10] as f32 } else { 3.0 };
    let baseline_window_ms = if params.len() > 11 { params[11] as u64 * 1000 } else { 30_000 };
    let max_absorber_ms = if params.len() > 12 { params[12] as u64 * 1000 } else { 30_000 };

    let ticks = &epicenter.ticks;
    if ticks.len() < 20 { return None; }

    // ═══ BASELINE: dynamic window (DE param [11]) ═══
    let start_ts = ticks[0].ts_ms;
    let baseline_end = start_ts + baseline_window_ms;
    
    let mut baseline_prices: Vec<f32> = Vec::new();
    let mut baseline_volume: f64 = 0.0;
    let mut baseline_trade_count: u32 = 0;
    
    for t in ticks.iter() {
        if t.ts_ms > baseline_end { break; }
        baseline_prices.push(t.price);
        baseline_volume += (t.price * t.qty) as f64;
        baseline_trade_count += 1;
    }
    
    if baseline_prices.len() < 10 { return None; }
    if baseline_volume < 10_000.0 { baseline_volume = 10_000.0; }
    if baseline_trade_count < 20 { baseline_trade_count = 20; }
    
    let baseline_secs = baseline_window_ms as f32 / 1000.0;
    let baseline_tps = baseline_trade_count as f32 / baseline_secs;
    
    // Baseline volatility
    let baseline_high = baseline_prices.iter().cloned().fold(f32::MIN, f32::max);
    let baseline_low = baseline_prices.iter().cloned().fold(f32::MAX, f32::min);
    let base_price = *baseline_prices.last().unwrap();
    let baseline_range_pct = (baseline_high - baseline_low) / base_price;
    let time_scale = (window_ms as f32 / (baseline_window_ms as f32)).sqrt();
    let window_std = (baseline_range_pct * time_scale).max(0.00005);
    let baseline_vol_per_window = baseline_volume * (window_ms as f64 / baseline_window_ms as f64);
    
    // Baseline absorption
    let baseline_absorption = if baseline_range_pct > 0.00001 {
        (baseline_volume as f32) / baseline_range_pct
    } else {
        1_000_000.0
    };

    // ═══ STATE MACHINE ═══
    let mut window: VecDeque<usize> = VecDeque::with_capacity(5000);
    let mut current_cvd = 0.0_f32;
    
    let mut squeeze_detected = false;
    let mut squeeze_cvd: f32 = 0.0;
    let mut dump_start_price: f32 = base_price;
    let mut local_extreme: f32 = base_price;
    let mut local_extreme_ts: u64 = 0;
    
    let mut entry_idx = None;

    for (i, tick) in ticks.iter().enumerate() {
        if tick.ts_ms <= baseline_end { continue; }
        
        // Update CVD
        let quote_qty = tick.qty * tick.price;
        let vol = if tick.is_buyer_maker { -quote_qty } else { quote_qty };
        current_cvd += vol;

        // Maintain time window
        window.push_back(i);
        while let Some(&old_idx) = window.front() {
            if tick.ts_ms.saturating_sub(ticks[old_idx].ts_ms) > window_ms {
                window.pop_front();
            } else {
                break;
            }
        }

        if !squeeze_detected {
            if let Some(&first_idx) = window.front() {
                let first_tick = &ticks[first_idx];
                
                let price_move_pct = if direction == "LONG" {
                    (first_tick.price - tick.price) / first_tick.price
                } else {
                    (tick.price - first_tick.price) / first_tick.price
                };
                
                let zscore = price_move_pct / window_std;
                
                if zscore >= min_zscore {
                    let window_volume: f64 = window.iter()
                        .map(|&idx| (ticks[idx].price * ticks[idx].qty) as f64)
                        .sum();
                    let vol_spike = if baseline_vol_per_window > 0.0 {
                        (window_volume / baseline_vol_per_window) as f32
                    } else { 1.0 };
                    
                    if vol_spike >= min_vol_spike {
                        squeeze_detected = true;
                        squeeze_cvd = current_cvd;
                        dump_start_price = first_tick.price;
                        local_extreme = tick.price;
                        local_extreme_ts = tick.ts_ms;
                    }
                }
            }
        } else {
            // Track local extreme
            if direction == "LONG" {
                if tick.price < local_extreme {
                    local_extreme = tick.price;
                    local_extreme_ts = tick.ts_ms;
                }
            } else {
                if tick.price > local_extreme {
                    local_extreme = tick.price;
                    local_extreme_ts = tick.ts_ms;
                }
            }
            
            // Need micro_window_ms after local extreme
            if tick.ts_ms < local_extreme_ts + micro_window_ms { continue; }
            
            // Timeout: DE param [12] (max_absorber_duration_sec)
            if tick.ts_ms > local_extreme_ts + max_absorber_ms { break; }
            
            // ── CHECK 1: CVD DIVERGENCE ──
            let cvd_divergence = if direction == "LONG" {
                current_cvd > squeeze_cvd
            } else {
                current_cvd < squeeze_cvd
            };
            if !cvd_divergence { continue; }
            
            // ── CHECK 2: ABSORPTION ──
            let target_micro_ts = tick.ts_ms.saturating_sub(micro_window_ms);
            let mut micro_volume = 0.0_f32;
            let mut micro_trades = 0_u32;
            let mut micro_high = f32::MIN;
            let mut micro_low = f32::MAX;
            
            for &idx in window.iter().rev() {
                let t = &ticks[idx];
                if t.ts_ms < target_micro_ts { break; }
                micro_volume += t.price * t.qty;
                micro_trades += 1;
                if t.price > micro_high { micro_high = t.price; }
                if t.price < micro_low { micro_low = t.price; }
            }
            
            if micro_trades < 3 { continue; }
            
            let micro_range = (micro_high - micro_low) / tick.price;
            let absorption_ratio = if micro_range > 0.000001 {
                micro_volume / (micro_range * tick.price)
            } else {
                micro_volume * 100.0
            };
            
            if absorption_ratio < baseline_absorption * min_absorption { continue; }
            
            // ── CHECK 3: SPEED ──
            let micro_seconds = micro_window_ms as f32 / 1000.0;
            let micro_tps = micro_trades as f32 / micro_seconds.max(0.001);
            if micro_tps > baseline_tps * max_speed_mult { continue; }
            
            // ── CHECK 4: RECLAIM ──
            let reclaim = if direction == "LONG" {
                (tick.price - local_extreme) / local_extreme
            } else {
                (local_extreme - tick.price) / local_extreme
            };
            if reclaim < min_reclaim_pct { continue; }
            
            // ═══ ALL CHECKS PASSED → ENTER! ═══
            entry_idx = Some(i);
            break;
        }
    }

    // ═══ GRID ENTRY + TRADE SIM ═══
    if let Some(e_idx) = entry_idx {
        if e_idx + 5 >= ticks.len() { return None; }
        
        let taker_fee = 0.0005;
        
        let grid_prices: Vec<f64> = (1..=5)
            .map(|offset| ticks[e_idx + offset].price as f64)
            .collect();
        let avg_entry_raw = grid_prices.iter().sum::<f64>() / grid_prices.len() as f64;
        
        let entry_price = if direction == "LONG" {
            avg_entry_raw * (1.0 + taker_fee)
        } else {
            avg_entry_raw * (1.0 - taker_fee)
        };
        
        // SL: below/above local extreme by sl_buffer_pct
        let mut sl_price;
        if direction == "LONG" {
            sl_price = (local_extreme as f64) * (1.0 - sl_buffer_pct);
        } else {
            sl_price = (local_extreme as f64) * (1.0 + sl_buffer_pct);
        }

        let initial_sl_price = sl_price;
        let risk = (entry_price - sl_price).abs();

        // FIX #1: TP calculated from ENTRY PRICE, not from local_extreme.
        // Old logic: tp = local_extreme + dump_size * 0.8 — this let GA enter
        // ABOVE the TP and claim instant 0R "wins" at 70% WR.
        // New logic: TP = entry + max(dump_recovery, risk * 1.5)
        let dump_size = (dump_start_price as f64 - local_extreme as f64).abs();
        let tp_from_dump = dump_size * tp_recovery_pct;
        // TP must be at least 1.5R from entry (matches live MIN_TP_PCT logic)
        let tp_dist = tp_from_dump.max(risk * 1.5);
        
        let tp_price;
        let be_trigger_price;
        if direction == "LONG" {
            tp_price = entry_price + tp_dist;
            be_trigger_price = entry_price * (1.0 + be_trigger_pct);
        } else {
            tp_price = entry_price - tp_dist;
            be_trigger_price = entry_price * (1.0 - be_trigger_pct);
        }

        // FIX #2: Reject trades with bad R:R (TP closer than SL)
        let reward = (tp_price - entry_price).abs();
        let rr_ratio = if risk > 0.0 { reward / risk } else { 0.0 };
        if rr_ratio < 1.0 {
            return None; // Skip: risk > reward
        }

        // FIX #4: Enforce minimum trail_pct = 0.4% (matches live orchestrator line 1885)
        let effective_trail_pct = trail_pct.max(0.004);

        let mut is_breakeven = false;
        let mut best_price = entry_price;
        let mut exit_price = entry_price;
        let mut pnl_r = 0.0;
        let mut mfe = 0.0_f64;

        for tick in &ticks[(e_idx + 6)..] {
            let p_raw = tick.price as f64;
            let p = if direction == "LONG" {
                p_raw * (1.0 - taker_fee)
            } else {
                p_raw * (1.0 + taker_fee)
            };
            
            let favorable = if direction == "LONG" {
                (p - entry_price) / entry_price * 100.0
            } else {
                (entry_price - p) / entry_price * 100.0
            };
            if favorable > mfe { mfe = favorable; }
            
            if direction == "LONG" {
                if p > best_price { best_price = p; }
                
                if !is_breakeven && p >= be_trigger_price {
                    sl_price = entry_price;
                    is_breakeven = true;
                }
                
                // TP check
                if p >= tp_price {
                    exit_price = p;
                    pnl_r = if risk > 0.0 { (exit_price - entry_price) / risk } else { 0.0 };
                    break;
                }
                
                // FIX #3: When breakeven is active, run trailing BEFORE SL check.
                // Old code checked raw SL first, so trail could never tighten past entry.
                if is_breakeven {
                    let trailing_sl = best_price * (1.0 - effective_trail_pct);
                    if trailing_sl > sl_price { sl_price = trailing_sl; }
                }

                // SL/Trail check (unified — uses tightened SL if trail moved it)
                if p <= sl_price {
                    exit_price = p;
                    pnl_r = if risk > 0.0 { (exit_price - entry_price) / risk } else { 0.0 };
                    break;
                }
            } else {
                if p < best_price { best_price = p; }
                
                if !is_breakeven && p <= be_trigger_price {
                    sl_price = entry_price;
                    is_breakeven = true;
                }
                
                // TP check
                if p <= tp_price {
                    exit_price = p;
                    pnl_r = if risk > 0.0 { (entry_price - exit_price) / risk } else { 0.0 };
                    break;
                }
                
                // FIX #3: Trail before SL for SHORT too
                if is_breakeven {
                    let trailing_sl = best_price * (1.0 + effective_trail_pct);
                    if trailing_sl < sl_price { sl_price = trailing_sl; }
                }

                // SL/Trail check (unified)
                if p >= sl_price {
                    exit_price = p;
                    pnl_r = if risk > 0.0 { (entry_price - exit_price) / risk } else { 0.0 };
                    break;
                }
            }
        }

        // FIX #5: TIMEOUT = penalty, not free money.
        // Old code used last tick price — GA exploited this for +0.03R/trade.
        // Now: timeout = -0.5R (partial loss for indecisive trade)
        if pnl_r == 0.0 && exit_price == entry_price {
            pnl_r = -0.5;
            exit_price = if direction == "LONG" {
                entry_price - risk * 0.5
            } else {
                entry_price + risk * 0.5
            };
        }

        return Some(Trade {
            entry_idx: e_idx,
            direction: direction.to_string(),
            entry_price,
            sl_price: initial_sl_price,
            tp_price,
            exit_price,
            pnl_r,
            risk_dist: risk,
            pnl_abs: 0.0,
            mfe_pct: mfe,
        });
    }

    None
}
