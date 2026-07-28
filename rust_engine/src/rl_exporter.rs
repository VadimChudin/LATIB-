use crate::tick_backtest::Epicenter;
use std::collections::VecDeque;

/// Exports continuous trajectories of features for a specific epicenter.
/// Stepping at fixed `step_ms` interval (e.g. 50ms).
pub fn export_trajectory(epicenter: &Epicenter, direction: &str, step_ms: u64) -> Option<Vec<Vec<f32>>> {
    let ticks = &epicenter.ticks;
    if ticks.len() < 100 { return None; }
    
    // Baselines
    let baseline_window_ms = 30_000;
    let window_ms = 1_000;
    
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
    
    let baseline_high = baseline_prices.iter().cloned().fold(f32::MIN, f32::max);
    let baseline_low = baseline_prices.iter().cloned().fold(f32::MAX, f32::min);
    let base_price = *baseline_prices.last().unwrap();
    let baseline_range_pct = (baseline_high - baseline_low) / base_price;
    let baseline_vol_per_window = baseline_volume * (window_ms as f64 / baseline_window_ms as f64);
    
    let baseline_absorption = if baseline_range_pct > 0.00001 {
        (baseline_volume as f32) / baseline_range_pct
    } else {
        1_000_000.0
    };

    let mut current_cvd = 0.0_f32;
    let mut squeeze_cvd = 0.0_f32;
    let mut squeeze_detected = false;
    let mut local_extreme = base_price;

    let mut window: VecDeque<usize> = VecDeque::with_capacity(5000);
    
    // We want to sample state every `step_ms`. 
    // We maintain a state frame that gets pushed when the clock hits the next interval.
    let mut trajectories = Vec::new();
    
    let mut current_time = baseline_end;
    let end_time = ticks.last().unwrap().ts_ms;
    
    let mut tick_idx = 0;
    
    // Fast forward to baseline_end
    while tick_idx < ticks.len() && ticks[tick_idx].ts_ms <= baseline_end {
        tick_idx += 1;
    }

    while current_time <= end_time {
        // Process all ticks up to current_time
        while tick_idx < ticks.len() && ticks[tick_idx].ts_ms <= current_time {
            let tick = &ticks[tick_idx];
            
            let quote_qty = tick.qty * tick.price;
            let vol = if tick.is_buyer_maker { -quote_qty } else { quote_qty };
            current_cvd += vol;
            
            window.push_back(tick_idx);
            
            if !squeeze_detected {
                // Determine if a squeeze occurred
                if let Some(&first_idx) = window.front() {
                    let first_tick = &ticks[first_idx];
                    let price_move_pct = if direction == "LONG" {
                        (first_tick.price - tick.price) / first_tick.price
                    } else {
                        (tick.price - first_tick.price) / first_tick.price
                    };
                    
                    let time_scale = (window_ms as f32 / (baseline_window_ms as f32)).sqrt();
                    let window_std = (baseline_range_pct * time_scale).max(0.00005);
                    let zscore = price_move_pct / window_std;
                    
                    if zscore >= 2.0 {
                        let window_volume: f64 = window.iter()
                            .map(|&idx| (ticks[idx].price * ticks[idx].qty) as f64)
                            .sum();
                        let vol_spike = if baseline_vol_per_window > 0.0 {
                            (window_volume / baseline_vol_per_window) as f32
                        } else { 1.0 };
                        
                        if vol_spike >= 1.5 {
                            squeeze_detected = true;
                            squeeze_cvd = current_cvd;
                            local_extreme = tick.price;
                        }
                    }
                }
            } else {
                // Update local extreme
                if direction == "LONG" {
                    if tick.price < local_extreme { local_extreme = tick.price; }
                } else {
                    if tick.price > local_extreme { local_extreme = tick.price; }
                }
            }
            tick_idx += 1;
        }
        
        // Trim window
        while let Some(&old_idx) = window.front() {
            if current_time.saturating_sub(ticks[old_idx].ts_ms) > window_ms {
                window.pop_front();
            } else {
                break;
            }
        }
        
        // If we crossed into a state where squeeze was detected, start collecting frames!
        if squeeze_detected {
            // Compute micro state (like Absorption, Speed over last micro_window)
            let micro_window_ms = 500;
            let target_micro_ts = current_time.saturating_sub(micro_window_ms);
            
            let mut micro_volume = 0.0_f32;
            let mut micro_trades = 0_u32;
            let mut micro_high = f32::MIN;
            let mut micro_low = f32::MAX;
            let current_price = if let Some(&last_idx) = window.back() {
                ticks[last_idx].price
            } else {
                base_price
            };
            
            for &idx in window.iter().rev() {
                let t = &ticks[idx];
                if t.ts_ms < target_micro_ts { break; }
                micro_volume += t.price * t.qty;
                micro_trades += 1;
                if t.price > micro_high { micro_high = t.price; }
                if t.price < micro_low { micro_low = t.price; }
            }
            
            let micro_range = (micro_high - micro_low) / current_price;
            let raw_absorption_ratio = if micro_range > 0.000001 {
                micro_volume / (micro_range * current_price)
            } else {
                micro_volume * 100.0
            };
            
            let normalized_absorption = raw_absorption_ratio / baseline_absorption.max(1.0);
            
            let micro_seconds = micro_window_ms as f32 / 1000.0;
            let micro_tps = micro_trades as f32 / micro_seconds.max(0.001);
            let normalized_tps = micro_tps / baseline_tps.max(1.0);
            
            let reclaim_pct = if direction == "LONG" {
                (current_price - local_extreme) / local_extreme
            } else {
                (local_extreme - current_price) / local_extreme
            };
            
            let cvd_divergence = if direction == "LONG" {
                current_cvd - squeeze_cvd
            } else {
                squeeze_cvd - current_cvd
            };

            // Build RL Observation Vector
            let mut obs = Vec::new();
            obs.push(current_time as f32);        // 0: Time
            obs.push(current_price);              // 1: Price
            obs.push(normalized_absorption);      // 2: Absorption
            obs.push(normalized_tps);             // 3: Speed
            obs.push(reclaim_pct * 1000.0);       // 4: Reclaim % (x1000 for network scaling)
            obs.push(cvd_divergence / 1000.0);    // 5: CVD Divergence
            obs.push((current_price - base_price) / base_price * 1000.0); // 6: Move since start
            obs.push(if direction == "LONG" { 1.0 } else { -1.0 }); // 7: Direction bias
            obs.push(micro_volume / 1000.0);      // 8: Micro Vol

            // Ensure our Gym env matches `num_features = 9` (or pad to 17)
            // Let's output 17 features:
            while obs.len() < 19 { // Wait, gym expects `ts, price, f1..f17` (total 19)
                obs.push(0.0);
            }

            trajectories.push(obs);
            
            // Limit trajectory max length (e.g. max 5 mins after squeeze = 6000 frames)
            if trajectories.len() > 6000 {
                break;
            }
        }
        
        current_time += step_ms;
    }
    
    if trajectories.is_empty() {
        None
    } else {
        Some(trajectories)
    }
}
