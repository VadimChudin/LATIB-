/// Density Breakout Strategy (Proxy for HFT logic)
/// ===============================================
///
/// This strategy simulates HFT density breakouts using candle proxies:
/// 1. Proxy Wall: Horizontal resistance/support with 2+ touches.
/// 2. Stop-Hunt Defense: Detects "shakeouts" (dip & recovery) before the move.
/// 3. Breakout Trigger: High volume + full body candle beyond resistance/support.
///
/// Supports both LONG (resistance breakout) and SHORT (support breakdown).

use crate::backtest::{Candle, Trade, PrecomputedData};

pub fn run_backtest_with_params(candles: &[Candle], data: &PrecomputedData, params: &[f64]) -> Vec<Trade> {
    if candles.len() < 200 { return vec![]; }

    // Parameters from GA
    let vol_spike_mult = params.get(0).copied().unwrap_or(2.5);
    let min_touches    = params.get(1).copied().unwrap_or(2.0) as usize;
    let shakeout_pct   = params.get(2).copied().unwrap_or(0.006); // 0.6% dip
    let tp_rr          = params.get(3).copied().unwrap_or(2.0);
    let sl_atr_mult    = params.get(4).copied().unwrap_or(1.0);

    let mut trades = Vec::new();
    let mut cooldown = 0;

    for i in 200..candles.len() {
        if cooldown > 0 {
            cooldown -= 1;
            continue;
        }

        let curr = &candles[i];
        let window = &candles[i-200..i];

        // Volume spike check (shared by LONG and SHORT)
        let vol_ma = {
            let start = i.saturating_sub(20);
            let sum: f64 = candles[start..i].iter().map(|c| c.volume).sum();
            sum / 20.0
        };
        let is_vol_spike = curr.volume > vol_ma * vol_spike_mult;
        let body_ratio = (curr.close - curr.open).abs() / (curr.high - curr.low).max(0.000001);

        // ATR for SL/TP
        let atr_v = data.atr[i];
        let sl_dist = if atr_v > 0.0 { atr_v * sl_atr_mult } else { curr.close * 0.005 * sl_atr_mult };

        // === LONG: Resistance Breakout ===
        let resistance = find_proxy_wall_resistance(window, min_touches);
        if let Some(res_level) = resistance {
            let is_full_body_up = body_ratio > 0.7 && curr.close > curr.open;
            // FIX: Use shakeout detection as quality filter
            let is_shaken = detect_shakeout(&candles[i.saturating_sub(15)..i], res_level, shakeout_pct);
            let touches_ok = min_touches >= 3 || is_shaken; // 3+ touches OR confirmed shakeout

            if curr.close > res_level && is_vol_spike && is_full_body_up && touches_ok {
                let sl_price = curr.close - sl_dist;
                let tp_price = curr.close + (sl_dist * tp_rr);

                trades.push(Trade {
                    entry_idx: i,
                    direction: "LONG".into(),
                    entry_price: curr.close,
                    sl_price,
                    tp_price,
                    exit_price: 0.0,
                    pnl_r: 0.0,
                    risk_dist: sl_dist,
                    pnl_abs: 0.0,
                    mfe_pct: 0.0,
                });
                cooldown = 10;
                continue; // Don't check SHORT on the same bar
            }
        }

        // === SHORT: Support Breakdown ===
        let support = find_proxy_wall_support(window, min_touches);
        if let Some(sup_level) = support {
            let is_full_body_down = body_ratio > 0.7 && curr.close < curr.open;
            let is_shaken = detect_shakeout_short(&candles[i.saturating_sub(15)..i], sup_level, shakeout_pct);
            let touches_ok = min_touches >= 3 || is_shaken;

            if curr.close < sup_level && is_vol_spike && is_full_body_down && touches_ok {
                let sl_price = curr.close + sl_dist;
                let tp_price = curr.close - (sl_dist * tp_rr);

                trades.push(Trade {
                    entry_idx: i,
                    direction: "SHORT".into(),
                    entry_price: curr.close,
                    sl_price,
                    tp_price,
                    exit_price: 0.0,
                    pnl_r: 0.0,
                    risk_dist: sl_dist,
                    pnl_abs: 0.0,
                    mfe_pct: 0.0,
                });
                cooldown = 10;
            }
        }
    }

    trades
}

/// Find resistance wall (highs clustering above current price)
fn find_proxy_wall_resistance(window: &[Candle], min_touches: usize) -> Option<f64> {
    let mut clusters: Vec<(f64, usize)> = Vec::new();
    let threshold = 0.0005; // 0.05% tolerance

    for c in window {
        let price = c.high;
        let mut found = false;
        for cluster in &mut clusters {
            if (price - cluster.0).abs() / cluster.0 < threshold {
                cluster.1 += 1;
                found = true;
                break;
            }
        }
        if !found {
            clusters.push((price, 1));
        }
    }

    let last_price = window.last().map(|c| c.close).unwrap_or(0.0);
    clusters.iter()
        .filter(|c| c.1 >= min_touches && c.0 > last_price)
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|c| c.0)
}

/// Find support wall (lows clustering below current price)
fn find_proxy_wall_support(window: &[Candle], min_touches: usize) -> Option<f64> {
    let mut clusters: Vec<(f64, usize)> = Vec::new();
    let threshold = 0.0005;

    for c in window {
        let price = c.low;
        let mut found = false;
        for cluster in &mut clusters {
            if (price - cluster.0).abs() / cluster.0 < threshold {
                cluster.1 += 1;
                found = true;
                break;
            }
        }
        if !found {
            clusters.push((price, 1));
        }
    }

    let last_price = window.last().map(|c| c.close).unwrap_or(0.0);
    clusters.iter()
        .filter(|c| c.1 >= min_touches && c.0 < last_price)
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|c| c.0)
}

/// Detect shakeout before LONG breakout (dip below level then recovery)
fn detect_shakeout(recent: &[Candle], level: f64, depth_pct: f64) -> bool {
    if recent.is_empty() { return false; }
    
    let mut dipped = false;
    let mut recovered = false;
    
    for c in recent {
        let dist = (level - c.low) / level;
        if dist > depth_pct {
            dipped = true;
        }
        if dipped && c.close > level * 0.995 {
            recovered = true;
        }
    }
    
    dipped && recovered
}

/// Detect shakeout before SHORT breakdown (spike above level then drop)
fn detect_shakeout_short(recent: &[Candle], level: f64, depth_pct: f64) -> bool {
    if recent.is_empty() { return false; }
    
    let mut spiked = false;
    let mut dropped = false;
    
    for c in recent {
        let dist = (c.high - level) / level;
        if dist > depth_pct {
            spiked = true;
        }
        if spiked && c.close < level * 1.005 {
            dropped = true;
        }
    }
    
    spiked && dropped
}
