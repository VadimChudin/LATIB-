/// Ultimate SMC Trail Strategy — FULL PORT from Python
/// ==================================================
/// 
/// Complete logic:
/// 1. Swing High/Low detection
/// 2. Fair Value Gap (FVG) detection
/// 3. Break of Structure (BOS)
/// 4. Order Block scoring (6 criteria, min_score threshold)
/// 5. EMA200 trend filter
/// 6. Trailing stop with activation at 1.0R
///
/// This matches the Python version 1:1.

use crate::backtest::{Candle, Trade, PrecomputedData};

/// FVG (Fair Value Gap) zone
#[derive(Clone, Copy)]
struct FvgZone {
    top: f64,
    bot: f64,
    direction: i8, // 1 = bull, -1 = bear
    volume: f64,   // Volume of the impulse candle that created the FVG
}

/// Phase 2: HFT Confluence Metrics
#[derive(Debug, Clone, Default)]
pub struct HftMetrics {
    pub nearby_wall_presence: f64,   // 0.0 to 1.0 (proximity and size weight)
    pub tape_delta_momentum: f64,    // Normalized delta (-1 to 1)
    pub liquidation_proximity: f64,
    pub order_book_imbalance: f64,
}

/// Swing point
#[derive(Clone, Copy)]
struct Swing {
    swing_type: i8, // 1 = high, -1 = low, 0 = none
    level: f64,     // NAN if no swing
}

fn find_swings(candles: &[Candle], swing_len: usize) -> Vec<Swing> {
    let n = candles.len();
    let mut swings = vec![Swing { swing_type: 0, level: f64::NAN }; n];
    if n < swing_len * 2 + 1 { return swings; }

    use std::collections::VecDeque;
    let mut max_dq: VecDeque<usize> = VecDeque::with_capacity(swing_len * 2 + 1);
    let mut min_dq: VecDeque<usize> = VecDeque::with_capacity(swing_len * 2 + 1);
    let window_size = swing_len * 2 + 1;

    for i in 0..n {
        while !max_dq.is_empty() && candles[*max_dq.back().unwrap()].high <= candles[i].high { max_dq.pop_back(); }
        max_dq.push_back(i);
        if *max_dq.front().unwrap() <= i.saturating_sub(window_size) { max_dq.pop_front(); }

        while !min_dq.is_empty() && candles[*min_dq.back().unwrap()].low >= candles[i].low { min_dq.pop_back(); }
        min_dq.push_back(i);
        if *min_dq.front().unwrap() <= i.saturating_sub(window_size) { min_dq.pop_front(); }

        if i >= window_size - 1 {
            let center = i - swing_len;
            if *max_dq.front().unwrap() == center {
                swings[center] = Swing { swing_type: 1, level: candles[center].high };
            } else if *min_dq.front().unwrap() == center {
                swings[center] = Swing { swing_type: -1, level: candles[center].low };
            }
        }
    }
    swings
}

fn find_fvgs(candles: &[Candle], atr: &[f64], fvg_min_atr: f64) -> Vec<Option<FvgZone>> {
    let n = candles.len();
    let mut fvgs = vec![None; n];
    if n < 3 { return fvgs; }

    for i in 2..n {
        let atr_v = atr[i];
        if atr_v <= 0.0 { continue; }
        if candles[i].low > candles[i - 2].high {
            let gap = candles[i].low - candles[i - 2].high;
            if gap > atr_v * fvg_min_atr && candles[i - 1].close > candles[i - 1].open {
                fvgs[i] = Some(FvgZone { top: candles[i].low, bot: candles[i - 2].high, direction: 1, volume: candles[i - 1].volume });
            }
        } else if candles[i - 2].low > candles[i].high {
            let gap = candles[i - 2].low - candles[i].high;
            if gap > atr_v * fvg_min_atr && candles[i - 1].close < candles[i - 1].open {
                fvgs[i] = Some(FvgZone { top: candles[i - 2].low, bot: candles[i].high, direction: -1, volume: candles[i - 1].volume });
            }
        }
    }
    fvgs
}

fn score_ob(
    candles: &[Candle],
    i: usize,
    direction: &str,
    fvg_top: f64,
    fvg_bot: f64,
    fvg_volume: f64,
    atr_v: f64,
    _swings: &[Swing],
    bos: &[i8],
    swept_long: bool,
    swept_short: bool,
    hft: Option<&HftMetrics>,
) -> i32 {
    let mut score: i32 = 0;

    // 1. Impulse candle
    if i >= 3 {
        let impulse = (candles[i - 1].close - candles[i - 1].open).abs();
        if impulse > atr_v * 0.8 { score += 1; }
    }

    // 2. Unmitigated FVG
    let mut mitigated = false;
    let start = i.saturating_sub(20);
    for k in start..i {
        if fvg_bot <= candles[k].close && candles[k].close <= fvg_top {
            mitigated = true;
            break;
        }
    }
    if !mitigated { score += 1; }

    // 3. Liquidity sweep (Precomputed)
    if direction == "LONG" && swept_long { score += 1; }
    else if direction == "SHORT" && swept_short { score += 1; }

    // 4. Rejection at FVG zone
    if direction == "LONG" && candles[i].low <= fvg_top && candles[i].close > candles[i].open {
        score += 1;
    } else if direction == "SHORT" && candles[i].high >= fvg_bot && candles[i].close < candles[i].open {
        score += 1;
    }

    // 5. Clean zone
    let zone_size = fvg_top - fvg_bot;
    let mut wicks_in_zone = 0;
    let wick_start = if i >= 10 { i - 10 } else { 0 };
    for k in wick_start..i {
        let upper_wick = candles[k].high - candles[k].close.max(candles[k].open);
        let lower_wick = candles[k].close.min(candles[k].open) - candles[k].low;
        if upper_wick > zone_size || lower_wick > zone_size { wicks_in_zone += 1; }
    }
    if wicks_in_zone <= 1 { score += 1; }

    // 6. BOS alignment
    let bos_start = if i >= 15 { i - 15 } else { 0 };
    if direction == "LONG" {
        if bos[bos_start..i].iter().any(|&b| b == 1) { score += 1; }
    } else if bos[bos_start..i].iter().any(|&b| b == -1) {
        score += 1;
    }

    // 7. Volume FVG
    if i >= 20 {
        let avg_vol: f64 = candles[i.saturating_sub(20)..i].iter().map(|c| c.volume).sum::<f64>() / 20.0;
        if avg_vol > 0.0 && fvg_volume > avg_vol * 2.0 { score += 1; }
    }

    // --- Phase 2: HFT Criteria ---
    if let Some(h) = hft {
        if h.nearby_wall_presence > 0.5 { score += 1; }
        if h.liquidation_proximity > 0.0 { score += 1; } // Changed from > 0.5 to > 0.0 as per instruction
        if (direction == "LONG" && h.tape_delta_momentum > 0.2) || 
           (direction == "SHORT" && h.tape_delta_momentum < -0.2) {
            score += 1;
        }
    }

    score
}

pub fn run_backtest(candles: &[Candle], data: &PrecomputedData) -> Vec<Trade> {
    run_backtest_with_params_hft(candles, data, &[5.0, 0.3, 3.0, 1.0, 1.0, 0.5], None)
}

pub fn run_backtest_with_params(candles: &[Candle], data: &PrecomputedData, params: &[f64]) -> Vec<Trade> {
    run_backtest_with_params_hft(candles, data, params, None)
}

pub fn run_live_signal(candles: &[Candle], data: &PrecomputedData, params: &[f64], hft: &HftMetrics) -> Option<Trade> {
    let trades = run_backtest_with_params_hft(candles, data, params, Some(hft));
    trades.into_iter().find(|t| t.exit_price == 0.0)
}

fn run_backtest_with_params_hft(candles: &[Candle], data: &PrecomputedData, params: &[f64], hft: Option<&HftMetrics>) -> Vec<Trade> {
    let n = candles.len();
    if n < 200 { return vec![]; }

    let atr = &data.atr;
    let ema200 = &data.ema_200;

    let swing_len = params.get(0).copied().unwrap_or(5.0) as usize;
    let fvg_min_atr = params.get(1).copied().unwrap_or(0.3);
    let ob_min_score = params.get(2).copied().unwrap_or(3.0) as i32;
    let sl_atr_mult = params.get(3).copied().unwrap_or(1.0);
    let trail_act_r = params.get(4).copied().unwrap_or(1.0);
    let trail_atr_mult = params.get(5).copied().unwrap_or(0.5);

    let swings = find_swings(candles, swing_len);
    let fvgs = find_fvgs(candles, atr, fvg_min_atr);
    let adx = &data.adx;

    let mut bos = vec![0i8; n];
    let mut swept_long = vec![false; n];
    let mut swept_short = vec![false; n];
    let mut last_high = f64::NAN;
    let mut last_low = f64::NAN;

    // Rolling min/max for last 10 candles to optimize sweep detection
    let mut min10 = vec![f64::MAX; n];
    let mut max10 = vec![0.0; n];
    use std::collections::VecDeque;
    let mut min_dq: VecDeque<usize> = VecDeque::with_capacity(11);
    let mut max_dq: VecDeque<usize> = VecDeque::with_capacity(11);

    for i in 0..n {
        while !min_dq.is_empty() && candles[*min_dq.back().unwrap()].low >= candles[i].low { min_dq.pop_back(); }
        min_dq.push_back(i);
        if *min_dq.front().unwrap() < i.saturating_sub(10) { min_dq.pop_front(); }
        min10[i] = candles[*min_dq.front().unwrap()].low;

        while !max_dq.is_empty() && candles[*max_dq.back().unwrap()].high <= candles[i].high { max_dq.pop_back(); }
        max_dq.push_back(i);
        if *max_dq.front().unwrap() < i.saturating_sub(10) { max_dq.pop_front(); }
        max10[i] = candles[*max_dq.front().unwrap()].high;

        let confirmed_idx = i.saturating_sub(swing_len);
        if confirmed_idx > 0 {
            let s = swings[confirmed_idx];
            if s.swing_type == 1 { last_high = s.level; }
            else if s.swing_type == -1 { last_low = s.level; }
        }
        if !last_high.is_nan() && candles[i].close > last_high { bos[i] = 1; }
        else if !last_low.is_nan() && candles[i].close < last_low { bos[i] = -1; }

        let sw_start = i.saturating_sub(30);
        let m10 = min10[i];
        let x10 = max10[i];
        for j in sw_start..i {
            if swings[j].swing_type == -1 && m10 < swings[j].level { swept_long[i] = true; break; }
        }
        for j in sw_start..i {
            if swings[j].swing_type == 1 && x10 > swings[j].level { swept_short[i] = true; break; }
        }
    }

    let mut trades = Vec::new();
    let start_idx = 200.max(swing_len * 2 + 5);
    let mut in_position = false;
    let mut direction = "";
    let mut entry_price = 0.0;
    let mut sl_price = 0.0;
    let mut init_risk = 0.0;
    let mut entry_atr = 0.0;
    let mut trail_sl = 0.0;
    let mut trail_active = false;
    let mut entry_idx = 0;

    for i in start_idx..(n - 1) {
        if !in_position {
            let price = candles[i].close;
            let e200 = ema200[i];
            let atr_v = atr[i];
            let adx_v = adx[i];
            if e200 == 0.0 || atr_v <= 0.0 || adx_v < 20.0 { continue; }

            if price > e200 {
                for j in 1..15 {
                    let k = i.saturating_sub(j);
                    if let Some(fvg) = &fvgs[k] {
                        if fvg.direction == 1 && score_ob(candles, i, "LONG", fvg.top, fvg.bot, fvg.volume, atr_v, &swings, &bos, swept_long[i], swept_short[i], if i == n - 1 { hft } else { None }) >= ob_min_score {
                            let sl = (candles[i].low - atr_v * sl_atr_mult).min(price - 0.0001);
                            let risk = price - sl;
                            if risk > 0.0 {
                                entry_price = price; sl_price = sl; in_position = true; direction = "LONG";
                                init_risk = risk; entry_atr = atr_v; trail_active = false; trail_sl = sl; entry_idx = i;
                                break;
                            }
                        }
                    }
                }
            } else if price < e200 {
                for j in 1..15 {
                    let k = i.saturating_sub(j);
                    if let Some(fvg) = &fvgs[k] {
                        if fvg.direction == -1 && score_ob(candles, i, "SHORT", fvg.top, fvg.bot, fvg.volume, atr_v, &swings, &bos, swept_long[i], swept_short[i], if i == n - 1 { hft } else { None }) >= ob_min_score {
                            let sl = (candles[i].high + atr_v * sl_atr_mult).max(price + 0.0001);
                            let risk = sl - price;
                            if risk > 0.0 {
                                entry_price = price; sl_price = sl; in_position = true; direction = "SHORT";
                                init_risk = risk; entry_atr = atr_v; trail_active = false; trail_sl = sl; entry_idx = i;
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            let high = candles[i].high;
            let low = candles[i].low;
            let current_atr = if atr[i] > 0.0 { atr[i] } else { entry_atr };
            if direction == "LONG" {
                if high >= entry_price + trail_act_r * init_risk { trail_active = true; }
                if trail_active {
                    let new_trail = high - current_atr * trail_atr_mult;
                    if new_trail > trail_sl { trail_sl = new_trail; }
                }
                let active_sl = if trail_active { trail_sl } else { sl_price };
                if low <= active_sl {
                    trades.push(Trade { entry_idx, direction: "LONG".into(), entry_price, sl_price, tp_price: 0.0, exit_price: active_sl, pnl_r: ((active_sl - entry_price) / init_risk * 1000.0).round() / 1000.0, risk_dist: init_risk, pnl_abs: 0.0, mfe_pct: 0.0 });
                    in_position = false;
                }
            } else {
                if low <= entry_price - trail_act_r * init_risk { trail_active = true; }
                if trail_active {
                    let new_trail = low + current_atr * trail_atr_mult;
                    if new_trail < trail_sl || trail_sl == sl_price { trail_sl = new_trail; }
                }
                let active_sl = if trail_active { trail_sl } else { sl_price };
                if high >= active_sl {
                    trades.push(Trade { entry_idx, direction: "SHORT".into(), entry_price, sl_price, tp_price: 0.0, exit_price: active_sl, pnl_r: ((entry_price - active_sl) / init_risk * 1000.0).round() / 1000.0, risk_dist: init_risk, pnl_abs: 0.0, mfe_pct: 0.0 });
                    in_position = false;
                }
            }
        }
    }
    if in_position {
        trades.push(Trade { entry_idx, direction: direction.into(), entry_price, sl_price: if trail_active { trail_sl } else { sl_price }, tp_price: 0.0, exit_price: 0.0, pnl_r: 0.0, risk_dist: init_risk, pnl_abs: 0.0, mfe_pct: 0.0 });
    }
    trades
}
