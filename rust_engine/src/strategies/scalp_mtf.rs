use crate::backtest::{Candle, Trade, PrecomputedData};
use crate::strategies::smc::HftMetrics;

pub fn run_backtest(candles: &[Candle], data: &PrecomputedData) -> Vec<Trade> {
    run_backtest_with_params_hft(candles, data, &[9.0, 50.0, 30.0, 1.5], None)
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

    let fast_ema_period = params.get(0).copied().unwrap_or(9.0) as usize;
    let slow_ema_period = params.get(1).copied().unwrap_or(50.0) as usize;
    let rsi_thresh = params.get(2).copied().unwrap_or(30.0);
    let tp_rr = params.get(3).copied().unwrap_or(1.5);
    let sl_atr_mult = 1.0;

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    
    // EMA periods are parameters, so we might need local calc if they differ from cache
    let ema_fast = if fast_ema_period == 9 { data.ema_fast.clone() } else { crate::backtest::calc_ema(&closes, fast_ema_period) };
    let ema_slow = if slow_ema_period == 50 { data.ema_slow.clone() } else { crate::backtest::calc_ema(&closes, slow_ema_period) };
    
    let atr = &data.atr;
    let rsi = &data.rsi;

    let mut vol_sma = vec![0.0; n];
    for i in 20..n {
        vol_sma[i] = volumes[i-20..i].iter().sum::<f64>() / 20.0;
    }

    let mut highest_high_5 = vec![0.0; n];
    let mut lowest_low_5 = vec![0.0; n];
    for i in 5..n {
        let mut max_h = candles[i-1].high;
        let mut min_l = candles[i-1].low;
        for j in 1..=5 {
            if candles[i-j].high > max_h { max_h = candles[i-j].high; }
            if candles[i-j].low < min_l { min_l = candles[i-j].low; }
        }
        highest_high_5[i] = max_h;
        lowest_low_5[i] = min_l;
    }

    let mut trades = Vec::new();
    let mut in_trade = false;
    let mut trade_dir = 0; 
    let mut entry_price = 0.0;
    let mut sl_price = 0.0;
    let mut tp_price = 0.0;
    let mut entry_idx = 0;

    for i in 21..n {
        let curr = &candles[i];

        if in_trade {
            if trade_dir == 1 {
                if curr.low <= sl_price {
                    trades.push(Trade { entry_idx, direction: "LONG".into(), entry_price, sl_price, tp_price, exit_price: sl_price, pnl_r: -1.0, risk_dist: (entry_price - sl_price).abs(), pnl_abs: 0.0, mfe_pct: 0.0 });
                    in_trade = false;
                } else if curr.high >= tp_price {
                    trades.push(Trade { entry_idx, direction: "LONG".into(), entry_price, sl_price, tp_price, exit_price: tp_price, pnl_r: tp_rr, risk_dist: (entry_price - sl_price).abs(), pnl_abs: 0.0, mfe_pct: 0.0 });
                    in_trade = false;
                }
            } else {
                if curr.high >= sl_price {
                    trades.push(Trade { entry_idx, direction: "SHORT".into(), entry_price, sl_price, tp_price, exit_price: sl_price, pnl_r: -1.0, risk_dist: (entry_price - sl_price).abs(), pnl_abs: 0.0, mfe_pct: 0.0 });
                    in_trade = false;
                } else if curr.low <= tp_price {
                    trades.push(Trade { entry_idx, direction: "SHORT".into(), entry_price, sl_price, tp_price, exit_price: tp_price, pnl_r: tp_rr, risk_dist: (entry_price - sl_price).abs(), pnl_abs: 0.0, mfe_pct: 0.0 });
                    in_trade = false;
                }
            }
            continue;
        }

        let trend_up = ema_fast[i] > ema_slow[i];
        let trend_down = ema_fast[i] < ema_slow[i];
        let atr_v = atr[i];
        if atr_v <= 0.0 { continue; }

        if vol_sma[i] > 0.0 && volumes[i] < vol_sma[i] * 1.3 { continue; }

        // --- HFT Confirmation (for Live only) ---
        let mut tape_confirms = true;
        
        if i == n - 1 {
            if let Some(h) = hft {
                // OBI Filter: Confirm pressure is in the right direction
                if ema_fast[i] > ema_slow[i] && h.order_book_imbalance < 0.1 { tape_confirms = false; }
                if ema_fast[i] < ema_slow[i] && h.order_book_imbalance > -0.1 { tape_confirms = false; }
                
                // Tape Delta Block: Ensure aggressive flow isn't against us
                if ema_fast[i] > ema_slow[i] && h.tape_delta_momentum < -0.1 { tape_confirms = false; }
                if ema_fast[i] < ema_slow[i] && h.tape_delta_momentum > 0.1 { tape_confirms = false; }
            }
        }

        if !tape_confirms { continue; }

        // LONG
        let is_oversold = rsi[i] < rsi_thresh || rsi[i-1] < rsi_thresh;
        if trend_up && is_oversold && curr.close > highest_high_5[i] {
            in_trade = true; trade_dir = 1; entry_price = curr.close;
            sl_price = entry_price - (atr_v * sl_atr_mult);
            tp_price = entry_price + (atr_v * sl_atr_mult * tp_rr);
            entry_idx = i;
        }

        // SHORT
        let overbought_thresh = 100.0 - rsi_thresh;
        let is_overbought = rsi[i] > overbought_thresh || rsi[i-1] > overbought_thresh;
        if !in_trade && trend_down && is_overbought && curr.close < lowest_low_5[i] {
            in_trade = true; trade_dir = -1; entry_price = curr.close;
            sl_price = entry_price + (atr_v * sl_atr_mult);
            tp_price = entry_price - (atr_v * sl_atr_mult * tp_rr);
            entry_idx = i;
        }
    }

    if in_trade {
        trades.push(Trade { entry_idx, direction: if trade_dir == 1 { "LONG" } else { "SHORT" }.into(), entry_price, sl_price, tp_price, exit_price: 0.0, pnl_r: 0.0, risk_dist: (entry_price - sl_price).abs(), pnl_abs: 0.0, mfe_pct: 0.0 });
    }
    trades
}
