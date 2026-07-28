use crate::backtest::{Candle, Trade, PrecomputedData};
use crate::strategies::smc::HftMetrics;

pub fn run_backtest(candles: &[Candle], data: &PrecomputedData) -> Vec<Trade> {
    run_backtest_with_params_hft(candles, data, &[0.03, 0.05, 1.5, 1.0, 0.5, 6.0], None)
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
    if n < 250 { return vec![]; }

    let fr_long_thresh = params.get(0).copied().unwrap_or(0.03);   // FR < -this → LONG
    let fr_short_thresh = params.get(1).copied().unwrap_or(0.05);  // FR > +this → SHORT
    let sl_atr_mult = params.get(2).copied().unwrap_or(1.5);
    let trail_act_r = params.get(3).copied().unwrap_or(1.0);
    let trail_atr_mult = params.get(4).copied().unwrap_or(0.5);
    let cooldown_post = params.get(5).copied().unwrap_or(6.0) as usize;

    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let atr = &data.atr;
    let rsi = &data.rsi;
    let ema200 = &data.ema_200;

    let mut vol_sma = vec![0.0; n];
    for i in 20..n {
        vol_sma[i] = volumes[i - 20..i].iter().sum::<f64>() / 20.0;
    }

    let mut simulated_fr = vec![0.0; n];
    for i in 20..n {
        let rsi_v = if i < rsi.len() { rsi[i] } else { 50.0 };
        let rsi_norm = (rsi_v - 50.0) / 50.0;
        let vol_factor = if vol_sma[i] > 0.0 { (volumes[i] / vol_sma[i]).min(3.0) } else { 1.0 };
        simulated_fr[i] = rsi_norm * vol_factor * 0.05;
    }

    const SETTLEMENT_PERIOD: usize = 96;
    let mut settlement_fr: Vec<(usize, f64)> = Vec::new();
    let mut bar = SETTLEMENT_PERIOD;
    while bar < n {
        let start = bar.saturating_sub(SETTLEMENT_PERIOD);
        let avg_fr: f64 = simulated_fr[start..bar].iter().sum::<f64>() / SETTLEMENT_PERIOD as f64;
        settlement_fr.push((bar, avg_fr));
        bar += SETTLEMENT_PERIOD;
    }

    let mut trades = Vec::new();
    let mut in_position = false;
    let mut direction = "";
    let mut entry_p = 0.0;
    let mut sl_p = 0.0;
    let mut init_risk = 0.0;
    let mut entry_atr = 0.0;
    let mut trail_sl = 0.0;
    let mut trail_active = false;
    let mut entry_idx = 0;

    let mut settlement_cursor = 0usize;
    for i in 210..(n - 1) {
        if !in_position {
            let price = candles[i].close;
            let atr_v = atr[i];
            if atr_v <= 0.0 { continue; }

            // Efficiently find the last settlement before this bar using the cursor
            while settlement_cursor + 1 < settlement_fr.len() && settlement_fr[settlement_cursor + 1].0 <= i {
                settlement_cursor += 1;
            }

            let mut current_fr = 0.0;
            let mut prev_fr = 0.0;
            let mut bars_since_settlement = 999;

            if !settlement_fr.is_empty() && settlement_fr[settlement_cursor].0 <= i {
                current_fr = settlement_fr[settlement_cursor].1;
                prev_fr = if settlement_cursor > 0 { settlement_fr[settlement_cursor - 1].1 } else { 0.0 };
                bars_since_settlement = i - settlement_fr[settlement_cursor].0;
            }

            if bars_since_settlement < cooldown_post { continue; }

            // --- HFT Confirmation (Live Only) ---
            let mut hft_confirm = true;
            if i == n - 2 {
                if let Some(h) = hft {
                    // Reversal confirmation: if longing, we want delta to turn positive
                    if current_fr < 0.0 && h.tape_delta_momentum < 0.0 { hft_confirm = false; }
                    // If shorting, we want delta to turn negative
                    if current_fr > 0.0 && h.tape_delta_momentum > 0.0 { hft_confirm = false; }
                }
            }
            if !hft_confirm { continue; }

            // LONG
            if current_fr < -fr_long_thresh / 100.0 && prev_fr < -fr_long_thresh / 100.0 {
                if candles[i].close > candles[i].open && price > ema200[i] * 0.98 {
                    let sl = price - atr_v * sl_atr_mult;
                    let risk = price - sl;
                    if risk > 0.0 {
                        entry_p = price; sl_p = sl; in_position = true; direction = "LONG";
                        init_risk = risk; entry_atr = atr_v; trail_active = false; trail_sl = sl; entry_idx = i;
                    }
                }
            }
            // SHORT
            else if current_fr > fr_short_thresh / 100.0 && prev_fr > fr_short_thresh / 100.0 {
                if candles[i].close < candles[i].open && price < ema200[i] * 1.02 {
                    let sl = price + atr_v * sl_atr_mult;
                    let risk = sl - price;
                    if risk > 0.0 {
                        entry_p = price; sl_p = sl; in_position = true; direction = "SHORT";
                        init_risk = risk; entry_atr = atr_v; trail_active = false; trail_sl = sl; entry_idx = i;
                    }
                }
            }
        } else {
            let high = candles[i].high;
            let low = candles[i].low;
            let current_atr = if atr[i] > 0.0 { atr[i] } else { entry_atr };

            if direction == "LONG" {
                if high >= entry_p + trail_act_r * init_risk { trail_active = true; }
                if trail_active {
                    let new_trail = high - current_atr * trail_atr_mult;
                    if new_trail > trail_sl { trail_sl = new_trail; }
                }
                let active_sl = if trail_active { trail_sl } else { sl_p };
                if low <= active_sl {
                    trades.push(Trade { entry_idx, direction: "LONG".into(), entry_price: entry_p, sl_price: sl_p, tp_price: 0.0, exit_price: active_sl, pnl_r: (active_sl - entry_p) / init_risk, risk_dist: init_risk, pnl_abs: 0.0, mfe_pct: 0.0 });
                    in_position = false;
                }
            } else {
                if low <= entry_p - trail_act_r * init_risk { trail_active = true; }
                if trail_active {
                    let new_trail = low + current_atr * trail_atr_mult;
                    if new_trail < trail_sl || trail_sl == sl_p { trail_sl = new_trail; }
                }
                let active_sl = if trail_active { trail_sl } else { sl_p };
                if high >= active_sl {
                    trades.push(Trade { entry_idx, direction: "SHORT".into(), entry_price: entry_p, sl_price: sl_p, tp_price: 0.0, exit_price: active_sl, pnl_r: (entry_p - active_sl) / init_risk, risk_dist: init_risk, pnl_abs: 0.0, mfe_pct: 0.0 });
                    in_position = false;
                }
            }
        }
    }

    if in_position {
        trades.push(Trade { entry_idx, direction: direction.into(), entry_price: entry_p, sl_price: if trail_active { trail_sl } else { sl_p }, tp_price: 0.0, exit_price: 0.0, pnl_r: 0.0, risk_dist: init_risk, pnl_abs: 0.0, mfe_pct: 0.0 });
    }
    trades
}
