use crate::backtest::{Candle, Trade, PrecomputedData};
use crate::bitset_engine::BitsetSignals;

/// Fast Binary version of Density strategy
pub fn run_backtest_with_params(candles: &[Candle], data: &PrecomputedData, params: &[f64]) -> Vec<Trade> {
    if candles.len() < 200 { return vec![]; }
    let bitsets = match &data.bitsets {
        Some(bs) => bs,
        None => return vec![], // Binary engine requires bitsets
    };

    // Parameters from GA
    let vol_spike_idx = (params[0] as usize).min(6);
    let body_ratio_idx = (params[1] as usize).min(4);
    let use_ema_filter = params[2] > 0.5;
    let tp_rr          = params[3];
    let sl_atr_mult    = params[4];

    // STEP 1: Fast Bitset Intersection (The Core Speedup)
    let entry_signals = bitsets.combine_signals(vol_spike_idx, body_ratio_idx, use_ema_filter);
    let signal_indices = BitsetSignals::get_signal_indices(&entry_signals);

    let mut trades = Vec::new();
    let mut last_exit_idx = 0;

    // STEP 2: Process only active signals (Sparse scanning)
    for i in signal_indices {
        if i < 200 || i <= last_exit_idx { continue; }

        let curr = &candles[i];
        let atr_v = data.atr[i];
        let sl_dist = if atr_v > 0.0 { atr_v * sl_atr_mult } else { curr.close * 0.005 * sl_atr_mult };
        let sl_price = curr.close - sl_dist;
        let tp_price = curr.close + (sl_dist * tp_rr);

        // Simple exit simulation (could also be binarized later)
        let mut exit_idx = i + 1;
        let mut final_price = tp_price;
        let mut pnl = tp_rr;

        while exit_idx < candles.len() {
            if candles[exit_idx].low <= sl_price {
                final_price = sl_price;
                pnl = -1.0;
                break;
            }
            if candles[exit_idx].high >= tp_price {
                break;
            }
            exit_idx += 1;
        }

        if exit_idx >= candles.len() {
            exit_idx = candles.len() - 1;
            final_price = candles[exit_idx].close;
            pnl = (final_price - curr.close) / sl_dist;
        }

        trades.push(Trade {
            entry_idx: i,
            direction: "LONG".into(),
            entry_price: curr.close,
            sl_price,
            tp_price,
            exit_price: final_price,
            pnl_r: pnl,
            risk_dist: sl_dist,
            pnl_abs: 0.0,
            mfe_pct: 0.0,
        });
        
        last_exit_idx = exit_idx + 10; // Cooldown
    }

    trades
}
