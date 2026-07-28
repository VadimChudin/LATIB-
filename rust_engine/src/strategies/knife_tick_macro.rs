use crate::backtest::{Candle, PrecomputedData, Trade};

/// Macro-level signal generator for the Knife Tick (HFT) strategy.
/// This runs on 1-minute candles to identify "Epicenters" (sudden drops/pumps).
/// Once an epicenter is found, it sends an unclosed Trade to the Orchestrator,
/// which will spawn the HFT Absorber to find the micro-level bottom/top.
pub fn run_backtest_with_params(
    candles: &[Candle],
    indicators: &PrecomputedData,
    params: &[f64],
) -> Vec<Trade> {
    let mut trades = Vec::new();
    if candles.len() < 2 {
        return trades;
    }

    // Parameters mapped from config_loader.rs knifetick params_vec():
    //   [0] window_ms        [1] min_zscore       [2] min_vol_spike
    //   [3] (unused_tp)      [4] sl_buffer_pct    [5] be_trigger_pct
    //   [6] trail_pct        [7] micro_window_ms  [8] min_absorption
    //   [9] min_reclaim_pct  [10] max_speed_mult
    //
    // BUG FIX: params[1] is min_zscore (1.5-3.7). The previous HFT Absorber wasn't checking
    // zscore because it assumed the macro trigger already did!
    let sl_pct = params.get(4).copied().unwrap_or(0.003);
    let min_zscore = params.get(1).copied().unwrap_or(2.5);
    let tp_pct = sl_pct * 5.0; // Macro TP = 5× SL (generous, absorber sets real TP)

    for i in 1..candles.len() {
        let current = &candles[i];
        
        let bb_upper = indicators.bb_upper[i];
        let bb_lower = indicators.bb_lower[i];
        let bb_std = if bb_upper > bb_lower { (bb_upper - bb_lower) / 4.0 } else { 0.0001 };
        
        let range_dist = current.high - current.low;
        let range_pct = if current.high > 0.0 { range_dist / current.high } else { 0.0 };
        let mid_price = (current.high + current.low) / 2.0;
        
        // ── Epicenter Trigger (Volatility Burst + Statistically Extreme) ──
        // 1. Must exceed 1.5x of our SL distance (to avoid noise)
        // 2. Must exceed the Z-score requested by the DE optimizer
        let required_dist = (mid_price * sl_pct * 1.5).max(bb_std * min_zscore);
        
        // If the candle's total range exceeds the required extreme drop/pump:
        // Spawn the HFT absorber!
        if range_dist >= required_dist {
            let entry_price = current.close;
            
            // If price closes in the lower half, it's a dump -> we want to LONG the bounce
            if current.close < mid_price {
                let sl_price = entry_price * (1.0 - sl_pct);
                trades.push(Trade {
                    direction: "LONG".to_string(),
                    entry_price,
                    entry_idx: i,
                    exit_price: 0.0,
                    pnl_r: 0.0,
                    sl_price,
                    tp_price: entry_price * (1.0 + tp_pct),
                    risk_dist: (entry_price - sl_price).abs(),
                    pnl_abs: 0.0,
                    mfe_pct: 0.0,
                });
            } 
            // If price closes in the upper half, it's a pump -> we want to SHORT the top
            else {
                let sl_price = entry_price * (1.0 + sl_pct);
                trades.push(Trade {
                    direction: "SHORT".to_string(),
                    entry_price,
                    entry_idx: i,
                    exit_price: 0.0,
                    pnl_r: 0.0,
                    sl_price,
                    tp_price: entry_price * (1.0 - tp_pct),
                    risk_dist: (entry_price - sl_price).abs(),
                    pnl_abs: 0.0,
                    mfe_pct: 0.0,
                });
            }
        }
    }

    // Return the history of MACRO level signals.
    // The Orchestrator will check if the `last()` trade is unclosed and fresh (on the current candle).
    trades
}
