/// Backtest core: CSV loading, trade execution, and result types.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct LevelInfo {
    pub price: f64,
    pub touch_count: usize,
    pub first_seen_idx: usize,
    pub last_seen_idx: usize,
    pub strength: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Candle {
    pub timestamp: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    // HFT fields (optional — backward compatible with old CSVs)
    #[serde(default)]
    pub num_trades: f64,
    #[serde(default)]
    pub taker_buy_volume: f64,
    #[serde(default)]
    pub quote_volume: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Trade {
    pub entry_idx: usize,
    pub direction: String,  // "LONG" or "SHORT"
    pub entry_price: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    pub exit_price: f64,
    pub pnl_r: f64,  // PnL in R-multiples
    pub risk_dist: f64,
    pub pnl_abs: f64,
    /// Phase 31: Maximum Favorable Excursion % (how far price went in our favor)
    pub mfe_pct: f64,
}

use crate::bitset_engine::BitsetSignals;

#[derive(Debug, Clone)]
pub struct PrecomputedData {
    pub atr: Vec<f64>,
    pub rsi: Vec<f64>,
    pub ema_fast: Vec<f64>,
    pub ema_slow: Vec<f64>,
    pub ema_200: Vec<f64>,
    pub adx: Vec<f64>,
    pub bb_upper: Vec<f64>,
    pub bb_lower: Vec<f64>,
    pub bb_mid: Vec<f64>,
    pub bitsets: Option<BitsetSignals>,
    pub btc_vol: Option<Vec<f64>>,
    // HFT: delta = 2*taker_buy_vol - vol (positive=buyers, negative=sellers)
    pub delta: Vec<f64>,
    // HFT: tape speed = num_trades per candle
    pub tape_speed: Vec<f64>,
}

// build_bitsets logic moved to bitset_engine.rs


/// Load OHLCV candles from CSV
pub fn load_csv(path: &Path) -> Vec<Candle> {
    let mut reader = csv::Reader::from_path(path).expect("Failed to open CSV");
    let mut candles = Vec::with_capacity(220_000);
    for result in reader.deserialize() {
        if let Ok(candle) = result {
            candles.push(candle);
        }
    }
    candles
}

/// Calculate ATR over a rolling window
pub fn calc_atr(candles: &[Candle], period: usize) -> Vec<f64> {
    let n = candles.len();
    let mut atr = vec![0.0; n];
    for i in 1..n {
        let tr = (candles[i].high - candles[i].low)
            .max((candles[i].high - candles[i - 1].close).abs())
            .max((candles[i].low - candles[i - 1].close).abs());
        if i < period {
            atr[i] = tr;
        } else {
            atr[i] = (atr[i - 1] * (period as f64 - 1.0) + tr) / period as f64;
        }
    }
    atr
}

/// Simple EMA
pub fn calc_ema(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut ema = vec![0.0; n];
    if n == 0 { return ema; }
    let mult = 2.0 / (period as f64 + 1.0);
    ema[0] = values[0];
    for i in 1..n {
        ema[i] = (values[i] - ema[i - 1]) * mult + ema[i - 1];
    }
    ema
}

/// RSI calculation
pub fn calc_rsi(closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    let mut rsi = vec![50.0; n];
    if n <= period { return rsi; }

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    // Initial average
    for i in 1..=period {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 { avg_gain += change; } else { avg_loss -= change; }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    for i in period..n {
        if i > period {
            let change = closes[i] - closes[i - 1];
            let (gain, loss) = if change > 0.0 { (change, 0.0) } else { (0.0, -change) };
            avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
        }
        if avg_loss == 0.0 {
            rsi[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    rsi
}

/// ADX calculation using Wilder's Smoothing (RMA)
pub fn calc_adx(candles: &[Candle], period: usize) -> Vec<f64> {
    let n = candles.len();
    let mut adx = vec![0.0; n];
    if n <= period * 2 { return adx; }

    let mut tr_smoothed = 0.0;
    let mut plus_dm_smoothed = 0.0;
    let mut minus_dm_smoothed = 0.0;
    let mut dx_history = vec![0.0; n];

    for i in 1..n {
        let up_move = candles[i].high - candles[i - 1].high;
        let down_move = candles[i - 1].low - candles[i].low;

        let plus_dm = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
        let minus_dm = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };

        let tr = (candles[i].high - candles[i].low)
            .max((candles[i].high - candles[i - 1].close).abs())
            .max((candles[i].low - candles[i - 1].close).abs());

        if i < period {
            tr_smoothed += tr;
            plus_dm_smoothed += plus_dm;
            minus_dm_smoothed += minus_dm;
        } else if i == period {
            tr_smoothed = (tr_smoothed + tr) / period as f64;
            plus_dm_smoothed = (plus_dm_smoothed + plus_dm) / period as f64;
            minus_dm_smoothed = (minus_dm_smoothed + minus_dm) / period as f64;
        } else {
            tr_smoothed = (tr_smoothed * (period as f64 - 1.0) + tr) / period as f64;
            plus_dm_smoothed = (plus_dm_smoothed * (period as f64 - 1.0) + plus_dm) / period as f64;
            minus_dm_smoothed = (minus_dm_smoothed * (period as f64 - 1.0) + minus_dm) / period as f64;
        }

        if i >= period && tr_smoothed > 0.0 {
            let di_plus = 100.0 * plus_dm_smoothed / tr_smoothed;
            let di_minus = 100.0 * minus_dm_smoothed / tr_smoothed;
            let sum = di_plus + di_minus;
            let dx = if sum > 0.0 { 100.0 * (di_plus - di_minus).abs() / sum } else { 0.0 };
            dx_history[i] = dx;

            if i == period * 2 - 1 {
                let dx_sum: f64 = dx_history[period..=i].iter().sum();
                adx[i] = dx_sum / period as f64;
            } else if i >= period * 2 {
                adx[i] = (adx[i - 1] * (period as f64 - 1.0) + dx) / period as f64;
            }
        }
    }
    adx
}

/// Bollinger Bands logic
pub fn calc_bollinger_bands(values: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = values.len();
    let mut mid = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut lower = vec![0.0; n];

    if n < period { return (upper, lower, mid); }

    for i in period..n {
        let window = &values[i-period..i];
        let sum: f64 = window.iter().sum();
        let mean = sum / period as f64;
        
        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std = variance.sqrt();

        mid[i] = mean;
        upper[i] = mean + std_dev * std;
        lower[i] = mean - std_dev * std;
    }
    (upper, lower, mid)
}

/// Simulate trade with SL/TP using future candles
pub fn _simulate_trade(
    candles: &[Candle],
    entry_idx: usize,
    direction: &str,
    entry_price: f64,
    sl_price: f64,
    tp_price: f64,
    max_bars: usize,
) -> Trade {
    let mut exit_price = entry_price;
    let mut pnl_r = 0.0;
    let risk = (entry_price - sl_price).abs();

    for i in (entry_idx + 1)..candles.len().min(entry_idx + max_bars) {
        let c = &candles[i];
        if direction == "LONG" {
            if c.low <= sl_price {
                exit_price = sl_price;
                pnl_r = -1.0;
                break;
            }
            if tp_price > 0.0 && c.high >= tp_price {
                exit_price = tp_price;
                pnl_r = if risk > 0.0 { (tp_price - entry_price) / risk } else { 1.0 };
                break;
            }
        } else {
            if c.high >= sl_price {
                exit_price = sl_price;
                pnl_r = -1.0;
                break;
            }
            if tp_price > 0.0 && c.low <= tp_price {
                exit_price = tp_price;
                pnl_r = if risk > 0.0 { (entry_price - tp_price) / risk } else { 1.0 };
                break;
            }
        }
    }

    Trade {
        entry_idx,
        direction: direction.to_string(),
        entry_price,
        sl_price,
        tp_price,
        exit_price,
        pnl_r,
        risk_dist: risk,
        pnl_abs: 0.0,
            mfe_pct: 0.0,
    }
}

/// Slippage constant: 0.04% per side (entry + exit = 0.08% total)
pub const SLIPPAGE_PCT: f64 = 0.0004;

pub fn apply_slippage(trades: &mut Vec<Trade>) {
    for trade in trades.iter_mut() {
        let slip_entry = trade.entry_price * SLIPPAGE_PCT;
        let slip_exit = trade.exit_price * SLIPPAGE_PCT;

        if trade.direction == "LONG" {
            let adj_entry = trade.entry_price + slip_entry;
            let adj_exit = trade.exit_price - slip_exit;
            let risk = (trade.entry_price - trade.sl_price).abs();
            if risk > 0.0 {
                trade.pnl_r = (adj_exit - adj_entry) / risk;
            }
        } else {
            let adj_entry = trade.entry_price - slip_entry;
            let adj_exit = trade.exit_price + slip_exit;
            let risk = (trade.sl_price - trade.entry_price).abs();
            if risk > 0.0 {
                trade.pnl_r = (adj_entry - adj_exit) / risk;
            }
        }
    }
}

pub fn apply_max_drawdown(trades: &[Trade], max_daily_loss_r: f64) -> Vec<Trade> {
    let mut result = Vec::new();
    let mut daily_pnl = 0.0;
    let mut last_day_idx = 0usize;

    for trade in trades {
        let current_day = trade.entry_idx / 288;
        if current_day != last_day_idx {
            daily_pnl = 0.0;
            last_day_idx = current_day;
        }
        if daily_pnl <= -max_daily_loss_r {
            continue;
        }
        daily_pnl += trade.pnl_r;
        result.push(trade.clone());
    }
    result
}

pub fn find_historical_levels(candles: &[Candle], tolerance_pct: f64) -> Vec<LevelInfo> {
    let mut levels = Vec::new();
    if candles.is_empty() { return levels; }
    let avg_price: f64 = candles.iter().take(100).map(|c| c.close).sum::<f64>() / (100.0_f64).min(candles.len() as f64);
    let bucket_size = avg_price * tolerance_pct;
    let mut touch_map: HashMap<i64, Vec<usize>> = HashMap::new();

    for (idx, c) in candles.iter().enumerate() {
        let high_key = (c.high / bucket_size).round() as i64;
        let low_key = (c.low / bucket_size).round() as i64;
        touch_map.entry(high_key).or_default().push(idx);
        if high_key != low_key {
            touch_map.entry(low_key).or_default().push(idx);
        }
    }

    for (key, indices) in touch_map {
        if indices.len() >= 3 {
            let first = indices[0];
            let last = *indices.last().unwrap();
            if last - first > 60 {
                let price = key as f64 * bucket_size;
                levels.push(LevelInfo {
                    price,
                    touch_count: indices.len(),
                    first_seen_idx: first,
                    last_seen_idx: last,
                    strength: indices.len() as f64 * (last - first) as f64 / 1000.0,
                });
            }
        }
    }
    levels.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
    levels
}
