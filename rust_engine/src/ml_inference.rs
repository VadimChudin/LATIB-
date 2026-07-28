/// ML Inference Engine — loads LightGBM JSON models and does prediction
/// 
/// LightGBM trees are just nested if/else — perfect for Rust.
/// Model format: JSON exported by export_models_json.py

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize, Debug)]
pub struct LgbmModel {
    pub num_trees: usize,
    pub num_features: usize,
    pub feature_names: Vec<String>,
    pub trees: Vec<serde_json::Value>,  // Raw JSON tree nodes
}

#[derive(Deserialize, Debug, Clone)]
pub struct MetaInfo {
    pub strategy_map: std::collections::HashMap<String, usize>,
    pub spot_probe_map: std::collections::HashMap<String, i32>,
    pub feature_cols: Vec<String>,
    pub train_size: usize,
    pub trained_at: String,
}

impl LgbmModel {
    /// Load model from JSON file
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let model: LgbmModel = serde_json::from_str(&data)?;
        Ok(model)
    }

    /// Predict probability (0.0 - 1.0) for binary classification
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        let mut raw_score = 0.0;

        for tree in &self.trees {
            raw_score += self.evaluate_tree(tree, features);
        }

        // Sigmoid to convert raw score to probability
        1.0 / (1.0 + (-raw_score).exp())
    }

    /// Recursively evaluate a single decision tree
    fn evaluate_tree(&self, node: &serde_json::Value, features: &[f64]) -> f64 {
        // Leaf node
        if let Some(leaf_value) = node.get("leaf_value") {
            return leaf_value.as_f64().unwrap_or(0.0);
        }

        // Decision node
        let split_feature = node.get("split_feature")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        
        let threshold = node.get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let decision_type = node.get("decision_type")
            .and_then(|v| v.as_str())
            .unwrap_or("<=");

        let feature_val = if split_feature < features.len() {
            features[split_feature]
        } else {
            0.0
        };

        let go_left = match decision_type {
            "<=" => feature_val <= threshold,
            "<" => feature_val < threshold,
            ">" => feature_val > threshold,
            ">=" => feature_val >= threshold,
            "==" => (feature_val - threshold).abs() < 1e-10,
            _ => feature_val <= threshold,
        };

        if go_left {
            if let Some(left) = node.get("left_child") {
                self.evaluate_tree(left, features)
            } else {
                0.0
            }
        } else {
            if let Some(right) = node.get("right_child") {
                self.evaluate_tree(right, features)
            } else {
                0.0
            }
        }
    }
}

/// Feature extraction from candles (mirrors Python ml_filter.py)
pub fn extract_features(
    candles: &[crate::backtest::Candle], 
    idx: usize,
    btc_trend: f64,
    btc_volatility_pct: f64,
    btc_dump_3c: f64,
    funding_rate: f64
) -> Vec<f64> {
    if idx < 200 { return vec![0.0; 26]; }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let opens: Vec<f64> = candles.iter().map(|c| c.open).collect();

    let close = closes[idx];
    let high = highs[idx];
    let low = lows[idx];
    let volume = volumes[idx];

    // ATR
    let atr = crate::backtest::calc_atr(candles, 14);
    let atr_val = atr[idx];
    let atr_pct = if close > 0.0 { (atr_val / close) * 100.0 } else { 0.0 };

    // Volume ratio
    let vol_sma: f64 = volumes[idx.saturating_sub(20)..idx].iter().sum::<f64>() / 20.0;
    let vol_ratio = if vol_sma > 0.0 { volume / vol_sma } else { 1.0 };

    // Hour (placeholder = 12 without timestamp)
    let hour = 12.0;

    // EMA50 distance
    let ema_50: f64 = closes[idx.saturating_sub(50)..idx].iter().sum::<f64>() / 50.0;
    let dist_to_ema = if ema_50 > 0.0 { (close - ema_50) / ema_50 * 100.0 } else { 0.0 };

    // RSI (simplified)
    let rsi = calc_rsi_at(&closes, idx, 14);
    
    // ADX — Average Directional Index (Wilder's method)
    let adx = calc_adx_at(&highs, &lows, &closes, idx, 14);

    // MACD histogram
    let ema_12 = ema_at(&closes, idx, 12);
    let ema_26 = ema_at(&closes, idx, 26);
    let macd_line = ema_12 - ema_26;
    let macd_hist = macd_line; // Simplified

    // Bollinger position
    let bb_mean: f64 = closes[idx.saturating_sub(20)..idx].iter().sum::<f64>() / 20.0;
    let bb_std: f64 = (closes[idx.saturating_sub(20)..idx].iter()
        .map(|c| (c - bb_mean).powi(2)).sum::<f64>() / 20.0).sqrt();
    let bb_pos = if bb_std > 0.0 { (close - bb_mean) / (2.0 * bb_std) + 0.5 } else { 0.5 };

    // Stochastic
    let h14 = highs[idx.saturating_sub(14)..=idx].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let l14 = lows[idx.saturating_sub(14)..=idx].iter().cloned().fold(f64::INFINITY, f64::min);
    let stoch = if h14 - l14 > 0.0 { (close - l14) / (h14 - l14) * 100.0 } else { 50.0 };

    // CCI
    let tp = (high + low + close) / 3.0;
    let tp_sma: f64 = (0..20).map(|j| {
        let k = idx.saturating_sub(j);
        (highs[k] + lows[k] + closes[k]) / 3.0
    }).sum::<f64>() / 20.0;
    let md: f64 = (0..20).map(|j| {
        let k = idx.saturating_sub(j);
        ((highs[k] + lows[k] + closes[k]) / 3.0 - tp_sma).abs()
    }).sum::<f64>() / 20.0;
    let cci = if md > 0.0 { (tp - tp_sma) / (0.015 * md) } else { 0.0 };

    // MFI — Money Flow Index (volume-weighted RSI)
    let mfi = calc_mfi_at(&highs, &lows, &closes, &volumes, idx, 14);

    // Hurst Exponent — Rescaled Range (R/S) method
    let hurst = calc_hurst_at(&closes, idx, 100);

    // === B1: Rolling features ===
    let rsi_5 = if idx >= 5 { calc_rsi_at(&closes, idx - 5, 14) } else { rsi };
    let rsi_change_5 = rsi - rsi_5;

    let vol_5 = if idx >= 5 { volumes[idx - 5] } else { volume };
    let vol_change_5 = if vol_5 > 0.0 { volume / vol_5 - 1.0 } else { 0.0 };

    let atr_5 = if idx >= 5 { atr[idx - 5] } else { atr_val };
    let atr_change_5 = if atr_5 > 0.0 { atr_val / atr_5 - 1.0 } else { 0.0 };

    let macd_5 = if idx >= 5 { ema_at(&closes, idx - 5, 12) - ema_at(&closes, idx - 5, 26) } else { macd_hist };
    let macd_slope = macd_hist - macd_5;

    let range_high = highs[idx.saturating_sub(10)..idx].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range_low = lows[idx.saturating_sub(10)..idx].iter().cloned().fold(f64::INFINITY, f64::min);
    let close_range = range_high - range_low;
    let close_pos = if close_range > 0.0 { (close - range_low) / close_range } else { 0.5 };

    let ema_10: f64 = closes[idx.saturating_sub(10)..idx].iter().sum::<f64>() / 10.0;
    let ema_20: f64 = closes[idx.saturating_sub(20)..idx].iter().sum::<f64>() / 20.0;
    let ema_slope = if ema_20 > 0.0 { (ema_10 - ema_20) / ema_20 * 100.0 } else { 0.0 };

    let bb_width = if close > 0.0 { bb_std * 4.0 / close * 100.0 } else { 0.0 };
    let stoch_dist_50 = (stoch - 50.0).abs();

    // === B3: HTF ===
    let ema_50_val: f64 = closes[idx.saturating_sub(50)..idx].iter().sum::<f64>() / 50.0;
    let ema_200_val: f64 = closes[idx.saturating_sub(200)..idx].iter().sum::<f64>() / 200.0;
    let htf_trend = if ema_200_val > 0.0 { (ema_50_val - ema_200_val) / ema_200_val * 100.0 } else { 0.0 };

    let h1_open = if idx >= 12 { opens[idx - 11] } else { opens[idx] };
    let h1_direction = if close > h1_open { 1.0 } else { -1.0 };

    // Feature vector (strict order matching Python pd.DataFrame columns after dropping 'index')
    vec![
        hurst,              // 0
        atr_pct,            // 1
        vol_ratio,          // 2
        hour,               // 3
        rsi,                // 4
        adx,                // 5
        macd_hist,          // 6
        stoch,              // 7
        cci,                // 8
        mfi,                // 9
        rsi_change_5,       // 10
        vol_change_5,       // 11
        atr_change_5,       // 12
        macd_slope,         // 13
        close_pos,          // 14
        ema_slope,          // 15
        dist_to_ema,        // 16
        htf_trend,          // 17
        stoch_dist_50,      // 18
        bb_pos,             // 19
        bb_width,           // 20
        h1_direction,       // 21
        funding_rate,       // 22
        btc_trend,          // 23
        btc_volatility_pct, // 24
        btc_dump_3c,        // 25
        // --- Phase 11: ScalpMTF Micro Features ---
        calc_zscore_atr(&atr, idx), // 26: volatility_zscore
        calc_micro_trend(&closes, idx, atr_val), // 27: micro_trend
        calc_tightness(&highs, &lows, idx, atr_val), // 28: tightness
    ]
}

fn calc_zscore_atr(atr: &[f64], idx: usize) -> f64 {
    if idx < 100 { return 0.0; }
    let window = &atr[idx.saturating_sub(99)..=idx];
    let mean: f64 = window.iter().sum::<f64>() / 100.0;
    let std: f64 = (window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 100.0).sqrt();
    if std > 0.0 { (atr[idx] - mean) / std } else { 0.0 }
}

fn calc_micro_trend(closes: &[f64], idx: usize, atr: f64) -> f64 {
    if idx < 5 || atr <= 0.0 { return 0.0; }
    let mean_5: f64 = closes[idx.saturating_sub(4)..=idx].iter().sum::<f64>() / 5.0;
    (closes[idx] - mean_5) / atr
}

fn calc_tightness(highs: &[f64], lows: &[f64], idx: usize, atr: f64) -> f64 {
    if idx < 10 || atr <= 0.0 { return 0.0; }
    let h10 = highs[idx.saturating_sub(9)..=idx].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let l10 = lows[idx.saturating_sub(9)..=idx].iter().cloned().fold(f64::INFINITY, f64::min);
    (h10 - l10) / atr
}

fn calc_rsi_at(closes: &[f64], idx: usize, period: usize) -> f64 {
    if idx < period + 1 { return 50.0; }
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in (idx - period)..idx {
        let change = closes[i + 1] - closes[i];
        if change > 0.0 { avg_gain += change; } else { avg_loss -= change; }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;
    if avg_loss == 0.0 { return 100.0; }
    let rs = avg_gain / avg_loss;
    100.0 - 100.0 / (1.0 + rs)
}

fn ema_at(values: &[f64], idx: usize, period: usize) -> f64 {
    let start = idx.saturating_sub(period * 3);
    let mult = 2.0 / (period as f64 + 1.0);
    let mut ema = values[start];
    for i in (start + 1)..=idx {
        ema = (values[i] - ema) * mult + ema;
    }
    ema
}

/// ADX — Average Directional Index (Wilder's smoothed method)
/// Measures trend strength: >25 = trending, <20 = ranging
fn calc_adx_at(highs: &[f64], lows: &[f64], closes: &[f64], idx: usize, period: usize) -> f64 {
    if idx < period * 2 + 1 { return 25.0; }

    let start = idx.saturating_sub(period * 2);
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;
    let mut tr_sum = 0.0;

    // Initial sums
    for i in (start + 1)..=(start + period) {
        let up_move = highs[i] - highs[i - 1];
        let down_move = lows[i - 1] - lows[i];

        if up_move > down_move && up_move > 0.0 { plus_dm_sum += up_move; }
        if down_move > up_move && down_move > 0.0 { minus_dm_sum += down_move; }

        let tr1 = highs[i] - lows[i];
        let tr2 = (highs[i] - closes[i - 1]).abs();
        let tr3 = (lows[i] - closes[i - 1]).abs();
        tr_sum += tr1.max(tr2).max(tr3);
    }

    // Wilder's smoothing for remaining bars
    let mut dx_sum = 0.0;
    let mut dx_count = 0;

    for i in (start + period + 1)..=idx {
        let up_move = highs[i] - highs[i - 1];
        let down_move = lows[i - 1] - lows[i];

        let mut plus_dm = 0.0;
        let mut minus_dm = 0.0;
        if up_move > down_move && up_move > 0.0 { plus_dm = up_move; }
        if down_move > up_move && down_move > 0.0 { minus_dm = down_move; }

        let tr1 = highs[i] - lows[i];
        let tr2 = (highs[i] - closes[i - 1]).abs();
        let tr3 = (lows[i] - closes[i - 1]).abs();
        let tr = tr1.max(tr2).max(tr3);

        let p = period as f64;
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / p) + plus_dm;
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / p) + minus_dm;
        tr_sum = tr_sum - (tr_sum / p) + tr;

        if tr_sum > 0.0 {
            let plus_di = (plus_dm_sum / tr_sum) * 100.0;
            let minus_di = (minus_dm_sum / tr_sum) * 100.0;
            let di_sum = plus_di + minus_di;
            if di_sum > 0.0 {
                let dx = ((plus_di - minus_di).abs() / di_sum) * 100.0;
                dx_sum += dx;
                dx_count += 1;
            }
        }
    }

    if dx_count > 0 { dx_sum / dx_count as f64 } else { 25.0 }
}

/// MFI — Money Flow Index (volume-weighted RSI)
/// Uses typical price × volume to measure buying/selling pressure
fn calc_mfi_at(highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64], idx: usize, period: usize) -> f64 {
    if idx < period + 1 { return 50.0; }

    let mut pos_flow = 0.0;
    let mut neg_flow = 0.0;

    for i in (idx - period)..idx {
        let tp_curr = (highs[i + 1] + lows[i + 1] + closes[i + 1]) / 3.0;
        let tp_prev = (highs[i] + lows[i] + closes[i]) / 3.0;
        let money_flow = tp_curr * volumes[i + 1];

        if tp_curr > tp_prev {
            pos_flow += money_flow;
        } else {
            neg_flow += money_flow;
        }
    }

    if neg_flow == 0.0 { return 100.0; }
    let mfr = pos_flow / neg_flow;
    100.0 - 100.0 / (1.0 + mfr)
}

/// Hurst Exponent — Rescaled Range (R/S) analysis
/// >0.5 = trending, <0.5 = mean-reverting, =0.5 = random walk
fn calc_hurst_at(closes: &[f64], idx: usize, window: usize) -> f64 {
    if idx < window { return 0.5; }

    let series = &closes[(idx - window)..idx];
    let n = series.len();
    if n < 20 { return 0.5; }

    // Calculate returns
    let returns: Vec<f64> = (1..n).map(|i| (series[i] / series[i - 1]).ln()).collect();
    let m = returns.len();
    if m < 10 { return 0.5; }

    // Try multiple sub-window sizes and fit log-log regression
    let mut log_ns = Vec::new();
    let mut log_rs = Vec::new();

    for div in &[4, 8, 16, 32] {
        let sub_n = m / div;
        if sub_n < 4 { continue; }

        let mut rs_values = Vec::new();
        for chunk in returns.chunks(sub_n) {
            if chunk.len() < 4 { continue; }

            let mean: f64 = chunk.iter().sum::<f64>() / chunk.len() as f64;
            let deviations: Vec<f64> = chunk.iter().map(|r| r - mean).collect();

            // Cumulative deviation
            let mut cum_dev = Vec::with_capacity(chunk.len());
            let mut running = 0.0;
            for d in &deviations {
                running += d;
                cum_dev.push(running);
            }

            let range = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                      - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);

            let std = (chunk.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                      / chunk.len() as f64).sqrt();

            if std > 0.0 {
                rs_values.push(range / std);
            }
        }

        if !rs_values.is_empty() {
            let avg_rs: f64 = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
            if avg_rs > 0.0 {
                log_ns.push((sub_n as f64).ln());
                log_rs.push(avg_rs.ln());
            }
        }
    }

    if log_ns.len() < 2 { return 0.5; }

    // Simple linear regression: slope = Hurst exponent
    let n_pts = log_ns.len() as f64;
    let sum_x: f64 = log_ns.iter().sum();
    let sum_y: f64 = log_rs.iter().sum();
    let sum_xy: f64 = log_ns.iter().zip(log_rs.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_ns.iter().map(|x| x * x).sum();

    let denom = n_pts * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-10 { return 0.5; }

    let slope = (n_pts * sum_xy - sum_x * sum_y) / denom;
    slope.clamp(0.0, 1.0)
}
