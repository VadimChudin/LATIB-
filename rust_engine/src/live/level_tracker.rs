///! Level Tracker — Автоматический S/R + Volume Profile + POC
///! ===========================================================
///! Анализирует часовые свечи (агрегрованные из 5-минуток) для:
///! 1. S/R кластеров: уровни с 2+ касаниями (ATR-нормализованные)
///! 2. Volume Profile: распределение объёма по ценовым бинам
///! 3. POC (Point of Control): цена с максимальным объёмом
///!
///! Все пороги динамические — зависят от ATR конкретного инструмента.

// use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use tracing::debug;

use crate::backtest::Candle;

// ── Config ──────────────────────────────────────────────────────────────────

/// Minimum number of touches for a valid S/R level
const MIN_TOUCHES: u32 = 2;

/// How close touches must be to cluster (fraction of ATR)
const CLUSTER_ATR_FRACTION: f64 = 0.3;

/// How many hourly candles to analyze
const HOURLY_LOOKBACK: usize = 100;

/// Number of bins for Volume Profile (splits the daily range into N price levels)
const VOLUME_PROFILE_BINS: usize = 50;

/// How many 5m candles make 1 hourly candle
const CANDLES_PER_HOUR: usize = 12;

// ── Types ───────────────────────────────────────────────────────────────────

/// A detected horizontal S/R level
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: f64,
    pub touches: u32,
    pub total_volume_at_touches: f64,
    pub last_touch: Instant,
    pub is_support: bool,
    /// How "heavy" this level is (normalized by avg hourly volume)
    pub weight: f64,
}

/// Volume Profile result for a symbol
#[derive(Debug, Clone)]
pub struct VolumeProfile {
    /// Price levels and their cumulated volume
    pub bins: Vec<(f64, f64)>,      // (price_center, volume)
    /// Point of Control — price with the highest volume
    pub poc_price: f64,
    pub poc_volume: f64,
    /// Value Area High / Low (70% of volume falls in this range)
    pub va_high: f64,
    pub va_low: f64,
}

/// Snapshot of levels for one symbol
#[derive(Debug, Clone)]
pub struct LevelSnapshot {
    pub levels: Vec<PriceLevel>,
    pub volume_profile: Option<VolumeProfile>,
    pub atr: f64,
}

impl LevelSnapshot {
    /// Find the nearest S/R level to a given price (within max_dist ATR)
    pub fn nearest_level(&self, price: f64, max_atr_dist: f64) -> Option<&PriceLevel> {
        let max_dist = self.atr * max_atr_dist;
        self.levels.iter()
            .filter(|l| (l.price - price).abs() < max_dist)
            .min_by(|a, b| {
                let da = (a.price - price).abs();
                let db = (b.price - price).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Does the POC confirm the given level? (POC within 1 bin of the level)
    pub fn poc_confirms_level(&self, level_price: f64) -> bool {
        if let Some(ref vp) = self.volume_profile {
            let bin_width = if vp.bins.len() > 1 {
                (vp.bins[1].0 - vp.bins[0].0).abs()
            } else {
                self.atr * 0.1
            };
            (vp.poc_price - level_price).abs() <= bin_width * 1.5
        } else {
            false
        }
    }
}

// ── Level Store ─────────────────────────────────────────────────────────────

/// Thread-safe shared store for all symbols' levels
pub type LevelStore = Arc<DashMap<String, LevelSnapshot>>;

pub fn new_store() -> LevelStore {
    Arc::new(DashMap::new())
}

// ── Core Algorithms ─────────────────────────────────────────────────────────

/// Build hourly candles from a buffer of 5-minute candles
fn aggregate_hourly(candles_5m: &[Candle]) -> Vec<Candle> {
    let mut hourly = Vec::new();

    let mut i = 0;
    while i + CANDLES_PER_HOUR <= candles_5m.len() {
        let chunk = &candles_5m[i..i + CANDLES_PER_HOUR];
        let open = chunk[0].open;
        let close = chunk[CANDLES_PER_HOUR - 1].close;
        let high = chunk.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
        let low = chunk.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let volume: f64 = chunk.iter().map(|c| c.volume).sum();
        let timestamp = chunk[0].timestamp.clone();

        hourly.push(Candle { open, high, low, close, volume, timestamp, num_trades: 0.0, taker_buy_volume: 0.0, quote_volume: 0.0 });
        i += CANDLES_PER_HOUR;
    }
    hourly
}

/// Calculate ATR from candles
fn calc_atr_simple(candles: &[Candle], period: usize) -> f64 {
    if candles.len() < period + 1 { return 0.0; }
    let mut sum = 0.0;
    let n = candles.len();
    for i in (n - period)..n {
        let tr = (candles[i].high - candles[i].low)
            .max((candles[i].high - candles[i - 1].close).abs())
            .max((candles[i].low - candles[i - 1].close).abs());
        sum += tr;
    }
    sum / period as f64
}

/// Find S/R clusters: prices where highs/lows cluster together (2+ touches)
pub fn find_sr_levels(candles: &[Candle], atr: f64) -> Vec<PriceLevel> {
    if atr <= 0.0 || candles.is_empty() { return vec![]; }

    let cluster_dist = atr * CLUSTER_ATR_FRACTION;
    let now = Instant::now();

    // Collect all swing points (highs = resistance, lows = support)
    let mut swing_points: Vec<(f64, bool, f64)> = Vec::new(); // (price, is_support, volume)

    for c in candles {
        swing_points.push((c.high, false, c.volume)); // Resistance candidate
        swing_points.push((c.low, true, c.volume));   // Support candidate
    }

    // Sort by price
    swing_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Cluster nearby points
    let mut levels: Vec<PriceLevel> = Vec::new();
    let mut i = 0;

    while i < swing_points.len() {
        let base_price = swing_points[i].0;
        let mut cluster_sum = base_price;
        let mut cluster_vol = swing_points[i].2;
        let mut count = 1u32;
        let mut support_count = 0u32;
        let mut resist_count = 0u32;

        if swing_points[i].1 { support_count += 1; } else { resist_count += 1; }

        // Absorb nearby points into this cluster
        let mut j = i + 1;
        while j < swing_points.len() && (swing_points[j].0 - base_price) < cluster_dist {
            cluster_sum += swing_points[j].0;
            cluster_vol += swing_points[j].2;
            count += 1;
            if swing_points[j].1 { support_count += 1; } else { resist_count += 1; }
            j += 1;
        }

        if count >= MIN_TOUCHES {
            let avg_price = cluster_sum / count as f64;
            levels.push(PriceLevel {
                price: avg_price,
                touches: count,
                total_volume_at_touches: cluster_vol,
                last_touch: now,
                is_support: support_count > resist_count,
                weight: 0.0, // Will be normalized later
            });
        }
        i = j;
    }

    // Normalize weights by average volume
    let avg_vol: f64 = if levels.is_empty() { 1.0 } else {
        levels.iter().map(|l| l.total_volume_at_touches).sum::<f64>() / levels.len() as f64
    };
    for level in &mut levels {
        level.weight = level.total_volume_at_touches / avg_vol.max(1.0);
    }

    // Sort by touches (most significant first)
    levels.sort_by(|a, b| b.touches.cmp(&a.touches));
    levels
}

/// Calculate Volume Profile and POC from candles
pub fn calc_volume_profile(candles: &[Candle]) -> Option<VolumeProfile> {
    if candles.is_empty() { return None; }

    let price_high = candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
    let price_low = candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    let range = price_high - price_low;

    if range <= 0.0 { return None; }

    let bin_width = range / VOLUME_PROFILE_BINS as f64;
    let mut bins = vec![0.0f64; VOLUME_PROFILE_BINS];

    // Distribute each candle's volume across the bins it spans
    for c in candles {
        let low_bin = ((c.low - price_low) / bin_width).floor() as usize;
        let high_bin = ((c.high - price_low) / bin_width).floor() as usize;
        let low_bin = low_bin.min(VOLUME_PROFILE_BINS - 1);
        let high_bin = high_bin.min(VOLUME_PROFILE_BINS - 1);

        let num_bins = (high_bin - low_bin + 1) as f64;
        let vol_per_bin = c.volume / num_bins;

        for b in low_bin..=high_bin {
            bins[b] += vol_per_bin;
        }
    }

    // Find POC (bin with max volume)
    let (poc_idx, &poc_volume) = bins.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    let poc_price = price_low + (poc_idx as f64 + 0.5) * bin_width;

    // Calculate Value Area (70% of total volume)
    let total_volume: f64 = bins.iter().sum();
    let va_target = total_volume * 0.70;

    let mut va_low_idx = poc_idx;
    let mut va_high_idx = poc_idx;
    let mut va_vol = bins[poc_idx];

    while va_vol < va_target {
        let expand_down = if va_low_idx > 0 { bins[va_low_idx - 1] } else { 0.0 };
        let expand_up = if va_high_idx < VOLUME_PROFILE_BINS - 1 { bins[va_high_idx + 1] } else { 0.0 };

        if expand_down >= expand_up && va_low_idx > 0 {
            va_low_idx -= 1;
            va_vol += bins[va_low_idx];
        } else if va_high_idx < VOLUME_PROFILE_BINS - 1 {
            va_high_idx += 1;
            va_vol += bins[va_high_idx];
        } else {
            break;
        }
    }

    let result_bins: Vec<(f64, f64)> = bins.iter().enumerate()
        .map(|(i, &v)| (price_low + (i as f64 + 0.5) * bin_width, v))
        .collect();

    Some(VolumeProfile {
        bins: result_bins,
        poc_price,
        poc_volume,
        va_high: price_low + (va_high_idx as f64 + 1.0) * bin_width,
        va_low: price_low + va_low_idx as f64 * bin_width,
    })
}

/// Full update: aggregate hourly, find S/R, build profile, update store
pub fn update_levels(
    store: &LevelStore,
    symbol: &str,
    candles_5m: &[Candle],
) {
    // Need enough history: 100 hours × 12 = 1200 5m candles 
    // We work with whatever we have (minimum ~50 hourly candles)
    let hourly = aggregate_hourly(candles_5m);
    if hourly.len() < 20 { return; }

    // Take last HOURLY_LOOKBACK candles
    let start = hourly.len().saturating_sub(HOURLY_LOOKBACK);
    let window = &hourly[start..];

    let atr = calc_atr_simple(window, 14);
    if atr <= 0.0 { return; }

    let levels = find_sr_levels(window, atr);
    let volume_profile = calc_volume_profile(window);

    let level_count = levels.len();
    let poc_info = volume_profile.as_ref()
        .map(|vp| format!("POC={:.4}", vp.poc_price))
        .unwrap_or_else(|| "no POC".to_string());

    store.insert(symbol.to_string(), LevelSnapshot {
        levels,
        volume_profile,
        atr,
    });

    debug!("📊 LevelTracker [{}]: {} S/R levels | ATR={:.4} | {}",
        symbol, level_count, atr, poc_info);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(o: f64, h: f64, l: f64, c: f64, v: f64) -> Candle {
        Candle { open: o, high: h, low: l, close: c, volume: v, timestamp: String::new(), num_trades: 0.0, taker_buy_volume: 0.0, quote_volume: 0.0 }
    }

    #[test]
    fn test_sr_clustering() {
        // Create candles that touch 100.0 multiple times
        let candles = vec![
            make_candle(99.0, 100.1, 98.5, 99.5, 1000.0),
            make_candle(99.5, 101.0, 99.0, 100.5, 1200.0),
            make_candle(100.5, 100.8, 99.8, 100.0, 900.0),
            make_candle(100.0, 100.2, 98.0, 98.5, 1500.0),  // Low touches ~98
            make_candle(98.5, 99.0, 97.9, 98.8, 1100.0),    // Low touches ~98 again
            make_candle(98.8, 100.3, 98.2, 100.0, 800.0),
        ];
        let atr = 1.5;
        let levels = find_sr_levels(&candles, atr);

        assert!(!levels.is_empty(), "Should find at least one S/R level");
        assert!(levels[0].touches >= 2, "Top level should have 2+ touches");
    }

    #[test]
    fn test_volume_profile_poc() {
        // Create candles with high volume around 100.0
        let mut candles = Vec::new();
        for i in 0..50 {
            let base = 99.0 + (i as f64 * 0.1) % 3.0;
            let vol = if (base - 100.5).abs() < 0.5 { 5000.0 } else { 1000.0 };
            candles.push(make_candle(base, base + 0.5, base - 0.3, base + 0.2, vol));
        }

        let vp = calc_volume_profile(&candles);
        assert!(vp.is_some(), "Should produce a volume profile");
        let vp = vp.unwrap();
        assert!(vp.poc_volume > 0.0, "POC should have volume");
        assert!(vp.va_high > vp.va_low, "Value area should be valid");
    }
}
