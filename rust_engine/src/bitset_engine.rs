use fixedbitset::FixedBitSet;

use crate::backtest::Candle;

#[derive(Debug, Clone)]
pub struct BitsetSignals {
    /// Bit is 1 if close > ema_200
    pub price_above_ema200: FixedBitSet,
    /// Bit is 1 if close < ema_200
    pub price_below_ema200: FixedBitSet,
    /// Bit is 1 if ADX > 20.0
    pub adx_strong: FixedBitSet,
    
    // Volume Spikes (Multi-threshold)
    pub vol_spikes: Vec<FixedBitSet>, // thresholds: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    // Candle Body (Multi-threshold)
    pub full_bodies_long: Vec<FixedBitSet>, // ratios: [0.5, 0.6, 0.7, 0.8, 0.9]
    pub full_bodies_short: Vec<FixedBitSet>,
    
    // SMA/EMA Crosses or proximity
    pub fvg_bullish: Vec<FixedBitSet>, 
    pub fvg_bearish: Vec<FixedBitSet>,
}

impl BitsetSignals {
    pub fn new(n: usize) -> Self {
        Self {
            price_above_ema200: FixedBitSet::with_capacity(n),
            price_below_ema200: FixedBitSet::with_capacity(n),
            adx_strong: FixedBitSet::with_capacity(n),
            vol_spikes: vec![FixedBitSet::with_capacity(n); 7],
            full_bodies_long: vec![FixedBitSet::with_capacity(n); 5],
            full_bodies_short: vec![FixedBitSet::with_capacity(n); 5],
            fvg_bullish: vec![FixedBitSet::with_capacity(n); 10],
            fvg_bearish: vec![FixedBitSet::with_capacity(n); 10],
        }
    }

    /// Combines multiple criteria into a single entry signal bitset
    /// This is where the x1000 speedup happens (bitwise AND)
    pub fn combine_signals(&self, 
        vol_spike_idx: usize, 
        body_long_idx: usize,
        use_ema_filter: bool
    ) -> FixedBitSet {
        let mut result = self.vol_spikes[vol_spike_idx].clone();
        result &= &self.full_bodies_long[body_long_idx];
        
        if use_ema_filter {
            result &= &self.price_above_ema200;
        }
        
        result
    }

    /// Returns a list of indices where signals are active
    pub fn get_signal_indices(bitset: &FixedBitSet) -> Vec<usize> {
        bitset.ones().collect()
    }

    /// Export raw u32 buffers for GPU
    pub fn export_raw_u32(&self) -> Vec<u32> {
        self.price_above_ema200.as_slice().to_vec()
    }

    /// Precalculate all 70 discrete combinations for GPU acceleration
    pub fn precalculate_combinations(&self) -> Vec<u32> {
        let mut all_buffers = Vec::with_capacity(70 * self.price_above_ema200.as_slice().len());
        
        for v_idx in 0..7 {
            for b_idx in 0..5 {
                for ema_f in 0..2 {
                    let mut combined = self.vol_spikes[v_idx].clone();
                    combined &= &self.full_bodies_long[b_idx];
                    if ema_f == 1 {
                        combined &= &self.price_above_ema200;
                    }
                    all_buffers.extend_from_slice(combined.as_slice());
                }
            }
        }
        all_buffers
    }
}

pub fn build_bitsets(candles: &[Candle], ema_200: &[f64], adx: &[f64]) -> BitsetSignals {
    let n = candles.len();
    let mut bs = BitsetSignals::new(n);

    // Volume MA (20)
    let mut sum_vol = 0.0;
    let mut vol_mas = vec![0.0; n];
    for i in 0..n {
        sum_vol += candles[i].volume;
        if i >= 20 {
            sum_vol -= candles[i-20].volume;
            vol_mas[i] = sum_vol / 20.0;
        }
    }

    let vol_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0];
    let body_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9];

    for i in 0..n {
        // EMA 200
        if i < ema_200.len() && ema_200[i] > 0.0 {
            if candles[i].close > ema_200[i] { bs.price_above_ema200.set(i, true); }
            if candles[i].close < ema_200[i] { bs.price_below_ema200.set(i, true); }
        }
        
        // ADX
        if i < adx.len() && adx[i] > 20.0 {
            bs.adx_strong.set(i, true);
        }

        // Volume Spikes
        if vol_mas[i] > 0.0 {
            for (idx, &t) in vol_thresholds.iter().enumerate() {
                if candles[i].volume > vol_mas[i] * t {
                    bs.vol_spikes[idx].set(i, true);
                }
            }
        }

        // Body Ratio
        let range = (candles[i].high - candles[i].low).max(1e-9);
        let body = (candles[i].close - candles[i].open).abs();
        let ratio = body / range;
        let is_long = candles[i].close > candles[i].open;
        
        for (idx, &t) in body_thresholds.iter().enumerate() {
            if ratio > t {
                if is_long {
                    bs.full_bodies_long[idx].set(i, true);
                } else {
                    bs.full_bodies_short[idx].set(i, true);
                }
            }
        }
    }
    bs
}
