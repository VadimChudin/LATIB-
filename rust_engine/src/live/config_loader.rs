///! Config Loader — reads active_config.json into Rust structs
///! ==========================================================
///! Parses the Python-generated configuration to know which
///! strategies and symbols are approved for live trading.

use serde::Deserialize;
use std::path::Path;
use tracing::{info, warn};

/// A single strategy-symbol configuration from active_config.json
#[derive(Debug, Clone, Deserialize)]
pub struct LiveConfig {
    pub symbol: String,
    pub timeframe: String,
    pub strategy: String,
    // Tier-based leverage system
    #[serde(default)]
    pub tier: u8,
    #[serde(default = "default_leverage")]
    pub leverage: u32,
    // Phase 21: Dual-Mode support
    pub conservative: Option<ModeConfig>,
    pub aggressive: Option<ModeConfig>,
    // Fallback for legacy
    pub params: Option<serde_json::Value>,
    #[serde(default)]
    pub metrics: ConfigMetrics,
}

fn default_leverage() -> u32 { 10 }

#[derive(Debug, Clone, Deserialize)]
pub struct ModeConfig {
    pub params: serde_json::Value,
    pub metrics: ConfigMetrics,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ConfigMetrics {
    #[serde(default)]
    pub win_rate: f64,
    #[serde(default)]
    pub total_trades: usize,
    #[serde(default)]
    pub score: f64,
}

impl LiveConfig {
    /// Map Python strategy name to Rust strategy identifier
    pub fn rust_strategy_name(&self) -> &str {
        match self.strategy.as_str() {
            "Ultimate_SMC_Trail" => "smc",
            "KnifeCatcher_ML" => "knife",
            "knife_tick" => "knifetick",
            "ScalpMTF" => "scalpmtf",
            "FundingRate_MR" => "fundingrate",
            "Density" => "density",
            _ => "smc",
        }
    }

    /// Map strategy to ML model name
    pub fn model_name(&self) -> &str {
        match self.strategy.as_str() {
            "Ultimate_SMC_Trail" => "ultimate_smc_trail",
            "KnifeCatcher_ML" => "knife_catcher",
            "knife_tick" => "knife_catcher",
            "ScalpMTF" => "scalpmtf_model",
            "FundingRate_MR" => "funding_rate_model",
            "Density" => "density_model",
            _ => "ultimate_smc_trail",
        }
    }

    /// Extract strategy parameters as f64 vector (same order as GA optimizer)
    pub fn params_vec(&self, mode: &str) -> Vec<f64> {
        let p = if mode == "aggressive" {
            if let Some(am) = &self.aggressive { &am.params } else { &self.params.as_ref().unwrap() }
        } else {
            if let Some(cm) = &self.conservative { &cm.params } else { &self.params.as_ref().unwrap() }
        };
        match self.rust_strategy_name() {
            "smc" => vec![
                p["ema_fast"].as_f64().unwrap_or(8.0),
                p["ema_slow"].as_f64().unwrap_or(21.0),
                p["sl_atr_mult"].as_f64().unwrap_or(1.5),
                p["trail_activate_r"].as_f64().unwrap_or(1.5),
                p["trail_atr_mult"].as_f64().unwrap_or(0.5),
                p["tp_rr"].as_f64().unwrap_or(3.0),
            ],
            "scalpmtf" => vec![
                p["fast_ema"].as_f64().unwrap_or(9.0),
                p["slow_ema"].as_f64().unwrap_or(50.0),
                p["rsi_thresh"].as_f64().unwrap_or(30.0),
                p["tp_rr"].as_f64().unwrap_or(1.0),
            ],
            "knifetick" => vec![
                p["window_ms"].as_f64().unwrap_or(2000.0),
                p["min_zscore"].as_f64().unwrap_or(2.5),
                p["min_vol_spike"].as_f64().unwrap_or(2.0),
                p["(unused_tp)"].as_f64().unwrap_or(0.0),
                p["sl_buffer_pct"].as_f64().unwrap_or(0.002),
                p["be_trigger_pct"].as_f64().unwrap_or(0.005),
                p["trail_pct"].as_f64().unwrap_or(0.004),
                p["micro_window_ms"].as_f64().unwrap_or(1000.0),
                p["min_absorption"].as_f64().unwrap_or(2.0),
                p["min_reclaim_pct"].as_f64().unwrap_or(0.001),
                p["max_speed_mult"].as_f64().unwrap_or(3.0),
                p["baseline_window_sec"].as_f64().unwrap_or(30.0),
                p["max_absorber_sec"].as_f64().unwrap_or(30.0),
                p["rewake_cooldown_sec"].as_f64().unwrap_or(60.0),
            ],
            "knife" => vec![
                p["drop_pct"].as_f64().unwrap_or(5.0),
                p["rsi_threshold"].as_f64().unwrap_or(25.0),
                p["vol_surge"].as_f64().unwrap_or(2.0),
                p["recovery_bars"].as_f64().unwrap_or(3.0),
                p["tp_rr"].as_f64().unwrap_or(2.0),
            ],
            "fundingrate" => vec![
                p["fr_long_thresh"].as_f64().unwrap_or(0.03),
                p["fr_short_thresh"].as_f64().unwrap_or(0.05),
                p["sl_atr_mult"].as_f64().unwrap_or(1.5),
                p["trail_activate_r"].as_f64().unwrap_or(1.0),
                p["trail_atr_mult"].as_f64().unwrap_or(0.5),
                p["cooldown_bars"].as_f64().unwrap_or(6.0),
            ],
            "density" => vec![
                p["vol_spike_mult"].as_f64().unwrap_or(2.5),
                p["min_touches"].as_f64().unwrap_or(2.0),
                p["shakeout_pct"].as_f64().unwrap_or(0.006),
                p["tp_rr"].as_f64().unwrap_or(2.0),
                p["sl_atr_mult"].as_f64().unwrap_or(1.0),
            ],
            _ => vec![],
        }
    }
}

/// Load all configs from active_config.json
pub fn load_active_configs(path: &Path) -> Vec<LiveConfig> {
    match std::fs::read_to_string(path) {
        Ok(data) => {
            match serde_json::from_str::<Vec<LiveConfig>>(&data) {
                Ok(mut configs) => {
                    info!("📋 Loaded {} active configs from {}", configs.len(), path.display());
                    for c in &mut configs {
                        // Normalize symbol to Binance WS format immediately (e.g. BTC_USDT -> BTC/USDT)
                        c.symbol = c.symbol.replace("_", "/");
                        info!("   {} | {} | {} | T{} {}x (score={:.1})",
                            c.symbol, c.strategy, c.timeframe, c.tier, c.leverage, c.metrics.score);
                    }
                    configs
                }
                Err(e) => {
                    warn!("⚠️ Failed to parse active_config.json: {}", e);
                    vec![]
                }
            }
        }
        Err(e) => {
            warn!("⚠️ Could not read {}: {}", path.display(), e);
            vec![]
        }
    }
}

/// Get unique symbols from configs (for WebSocket subscription) 
pub fn get_active_symbols(configs: &[LiveConfig]) -> Vec<String> {
    let mut symbols: Vec<String> = configs.iter()
        .map(|c| c.symbol.clone())
        .collect();
    symbols.sort();
    symbols.dedup();
    symbols
}
