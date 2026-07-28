"""
AEGIS 2.0 — Shared Constants
==============================
Single source of truth for paths, strategy mappings, and config values
used across the entire Python pipeline.
"""
import os
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = BASE_DIR / "data" / "cache"
TICK_CACHE_DIR = CACHE_DIR / "ticks"
GA_RESULTS_DIR = BASE_DIR / "data" / "ga_results"
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_JSON_DIR = BASE_DIR / "data" / "models_json"

ACTIVE_CONFIG_PATH = BASE_DIR / "data" / "active_config.json"
AGGREGATED_GA_PATH = BASE_DIR / "data" / "ga_aggregated_results.json"
TOP_SYMBOLS_PATH = BASE_DIR / "data" / "top_symbols.json"
PIPELINE_CHECKPOINT_PATH = BASE_DIR / "data" / "pipeline_checkpoint.json"
RETRAIN_FLAG_PATH = MODELS_DIR / "retrain_flag.txt"

# Rust binary
BINARY_PATH = BASE_DIR / "rust_engine" / "target" / "release" / "aegis_engine.exe"
if not BINARY_PATH.exists():
    BINARY_PATH = BINARY_PATH.with_suffix("")  # Linux/Mac fallback

# ── Strategy Mappings ──
# Rust strategy name → Python strategy name
STRAT_MAP = {
    "smc": "Ultimate_SMC_Trail",
    "knife": "KnifeCatcher_ML",
    "scalpmtf": "ScalpMTF",
    "fundingrate": "FundingRate_MR",
    "density": "Density",
}

# Python strategy name → Rust strategy name
REVERSE_STRAT_MAP = {v: k for k, v in STRAT_MAP.items()}

# ── GA Parameters ──
PARAM_NAMES = {
    "smc": ["swing_length", "fvg_min_atr", "ob_min_score", "sl_atr_mult", "trail_activate_r", "trail_atr_mult"],
    "knife": ["score_threshold", "price_vol_weight", "flow_weight", "tech_weight", "pattern_weight", "lookback_bars", "cum_delta_bars", "min_red_candles", "tp_rr", "sl_atr_mult"],
    "scalpmtf": ["fast_ema", "slow_ema", "rsi_thresh", "tp_rr"],
    "fundingrate": ["fr_long_thresh", "fr_short_thresh", "sl_atr_mult", "trail_activate_r", "trail_atr_mult", "cooldown_bars"],
    "density": ["vol_spike_mult", "min_touches", "shakeout_pct", "tp_rr", "sl_atr_mult"],
}
COMMON_PARAMS = ["cooldown_bars", "max_trades_day", "max_drawdown_r"]

# ── Pipeline Config ──
TOP_N_CONFIGS = 15
TOP_PER_STRAT = 25
MAX_DATA_AGE_HOURS = 72
CHECKPOINT_MAX_AGE_HOURS = 48
