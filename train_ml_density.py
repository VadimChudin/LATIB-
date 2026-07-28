"""
ML Training Script — Density Breakout
====================================
Pools 730-day 5m data and trains Triple-AI ensemble for Density strategy.
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ml_filter import RegimeMLFilter
from strategies.density import DensityStrategy

with open("data/top_symbols.json") as f:
    SYMBOLS = json.load(f)[:50]
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')

def get_best_params():
    default_params = {
        "vol_spike_mult": 2.5, "min_touches": 3, "shakeout_pct": 0.006,
        "tp_rr": 2.0, "sl_atr_mult": 1.0
    }
    config_path = "data/active_config.json"
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs = json.load(f)
            for c in configs:
                if c.get("strategy") == "Density":
                    return c.get("params", default_params)
    except: pass
    return default_params

PARAMS = get_best_params()

def load_symbol(symbol: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f'{symbol}_5m_730d.csv')
    if not os.path.exists(path): return None
    df = pd.read_csv(path, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    return df

def main():
    logger.info("  ML TRAINING: Density Breakout")
    btc_df = load_symbol("BTC_USDT")
    strat = DensityStrategy()
    all_features, all_labels = [], []

    for symbol in SYMBOLS:
        df = load_symbol(symbol)
        if df is None: continue
        
        result = strat.backtest_logic(df.copy(), PARAMS)
        trades_df = result[result['trade_pnl_r'] != 0].copy()
        if len(trades_df) < 5: continue

        trade_indices = trades_df['entry_idx'].tolist()
        labels = (trades_df['trade_pnl_r'] > 0).astype(int).values
        
        ml_filter = RegimeMLFilter(model_name="density_model")
        features_df = ml_filter.prepare_features(df, trade_indices, btc_df=btc_df)
        
        if len(features_df) > 0:
            all_features.append(features_df)
            all_labels.append(pd.Series(labels[:len(features_df)]))
        
        import gc; gc.collect()

    if not all_features:
        logger.error("❌ No trades found for Density training.")
        return

    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = pd.concat(all_labels, ignore_index=True)
    
    ml_filter = RegimeMLFilter(model_name="density_model")
    ml_filter.train(combined_features, combined_labels)
    logger.info("✅ Density ML Model Saved.")

if __name__ == '__main__':
    main()
