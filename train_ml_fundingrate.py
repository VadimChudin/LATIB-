"""
ML Training Script — Funding Rate Mean Reversion
==============================================
Pools 730-day 5m data, runs backtest, and trains Triple-AI ensemble.
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
from strategies.funding_rate import FundingRateStrategy

with open("data/top_symbols.json") as f:
    SYMBOLS = json.load(f)[:50]
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')

def get_best_params():
    default_params = {
        "fr_long_thresh": 0.03, "fr_short_thresh": 0.05,
        "sl_atr_mult": 1.5, "tp_rr": 2.0
    }
    config_path = "data/active_config.json"
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs = json.load(f)
            for c in configs:
                if c.get("strategy") == "FundingRate_MR":
                    return c.get("params", default_params)
    except: pass
    return default_params

PARAMS = get_best_params()

def load_symbol(symbol: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f'{symbol}_5m_730d.csv')
    if not os.path.exists(path): return None
    df = pd.read_csv(path, engine='c', low_memory=False)
    # Ensure funding_rate exists (simulated if missing preserved from strategy logic)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    return df

def main():
    logger.info("  ML TRAINING: Funding Rate MR")
    btc_df = load_symbol("BTC_USDT")
    strat = FundingRateStrategy()
    all_features, all_labels = [], []

    for symbol in SYMBOLS:
        df = load_symbol(symbol)
        if df is None: continue
        
        result = strat.backtest_logic(df.copy(), PARAMS)
        trades_df = result[result['trade_pnl_r'] != 0].copy()
        if len(trades_df) < 5: continue

        trade_indices = trades_df['entry_idx'].tolist() if 'entry_idx' in trades_df.columns else trades_df.index.get_indexer(trades_df.index)
        labels = (trades_df['trade_pnl_r'] > 0).astype(int).values
        
        ml_filter = RegimeMLFilter(model_name="funding_rate_model")
        features_df = ml_filter.prepare_features(df, trade_indices, btc_df=btc_df)
        
        if len(features_df) > 0:
            all_features.append(features_df)
            all_labels.append(pd.Series(labels[:len(features_df)]))
        
        import gc; gc.collect()

    if not all_features:
        logger.error("❌ No trades found for FundingRate training.")
        return

    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = pd.concat(all_labels, ignore_index=True)
    
    ml_filter = RegimeMLFilter(model_name="funding_rate_model")
    ml_filter.train(combined_features, combined_labels)
    logger.info("✅ FundingRate ML Model Saved.")

if __name__ == '__main__':
    main()
