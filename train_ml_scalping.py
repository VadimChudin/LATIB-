"""
ML Training Script — Scalp MTF (Multi-Symbol)
=================================================
Pools 90-day 1m data from top symbols, runs backtest on each,
then trains one Triple-AI ensemble on all combined trades.

Run: python train_ml_scalping.py
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
from strategies.scalp_mtf import ScalpMTFStrategy
import ta

def discover_1m_symbols():
    """Auto-discover all symbols with 1m_730d data in cache."""
    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')
    symbols = []
    if os.path.isdir(cache_dir):
        for f in sorted(os.listdir(cache_dir)):
            if f.endswith('_1m_730d.csv'):
                # "BTC_USDT_1m_730d.csv" -> "BTC_USDT"
                sym = f.replace('_1m_730d.csv', '')
                symbols.append(sym)
    return symbols

SYMBOLS = discover_1m_symbols()
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')

def get_best_params():
    default_params = {
        'fast_ema': 9,
        'slow_ema': 50,
        'rsi_thresh': 30,
        'tp_rr': 1.0
    }
    config_path = os.path.join(os.path.dirname(__file__), 'data', 'active_config.json')
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs = json.load(f)
            for c in configs:
                if c.get("strategy") == "ScalpMTF":
                    logger.info("  🔧 Loaded GA dynamic params from active_config.json")
                    p = c.get("params", default_params)
                    return p
    except Exception as e:
        logger.warning(f"  ⚠️ Could not load GA params: {e}")
    
    logger.info("  🔧 Using default fallback params")
    return default_params

PARAMS = get_best_params()

def load_symbol(symbol: str) -> pd.DataFrame:
    # Notice we use 1m_730d for the Scalping ML Train
    path = os.path.join(CACHE_DIR, f'{symbol}_1m_730d.csv')
    if not os.path.exists(path):
        logger.warning(f"  ⚠️ {symbol}: cache not found at {path}")
        return None
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(path, dtype=dtypes, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_indexed = df.set_index('timestamp', drop=False)
    logger.info(f"  ✅ {symbol}: loaded {len(df)} candles")
    return df_indexed

def main():
    logger.info("=" * 60)
    logger.info("  ML TRAINING: Scalp MTF (1m Multi-Symbol)")
    logger.info("=" * 60)

    strat = ScalpMTFStrategy()
    all_features = []
    all_labels = []

    for symbol in SYMBOLS:
        logger.info(f"\n🔄 Processing {symbol}...")
        df = load_symbol(symbol)
        if df is None:
            continue

        try:
            result = strat.backtest_logic(df.copy(), PARAMS)
            f_df = strat.get_features(result)
        except Exception as e:
            logger.error(f"  ❌ Backtest failed for {symbol}: {e}")
            continue

        trades_df = result[result['trade_pnl_r'] != 0].copy()
        if len(trades_df) < 10:
            logger.warning(f"  ⚠️ {symbol}: only {len(trades_df)} trades, skipping")
            continue

        for idx in trades_df['entry_idx']:
            # Ensure we have enough history for the features calculations safely
            if pd.isna(idx) or idx < 200:
                continue
                
            idx_int = int(idx)
            current = result.iloc[idx_int]
            timestamp = result.index[idx_int]
            pnl_r = current['trade_pnl_r']
            
            features = f_df.iloc[idx_int].to_dict()
                
            # Time features
            hour = timestamp.hour
            features['kill_zone'] = 1 if (7 <= hour < 11) or (13 <= hour < 17) else 0

            all_features.append(features)
            all_labels.append(1 if pnl_r > 0 else 0)

        wr = (trades_df['trade_pnl_r'] > 0).mean() * 100
        logger.info(f"  📊 {symbol}: {len(trades_df)} trades, WR={wr:.1f}%")
        
        del df
        del trades_df
        del result
        del f_df
        import gc; gc.collect()

    if not all_features:
        logger.error("❌ No trades generated across all symbols. Abandoning ML training.")
        return

    logger.info("=" * 60)
    logger.info(f"  POOLED DATASET: {len(all_labels)} trades from {len(SYMBOLS)} symbols")
    overall_wr = sum(all_labels) / len(all_labels) * 100
    if len(all_labels) > 0:
        logger.info(f"  Overall Base WR: {overall_wr:.1f}%")
    logger.info("=" * 60)

    # 3. Train Triple-AI
    logger.info("\n3. Training Triple-AI Ensemble (XGB+LGBM+RF) for ScalpMTF...")
    X_df = pd.DataFrame(all_features)
    y_series = pd.Series(all_labels)

    ml_filter = RegimeMLFilter("scalpmtf_model")
    ml_filter.train(X_df, y_series)

    rf_model = ml_filter.models.get('rf')
    if rf_model and hasattr(rf_model, 'feature_importances_'):
        importances = list(zip(X_df.columns, rf_model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        top = importances[:3]
        logger.info(f"Top 3 ML Features (via RF sub-model): {top}")

    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info(f"Model saved to: {ml_filter.model_path}")

if __name__ == "__main__":
    main()
