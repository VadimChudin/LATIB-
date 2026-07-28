"""
ML Training Script — Ultimate SMC (Multi-Symbol)
=================================================
Pools 730-day 5m data from 5 symbols, runs backtest on each,
then trains one Triple-AI ensemble on all combined trades.

Run: python train_ml_smc.py
"""
import os
import sys
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ml_filter import RegimeMLFilter
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy

import json
with open("data/top_symbols.json") as f:
    SYMBOLS = json.load(f)[:50]
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')

import json

def get_best_params():
    default_params = {
        'swing_length': 3, 'fvg_min_atr': 0.3, 'ob_min_score': 4,
        'sl_atr_mult': 1.0, 'trail_activate_r': 0.8, 'trail_atr_mult': 0.2
    }
    config_path = os.path.join(os.path.dirname(__file__), 'data', 'active_config.json')
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs = json.load(f)
            for c in configs:
                if c.get("strategy") == "Ultimate_SMC_Trail":
                    logger.info("  🔧 Loaded GA dynamic params from active_config.json")
                    p = c.get("params", default_params)
                    for k in ['swing_length', 'ob_min_score', 'cooldown_bars', 'max_trades_day']:
                        if k in p: p[k] = int(p[k])
                    return p
    except Exception as e:
        logger.warning(f"  ⚠️ Could not load GA params: {e}")
    
    logger.info("  🔧 Using default fallback params")
    return default_params

PARAMS = get_best_params()


def load_symbol(symbol: str) -> pd.DataFrame:
    """Load cached CSV for a symbol."""
    path = os.path.join(CACHE_DIR, f'{symbol}_5m_730d.csv')
    if not os.path.exists(path):
        logger.warning(f"  ⚠️ {symbol}: cache not found at {path}")
        return None
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(path, dtype=dtypes, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    logger.info(f"  ✅ {symbol}: loaded {len(df)} candles")
    return df


def main():
    logger.info("=" * 60)
    logger.info("  ML TRAINING: Ultimate SMC Trail (Multi-Symbol)")
    logger.info("=" * 60)

    # Pre-load BTC for Gravity Correlation
    btc_df = load_symbol("BTC_USDT")
    if btc_df is None:
        logger.warning("  ⚠️ BTC_USDT missing. BTC Gravity will be disabled for this training.")

    strat = UltimateSMCTrailStrategy()
    all_features = []
    all_labels = []

    for symbol in SYMBOLS:
        logger.info(f"\n🔄 Processing {symbol}...")
        df = load_symbol(symbol)
        if df is None:
            continue

        # Run backtest
        try:
            result = strat.backtest_logic(df.copy(), PARAMS)
        except Exception as e:
            logger.error(f"  ❌ Backtest failed for {symbol}: {e}")
            continue

        trades_df = result[result['trade_pnl_r'] != 0].copy()
        if len(trades_df) < 10:
            logger.warning(f"  ⚠️ {symbol}: only {len(trades_df)} trades, skipping")
            continue

        trade_indices = trades_df['entry_idx'].tolist()
        labels = (trades_df['trade_pnl_r'] > 0).astype(int).values
        wr = sum(labels) / len(labels)
        logger.info(f"  📊 {symbol}: {len(trades_df)} trades, WR={wr:.1%}")

        # Compute features
        ml_filter = RegimeMLFilter(model_name="ultimate_smc_trail_model")
        features_df = ml_filter.prepare_features(df, trade_indices, btc_df=btc_df)

        if len(features_df) != len(labels):
            valid_indices = features_df['index'].values
            valid_mask = [idx in valid_indices for idx in trade_indices]
            labels = labels[valid_mask]

        all_features.append(features_df)
        all_labels.append(pd.Series(labels))
        
        # Prevent ArrayMemoryError by aggressively freeing RAM after each symbol
        del df
        del trades_df
        del result
        del features_df
        import gc; gc.collect()

    if not all_features:
        logger.error("\n❌ No valid data from any symbol!")
        return

    # Pool all data
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = pd.concat(all_labels, ignore_index=True)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  POOLED DATASET: {len(combined_features)} trades from {len(all_features)} symbols")
    logger.info(f"  Overall WR: {combined_labels.mean():.1%}")
    logger.info(f"{'=' * 60}")

    # Train
    logger.info("\n3. Training Triple-AI Ensemble (XGB+LGBM+RF)...")
    ml_filter = RegimeMLFilter(model_name="ultimate_smc_trail_model")
    metrics = ml_filter.train(combined_features, combined_labels)

    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info(f"Model saved to: {ml_filter.model_path}")


if __name__ == '__main__':
    main()
