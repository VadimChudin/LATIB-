"""
ML Training Script
==================
Runs a historical backtest to collect trade decisions, computes
features for every trade, and trains the Regime-Adaptive Random Forest.

Run: python train_ml.py
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
import pandas_ta as ta

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ml_filter import RegimeMLFilter
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy

def main():
    # ── Load history ──────────────────────────────────────────────────────────
    cache_file = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'BTC_USDT_5m_730d.csv')
    if not os.path.exists(cache_file):
        logger.error(f"Cache missing: {cache_file}. Run diagnose.py first.")
        return

    logger.info("Loading market data...")
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(cache_file, dtype=dtypes, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    
    # ── Run Backtest to get trades ───────────────────────────────────────────
    logger.info("\n1. Running Backtest to generate training labels...")
    strat = UltimateSMCTrailStrategy()
    # Optimized params from extreme brute_force.py Grid Search (#5 combination provides best PF/WR ratio)
    params = {'ema_fast': 5, 'ema_slow': 89, 'sl_atr_mult': 0.75, 
              'trail_activate_r': 0.8, 'trail_atr_mult': 0.3}
              
    r_trail = strat.backtest_logic(df.copy(), params)
    
    # Extract trades (non-zero pnl_r)
    trades_df = r_trail[r_trail['trade_pnl_r'] != 0].copy()
    # The crucial fix: use the exact index where the trade OPENED
    trade_indices = trades_df['entry_idx'].tolist()
    
    logger.info(f"   Found {len(trades_df)} total trades.")
    
    # Define success: for trailing stop, a win is any trade that captured > 0.0R
    # We want the ML to predict if a trade will be profitable AT ALL.
    labels = (trades_df['trade_pnl_r'] > 0).astype(int).values
    logger.info(f"   Labels: {sum(labels)} wins ({sum(labels)/len(labels):.1%}), {len(labels)-sum(labels)} losses.")
    
    # ── Compute ML Features ──────────────────────────────────────────────────
    logger.info("\n2. Computing ML Features for all trades (This may take a minute)...")
    ml_filter = RegimeMLFilter(model_name="swing_ict_kz_model")
    
    features_df = ml_filter.prepare_features(df, trade_indices)
    
    if len(features_df) != len(labels):
        # Align them (prepare_features skips early indices < 100)
        valid_indices = features_df['index'].values
        valid_mask = [idx in valid_indices for idx in trade_indices]
        labels = labels[valid_mask]
        
    logger.info(f"   Generated {len(features_df)} feature rows.")
    
    # ── Train Model ──────────────────────────────────────────────────────────
    logger.info("\n3. Training Regime-Adaptive Random Forest...")
    metrics = ml_filter.train(features_df, pd.Series(labels))
    
    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info(f"Model saved to: {ml_filter.model_path}")

if __name__ == '__main__':
    main()
