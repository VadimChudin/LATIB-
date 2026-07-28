"""
ML Filter Diagnostic — Ultimate SMC
===================================
Applies the trained Regime-Adaptive Random Forest model to the
Ultimate SMC Trailing backtest history to evaluate performance.

Run: python diagnose_ml_smc.py
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ml_filter import RegimeMLFilter
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy
from diagnose_trailing import simulate_compound, wr_needed_for_target

def main():
    cache_file = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'BTC_USDT_5m_730d.csv')
    if not os.path.exists(cache_file):
        print("Cache missing.")
        return

    print("Loading data...")
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(cache_file, dtype=dtypes, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days or 1

    # ── 1. Run Baseline Trailing Backtest (SMC) ─────────────────────────────
    print("\n1. Running Ultimate SMC Trailing backtest...")
    strat = UltimateSMCTrailStrategy()
    params = {
        'swing_length': 5,
        'fvg_min_atr': 0.3,
        'ob_min_score': 3,
        'sl_atr_mult': 1.0,
        'trail_activate_r': 1.0,
        'trail_atr_mult': 0.5
    }
    r_trail = strat.backtest_logic(df.copy(), params)
    
    trades_df = r_trail[r_trail['trade_pnl_r'] != 0].copy()
    trade_indices = [df.index.get_loc(ts) for ts in trades_df.index]
    
    t_base = trades_df['trade_pnl_r']
    wr_base = (t_base > 0).mean()
    avg_win_base = t_base[t_base > 0].mean()
    avg_loss_base = t_base[t_base < 0].mean()

    # ── 2. Apply ML Filter ──────────────────────────────────────────────────
    print("\n2. Applying Regime-Adaptive ML Filter (SMC Model)...")
    ml_filter = RegimeMLFilter(model_name="ultimate_smc_trail_model")
    if not ml_filter.is_fitted:
        print("Model not trained! Run train_ml_smc.py first.")
        return

    features_df = ml_filter.prepare_features(df, trade_indices)
    
    valid_indices = features_df['index'].values
    valid_mask = [idx in valid_indices for idx in trade_indices]
    trades_filtered = trades_df[valid_mask].copy()
    
    X = features_df.drop(columns=['index'])
    probs = ml_filter.clf.predict_proba(X)[:, 1]
    trades_filtered['ml_prob'] = probs
    
    # ── 3. Test Thresholds ──────────────────────────────────────────────────
    print("\n=== ML FILTER RESULTS (ULTIMATE SMC) ===")
    print(f"{'Threshold':>9} | {'Trades':>6} | {'Per Day':>7} | {'Win Rate':>8} | {'Avg Win':>7} | {'EV/Trade':>8}")
    print("-" * 65)
    
    print(f"{'0.00 (Base)':>9} | {len(trades_filtered):>6} | {len(trades_filtered)/days:>7.1f} | {wr_base:>7.1%}! | {avg_win_base:>6.2f}R | {wr_base*avg_win_base+(1-wr_base)*avg_loss_base:+.4f}R")
    
    best_results = None
    best_threshold = 0.50
    
    for thresh in [0.50, 0.55, 0.60, 0.63, 0.65, 0.70]:
        approved = trades_filtered[trades_filtered['ml_prob'] >= thresh]['trade_pnl_r']
        if len(approved) == 0: continue
            
        wr = (approved > 0).mean()
        avg_w = approved[approved > 0].mean() if wr > 0 else 0
        avg_l = approved[approved < 0].mean() if wr < 1 else -1.0
        ev = wr * avg_w + (1 - wr) * avg_l
        trades_day = len(approved) / days
        
        print(f"{thresh:>9.2f} | {len(approved):>6} | {trades_day:>7.1f} | {wr:>8.1%} | {avg_w:>6.2f}R | {ev:+.4f}R")
        
        if thresh == 0.63:
            best_results = approved
            best_threshold = thresh

    if best_results is None:
        return

    # ── 4. Full Analysis ────────────────────────────────────────────────────
    wr_best = (best_results > 0).mean()
    avg_win_best = best_results[best_results > 0].mean()
    tpd_best = int(len(best_results) / days)
    
    print("\n" + "=" * 60)
    print(f"  FULL SYSTEM CAPABILITY ADDITION (SMC Threshold >= {best_threshold})")
    print("=" * 60)
    print(f"  Adding {len(best_results)/days:.1f} trades/day at {wr_best:.1%} Win Rate")
    print(f"  Average profit per win: {avg_win_best:.2f}R")
    
if __name__ == '__main__':
    main()
