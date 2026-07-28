"""
Quick Diagnostic Script — runs a fast backtest on CACHED data only.
Loads the 730d BTC cache from disk and tests ONE strategy's parameters 
to confirm we get actual trades. No exchange connection needed.

Run: python diagnose.py
"""
import asyncio
import os
import sys
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    # Load cached data directly
    cache_file = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'BTC_USDT_5m_730d.csv')
    if not os.path.exists(cache_file):
        logger.error(f"Cache not found: {cache_file}. Run run_test.py first to download data.")
        return

    logger.info(f"Loading cache: {cache_file}")
    dtypes = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
    df = pd.read_csv(cache_file, dtype=dtypes, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"Loaded {len(df)} rows.")

    # ── Test 1: SwingICT (baseline) ──────────────────────────────────────────
    logger.info("\n=== Testing SwingICT Strategy (baseline) ===")
    from strategies.swing_ict import SwingICTStrategy
    swing = SwingICTStrategy()
    params = {'ema_fast': 13, 'ema_slow': 34, 'tp_rr': 0.6, 'sl_atr_mult': 1.0}
    result = swing.backtest_logic(df.copy(), params)
    trades = result[result['trade_pnl_r'] != 0]['trade_pnl_r']
    days   = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days or 1
    logger.info(f"Swing: {len(trades)} trades ({len(trades)/days:.1f}/day), WR={( trades>0).mean():.1%}")

    # ── Test 2: SwingICT + Kill Zones ────────────────────────────────────────
    logger.info("\n=== Testing SwingICT + Kill Zones ===")
    df_ts = df.copy()
    df_ts.index = df_ts['timestamp']
    from strategies.swing_ict_kz import SwingICTKZStrategy
    swing_kz = SwingICTKZStrategy()
    params_kz = {'ema_fast': 13, 'ema_slow': 34, 'tp_rr': 0.6, 'sl_atr_mult': 1.0}
    result_kz = swing_kz.backtest_logic(df_ts, params_kz)
    trades_kz = result_kz[result_kz['trade_pnl_r'] != 0]['trade_pnl_r']
    logger.info(f"SwingKZ: {len(trades_kz)} trades ({len(trades_kz)/days:.1f}/day), WR={(trades_kz>0).mean():.1%}")


    logger.info("\n=== Testing Ultimate SMC Strategy (Parameter Sweep) ===")
    from strategies.ultimate_smc import UltimateSMCStrategy
    smc = UltimateSMCStrategy()
    
    best_results = []
    for swing_len in [5, 10]:
        for fvg_atr in [0.3, 0.5]:
            for ob_score in [2, 3, 4]:
                for tp in [0.6, 0.8, 1.0, 1.2]:
                    for sl_mult in [0.5, 1.0]:
                        p = {'swing_length': swing_len, 'fvg_min_atr': fvg_atr,
                             'ob_min_score': ob_score, 'tp_rr': tp, 'sl_atr_mult': sl_mult}
                        try:
                            res = smc.backtest_logic(df.copy(), p)
                            trades = res[res['trade_pnl_r'] != 0]['trade_pnl_r']
                            if len(trades) >= 50:
                                wr = float((trades > 0).mean())
                                gw = trades[trades > 0].sum()
                                gl = abs(trades[trades < 0].sum()) + 1e-9
                                pf = float(gw / gl)
                                n  = len(trades)
                                metric = (wr * pf) / max(0.001, 0.1)  # rough metric
                                best_results.append((wr, pf, n, p))
                        except Exception as e:
                            pass
    
    best_results.sort(key=lambda x: x[0] * x[1], reverse=True)
    logger.info(f"\n{'WR':>7} {'PF':>6} {'Trades':>7}  Params")
    for wr, pf, n, p in best_results[:10]:
        logger.info(f"{wr:>7.1%} {pf:>6.2f} {n:>7}  swing={p['swing_length']} fvg={p['fvg_min_atr']} score≥{p['ob_min_score']} tp={p['tp_rr']} sl={p['sl_atr_mult']}")
    
    logger.info("\n=== DIAGNOSIS COMPLETE ===")

if __name__ == '__main__':
    main()
