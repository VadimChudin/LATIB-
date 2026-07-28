"""
Extreme Brute Force Optimizer - Ultimate SMC
============================================
Grid Search for Ultimate SMC.
"""
import os
import sys
import pandas as pd
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy

GLOBAL_DF = None

def init_worker(df_data):
    global GLOBAL_DF
    GLOBAL_DF = df_data

def evaluate_params_smc(params):
    strat = UltimateSMCTrailStrategy()
    
    res_df = strat.backtest_logic(GLOBAL_DF.copy(), params)
    trades = res_df[res_df['trade_pnl_r'] != 0]['trade_pnl_r']
    num_trades = len(trades)
    
    if num_trades < 30:
        return (params, 0.0, 0.0, num_trades)
        
    wins = len(trades[trades > 0])
    win_rate = wins / num_trades
    
    gross_profit = trades[trades > 0].sum()
    gross_loss = abs(trades[trades < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return (params, win_rate, profit_factor, num_trades)

def main():
    cache_file = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'BTC_USDT_5m_730d.csv')
    if not os.path.exists(cache_file):
        print(f"Cache missing: {cache_file}")
        return

    print("Loading market data...")
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(cache_file, dtype=dtypes, engine='c', low_memory=False)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    
    # Use last 6 months (approx 50,000 5m candles)
    df = df.tail(50000) 

    # --- ULTIMATE SMC MASSIVE GRID ---
    grid_smc = {
        'swing_length': [3, 5, 8],
        'fvg_min_atr': [0.2, 0.3, 0.5],
        'ob_min_score': [2, 3, 4],
        'sl_atr_mult': [0.5, 0.75, 1.0, 1.25],
        'trail_activate_r': [0.8, 1.0, 1.25, 1.5],
        'trail_atr_mult': [0.2, 0.3, 0.5, 0.8]
    }
    
    keys, values = zip(*grid_smc.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Starting EXTREME Optimization for ULTIMATE SMC ---")
    print(f"Total combinations to test: {len(combinations)}")
    
    results_smc = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=min(8, max(1, os.cpu_count() - 2)), initializer=init_worker, initargs=(df,)) as executor:
        futures = {executor.submit(evaluate_params_smc, params): params for params in combinations}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res[3] >= 50:
                results_smc.append(res)
            if completed % 100 == 0:
                print(f"Progress: {completed}/{len(combinations)}...")
                
    end_time = time.time()
    print(f"Ultimate SMC Optimization complete in {end_time - start_time:.1f} seconds.")
    
    results_smc.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    print("\n--- TOP 5 ULTIMATE SMC PARAMETERS ---")
    for i, (params, wr, pf, trades) in enumerate(results_smc[:5]):
        print(f"#{i+1} | WR: {wr:.2%} | PF: {pf:.2f} | Trades: {trades} | Params: {params}")

if __name__ == '__main__':
    main()
