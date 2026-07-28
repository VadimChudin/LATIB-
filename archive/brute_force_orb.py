"""
Extreme Brute Force Optimizer - ORB Strategy
============================================
Grid Search for Opening Range Breakout.
"""
import os
import sys
import pandas as pd
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategies.orb_strategy import ORBStrategy

GLOBAL_DF = None

def init_worker(df_data):
    global GLOBAL_DF
    # For ORB, we likely need 15m data, but let's pass what we have and let the strategy handle it 
    # OR we resample it here if needed. The ORB strategy expects regular OHLCV. 
    # Specifically, it expects 'timestamp' to be accessible.
    GLOBAL_DF = df_data

def evaluate_params_orb(params):
    strat = ORBStrategy()
    
    res_df = strat.backtest_logic(GLOBAL_DF.copy(), params)
    
    # Filter out empty or unfinished trades
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
    cache_file = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'BTC_USDT_15m_730d.csv')
    
    # If 15m doesn't exist, fallback to 5m. ORB works best on 15m, but the strategy logic
    # just counts `opening_bars`. If we use 5m, 4 bars = 20 mins. If 15m, 4 bars = 1 hour.
    if not os.path.exists(cache_file):
        cache_file = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'BTC_USDT_5m_730d.csv')

    print(f"Loading market data from {cache_file}...")
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(cache_file, dtype=dtypes, engine='c', low_memory=False)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # ORB strategy relies on iterating through df_sim, so we don't necessarily need timestamp as index,
    # but let's keep it consistent.
    
    # Use last 6 months (approx data limit to keep it fast)
    # 6 months of 15m = ~17500 candles. 6 months of 5m = ~52500 candles.
    df = df.tail(50000) 

    grid_orb = {
        'opening_bars': [2, 4, 6, 8, 12],
        'volume_mult': [1.0, 1.2, 1.5, 2.0],
        'tp_mult': [1.0, 1.5, 2.0, 2.5, 3.0]
    }
    
    keys, values = zip(*grid_orb.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Starting EXTREME Optimization for ORB ---")
    print(f"Total combinations to test: {len(combinations)}")
    
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=min(8, max(1, os.cpu_count() - 2)), initializer=init_worker, initargs=(df,)) as executor:
        futures = {executor.submit(evaluate_params_orb, params): params for params in combinations}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res[3] >= 30: # At least 30 trades to be statistically somewhat relevant
                results.append(res)
            if completed % 20 == 0:
                print(f"Progress: {completed}/{len(combinations)}...")
                
    end_time = time.time()
    print(f"ORB Optimization complete in {end_time - start_time:.1f} seconds.")
    
    results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    print("\n--- TOP 5 ORB PARAMETERS ---")
    for i, (params, wr, pf, trades) in enumerate(results[:5]):
        print(f"#{i+1} | WR: {wr:.2%} | PF: {pf:.2f} | Trades: {trades} | Params: {params}")

if __name__ == '__main__':
    main()
