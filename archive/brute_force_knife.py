"""
Extreme Brute Force Optimizer - Knife Catcher Strategy
======================================================
Grid Search for Mean Reversion (Liquidity Walls & Deviations).
"""
import os
import sys
import pandas as pd
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategies.knife_catcher import KnifeCatcherStrategy

GLOBAL_DF = None

def init_worker(df_data):
    global GLOBAL_DF
    # Needs basic OHLCV, the strategy calculates TA internally
    GLOBAL_DF = df_data

def evaluate_params_knife(params):
    strat = KnifeCatcherStrategy()
    
    # Needs to copy data as backtest_logic adds TA columns
    res_df = strat.backtest_logic(GLOBAL_DF.copy(), params)
    
    # Extract trades
    trades = res_df[res_df['trade_pnl_r'] != 0]['trade_pnl_r']
    num_trades = len(trades)
    
    if num_trades < 5: # Lowered from 30 because extreme panics are rare
        return (params, 0.0, 0.0, num_trades)
        
    wins = len(trades[trades > 0])
    win_rate = wins / num_trades
    
    gross_profit = trades[trades > 0].sum()
    gross_loss = abs(trades[trades < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return (params, win_rate, profit_factor, num_trades)

def main():
    cache_file = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'BTC_USDT_5m_730d.csv')

    print(f"Loading market data from {cache_file}...")
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(cache_file, dtype=dtypes, engine='c', low_memory=False)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Pre-calculating TA features to save RAM during multiprocessing...")
    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.atr(length=14, append=True)
    for std in [2.0, 2.5, 3.0]:
        df.ta.bbands(length=20, std=std, append=True)
    
    # 200,000 candles = ~2 years of 5m data.
    # We will test the deepest possible panics over the longest possible history.
    df = df.tail(200000) 

    grid_knife = {
        'rsi_oversold': [20, 25, 30],
        'bb_std': [2.0, 2.5, 3.0], # Reduced from 4.0 so we actually get trades
        'vol_spike_mult': [1.5, 2.0, 2.5], # Reduced from 4.0x
        'tp_rr': [1.0, 1.5, 2.0]
    }
    
    keys, values = zip(*grid_knife.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Starting EXTREME Optimization for Knife Catcher ---")
    print(f"Total combinations to test: {len(combinations)}")
    
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker, initargs=(df,)) as executor:
        futures = {executor.submit(evaluate_params_knife, params): params for params in combinations}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res[3] >= 5: # At least 5 knife catches (extremely rare events)
                results.append(res)
            if completed % 20 == 0:
                print(f"Progress: {completed}/{len(combinations)}...")
                
    end_time = time.time()
    print(f"Knife Catcher Optimization complete in {end_time - start_time:.1f} seconds.")
    
    results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    print("\n--- TOP 5 KNIFE CATCHER PARAMETERS (BASE EDGE) ---")
    for i, (params, wr, pf, trades) in enumerate(results[:5]):
        print(f"#{i+1} | WR: {wr:.2%} | PF: {pf:.2f} | Trades: {trades} | Params: {params}")

if __name__ == '__main__':
    main()
