"""
Brute Force Optimizer for Swing ICT Baseline
============================================
Runs a Grid Search across hundreds of parameter combinations to find
the most naturally profitable baseline for the Swing ICT Strategy,
before applying any ML.

Run: python brute_force.py
"""
import os
import sys
import pandas as pd
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategies.swing_ict_trail import SwingICTTrailStrategy

def evaluate_params(args):
    """Evaluates a single parameter combination."""
    df, params = args
    strat = SwingICTTrailStrategy()
    
    # Run backtest
    res_df = strat.backtest_logic(df.copy(), params)
    
    # Extract trades
    trades = res_df[res_df['trade_pnl_r'] != 0]['trade_pnl_r']
    num_trades = len(trades)
    
    if num_trades < 50:
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
    
    # Set index to timestamp to ensure strategy can read hours for kill zones
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    
    # We only need the last 6 months for a fast optimization that is relevant to current market
    df = df.tail(50000) 

    # Define Parameter Grid
    grid = {
        'ema_fast': [8, 13, 21],
        'ema_slow': [34, 50, 89, 144],
        'sl_atr_mult': [0.5, 1.0, 1.5],
        'trail_activate_r': [1.0, 1.5, 2.0],
        'trail_atr_mult': [0.3, 0.5, 0.8]
    }
    
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(combinations)}")
    
    args_list = [(df, params) for params in combinations]
    results = []
    
    print("Starting Grid Search using all CPU cores...")
    start_time = time.time()
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_params, args): args for args in args_list}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res[3] >= 50: # Only keep results with >50 trades
                results.append(res)
                
            if completed % 50 == 0:
                print(f"Progress: {completed}/{len(combinations)}...")
                
    end_time = time.time()
    print(f"\nOptimization complete in {end_time - start_time:.1f} seconds.")
    
    if not results:
        print("No combinations met the minimum trade count criteria.")
        return
        
    # Sort by Win Rate (primary) and Profit Factor (secondary)
    results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    print("\n--- TOP 10 PARAMETER COMBINATIONS ---")
    for i, (params, wr, pf, trades) in enumerate(results[:10]):
        print(f"#{i+1} | WR: {wr:.2%} | PF: {pf:.2f} | Trades: {trades}")
        print(f"      Params: {params}")

if __name__ == '__main__':
    main()
