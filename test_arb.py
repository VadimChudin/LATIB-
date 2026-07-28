import os
import sys
import json
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.stat_arb import StatArbStrategy

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')
PAIRS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'arb_pairs.json')

def load_pairs() -> list:
    if os.path.exists(PAIRS_FILE):
        try:
            with open(PAIRS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('pairs', [])
        except Exception as e:
            logger.error(f"Failed to load arb pairs: {e}")
    return []

def load_symbol(symbol: str) -> pd.DataFrame:
    """Load cached CSV for a symbol."""
    sym_safe = symbol.replace("/", "_")
    path = os.path.join(CACHE_DIR, f'{sym_safe}_5m_730d.csv')
    if not os.path.exists(path):
        return None
        
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(path, dtype=dtypes, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', drop=False, inplace=True)
    return df

def backtest_pair(sym_a: str, sym_b: str, strat: StatArbStrategy, risk_usd: float = 1000.0):
    logger.info(f"\nEvaluating Pair: {sym_a} / {sym_b}")
    df_a = load_symbol(sym_a)
    df_b = load_symbol(sym_b)
    
    if df_a is None or df_b is None:
        logger.warning(f"  Skipping: Missing cached data for {sym_a} or {sym_b}")
        return None
        
    logger.info(f"  Loaded {len(df_a)} candles for {sym_a}, {len(df_b)} candles for {sym_b}")
    
    # Generate signals
    res = strat.generate_signals(df_a, df_b)
    
    # Safely merge close_B price matching timestamps without index ambiguity
    res = res.reset_index(drop=True)
    df_b_safe = df_b.copy().reset_index(drop=True)
    
    temp_merge = pd.merge(res[['timestamp']], df_b_safe[['timestamp', 'close']], on='timestamp', how='left')
    res['close_B'] = temp_merge['close'].values
    res.rename(columns={'close': 'close_A'}, inplace=True)
    
    # Backtest Execution Loop
    trades = []
    current_pos = None # None, "SHORT_A_LONG_B", "LONG_A_SHORT_B"
    entry_price_a = 0.0
    entry_price_b = 0.0
    qty_a = 0.0
    qty_b = 0.0
    entry_time = None
    
    for i in range(len(res)):
        row = res.iloc[i]
        
        sig = row['signal']
        exit_sig = row['exit_signal']
        stop_loss = row['stop_loss']
        
        ca = row['close_A']
        cb = row['close_B']
        ts = row['timestamp']
        
        if pd.isna(ca) or pd.isna(cb):
            continue
            
        if current_pos:
            if exit_sig or stop_loss:
                # Close trade
                pnl_a, pnl_b = 0, 0
                if current_pos == "SHORT_A_LONG_B":
                    pnl_a = (entry_price_a - ca) * qty_a
                    pnl_b = (cb - entry_price_b) * qty_b
                else:
                    pnl_a = (ca - entry_price_a) * qty_a
                    pnl_b = (entry_price_b - cb) * qty_b
                    
                total_pnl = pnl_a + pnl_b
                trades.append({
                    'pair': f"{sym_a}-{sym_b}",
                    'pos': current_pos,
                    'entry_time': entry_time,
                    'exit_time': ts,
                    'bars_held': i - entry_idx,
                    'pnl_usd': total_pnl,
                    'pnl_pct': (total_pnl / (risk_usd * 2)) * 100,
                    'reason': 'STOP_LOSS' if stop_loss else 'TAKE_PROFIT'
                })
                current_pos = None
        else:
            if sig == 1:
                current_pos = "SHORT_A_LONG_B"
                entry_price_a = ca
                entry_price_b = cb
                qty_a = risk_usd / ca
                qty_b = risk_usd / cb
                entry_time = ts
                entry_idx = i
            elif sig == -1:
                current_pos = "LONG_A_SHORT_B"
                entry_price_a = ca
                entry_price_b = cb
                qty_a = risk_usd / ca
                qty_b = risk_usd / cb
                entry_time = ts
                entry_idx = i
                
    if not trades:
        logger.info("  No trades taken.")
        return None
        
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['pnl_usd'] > 0).mean() * 100
    total_pnl = trades_df['pnl_usd'].sum()
    avg_pnl = trades_df['pnl_usd'].mean()
    
    logger.info(f"  [RESULT] Trades: {len(trades_df)} | Win Rate: {win_rate:.1f}% | Total PnL: ${total_pnl:.2f} | Avg PnL: ${avg_pnl:.2f}")
    return trades_df

def main():
    pairs = load_pairs()
    if not pairs:
        logger.error("No pairs found in arb_pairs.json")
        return
        
    logger.info(f"Loaded {len(pairs)} pairs for Stat Arb backtesting")
    strat = StatArbStrategy(lookback_bars=100, entry_z=2.0, exit_z=0.2, sl_z=4.0)
    
    all_trades = []
    
    for pair in pairs:
        if len(pair) == 2:
            trades = backtest_pair(pair[0], pair[1], strat)
            if trades is not None:
                all_trades.append(trades)
                
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        total_wr = (combined['pnl_usd'] > 0).mean() * 100
        logger.info("="*60)
        logger.info(f"OVERALL STAT ARB PERFORMANCE (Risking $1000 per leg)")
        logger.info(f"Total Trades: {len(combined)}")
        logger.info(f"Total Win Rate: {total_wr:.1f}%")
        logger.info(f"Total Net PnL: ${combined['pnl_usd'].sum():.2f}")
        logger.info("="*60)
        
if __name__ == '__main__':
    main()
