import pandas as pd
import numpy as np

# Load data
print("Loading data...")
df = pd.read_csv("data/BTC_USDT_5m.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.iloc[-50000:].copy() # Test on last 50K candles

print(f"Data shape: {df.shape}")

# VWAP Test
print("\n--- Testing VWAP ---")
from strategies.vwap_squeeze import VWAPSqueezeStrategy
vwap = VWAPSqueezeStrategy()
res_vwap = vwap.backtest_logic(df, {'stdev_mult': 2.5, 'period': 50, 'tp_rr': 1.5, 'rsi_oversold': 35})
trades_vwap = res_vwap[res_vwap['trade_pnl_r'] != 0].copy()
print(f"VWAP Total Trades: {len(trades_vwap)}")
print(f"VWAP Win Rate: {len(trades_vwap[trades_vwap['trade_pnl_r'] > 0]) / len(trades_vwap) * 100:.2f}%" if len(trades_vwap) > 0 else "N/A")

# TTM Test
print("\n--- Testing TTM ---")
from strategies.ttm_squeeze import TTMSqueezeStrategy
ttm = TTMSqueezeStrategy()
res_ttm = ttm.backtest_logic(df, {'bb_len': 20, 'bb_mult': 2.0, 'kc_len': 20, 'kc_mult': 2.0, 'mom_len': 12, 'tp_rr': 2.0})
trades_ttm = res_ttm[res_ttm['trade_pnl_r'] != 0].copy()
print(f"TTM Total Trades: {len(trades_ttm)}")
print(f"TTM Squeeze ON Count: {res_ttm['sqz_on'].sum()}")
print(f"TTM Win Rate: {len(trades_ttm[trades_ttm['trade_pnl_r'] > 0]) / len(trades_ttm) * 100:.2f}%" if len(trades_ttm) > 0 else "N/A")

# SwingICT Test
print("\n--- Testing Ultimate SMC Trail ---")
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy
sict = UltimateSMCTrailStrategy()
res_sict = sict.backtest_logic(df, {'swing_len': 5.0, 'fvg_min_atr': 0.3, 'sl_atr_mult': 1.0, 'trail_activate_r': 1.0, 'trail_atr_mult': 0.5})
trades_sict = res_sict[res_sict['trade_pnl_r'] != 0].copy()
print(f"Ultimate SMC Total Trades: {len(trades_sict)}")
wins_sict = len(trades_sict[trades_sict['trade_pnl_r'] > 0])
total_sict = len(trades_sict)
print(f"Ultimate SMC Win Rate: {wins_sict / total_sict * 100:.2f}%" if total_sict > 0 else "N/A")
print(f"Ultimate SMC PnL Distribution:\n{trades_sict['trade_pnl_r'].value_counts().head()}")
