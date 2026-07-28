import sys
import pandas as pd
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy

df = pd.read_csv("data/cache/BTC_USDT_5m_730d.csv", low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.index = df['timestamp']
df = df.iloc[-50000:]
df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)
df['close'] = df['close'].astype(float)
df['volume'] = df['volume'].astype(float)

strat = UltimateSMCTrailStrategy()
params = {
    'swing_length': 5,
    'fvg_min_atr': 0.3,
    'ob_min_score': 3,
    'sl_atr_mult': 1.0,
    'trail_activate_r': 1.0,
    'trail_atr_mult': 0.5
}
print("Starting backtest...")
res = strat.backtest_logic(df, params)
trades = res[res['trade_pnl_r'] != 0]
print(f"Total trades: {len(trades)}")
if len(trades) > 0:
    print(trades[['timestamp', 'close', 'trade_pnl_r']].head())
