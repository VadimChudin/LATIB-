import pandas as pd
from strategies.scalping_strategy import ScalpingStrategy
import sys

def main():
    print("Loading 1m cache...")
    try:
        df1m = pd.read_csv('data/cache/BTC_USDT_1m_30d.csv', dtype={'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'})
        df1m['timestamp'] = pd.to_datetime(df1m['timestamp'])
        df1m.index = df1m['timestamp']
    except FileNotFoundError:
        print("1m cache missing. Run diagnose.py first.")
        sys.exit(1)

    print("Loading 5m cache...")
    try:
        df5m = pd.read_csv('data/cache/BTC_USDT_5m_730d.csv', dtype={'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'})
        df5m['timestamp'] = pd.to_datetime(df5m['timestamp'])
        df5m.index = df5m['timestamp']
    except FileNotFoundError:
        print("5m cache missing. Run diagnose.py first.")
        sys.exit(1)

    strat = ScalpingStrategy()
    params = {'ema_fast': 8, 'ema_slow': 21, 'tp_atr_mult': 1.0, 'sl_atr_mult': 0.8}

    print("\n=== RAW SCALPING (1m) ===")
    r1 = strat.backtest_logic(df1m.copy(), params)
    t1 = r1[r1['trade_pnl_r'] != 0]['trade_pnl_r']
    print(f"Trades: {len(t1)}")
    if len(t1) > 0:
        print(f"Win Rate: {(t1 > 0).mean():.1%}")
        print(f"EV / Trade: {t1.mean():.3f}R")
        
    print("\n=== RAW SCALPING (5m) ===")
    r5 = strat.backtest_logic(df5m.copy(), params)
    t5 = r5[r5['trade_pnl_r'] != 0]['trade_pnl_r']
    print(f"Trades: {len(t5)}")
    if len(t5) > 0:
        print(f"Win Rate: {(t5 > 0).mean():.1%}")
        print(f"EV / Trade: {t5.mean():.3f}R")

if __name__ == '__main__':
    main()
