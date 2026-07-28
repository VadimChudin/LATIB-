import asyncio
import pandas as pd
import ccxt.async_support as ccxt
from strategies.scalping_strategy import ScalpingStrategy

async def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.reset_index(drop=True)

async def main():
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    await ex.load_markets()
    
    print("Fetching 1m data for BTC/USDT (last 1500 candles)...")
    # Fetch maximum allowed by binance in one go (typically 1500)
    df1m = await fetch_ohlcv(ex, 'BTC/USDT', '1m', limit=1500)
    print(f"Loaded {len(df1m)} rows of 1m data.")
    
    print("\nFetching 5m data for BTC/USDT (last 1500 candles)...")
    df5m = await fetch_ohlcv(ex, 'BTC/USDT', '5m', limit=1500)
    print(f"Loaded {len(df5m)} rows of 5m data.")
    
    await ex.close()
    
    strat = ScalpingStrategy()
    params = {'ema_fast': 8, 'ema_slow': 21, 'tp_rr': 1.5, 'sl_atr_mult': 0.3}

    print("\n=== RAW SCALPING (1m) ===")
    try:
        r1 = strat.backtest_logic(df1m.copy(), params)
        t1 = r1[r1['trade_pnl_r'] != 0]['trade_pnl_r']
        print(f"Trades: {len(t1)}")
        if len(t1) > 0:
            print(f"Win Rate: {(t1 > 0).mean():.1%}")
            print(f"Avg EV / Trade: {t1.mean():.3f}R")
            print(f"Gross Win R: {t1[t1>0].sum():.1f}R  | Gross Loss R: {t1[t1<0].sum():.1f}R")
            print(f"Net R: {t1.sum():.1f}R")
    except Exception as e:
        print(f"Error testing 1m: {e}")
        
    print("\n=== RAW SCALPING (5m) ===")
    try:
        r5 = strat.backtest_logic(df5m.copy(), params)
        t5 = r5[r5['trade_pnl_r'] != 0]['trade_pnl_r']
        print(f"Trades: {len(t5)}")
        if len(t5) > 0:
            print(f"Win Rate: {(t5 > 0).mean():.1%}")
            print(f"Avg EV / Trade: {t5.mean():.3f}R")
            print(f"Gross Win R: {t5[t5>0].sum():.1f}R  | Gross Loss R: {t5[t5<0].sum():.1f}R")
            print(f"Net R: {t5.sum():.1f}R")
    except Exception as e:
        print(f"Error testing 5m: {e}")

if __name__ == "__main__":
    asyncio.run(main())
