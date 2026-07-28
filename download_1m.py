import ccxt
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta

DAYS = 30
TIMEFRAME = "1m"
CACHE_DIR = "data/cache"


def download_ohlcv(symbol, timeframe=TIMEFRAME, days=DAYS):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    safe_name = symbol.replace("/", "_")
    filename = f"{CACHE_DIR}/{safe_name}_{timeframe}_{days}d.csv"
    
    # Check if fresh enough (< 6h old)
    if os.path.exists(filename):
        age_h = (time.time() - os.path.getmtime(filename)) / 3600
        if age_h < 6:
            print(f"  ⏭️  {symbol}: Up to date ({age_h:.1f}h old)")
            return filename
    
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_ohlcv = []
    
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"   Error: {e}. Retrying...")
            time.sleep(5)
            
    if not all_ohlcv:
        print(f"  ❌ {symbol}: No data")
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"  ✅ {symbol}: {len(df)} candles saved")
    return filename


if __name__ == "__main__":
    # Load top symbols
    with open("data/top_symbols.json") as f:
        symbols = json.load(f)
    
    print(f"📥 Downloading {DAYS}d of {TIMEFRAME} candles for {len(symbols)} symbols\n")
    
    ok = 0
    for i, sym in enumerate(symbols):
        sym_ccxt = sym.replace("_", "/")
        print(f"[{i+1}/{len(symbols)}] {sym_ccxt}")
        result = download_ohlcv(sym_ccxt)
        if result:
            ok += 1
    
    print(f"\n🏆 Done: {ok}/{len(symbols)} symbols downloaded")
