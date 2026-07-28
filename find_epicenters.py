import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = "data/cache"
OUT_DIR = "data/epicenters"
os.makedirs(OUT_DIR, exist_ok=True)

# Define Epicenter Criteria
MIN_SPREAD_PCT = 0.004      # High - Low must be > 0.4%
MIN_VOLUME_MULT = 3.0       # Volume must be 3x the 20-period moving average

def find_epicenters(symbol: str, timeframe="1m") -> list:
    csv_path = Path(CACHE_DIR) / f"{symbol}_{timeframe}_730d.csv"
    if not csv_path.exists():
        print(f"❌ Not found: {csv_path}")
        return []

    print(f"📊 Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Calculate indicators
    df['spread_pct'] = (df['high'] - df['low']) / df['close']
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['vol_mult'] = df['volume'] / df['vol_sma_20']

    # Smart Filters: RSI and SMA
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['sma_60'] = df['close'].rolling(window=60).mean()

    # ═══ COUNTER-TREND KNIFE CATCHER ═══
    # The key insight: knives that fall AGAINST the trend bounce hardest.
    # A flash crash in a BULL market is a dislocation — the trend pulls price back up.
    # A flash crash in a BEAR market is just... the trend continuing. No edge.

    # Identify LONG Epicenters (Flash Crash in BULLISH context)
    # 1. Red candle (sudden dump)
    # 2. RSI > 50 BEFORE the dump (market was healthy/bullish)
    # 3. Price still above 1-hour average (this is a dip, not a trend change)
    # 4. High volatility and volume (something big happened — liquidations, fat finger, etc.)
    mask_long = (
        (df['close'] < df['open']) & 
        (df['rsi_14'] > 50) &
        (df['close'] > df['sma_60'] * 0.99) &
        (df['spread_pct'] > MIN_SPREAD_PCT) & 
        (df['vol_mult'] > MIN_VOLUME_MULT)
    )

    # Identify SHORT Epicenters (Flash Pump in BEARISH context)
    # 1. Green candle (sudden pump)
    # 2. RSI < 50 BEFORE the pump (market was weak/bearish)
    # 3. Price still below 1-hour average (this is a squeeze, not a trend change)
    # 4. High volatility and volume
    mask_short = (
        (df['close'] > df['open']) & 
        (df['rsi_14'] < 50) &
        (df['close'] < df['sma_60'] * 1.01) &
        (df['spread_pct'] > MIN_SPREAD_PCT) & 
        (df['vol_mult'] > MIN_VOLUME_MULT)
    )

    df.loc[mask_long, 'direction'] = 'LONG'
    df.loc[mask_short, 'direction'] = 'SHORT'
    
    mask = mask_long | mask_short
    epicenters_df = df[mask].copy()
    
    epicenters = []
    for _, row in epicenters_df.iterrows():
        epicenters.append({
            "timestamp": row['timestamp'],
            "ts_ms": int(pd.Timestamp(row['timestamp']).timestamp() * 1000),
            "close": float(row['close']),
            "spread_pct": float(row['spread_pct']),
            "vol_mult": float(row['vol_mult']),
            "direction": str(row['direction'])
        })

    print(f"🎯 Found {len(epicenters)} epicenters for {symbol}")
    
    # Save to JSON
    out_file = Path(OUT_DIR) / f"{symbol}_epicenters.json"
    with open(out_file, "w") as f:
        json.dump(epicenters, f, indent=4)
    print(f"💾 Saved to {out_file}")
    
    return epicenters

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTC_USDT")
    args = parser.parse_args()
    
    find_epicenters(args.symbol)
