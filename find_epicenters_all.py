#!/usr/bin/env python3
"""
Find Epicenters for Multiple Symbols (Top 50)
==============================================
Scans 1m candles (or 5m fallback) for flash crash/pump events.
Limits to last 6 months of data.

Usage: python find_epicenters_all.py [--top 50]
"""
import os, json, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

CACHE_DIR = "data/cache"
OUT_DIR = "data/epicenters"
os.makedirs(OUT_DIR, exist_ok=True)

# Epicenter detection thresholds
MIN_SPREAD_PCT_1M = 0.004     # 0.4% for 1m candles
MIN_SPREAD_PCT_5M = 0.008     # 0.8% for 5m candles (higher bar)
MIN_VOLUME_MULT = 3.0         # Volume must be 3x the 20-period SMA

MONTHS_LOOKBACK = 1  # Last 30 days only


def find_epicenters_for_symbol(symbol: str) -> list:
    """Find epicenters for a single symbol. Returns list of epicenter dicts."""
    
    # Try 1m candles first, fall back to 5m
    csv_1m = Path(CACHE_DIR) / f"{symbol}_1m_30d.csv"
    csv_1m_old = Path(CACHE_DIR) / f"{symbol}_1m_730d.csv"
    csv_5m = Path(CACHE_DIR) / f"{symbol}_5m_730d.csv"
    
    if csv_1m.exists():
        csv_path = csv_1m
        min_spread = MIN_SPREAD_PCT_1M
        tf = "1m"
    elif csv_1m_old.exists():
        csv_path = csv_1m_old
        min_spread = MIN_SPREAD_PCT_1M
        tf = "1m"
    elif csv_5m.exists():
        csv_path = csv_5m
        min_spread = MIN_SPREAD_PCT_5M
        tf = "5m"
    else:
        return []

    try:
        df = pd.read_csv(csv_path, engine='c', low_memory=False)
    except Exception:
        return []
    
    if len(df) < 200:
        return []

    # Parse timestamps and filter to last 6 months
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cutoff = datetime.utcnow() - timedelta(days=MONTHS_LOOKBACK * 30)
    df = df[df['timestamp'] >= cutoff].copy()
    
    if len(df) < 100:
        return []

    # Calculate indicators
    df['spread_pct'] = (df['high'] - df['low']) / df['close']
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['vol_mult'] = df['volume'] / df['vol_sma_20']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # SMA for trend context
    sma_window = 60 if tf == "1m" else 12  # 12 × 5m = 60min
    df['sma_60'] = df['close'].rolling(window=sma_window).mean()

    # LONG epicenters: flash crash in bullish context
    mask_long = (
        (df['close'] < df['open']) &
        (df['rsi_14'] > 50) &
        (df['close'] > df['sma_60'] * 0.99) &
        (df['spread_pct'] > min_spread) &
        (df['vol_mult'] > MIN_VOLUME_MULT)
    )

    # SHORT epicenters: flash pump in bearish context  
    mask_short = (
        (df['close'] > df['open']) &
        (df['rsi_14'] < 50) &
        (df['close'] < df['sma_60'] * 1.01) &
        (df['spread_pct'] > min_spread) &
        (df['vol_mult'] > MIN_VOLUME_MULT)
    )

    df.loc[mask_long, 'direction'] = 'LONG'
    df.loc[mask_short, 'direction'] = 'SHORT'

    mask = mask_long | mask_short
    epicenters_df = df[mask].copy()

    epicenters = []
    for _, row in epicenters_df.iterrows():
        try:
            ts_ms = int(row['timestamp'].timestamp() * 1000)
        except Exception:
            continue
        epicenters.append({
            "timestamp": str(row['timestamp']),
            "ts_ms": ts_ms,
            "close": float(row['close']),
            "spread_pct": float(row['spread_pct']),
            "vol_mult": float(row['vol_mult']),
            "direction": str(row['direction'])
        })

    return epicenters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=50, help='Number of top symbols to process')
    args = parser.parse_args()

    symbols_path = Path("data/top_symbols.json")
    with open(symbols_path) as f:
        all_symbols = json.load(f)
    
    symbols = all_symbols[:args.top]
    print(f"🎯 Finding epicenters for top {len(symbols)} symbols (last {MONTHS_LOOKBACK} months)")
    print("=" * 60)

    total_epicenters = 0
    results = []

    for i, sym in enumerate(symbols):
        epicenters = find_epicenters_for_symbol(sym)
        
        if epicenters:
            out_file = Path(OUT_DIR) / f"{sym}_epicenters.json"
            with open(out_file, "w") as f:
                json.dump(epicenters, f, indent=2)
            
            n_long = sum(1 for e in epicenters if e['direction'] == 'LONG')
            n_short = len(epicenters) - n_long
            total_epicenters += len(epicenters)
            results.append((sym, len(epicenters), n_long, n_short))
            print(f"  [{i+1:2d}/{len(symbols)}] {sym:20s}: {len(epicenters):5d} epicenters ({n_long}L / {n_short}S)")
        else:
            print(f"  [{i+1:2d}/{len(symbols)}] {sym:20s}: SKIP (no candle data)")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Total: {total_epicenters} epicenters from {len(results)}/{len(symbols)} symbols")
    print(f"  📁 Saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
