"""
Phase 31: Find Epicenters V2 — with FALSE epicenters
=====================================================
Finds both REAL (bounce happened) and FALSE (drop continued) epicenters.
Downloads fresh tick data from Binance and saves per-epicenter CSVs.

Usage:
  python find_epicenters_v2.py --symbol DOT_USDT --days 7
  python find_epicenters_v2.py --symbol DOT_USDT --days 7 --all-symbols
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

CACHE_DIR = "data/cache"
OUT_DIR = "data/epicenters_ticks"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Epicenter criteria ──
MIN_SPREAD_PCT = 0.004      # High - Low must be > 0.4%
MIN_VOLUME_MULT = 3.0       # Volume must be 3x the 20-period MA
BOUNCE_THRESHOLD_PCT = 0.003 # 0.3% bounce = real epicenter
LOOK_AHEAD_MINUTES = 5       # How far ahead to check for bounce
TICK_WINDOW_SECS = 120        # Download 2 minutes of ticks per epicenter


def download_agg_trades(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download aggTrades from Binance Futures for a time range."""
    binance_symbol = symbol.replace("_", "")
    url = "https://fapi.binance.com/fapi/v1/aggTrades"
    
    all_trades = []
    current_start = start_ms
    
    while current_start < end_ms:
        params = {
            "symbol": binance_symbol,
            "startTime": current_start,
            "endTime": min(current_start + 3600000, end_ms),  # 1h chunks
            "limit": 1000
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                print("  Rate limited, sleeping 30s...")
                time.sleep(30)
                continue
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                break
                
            for t in data:
                all_trades.append({
                    "ts_ms": t["T"],
                    "price": float(t["p"]),
                    "qty": float(t["q"]),
                    "is_buyer_maker": t["m"]
                })
            
            # Move past latest trade
            current_start = data[-1]["T"] + 1
            time.sleep(0.15)  # Rate limit
            
        except Exception as e:
            print(f"  Error downloading trades: {e}")
            time.sleep(5)
            current_start += 3600000
    
    if not all_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_trades)
    df.sort_values("ts_ms", inplace=True)
    return df


def download_klines(symbol: str, days: int) -> pd.DataFrame:
    """Download 1m klines from Binance Futures."""
    binance_symbol = symbol.replace("_", "")
    url = "https://fapi.binance.com/fapi/v1/klines"
    
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    
    all_candles = []
    current_start = start_ms
    
    print(f"📊 Downloading {days}d of 1m klines for {symbol}...")
    
    while current_start < end_ms:
        params = {
            "symbol": binance_symbol,
            "interval": "1m",
            "startTime": current_start,
            "limit": 1500
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                time.sleep(30)
                continue
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                break
            
            for k in data:
                all_candles.append({
                    "timestamp": datetime.fromtimestamp(k[0]/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "num_trades": float(k[8]),
                    "taker_buy_volume": float(k[9]),
                })
            
            current_start = data[-1][0] + 60000
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(5)
            current_start += 1500 * 60000
    
    df = pd.DataFrame(all_candles)
    print(f"  Got {len(df)} candles")
    return df


def find_epicenters_v2(symbol: str, days: int = 7, download_ticks: bool = True) -> dict:
    """
    Find REAL and FALSE epicenters for a symbol.
    Returns dict with counts.
    """
    # Step 1: Get klines
    cache_path = Path(CACHE_DIR) / f"{symbol}_1m_{days}d_fresh.csv"
    
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 3600:
        print(f"📂 Using cached klines: {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        df = download_klines(symbol, days)
        if df.empty:
            print(f"❌ No data for {symbol}")
            return {"real_long": 0, "real_short": 0, "false_long": 0, "false_short": 0}
        df.to_csv(cache_path, index=False)
    
    if len(df) < 100:
        print(f"❌ Not enough data for {symbol}: {len(df)} candles")
        return {"real_long": 0, "real_short": 0, "false_long": 0, "false_short": 0}
    
    # Step 2: Calculate indicators
    df['spread_pct'] = (df['high'] - df['low']) / df['close']
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['vol_mult'] = df['volume'] / df['vol_sma_20']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['sma_60'] = df['close'].rolling(window=60).mean()
    
    # Step 3: Find potential epicenters (both real and false)
    # LONG candidates: red candle + high vol + big spread
    mask_long_candidate = (
        (df['close'] < df['open']) &
        (df['spread_pct'] > MIN_SPREAD_PCT) & 
        (df['vol_mult'] > MIN_VOLUME_MULT)
    )
    
    # SHORT candidates: green candle + high vol + big spread  
    mask_short_candidate = (
        (df['close'] > df['open']) &
        (df['spread_pct'] > MIN_SPREAD_PCT) & 
        (df['vol_mult'] > MIN_VOLUME_MULT)
    )
    
    # Step 4: Classify each candidate as REAL or FALSE by looking ahead
    results = {"real_long": 0, "real_short": 0, "false_long": 0, "false_short": 0}
    
    for idx in df.index[mask_long_candidate]:
        if idx + LOOK_AHEAD_MINUTES >= len(df):
            continue
        
        candle = df.iloc[idx]
        future = df.iloc[idx+1 : idx+1+LOOK_AHEAD_MINUTES]
        
        # For LONG: did price bounce UP after the dump?
        max_future_high = future['high'].max()
        bounce_pct = (max_future_high - candle['close']) / candle['close']
        
        has_bounce = bounce_pct >= BOUNCE_THRESHOLD_PCT
        direction = "LONG"
        subdir = f"{direction}" if has_bounce else f"{direction}_FALSE"
        
        if has_bounce:
            results["real_long"] += 1
        else:
            results["false_long"] += 1
        
        if download_ticks:
            _save_epicenter_ticks(symbol, candle, idx, subdir)
    
    for idx in df.index[mask_short_candidate]:
        if idx + LOOK_AHEAD_MINUTES >= len(df):
            continue
        
        candle = df.iloc[idx]
        future = df.iloc[idx+1 : idx+1+LOOK_AHEAD_MINUTES]
        
        # For SHORT: did price drop DOWN after the pump?
        min_future_low = future['low'].min()
        bounce_pct = (candle['close'] - min_future_low) / candle['close']
        
        has_bounce = bounce_pct >= BOUNCE_THRESHOLD_PCT
        direction = "SHORT"
        subdir = f"{direction}" if has_bounce else f"{direction}_FALSE"
        
        if has_bounce:
            results["real_short"] += 1
        else:
            results["false_short"] += 1
        
        if download_ticks:
            _save_epicenter_ticks(symbol, candle, idx, subdir)
    
    total = sum(results.values())
    real = results["real_long"] + results["real_short"]
    false_ = results["false_long"] + results["false_short"]
    print(f"\n🎯 {symbol}: {total} epicenters ({real} real + {false_} false)")
    print(f"   LONG: {results['real_long']} real + {results['false_long']} false")
    print(f"   SHORT: {results['real_short']} real + {results['false_short']} false")
    
    return results


def _save_epicenter_ticks(symbol: str, candle, candle_idx: int, subdir: str):
    """Download and save tick data for a single epicenter."""
    ts_str = candle.get('timestamp', '')
    if not ts_str:
        return
    
    try:
        ts = pd.Timestamp(ts_str, tz='UTC')
        ts_ms = int(ts.timestamp() * 1000)
    except:
        return
    
    # Create output directory
    out_path = Path(OUT_DIR) / symbol / subdir
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_path / f"{ts_ms}.csv"
    if csv_path.exists():
        return  # Already downloaded
    
    # Download ticks: 30s before to 90s after the candle
    start_ms = ts_ms - 30_000
    end_ms = ts_ms + TICK_WINDOW_SECS * 1000
    
    tick_df = download_agg_trades(symbol, start_ms, end_ms)
    if tick_df.empty:
        return
    
    tick_df.to_csv(csv_path, index=False)


def get_all_futures_symbols() -> list:
    """Get all USDT perpetual futures symbols from Binance."""
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        symbols = []
        for s in data['symbols']:
            if (s['contractType'] == 'PERPETUAL' and 
                s['quoteAsset'] == 'USDT' and 
                s['status'] == 'TRADING'):
                # Format: DOTUSDT -> DOT_USDT
                base = s['baseAsset']
                symbol = f"{base}_USDT"
                symbols.append(symbol)
        
        # Exclude high-cap
        exclude = {'BTC_USDT', 'ETH_USDT'}
        symbols = [s for s in symbols if s not in exclude]
        
        print(f"📋 Found {len(symbols)} tradeable USDT perpetual symbols (excl BTC/ETH)")
        return sorted(symbols)
        
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 31: Epicenter Finder V2")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol (e.g. DOT_USDT)")
    parser.add_argument("--days", type=int, default=7, help="Days of data to analyze")
    parser.add_argument("--all-symbols", action="store_true", help="Run on all Binance futures")
    parser.add_argument("--no-ticks", action="store_true", help="Skip tick download (just find epicenters)")
    args = parser.parse_args()
    
    if args.all_symbols:
        symbols = get_all_futures_symbols()
        summary = {}
        for i, sym in enumerate(symbols):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(symbols)}] Processing {sym}...")
            print(f"{'='*60}")
            result = find_epicenters_v2(sym, args.days, download_ticks=not args.no_ticks)
            summary[sym] = result
            time.sleep(1)  # Be nice to API
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(summary)} symbols processed")
        print(f"{'='*60}")
        
        total_real = sum(r["real_long"] + r["real_short"] for r in summary.values())
        total_false = sum(r["false_long"] + r["false_short"] for r in summary.values())
        print(f"Total epicenters: {total_real + total_false} ({total_real} real + {total_false} false)")
        
        # Save summary
        with open("data/epicenters_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("📁 Saved summary to data/epicenters_summary.json")
        
    elif args.symbol:
        find_epicenters_v2(args.symbol, args.days, download_ticks=not args.no_ticks)
    else:
        print("Usage: python find_epicenters_v2.py --symbol DOT_USDT --days 7")
        print("       python find_epicenters_v2.py --all-symbols --days 7")
