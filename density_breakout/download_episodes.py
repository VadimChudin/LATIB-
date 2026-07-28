"""
Phase 30.2: Download Hybrid Episode Data (1m body + ticks head + L2 book depth)
================================================================================
For each density episode found by find_density_episodes.py, downloads:
  1. body_1m.csv   — 1m klines for the consolidation period (candle shadows visible)
  2. head_ticks.csv — aggTrades for the last 10-15 min when price is ±0.5% from level
  3. book_depth.csv — L2 order book snapshots from data.binance.vision (REAL walls!)
  4. outcome_1m.csv — 1m klines for 30 min AFTER the breakout/rejection (for labeling)

L2 Source: https://data.binance.vision/data/futures/um/daily/bookDepth/{SYMBOL}/

Usage:
  python density_breakout/download_episodes.py --symbol ENJUSDT
  python density_breakout/download_episodes.py --all-symbols
"""

import os
import io
import json
import time
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Paths (SHARED cache + strategy-specific episodes) ──
CACHE_DIR = Path("data/cache")
EPISODES_DIR = Path("density_breakout/data/episodes")
HYBRID_DIR = Path("density_breakout/data/episodes_hybrid")
BOOKDEPTH_CACHE = Path("data/cache/bookdepth")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
HYBRID_DIR.mkdir(parents=True, exist_ok=True)
BOOKDEPTH_CACHE.mkdir(parents=True, exist_ok=True)

# ── Config ──
HEAD_ZONE_PCT = 0.005          # ±0.5% from level = "head" zone for tick download
HEAD_LOOKBACK_MINS = 15        # Download ticks for last 15 min of episode
OUTCOME_WINDOW_MINS = 30       # 30 min after episode


# ═══════════════════════════════════════════════════════
#  1m Klines Download (reuses shared cache)
# ═══════════════════════════════════════════════════════

def download_1m_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download 1m klines for a specific time range."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_candles = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": current_start,
            "endTime": end_ms,
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
                    "ts_ms": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "num_trades": int(k[8]),
                    "taker_buy_volume": float(k[9]),
                })

            current_start = data[-1][0] + 60_000
            time.sleep(0.1)

        except Exception as e:
            print(f"    Error downloading 1m klines: {e}")
            time.sleep(5)
            current_start += 1500 * 60_000

    return pd.DataFrame(all_candles) if all_candles else pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  AggTrades Download (ticks)
# ═══════════════════════════════════════════════════════

def download_agg_trades(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download aggTrades from Binance Futures."""
    url = "https://fapi.binance.com/fapi/v1/aggTrades"
    all_trades = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": min(current_start + 3_600_000, end_ms),
            "limit": 1000
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

            for t in data:
                all_trades.append({
                    "ts_ms": t["T"],
                    "price": float(t["p"]),
                    "qty": float(t["q"]),
                    "is_buyer_maker": t["m"]
                })

            current_start = data[-1]["T"] + 1
            time.sleep(0.15)

        except Exception as e:
            print(f"    Error downloading trades: {e}")
            time.sleep(5)
            current_start += 3_600_000

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  L2 BookDepth from data.binance.vision
# ═══════════════════════════════════════════════════════

def download_bookdepth_day(symbol: str, date_str: str) -> Optional[Path]:
    """
    Download a day's bookDepth ZIP from data.binance.vision.
    Returns path to local CSV or None.
    Format: {SYMBOL}-bookDepth-{date}.zip
    """
    local_dir = BOOKDEPTH_CACHE / symbol
    local_dir.mkdir(parents=True, exist_ok=True)

    csv_path = local_dir / f"{date_str}.csv"
    if csv_path.exists():
        return csv_path  # Already cached

    # Try multiple filename formats
    base_url = f"https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}"
    filenames = [
        f"{symbol}-bookDepth-{date_str}.zip",
    ]

    for fname in filenames:
        url = f"{base_url}/{fname}"
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 404:
                continue
            if resp.status_code == 429:
                time.sleep(30)
                continue
            resp.raise_for_status()

            # Extract ZIP
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_name = z.namelist()[0]  # Usually single CSV inside
            z.extract(csv_name, local_dir)

            # Rename to standard format
            extracted = local_dir / csv_name
            if extracted != csv_path:
                extracted.rename(csv_path)

            print(f"    L2: Downloaded {date_str} ({csv_path.stat().st_size // 1024}KB)")
            return csv_path

        except Exception as e:
            continue

    return None


def extract_bookdepth_for_episode(
    symbol: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """
    Get L2 depth snapshots for a specific time range from cached daily files.
    
    Real Binance bookDepth format (data.binance.vision):
      timestamp (datetime string), percentage (float), depth (float), notional (float)
    
    - percentage: distance from mid price in % (-5, -4, ..., +4, +5)
      Negative = bid side, Positive = ask side
    - depth: cumulative quantity at that % level
    - notional: cumulative USD value at that % level
    """
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    all_rows = []

    current_date = start_dt.date()
    end_date = end_dt.date()

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        csv_path = download_bookdepth_day(symbol, date_str)

        if csv_path and csv_path.exists():
            try:
                # Real format: timestamp,percentage,depth,notional (with header row)
                chunks = pd.read_csv(
                    csv_path,
                    chunksize=100_000,
                    on_bad_lines="skip",
                )

                for chunk in chunks:
                    # Parse timestamp string → unix ms for filtering
                    chunk["ts_ms"] = pd.to_datetime(
                        chunk["timestamp"], utc=True
                    ).astype(np.int64) // 10**6

                    # Filter to our time range
                    mask = (chunk["ts_ms"] >= start_ms) & (chunk["ts_ms"] <= end_ms)
                    filtered = chunk[mask]
                    if len(filtered) > 0:
                        all_rows.append(filtered[["ts_ms", "percentage", "depth", "notional"]].copy())

            except Exception as e:
                print(f"    Warning: Could not parse {csv_path}: {e}")

        current_date += timedelta(days=1)

    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        df.sort_values("ts_ms", inplace=True)
        return df

    return pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════

def download_episode_data(symbol: str, episode: dict, episode_dir: Path) -> dict:
    """
    Download all hybrid data for a single episode.
    Returns dict with download status.
    """
    status = {"body": False, "head": False, "book": False, "outcome": False}

    start_ms = episode["start_ts"]
    end_ms = episode["end_ts"]
    level_price = episode["level_price"]

    # ── 1. Body: 1m klines for the consolidation period ──
    body_path = episode_dir / "body_1m.csv"
    if not body_path.exists():
        df_body = download_1m_klines(symbol, start_ms, end_ms)
        if not df_body.empty:
            df_body.to_csv(body_path, index=False)
            status["body"] = True
    else:
        status["body"] = True

    # ── 2. Head: aggTrades for last 15 min of episode (near level) ──
    head_path = episode_dir / "head_ticks.csv"
    if not head_path.exists():
        head_start = end_ms - HEAD_LOOKBACK_MINS * 60_000
        df_head = download_agg_trades(symbol, head_start, end_ms + 60_000)
        if not df_head.empty:
            # Filter to ticks within ±0.5% of level
            zone_hi = level_price * (1 + HEAD_ZONE_PCT)
            zone_lo = level_price * (1 - HEAD_ZONE_PCT)
            df_head = df_head[
                (df_head["price"] >= zone_lo) & (df_head["price"] <= zone_hi)
            ]
            if not df_head.empty:
                df_head.to_csv(head_path, index=False)
                status["head"] = True
    else:
        status["head"] = True

    # ── 3. Book Depth: L2 snapshots from data.binance.vision ──
    book_path = episode_dir / "book_depth.csv"
    if not book_path.exists():
        # Get L2 data for episode period + 5 min buffer
        df_book = extract_bookdepth_for_episode(
            symbol, start_ms - 300_000, end_ms + 300_000
        )
        if not df_book.empty:
            df_book.to_csv(book_path, index=False)
            status["book"] = True
    else:
        status["book"] = True

    # ── 4. Outcome: 1m klines for 30 min after episode ──
    outcome_path = episode_dir / "outcome_1m.csv"
    if not outcome_path.exists():
        outcome_end = end_ms + OUTCOME_WINDOW_MINS * 60_000
        df_outcome = download_1m_klines(symbol, end_ms, outcome_end)
        if not df_outcome.empty:
            df_outcome.to_csv(outcome_path, index=False)
            status["outcome"] = True
    else:
        status["outcome"] = True

    return status


def process_symbol(symbol: str):
    """Download hybrid data for all episodes of a symbol."""
    episodes_path = EPISODES_DIR / f"{symbol}_episodes.json"

    if not episodes_path.exists():
        print(f"  No episodes file for {symbol}. Run find_density_episodes.py first.")
        return

    with open(episodes_path) as f:
        episodes = json.load(f)

    if not episodes:
        print(f"  No episodes for {symbol}")
        return

    print(f"\n{'='*60}")
    print(f"Downloading hybrid data for {symbol} ({len(episodes)} episodes)")
    print(f"{'='*60}")

    stats = {"body": 0, "head": 0, "book": 0, "outcome": 0, "total": len(episodes)}

    for i, ep in enumerate(episodes):
        ts_id = ep["start_ts"]
        episode_dir = HYBRID_DIR / symbol / str(ts_id)
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save episode metadata
        meta_path = episode_dir / "meta.json"
        if not meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(ep, f, indent=2)

        result = download_episode_data(symbol, ep, episode_dir)

        for key in stats:
            if key != "total" and result.get(key):
                stats[key] += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(episodes)}] body={stats['body']} head={stats['head']} "
                  f"book={stats['book']} outcome={stats['outcome']}")

        time.sleep(0.2)  # Be nice to API

    print(f"\n  Done: {stats}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 30.2: Download Hybrid Episode Data")
    parser.add_argument("--symbol", type=str, help="Single symbol (e.g. ENJUSDT)")
    parser.add_argument("--all-symbols", action="store_true", help="Process all symbols with episodes")
    args = parser.parse_args()

    if args.all_symbols:
        episode_files = list(EPISODES_DIR.glob("*_episodes.json"))
        print(f"Found {len(episode_files)} symbols with episodes")
        for ef in episode_files:
            sym = ef.stem.replace("_episodes", "")
            process_symbol(sym)
            time.sleep(1)

    elif args.symbol:
        process_symbol(args.symbol)
    else:
        print("Usage:")
        print("  python density_breakout/download_episodes.py --symbol ENJUSDT")
        print("  python density_breakout/download_episodes.py --all-symbols")
