"""
Phase 30.1: Find Density Episodes — consolidation zones near S/R levels
=======================================================================
Scans 5m candles for periods where price consolidates near a S/R level,
indicating a potential breakout setup (density zone / wall probing).

Pipeline:
  1. Download 5m candles for N days
  2. Aggregate 5m → 1H for S/R level detection (touch clustering)
  3. Scan 5m candles for consolidation episodes near levels
  4. Save episodes as JSON for download_episodes.py

Usage:
  python density_breakout/find_density_episodes.py --symbol ENJUSDT --days 30
  python density_breakout/find_density_episodes.py --all-symbols --days 30
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

# ── Paths ──
# SHARED: raw market data (klines, bookDepth) — reused across all strategies
CACHE_DIR = Path("data/cache")
# STRATEGY-SPECIFIC: density episodes only
EPISODES_DIR = Path("density_breakout/data/episodes")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EPISODES_DIR.mkdir(parents=True, exist_ok=True)

# ── S/R Level Detection ──
SR_CLUSTER_ATR_MULT = 0.3       # Cluster radius = 0.3 × ATR
SR_MIN_TOUCHES = 2              # Minimum touches to qualify as level
SR_LOOKBACK_1H = 168            # 7 days of 1H candles for level detection

# ── Episode Detection ──
EPISODE_ZONE_PCT = 0.01         # ±1% from level = consolidation zone
EPISODE_MIN_DURATION_MINS = 30  # Must stay in zone ≥ 30 min
EPISODE_MAX_DURATION_MINS = 480 # Cap at 8 hours
EPISODE_MIN_CANDLES = 6         # At least 6 × 5m = 30 min
EPISODE_MAX_GAP_MINS = 15      # Max exit from zone before reset (3 × 5m)
OUTCOME_WINDOW_MINS = 30        # Check 30 min after episode ends


# ═══════════════════════════════════════════════════════
#  Data Download (reused pattern from find_epicenters_v2)
# ═══════════════════════════════════════════════════════

def download_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Download klines from Binance Futures."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000

    all_candles = []
    current_start = start_ms

    print(f"  📊 Downloading {days}d of {interval} klines for {symbol}...")

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1500
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                print("  ⚠️ Rate limited, sleeping 30s...")
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

            # Interval step sizes in ms
            interval_ms = {
                "1m": 60_000, "5m": 300_000, "15m": 900_000,
                "1h": 3_600_000, "4h": 14_400_000
            }
            step = interval_ms.get(interval, 300_000)
            current_start = data[-1][0] + step
            time.sleep(0.1)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            time.sleep(5)
            current_start += 1500 * 300_000  # skip ahead

    df = pd.DataFrame(all_candles)
    print(f"  ✅ Got {len(df)} candles")
    return df


# ═══════════════════════════════════════════════════════
#  S/R Level Detection (mirrors level_tracker.rs logic)
# ═══════════════════════════════════════════════════════

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

    if len(tr) < period:
        return np.mean(tr) if len(tr) > 0 else 0.0
    return np.mean(tr[-period:])


def find_sr_levels(candles_1h: pd.DataFrame, atr: float) -> List[Dict]:
    """
    Find S/R levels via touch clustering (same algo as level_tracker.rs).
    Groups highs/lows within ±0.3 ATR into clusters.
    """
    if len(candles_1h) < 10 or atr <= 0:
        return []

    cluster_dist = atr * SR_CLUSTER_ATR_MULT
    extremes = []

    for _, c in candles_1h.iterrows():
        extremes.append(c["high"])
        extremes.append(c["low"])

    extremes.sort()

    # Cluster nearby prices
    levels = []
    i = 0
    while i < len(extremes):
        cluster = [extremes[i]]
        j = i + 1
        while j < len(extremes) and extremes[j] - extremes[i] <= cluster_dist:
            cluster.append(extremes[j])
            j += 1

        if len(cluster) >= SR_MIN_TOUCHES:
            avg_price = np.mean(cluster)
            levels.append({
                "price": round(float(avg_price), 8),
                "touches": len(cluster),
                "weight": len(cluster) * avg_price,  # volume-weighted proxy
            })

        i = j if j > i + 1 else i + 1

    # Sort by touches (strongest first)
    levels.sort(key=lambda x: x["touches"], reverse=True)

    # Deduplicate close levels
    filtered = []
    for lev in levels:
        too_close = False
        for existing in filtered:
            if abs(lev["price"] - existing["price"]) / existing["price"] < 0.003:
                too_close = True
                break
        if not too_close:
            filtered.append(lev)

    return filtered[:20]  # Top 20 levels max


# ═══════════════════════════════════════════════════════
#  Episode Detection
# ═══════════════════════════════════════════════════════

def find_episodes_near_level(
    candles_5m: pd.DataFrame,
    level: Dict,
    existing_episodes: List[Dict],
) -> List[Dict]:
    """
    Find consolidation episodes near a S/R level.
    An episode = price stays within ±1% of level for ≥30 minutes.
    """
    level_price = level["price"]
    zone_high = level_price * (1 + EPISODE_ZONE_PCT)
    zone_low = level_price * (1 - EPISODE_ZONE_PCT)

    episodes = []
    episode_start_idx = None
    episode_candles = []
    gap_count = 0

    for idx in range(len(candles_5m)):
        row = candles_5m.iloc[idx]
        price = row["close"]
        in_zone = zone_low <= price <= zone_high

        if in_zone:
            if episode_start_idx is None:
                episode_start_idx = idx
                episode_candles = []
                gap_count = 0

            episode_candles.append(idx)
            gap_count = 0  # Reset gap counter

        else:
            if episode_start_idx is not None:
                gap_count += 1
                # Allow small excursions (3 candles × 5m = 15 min)
                if gap_count > EPISODE_MAX_GAP_MINS // 5:
                    # Episode ended — evaluate it
                    ep = _evaluate_episode(
                        candles_5m, episode_candles, level, existing_episodes
                    )
                    if ep is not None:
                        episodes.append(ep)

                    episode_start_idx = None
                    episode_candles = []
                    gap_count = 0

        # Cap episode length
        if (episode_start_idx is not None and
                len(episode_candles) > EPISODE_MAX_DURATION_MINS // 5):
            ep = _evaluate_episode(
                candles_5m, episode_candles, level, existing_episodes
            )
            if ep is not None:
                episodes.append(ep)
            episode_start_idx = None
            episode_candles = []
            gap_count = 0

    # Handle episode still open at end of data
    if episode_start_idx is not None and len(episode_candles) >= EPISODE_MIN_CANDLES:
        ep = _evaluate_episode(
            candles_5m, episode_candles, level, existing_episodes
        )
        if ep is not None:
            episodes.append(ep)

    return episodes


def _evaluate_episode(
    candles_5m: pd.DataFrame,
    candle_indices: List[int],
    level: Dict,
    existing_episodes: List[Dict],
) -> Optional[Dict]:
    """Evaluate a consolidation episode and return it if it qualifies."""
    if len(candle_indices) < EPISODE_MIN_CANDLES:
        return None

    first_idx = candle_indices[0]
    last_idx = candle_indices[-1]

    first_candle = candles_5m.iloc[first_idx]
    last_candle = candles_5m.iloc[last_idx]

    start_ts = int(first_candle["ts_ms"])
    end_ts = int(last_candle["ts_ms"])
    duration_mins = (end_ts - start_ts) / 60_000

    if duration_mins < EPISODE_MIN_DURATION_MINS:
        return None

    # Check overlap with existing episodes (avoid duplicates)
    for existing in existing_episodes:
        if (abs(existing["start_ts"] - start_ts) < 600_000 and  # 10 min overlap
                abs(existing["level_price"] - level["price"]) / level["price"] < 0.005):
            return None

    # Calculate box metrics
    episode_data = candles_5m.iloc[candle_indices]
    box_high = float(episode_data["high"].max())
    box_low = float(episode_data["low"].min())
    box_width_pct = (box_high - box_low) / level["price"]

    # Volume trend: slope of volume over the episode
    volumes = episode_data["volume"].values
    if len(volumes) >= 3:
        x = np.arange(len(volumes))
        volume_slope = float(np.polyfit(x, volumes, 1)[0])
        avg_volume = float(np.mean(volumes))
        volume_trend = volume_slope / avg_volume if avg_volume > 0 else 0.0
    else:
        volume_trend = 0.0

    # Which side is the level? (resistance = above, support = below)
    avg_price = float(episode_data["close"].mean())
    side = "resistance" if avg_price < level["price"] else "support"

    # Count touches: how many times price came within 0.2% of level and reversed
    touch_count = 0
    for _, row in episode_data.iterrows():
        dist_to_level = abs(row["close"] - level["price"]) / level["price"]
        if dist_to_level < 0.002:
            touch_count += 1

    # Determine outcome (if we have enough data after)
    outcome = _classify_outcome(candles_5m, last_idx, level, side)

    episode = {
        "symbol": None,  # Will be filled by caller
        "level_price": round(level["price"], 8),
        "level_touches": level["touches"],
        "side": side,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_mins": round(duration_mins, 1),
        "box_high": round(box_high, 8),
        "box_low": round(box_low, 8),
        "box_width_pct": round(box_width_pct, 5),
        "episode_touches": touch_count,
        "volume_trend": round(volume_trend, 6),
        "num_candles": len(candle_indices),
        "outcome": outcome,
    }

    return episode


def _classify_outcome(
    candles_5m: pd.DataFrame,
    last_idx: int,
    level: Dict,
    side: str,
) -> Optional[str]:
    """
    Classify episode outcome by looking 30 min ahead.
    BROKE = breakout through level (≥0.5% beyond)
    REJECTED = bounced back away from level
    FAKE = broke through but returned within 10 min
    """
    outcome_candles = OUTCOME_WINDOW_MINS // 5  # 6 candles of 5m

    if last_idx + outcome_candles >= len(candles_5m):
        return None  # Not enough data

    future = candles_5m.iloc[last_idx + 1 : last_idx + 1 + outcome_candles]
    level_price = level["price"]

    if side == "resistance":
        # Price was below level, check if it broke above
        max_price = float(future["high"].max())
        break_pct = (max_price - level_price) / level_price

        if break_pct > 0.005:  # Broke >0.5% above
            # Check for fakeout: did it come back below?
            # Look at last 2 candles (10 min)
            late_close = float(future.iloc[-1]["close"]) if len(future) > 0 else max_price
            if late_close < level_price:
                return "FAKE"
            return "BROKE"
        else:
            return "REJECTED"

    else:  # support
        # Price was above level, check if it broke below
        min_price = float(future["low"].min())
        break_pct = (level_price - min_price) / level_price

        if break_pct > 0.005:
            late_close = float(future.iloc[-1]["close"]) if len(future) > 0 else min_price
            if late_close > level_price:
                return "FAKE"
            return "BROKE"
        else:
            return "REJECTED"


# ═══════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════

def find_density_episodes(symbol: str, days: int = 30) -> List[Dict]:
    """
    Full pipeline: download → levels → episodes → save.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Finding density episodes for {symbol} ({days} days)")
    print(f"{'='*60}")

    # Step 1: Download 5m candles
    cache_5m = CACHE_DIR / f"{symbol}_5m_{days}d.csv"
    if cache_5m.exists() and (time.time() - cache_5m.stat().st_mtime) < 3600:
        print(f"  📂 Using cached 5m klines")
        df_5m = pd.read_csv(cache_5m)
    else:
        df_5m = download_klines(symbol, "5m", days)
        if df_5m.empty:
            print(f"  ❌ No data for {symbol}")
            return []
        df_5m.to_csv(cache_5m, index=False)

    if len(df_5m) < 100:
        print(f"  ❌ Not enough data: {len(df_5m)} candles")
        return []

    # Step 2: Aggregate 5m → 1H for S/R levels
    print("  📐 Aggregating 5m → 1H for S/R detection...")
    df_5m["hour_group"] = df_5m["ts_ms"] // 3_600_000

    df_1h = df_5m.groupby("hour_group").agg({
        "ts_ms": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index(drop=True)

    print(f"  ✅ Got {len(df_1h)} hourly candles")

    # Step 3: Find S/R levels
    atr = calc_atr(df_1h)
    levels = find_sr_levels(df_1h, atr)
    print(f"  🎯 Found {len(levels)} S/R levels (ATR={atr:.6f})")

    for i, lev in enumerate(levels[:5]):
        print(f"    [{i+1}] {lev['price']:.6f} ({lev['touches']} touches)")

    # Step 4: Find consolidation episodes near each level
    all_episodes = []

    for lev in levels:
        eps = find_episodes_near_level(df_5m, lev, all_episodes)
        for ep in eps:
            ep["symbol"] = symbol
        all_episodes.extend(eps)

    # Step 5: Statistics
    outcomes = {}
    for ep in all_episodes:
        o = ep.get("outcome")
        if o is None:
            o = "UNKNOWN"
        outcomes[o] = outcomes.get(o, 0) + 1

    print(f"\n  📊 Found {len(all_episodes)} density episodes")
    for k, v in sorted(outcomes.items()):
        print(f"    {k}: {v}")

    # Step 6: Save
    out_path = EPISODES_DIR / f"{symbol}_episodes.json"
    with open(out_path, "w") as f:
        json.dump(all_episodes, f, indent=2)
    print(f"  💾 Saved to {out_path}")

    return all_episodes


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
                symbols.append(s['symbol'])

        # Exclude high-cap (too efficient for density play)
        exclude = {'BTCUSDT', 'ETHUSDT'}
        symbols = [s for s in symbols if s not in exclude]

        print(f"📋 Found {len(symbols)} tradeable USDT perpetual symbols")
        return sorted(symbols)

    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 30.1: Density Episode Finder")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol (e.g. ENJUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Days of data (default: 30)")
    parser.add_argument("--all-symbols", action="store_true", help="Run on all Binance futures")
    parser.add_argument("--top", type=int, default=50, help="Only top N symbols by volume (with --all-symbols)")
    args = parser.parse_args()

    if args.all_symbols:
        symbols = get_all_futures_symbols()
        if args.top:
            symbols = symbols[:args.top]  # TODO: sort by volume

        summary = {}
        for i, sym in enumerate(symbols):
            episodes = find_density_episodes(sym, args.days)
            summary[sym] = {
                "total": len(episodes),
                "broke": sum(1 for e in episodes if e.get("outcome") == "BROKE"),
                "rejected": sum(1 for e in episodes if e.get("outcome") == "REJECTED"),
                "fake": sum(1 for e in episodes if e.get("outcome") == "FAKE"),
            }
            time.sleep(1)

        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(summary)} symbols processed")
        print(f"{'='*60}")

        total_ep = sum(s["total"] for s in summary.values())
        total_broke = sum(s["broke"] for s in summary.values())
        total_rejected = sum(s["rejected"] for s in summary.values())
        total_fake = sum(s["fake"] for s in summary.values())
        print(f"Total episodes: {total_ep}")
        print(f"  BROKE: {total_broke} ({total_broke/max(total_ep,1)*100:.0f}%)")
        print(f"  REJECTED: {total_rejected} ({total_rejected/max(total_ep,1)*100:.0f}%)")
        print(f"  FAKE: {total_fake} ({total_fake/max(total_ep,1)*100:.0f}%)")

        with open(EPISODES_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    elif args.symbol:
        find_density_episodes(args.symbol, args.days)
    else:
        print("Usage:")
        print("  python density_breakout/find_density_episodes.py --symbol ENJUSDT --days 30")
        print("  python density_breakout/find_density_episodes.py --all-symbols --days 30")
