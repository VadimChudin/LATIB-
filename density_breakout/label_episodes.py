"""
Phase 30.3: Label Density Episodes
====================================
Reads downloaded episode data and labels each episode:
  BROKE     — price broke through level and moved >0.5% beyond
  FAKE      — price broke through but returned within 10 min
  REJECTED  — price couldn't break, retreated from level

Also extracts features for RL training from body/head/book data.

Usage:
  python density_breakout/label_episodes.py --symbol ENJUSDT
  python density_breakout/label_episodes.py --all-symbols
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List

# ── Paths ──
EPISODES_DIR = Path("density_breakout/data/episodes")
HYBRID_DIR = Path("density_breakout/data/episodes_hybrid")
LABELED_DIR = Path("density_breakout/data/episodes_labeled")
LABELED_DIR.mkdir(parents=True, exist_ok=True)

# ── Labeling thresholds ──
BREAKOUT_PCT = 0.005    # 0.5% beyond level = breakout
FAKE_RETURN_MINS = 10   # If returns within 10 min = fake breakout


def label_single_episode(episode_dir: Path, meta: dict) -> Optional[dict]:
    """
    Label one episode using outcome data and extract features.
    """
    # Load available data
    body_path = episode_dir / "body_1m.csv"
    head_path = episode_dir / "head_ticks.csv"
    book_path = episode_dir / "book_depth.csv"
    outcome_path = episode_dir / "outcome_1m.csv"

    if not outcome_path.exists():
        return None  # Can't label without outcome

    outcome_df = pd.read_csv(outcome_path)
    if outcome_df.empty:
        return None

    level_price = meta["level_price"]
    side = meta["side"]

    # ── Classify outcome ──
    label = _classify_outcome(outcome_df, level_price, side)

    # ── Extract features ──
    features = {}

    # Body features (from 1m consolidation candles)
    if body_path.exists():
        body_df = pd.read_csv(body_path)
        if not body_df.empty:
            features.update(_extract_body_features(body_df, level_price))

    # Head features (from tick-level data near level)
    if head_path.exists():
        head_df = pd.read_csv(head_path)
        if not head_df.empty:
            features.update(_extract_head_features(head_df, level_price, side))

    # Book depth features (from L2 snapshots)
    if book_path.exists():
        book_df = pd.read_csv(book_path)
        if not book_df.empty:
            features.update(_extract_book_features(book_df, level_price, side))

    # Meta features
    features["box_width_pct"] = meta.get("box_width_pct", 0)
    features["duration_mins"] = meta.get("duration_mins", 0)
    features["episode_touches"] = meta.get("episode_touches", 0)
    features["volume_trend"] = meta.get("volume_trend", 0)
    features["level_touches"] = meta.get("level_touches", 0)

    return {
        "symbol": meta.get("symbol"),
        "start_ts": meta["start_ts"],
        "end_ts": meta["end_ts"],
        "level_price": level_price,
        "side": side,
        "label": label,
        "features": features,
    }


def _classify_outcome(
    outcome_df: pd.DataFrame, level_price: float, side: str
) -> str:
    """Classify episode outcome from 1m outcome candles."""
    if side == "resistance":
        max_price = float(outcome_df["high"].max())
        break_pct = (max_price - level_price) / level_price

        if break_pct > BREAKOUT_PCT:
            # Check for fakeout
            n_fake_candles = FAKE_RETURN_MINS  # 10 candles of 1m
            if len(outcome_df) >= n_fake_candles:
                late = outcome_df.iloc[:n_fake_candles]
                # If price broke above but then all late closes are below level
                late_closes_below = (late["close"] < level_price).sum()
                if late_closes_below >= n_fake_candles * 0.6:
                    return "FAKE"
            return "BROKE"
        return "REJECTED"

    else:  # support
        min_price = float(outcome_df["low"].min())
        break_pct = (level_price - min_price) / level_price

        if break_pct > BREAKOUT_PCT:
            n_fake_candles = FAKE_RETURN_MINS
            if len(outcome_df) >= n_fake_candles:
                late = outcome_df.iloc[:n_fake_candles]
                late_closes_above = (late["close"] > level_price).sum()
                if late_closes_above >= n_fake_candles * 0.6:
                    return "FAKE"
            return "BROKE"
        return "REJECTED"


def _extract_body_features(body_df: pd.DataFrame, level_price: float) -> dict:
    """Extract features from 1m consolidation candles."""
    features = {}

    closes = body_df["close"].values
    volumes = body_df["volume"].values
    highs = body_df["high"].values
    lows = body_df["low"].values

    # Volume trend slope (normalized)
    if len(volumes) >= 3:
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes, 1)[0]
        avg_vol = np.mean(volumes)
        features["body_volume_slope"] = float(slope / avg_vol) if avg_vol > 0 else 0.0
    else:
        features["body_volume_slope"] = 0.0

    # Price return kurtosis (fat tails = big move coming)
    returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0])
    features["price_return_kurtosis"] = float(
        pd.Series(returns).kurtosis()
    ) if len(returns) > 3 else 0.0

    # Volume autocorrelation (lag 1-5 min)
    vol_series = pd.Series(volumes)
    features["volume_autocorrelation"] = float(
        vol_series.autocorr(lag=1)
    ) if len(volumes) > 5 else 0.0

    # POC distance from level (where most volume traded)
    # Bucket prices to find POC
    if len(closes) > 5:
        price_bins = np.linspace(min(lows), max(highs), 20)
        vol_profile = np.zeros(len(price_bins) - 1)
        for i in range(len(closes)):
            bin_idx = np.searchsorted(price_bins, closes[i]) - 1
            bin_idx = max(0, min(bin_idx, len(vol_profile) - 1))
            vol_profile[bin_idx] += volumes[i]
        poc_idx = np.argmax(vol_profile)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        features["poc_to_level_dist"] = float(
            (poc_price - level_price) / level_price
        )
    else:
        features["poc_to_level_dist"] = 0.0

    # Squeeze ratio: ratio of last 3 candle ranges vs first 3
    if len(highs) >= 6:
        first_ranges = highs[:3] - lows[:3]
        last_ranges = highs[-3:] - lows[-3:]
        avg_first = np.mean(first_ranges)
        avg_last = np.mean(last_ranges)
        features["squeeze_ratio"] = float(
            avg_last / avg_first
        ) if avg_first > 0 else 1.0
    else:
        features["squeeze_ratio"] = 1.0

    # Taker ratio at level
    if "taker_buy_volume" in body_df.columns:
        total_vol = body_df["volume"].sum()
        taker_buy = body_df["taker_buy_volume"].sum()
        features["taker_ratio"] = float(
            taker_buy / total_vol
        ) if total_vol > 0 else 0.5
    else:
        features["taker_ratio"] = 0.5

    return features


def _extract_head_features(
    head_df: pd.DataFrame, level_price: float, side: str
) -> dict:
    """Extract features from tick data near the level."""
    features = {}

    if head_df.empty:
        return {"tick_speed_accel": 0, "trade_size_entropy": 0, "buy_sell_clustering": 0}

    # Tick speed acceleration (2nd derivative)
    if len(head_df) > 10:
        ts = head_df["ts_ms"].values
        # Count trades per second
        bins = np.arange(ts[0], ts[-1] + 1000, 1000)
        counts, _ = np.histogram(ts, bins=bins)
        if len(counts) > 3:
            speed = np.diff(counts.astype(float))
            accel = np.diff(speed)
            features["tick_speed_accel"] = float(np.mean(accel[-5:])) if len(accel) >= 5 else 0.0
        else:
            features["tick_speed_accel"] = 0.0
    else:
        features["tick_speed_accel"] = 0.0

    # Trade size entropy
    qty = head_df["qty"].values
    if len(qty) > 5:
        hist, _ = np.histogram(qty, bins=20)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        features["trade_size_entropy"] = float(-np.sum(probs * np.log2(probs)))
    else:
        features["trade_size_entropy"] = 0.0

    # Buy/sell clustering (serial correlation of buy/sell)
    if "is_buyer_maker" in head_df.columns and len(head_df) > 5:
        bm = head_df["is_buyer_maker"].astype(int).values
        # Count runs
        runs = 1 + np.sum(np.diff(bm) != 0)
        expected_runs = 1 + 2 * np.sum(bm) * np.sum(1 - bm) / len(bm)
        features["buy_sell_clustering"] = float(
            runs / expected_runs
        ) if expected_runs > 0 else 1.0
    else:
        features["buy_sell_clustering"] = 1.0

    return features


def _extract_book_features(
    book_df: pd.DataFrame, level_price: float, side: str
) -> dict:
    """
    Extract features from L2 bookDepth snapshots.
    
    Real Binance bookDepth format:
      ts_ms, percentage, depth, notional
    - percentage: distance from mid price in % (-5,-4,...,+4,+5)
      Negative = bid side, Positive = ask side
    - notional: cumulative USD at that depth level
    """
    features = {}

    if book_df.empty or len(book_df) < 2:
        return {
            "wall_size_usd": 0, "wall_eaten_pct": 0,
            "book_depth_behind": 0, "book_imbalance": 0
        }

    try:
        # Wall size: notional at ±1% level (closest to the level being tested)
        # For resistance: ask side (+1%) is the wall
        # For support: bid side (-1%) is the wall
        wall_pct = 1.0 if side == "resistance" else -1.0
        wall_rows = book_df[book_df["percentage"] == wall_pct]

        if not wall_rows.empty:
            # Last snapshot wall size
            last_ts = wall_rows["ts_ms"].max()
            last_wall = wall_rows[wall_rows["ts_ms"] == last_ts]["notional"]
            features["wall_size_usd"] = float(last_wall.values[0]) if len(last_wall) > 0 else 0.0

            # Wall eaten pct: first vs last snapshot
            first_ts = wall_rows["ts_ms"].min()
            first_wall = wall_rows[wall_rows["ts_ms"] == first_ts]["notional"]
            first_val = float(first_wall.values[0]) if len(first_wall) > 0 else 0.0
            last_val = features["wall_size_usd"]
            features["wall_eaten_pct"] = float(
                1.0 - last_val / first_val
            ) if first_val > 0 else 0.0
        else:
            features["wall_size_usd"] = 0.0
            features["wall_eaten_pct"] = 0.0

        # Book depth behind level: liquidity BEYOND the wall (deeper levels)
        # For resistance: ask at +2%, +3%, +4%, +5%
        # For support: bid at -2%, -3%, -4%, -5%
        if side == "resistance":
            behind = book_df[book_df["percentage"] >= 2.0]
        else:
            behind = book_df[book_df["percentage"] <= -2.0]

        if not behind.empty:
            last_ts = behind["ts_ms"].max()
            behind_snap = behind[behind["ts_ms"] == last_ts]
            # Max notional at deepest level = total cumulative depth
            features["book_depth_behind"] = float(behind_snap["notional"].max())
        else:
            features["book_depth_behind"] = 0.0

        # Book imbalance: bid total vs ask total at ±1%
        last_ts = book_df["ts_ms"].max()
        last_snap = book_df[book_df["ts_ms"] == last_ts]
        
        bid_notional = last_snap[last_snap["percentage"] == -1.0]["notional"]
        ask_notional = last_snap[last_snap["percentage"] == 1.0]["notional"]
        
        bid_val = float(bid_notional.values[0]) if len(bid_notional) > 0 else 0
        ask_val = float(ask_notional.values[0]) if len(ask_notional) > 0 else 0
        total = bid_val + ask_val
        features["book_imbalance"] = float(
            (bid_val - ask_val) / total
        ) if total > 0 else 0.0

    except Exception as e:
        features.setdefault("wall_size_usd", 0.0)
        features.setdefault("wall_eaten_pct", 0.0)
        features.setdefault("book_depth_behind", 0.0)
        features.setdefault("book_imbalance", 0.0)

    return features


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def label_symbol(symbol: str):
    """Label all episodes for a symbol."""
    symbol_dir = HYBRID_DIR / symbol
    if not symbol_dir.exists():
        print(f"  No hybrid data for {symbol}. Run download_episodes.py first.")
        return

    episode_dirs = [d for d in symbol_dir.iterdir() if d.is_dir()]
    print(f"\nLabeling {symbol}: {len(episode_dirs)} episodes")

    labeled = []

    for ep_dir in sorted(episode_dirs):
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        result = label_single_episode(ep_dir, meta)
        if result:
            labeled.append(result)

    # Save labeled dataset
    out_path = LABELED_DIR / f"{symbol}_labeled.json"
    with open(out_path, "w") as f:
        json.dump(labeled, f, indent=2)

    # Stats
    labels = {}
    for r in labeled:
        l = r["label"]
        labels[l] = labels.get(l, 0) + 1

    print(f"  Labeled {len(labeled)} episodes:")
    for k, v in sorted(labels.items()):
        print(f"    {k}: {v} ({v/len(labeled)*100:.0f}%)")
    print(f"  Saved to {out_path}")

    # Also generate flat CSV for quick analysis
    flat_rows = []
    for r in labeled:
        row = {
            "symbol": r["symbol"],
            "start_ts": r["start_ts"],
            "level_price": r["level_price"],
            "side": r["side"],
            "label": r["label"],
        }
        row.update(r.get("features", {}))
        flat_rows.append(row)

    if flat_rows:
        flat_df = pd.DataFrame(flat_rows)
        flat_df.to_csv(LABELED_DIR / f"{symbol}_labeled.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 30.3: Label Density Episodes")
    parser.add_argument("--symbol", type=str, help="Single symbol")
    parser.add_argument("--all-symbols", action="store_true", help="Label all symbols")
    args = parser.parse_args()

    if args.all_symbols:
        for sym_dir in sorted(HYBRID_DIR.iterdir()):
            if sym_dir.is_dir():
                label_symbol(sym_dir.name)
    elif args.symbol:
        label_symbol(args.symbol)
    else:
        print("Usage:")
        print("  python density_breakout/label_episodes.py --symbol ENJUSDT")
        print("  python density_breakout/label_episodes.py --all-symbols")
