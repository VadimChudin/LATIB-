"""
AEGIS Phase 11.4 — Journal Aggregator & Meta-Label Pipeline
============================================================
Reads `data/trade_log.jsonl` (produced by Rust TradeLogger),
computes rolling metrics, trains a Meta-Model when enough data
exists (>= 20 trades), and exports it for Rust inference.

Usage:
    python aggregate_journal.py          # Analyze + train if enough data
    python aggregate_journal.py --stats  # Print stats only (no training)

Auto-skip: If trade_log.jsonl doesn't exist or has < 20 EXIT records,
           training is skipped gracefully.
"""

import json
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AggregateJournal")

# ── Configuration ────────────────────────────────────────────────────────────

TRADE_LOG_PATH = "data/trade_log.jsonl"
STATS_OUTPUT_PATH = "data/journal_stats.json"
META_MODEL_PATH = "data/models/meta_model.joblib"
META_MODEL_JSON = "data/models/meta_model_info.json"
BLACKLIST_PATH = "data/coin_blacklist.json"

MIN_TRADES_FOR_TRAINING = 20      # Minimum EXIT records to train Meta-Model
BLACKLIST_LOSS_STREAK = 3         # Consecutive losses to trigger blacklist
BLACKLIST_COOLDOWN_HOURS = 12     # Hours a coin stays blacklisted
ROLLING_WINDOW_7D = 7 * 24 * 3600  # 7 days in seconds

# ── Load Trades ──────────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    """Load trade log JSONL into a DataFrame. Returns empty DF if file missing."""
    if not os.path.exists(TRADE_LOG_PATH):
        logger.warning(f"📝 {TRADE_LOG_PATH} not found. No trades recorded yet.")
        return pd.DataFrame()

    records = []
    with open(TRADE_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        logger.warning("📝 Trade log is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(f"📝 Loaded {len(df)} records from trade log "
                f"({len(df[df['event'] == 'ENTRY'])} entries, "
                f"{len(df[df['event'] == 'EXIT'])} exits)")
    return df


# ── Rolling Metrics ──────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> dict:
    """Compute rolling aggregate metrics from trade exits."""
    exits = df[df["event"] == "EXIT"].copy()

    if exits.empty:
        return {"total_trades": 0, "message": "No completed trades yet"}

    stats = {
        "total_trades": len(exits),
        "total_pnl_r": round(float(exits["pnl_r"].sum()), 2),
        "avg_pnl_r": round(float(exits["pnl_r"].mean()), 4),
        "win_rate": round(float((exits["pnl_r"] > 0).mean()), 4),
        "avg_mfe_pct": round(float(exits["mfe_pct"].mean()), 4),
        "avg_mae_pct": round(float(exits["mae_pct"].mean()), 4),
        "avg_duration_secs": round(float(exits["duration_secs"].mean()), 0),
    }

    # Per-strategy stats
    strat_stats = {}
    for strat, group in exits.groupby("strategy"):
        strat_stats[strat] = {
            "trades": len(group),
            "win_rate": round(float((group["pnl_r"] > 0).mean()), 4),
            "avg_pnl_r": round(float(group["pnl_r"].mean()), 4),
            "total_pnl_r": round(float(group["pnl_r"].sum()), 2),
            "profit_factor": round(
                float(group[group["pnl_r"] > 0]["pnl_r"].sum()) /
                max(float(group[group["pnl_r"] < 0]["pnl_r"].abs().sum()), 0.01),
                2
            ),
        }
    stats["per_strategy"] = strat_stats

    # Per-coin stats
    coin_stats = {}
    for coin, group in exits.groupby("symbol"):
        coin_stats[coin] = {
            "trades": len(group),
            "win_rate": round(float((group["pnl_r"] > 0).mean()), 4),
            "total_pnl_r": round(float(group["pnl_r"].sum()), 2),
        }
    stats["per_coin"] = coin_stats

    # Exit reason distribution
    stats["exit_reasons"] = exits["exit_reason"].value_counts().to_dict()

    return stats


# ── Coin Blacklist ───────────────────────────────────────────────────────────

def update_blacklist(df: pd.DataFrame) -> dict:
    """
    Check for coins with >= BLACKLIST_LOSS_STREAK consecutive losses
    in the last 24h. Blacklist them for BLACKLIST_COOLDOWN_HOURS.
    """
    exits = df[df["event"] == "EXIT"].copy()
    if exits.empty:
        return {}

    blacklist = {}
    now_str = datetime.now(timezone.utc).isoformat()

    for coin, group in exits.groupby("symbol"):
        # Get last N trades for this coin, sorted by timestamp
        recent = group.sort_values("ts").tail(BLACKLIST_LOSS_STREAK + 2)
        pnls = recent["pnl_r"].tolist()

        # Check if the last BLACKLIST_LOSS_STREAK trades are all losses
        if len(pnls) >= BLACKLIST_LOSS_STREAK:
            last_n = pnls[-BLACKLIST_LOSS_STREAK:]
            if all(p < 0 for p in last_n):
                expire = (datetime.now(timezone.utc) +
                          timedelta(hours=BLACKLIST_COOLDOWN_HOURS)).isoformat()
                blacklist[coin] = {
                    "reason": f"{BLACKLIST_LOSS_STREAK} consecutive losses",
                    "streak_pnl": [round(p, 2) for p in last_n],
                    "blacklisted_at": now_str,
                    "expires_at": expire,
                }
                logger.warning(f"🚫 BLACKLISTED: {coin} — {BLACKLIST_LOSS_STREAK} losses in a row "
                               f"(cooldown {BLACKLIST_COOLDOWN_HOURS}h)")

    # Load existing blacklist and merge (keep unexpired entries)
    existing = {}
    if os.path.exists(BLACKLIST_PATH):
        try:
            with open(BLACKLIST_PATH, "r") as f:
                existing = json.load(f)
        except Exception:
            pass

    # Remove expired entries from existing
    now = datetime.now(timezone.utc)
    for coin, info in list(existing.items()):
        try:
            expires = datetime.fromisoformat(info["expires_at"])
            if now > expires:
                logger.info(f"✅ UNBLACKLISTED: {coin} — cooldown expired")
                del existing[coin]
        except Exception:
            del existing[coin]

    # Merge new blacklist entries
    existing.update(blacklist)

    # Save
    os.makedirs("data", exist_ok=True)
    with open(BLACKLIST_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    if existing:
        logger.info(f"🚫 Active blacklist: {list(existing.keys())}")

    return existing


# ── Regime Detection ─────────────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame) -> dict:
    """
    Simple regime detection based on recent trade outcomes.
    Outputs regime labels that the Meta-Model can use as features.
    """
    exits = df[df["event"] == "EXIT"].copy()
    if len(exits) < 5:
        return {"regime": "unknown", "recent_winrate": None, "loss_streak": 0}

    recent = exits.tail(10)
    wins = (recent["pnl_r"] > 0).sum()
    winrate = wins / len(recent)

    # Calculate current loss streak
    pnls = exits["pnl_r"].tolist()
    loss_streak = 0
    for p in reversed(pnls):
        if p < 0:
            loss_streak += 1
        else:
            break

    # Regime classification
    if winrate >= 0.6:
        regime = "favorable"
    elif winrate <= 0.3:
        regime = "hostile"
    else:
        regime = "neutral"

    return {
        "regime": regime,
        "recent_winrate": round(float(winrate), 2),
        "loss_streak": loss_streak,
        "last_10_avg_pnl_r": round(float(recent["pnl_r"].mean()), 4),
    }


# ── Meta-Model Training ─────────────────────────────────────────────────────

def train_meta_model(df: pd.DataFrame) -> bool:
    """
    Train a LightGBM Meta-Model to predict trade success.
    Target: is_win (1 if pnl_r > 0, else 0)
    Features: strategy encoded, spot_probe status, wall metrics, flow metrics
    
    Returns True if model was trained, False if skipped.
    """
    exits = df[df["event"] == "EXIT"]
    entries = df[df["event"] == "ENTRY"]

    if len(exits) < MIN_TRADES_FOR_TRAINING:
        logger.info(f"⏭️  Meta-Model training SKIPPED: only {len(exits)} trades "
                    f"(need {MIN_TRADES_FOR_TRAINING}). Collecting more data...")
        return False

    # Merge entry features with exit outcomes
    # Match by trade_id
    merged = entries.merge(
        exits[["trade_id", "pnl_r", "exit_reason", "mfe_pct", "mae_pct", "duration_secs"]],
        on="trade_id",
        how="inner",
        suffixes=("", "_exit"),
    )

    if len(merged) < MIN_TRADES_FOR_TRAINING:
        logger.info(f"⏭️  Meta-Model: only {len(merged)} matched entry-exit pairs. Skipping.")
        return False

    logger.info(f"🧠 Training Meta-Model on {len(merged)} completed trades...")

    # Prepare features
    feature_cols = []

    # Encode strategy as numeric
    strat_map = {s: i for i, s in enumerate(merged["strategy"].unique())}
    merged["strategy_enc"] = merged["strategy"].map(strat_map)
    feature_cols.append("strategy_enc")

    # Spot probe encoding
    spot_map = {"confirmed": 2, "neutral": 1, "blocked": 0, "unavailable": -1}
    merged["spot_probe_enc"] = merged["spot_probe"].map(spot_map).fillna(-1)
    feature_cols.append("spot_probe_enc")

    # Numeric features from entry
    for col in ["wall_size_usd", "wall_age_h", "wall_eaten_pct",
                "cvd_delta", "imbalance_ratio", "tape_speed",
                "entry_price", "risk_dist"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
            feature_cols.append(col)

    # Target
    merged["is_win"] = (merged["pnl_r"] > 0).astype(int)

    X = merged[feature_cols].values
    y = merged["is_win"].values

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        import joblib

        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )

        # Cross-validate if enough data
        if len(merged) >= 30:
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(merged) // 5), scoring="accuracy")
            logger.info(f"🧠 Meta-Model CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Train on full data
        model.fit(X, y)

        # Save model
        os.makedirs("data/models", exist_ok=True)
        joblib.dump(model, META_MODEL_PATH)
        logger.info(f"🧠 Meta-Model saved to {META_MODEL_PATH}")

        # Save feature importance
        importances = dict(zip(feature_cols, model.feature_importances_))
        logger.info(f"🧠 Feature importance: {importances}")

        # Save strategy map for inference
        meta_info = {
            "strategy_map": strat_map,
            "spot_probe_map": spot_map,
            "feature_cols": feature_cols,
            "train_size": len(merged),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(META_MODEL_JSON, "w") as f:
            json.dump(meta_info, f, indent=2)

        return True

    except ImportError as e:
        logger.warning(f"🧠 Meta-Model training requires sklearn: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    stats_only = "--stats" in sys.argv

    logger.info("=" * 50)
    logger.info("  AEGIS Journal Aggregator")
    logger.info("=" * 50)

    # Load trades
    df = load_trades()
    if df.empty:
        logger.info("⏭️  No trades to analyze. Waiting for data from live engine...")
        # Write empty stats
        os.makedirs("data", exist_ok=True)
        with open(STATS_OUTPUT_PATH, "w") as f:
            json.dump({"total_trades": 0, "message": "Waiting for trade data"}, f, indent=2)
        return

    # Compute stats
    stats = compute_stats(df)
    regime = detect_regime(df)
    stats["regime"] = regime

    # Print summary
    logger.info(f"\n📊 Overall: {stats['total_trades']} trades | "
                f"WR={stats.get('win_rate', 0) * 100:.1f}% | "
                f"PnL={stats.get('total_pnl_r', 0):.2f}R")
    logger.info(f"📊 Regime: {regime['regime']} | "
                f"Recent WR: {regime.get('recent_winrate', 'N/A')} | "
                f"Loss streak: {regime['loss_streak']}")

    if "per_strategy" in stats:
        for strat, s in stats["per_strategy"].items():
            logger.info(f"   📈 {strat}: {s['trades']} trades, "
                        f"WR={s['win_rate'] * 100:.1f}%, "
                        f"PF={s['profit_factor']:.2f}, "
                        f"PnL={s['total_pnl_r']:.2f}R")

    # Save stats
    os.makedirs("data", exist_ok=True)
    with open(STATS_OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"💾 Stats saved to {STATS_OUTPUT_PATH}")

    # Update blacklist
    blacklist = update_blacklist(df)
    stats["blacklisted_coins"] = list(blacklist.keys())

    if stats_only:
        logger.info("📊 Stats-only mode. Skipping Meta-Model training.")
        return

    # Train Meta-Model (auto-skips if < 20 trades)
    trained = train_meta_model(df)
    if trained:
        logger.info("✅ Meta-Model trained and exported successfully!")
    else:
        logger.info("⏭️  Meta-Model training deferred (not enough data yet)")


if __name__ == "__main__":
    main()
