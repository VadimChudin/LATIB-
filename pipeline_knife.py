"""
Pipeline: Knife Tick HFT (Batch Optimize → Config → Ready)
===========================================================
Runs DE optimization for each TOP symbol individually,
saves per-symbol best params into active_config.json.

Usage:  python pipeline_knife.py [--generations 200] [--skip-optimize]
"""
import os
import sys
import json
import subprocess
import argparse
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RUST_DIR = os.path.join(BASE_DIR, 'rust_engine')

BEST_PARAMS_PATH = os.path.join(DATA_DIR, 'ga_best_tick_params.json')
ACTIVE_CONFIG_PATH = os.path.join(DATA_DIR, 'active_config.json')
TICK_PARAMS_DIR = os.path.join(DATA_DIR, 'tick_params')

# Top-15 from batch optimization (OOS WR >= 55%, PnL > 0)
TOP_SYMBOLS = [
    "KITE_USDT", "WIF_USDT", "H_USDT", "XAN_USDT", "ALT_USDT",
    "STO_USDT", "CHR_USDT", "AIOT_USDT", "BANK_USDT", "GUA_USDT",
    "ZIL_USDT", "POLYX_USDT", "POWER_USDT", "ONT_USDT", "BARD_USDT",
]


def step1_optimize_all(generations: int):
    """Run Rust DE optimizer for each top symbol."""
    logger.info("=" * 60)
    logger.info("  STEP 1: Per-Symbol HFT Optimization (Rust DE)")
    logger.info("=" * 60)

    os.makedirs(TICK_PARAMS_DIR, exist_ok=True)
    results = []

    for i, sym in enumerate(TOP_SYMBOLS):
        logger.info(f"\n[{i+1}/{len(TOP_SYMBOLS)}] Optimizing {sym}...")

        cmd = [
            "cargo", "run", "--release", "--",
            "optimize-ticks",
            "--symbol", sym,
            "--direction", "ALL",
            "--generations", str(generations),
        ]

        result = subprocess.run(cmd, cwd=RUST_DIR, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"  ❌ {sym}: Optimizer failed")
            continue

        if not os.path.exists(BEST_PARAMS_PATH):
            logger.error(f"  ❌ {sym}: No params file")
            continue

        with open(BEST_PARAMS_PATH) as f:
            best = json.load(f)

        # Save per-symbol params
        sym_path = os.path.join(TICK_PARAMS_DIR, f"{sym}.json")
        with open(sym_path, 'w') as f:
            json.dump(best, f, indent=2)

        test_wr = best.get("test_wr", 0)
        test_pnl = best.get("test_pnl_r", 0)
        train_wr = best.get("train_wr", 0)
        total = best.get("train_trades", 0) + best.get("test_trades", 0)

        logger.info(f"  ✅ {sym}: Train {train_wr:.0f}% | Test {test_wr:.0f}% | PnL {test_pnl:+.2f}R | {total} trades")
        results.append((sym, best))

    logger.info(f"\n✅ Optimized {len(results)}/{len(TOP_SYMBOLS)} symbols")
    return results


def step2_build_config(results):
    """Build active_config.json from per-symbol optimization results."""
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 2: Build active_config.json")
    logger.info("=" * 60)

    configs = []
    now_str = datetime.now(timezone.utc).isoformat()

    for sym, best in results:
        params = best.get("params", {})
        test_wr = best.get("test_wr", 0)
        test_pnl = best.get("test_pnl_r", 0)
        train_wr = best.get("train_wr", 0)
        total = best.get("train_trades", 0) + best.get("test_trades", 0)

        # Skip symbols with bad OOS
        if test_wr < 40 or total < 5:
            logger.info(f"  ⏭️ {sym}: Skipped (Test WR={test_wr:.0f}%, trades={total})")
            continue

        # Tier based on OOS performance
        if test_wr >= 70 and test_pnl > 0:
            tier = 1
            leverage = 20
        elif test_wr >= 55 and test_pnl >= 0:
            tier = 2
            leverage = 10
        else:
            tier = 3
            leverage = 5

        configs.append({
            "symbol": sym,
            "timeframe": "tick",
            "strategy": "knife_tick",
            "tier": tier,
            "leverage": leverage,
            "params": params,
            "metrics": {
                "win_rate": test_wr,
                "train_wr": train_wr,
                "test_pnl_r": test_pnl,
                "total_trades": total,
                "fitness": best.get("fitness", 0),
            },
            "evaluated_at": now_str,
            "version": "phase31",
        })

    # Sort by tier then test_wr
    configs.sort(key=lambda c: (c["tier"], -c["metrics"]["win_rate"]))

    with open(ACTIVE_CONFIG_PATH, 'w') as f:
        json.dump(configs, f, indent=2)

    t1 = sum(1 for c in configs if c["tier"] == 1)
    t2 = sum(1 for c in configs if c["tier"] == 2)
    t3 = sum(1 for c in configs if c["tier"] == 3)
    logger.info(f"  ✅ Saved {len(configs)} entries: Tier1={t1} Tier2={t2} Tier3={t3}")


def main():
    parser = argparse.ArgumentParser(description="Knife Tick HFT Pipeline (Batch)")
    parser.add_argument('--generations', type=int, default=200, help='DE generations per symbol')
    parser.add_argument('--skip-optimize', action='store_true', help='Use existing tick_params/*.json')
    args = parser.parse_args()

    logger.info("🔪 KNIFE TICK HFT PIPELINE (BATCH)")
    logger.info(f"   Symbols: {len(TOP_SYMBOLS)} | Generations: {args.generations}")
    logger.info("")

    if not args.skip_optimize:
        results = step1_optimize_all(args.generations)
    else:
        logger.info("⏭️ Skipping optimization, loading from tick_params/")
        results = []
        for sym in TOP_SYMBOLS:
            sym_path = os.path.join(TICK_PARAMS_DIR, f"{sym}.json")
            if os.path.exists(sym_path):
                with open(sym_path) as f:
                    results.append((sym, json.load(f)))
            else:
                logger.warning(f"  ⚠️ {sym}: No cached params")

    if results:
        step2_build_config(results)

    logger.info("\n" + "=" * 60)
    logger.info("  ✅ PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
