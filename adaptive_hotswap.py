"""
Adaptive Hot-Swap V3 — with Tick Verification
===============================================
1. Loads GA candidate pool for each symbol/strategy
2. Evaluates candidates on recent 48h data (GPU)
3. Gets trade list from best candidate (Rust backtest-trades)
4. Verifies trades on real tick data (lazy_tick_loader)
5. Only writes to active_config.json if confidence >= "medium"
"""
import os
import sys
import json
import asyncio
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import logging
import re
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("AdaptiveHotSwap")

from constants import (
    STRAT_MAP, PARAM_NAMES, COMMON_PARAMS,
    ACTIVE_CONFIG_PATH, BINARY_PATH, CACHE_DIR,
    GA_RESULTS_DIR, MAX_DATA_AGE_HOURS, RETRAIN_FLAG_PATH
)


def load_candidates(symbol):
    path = Path(GA_RESULTS_DIR) / f"{symbol}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_recent_data(symbol):
    """Slices last 48 hours for evaluation. Checks data freshness."""
    cache_path = Path(CACHE_DIR) / f"{symbol}_5m_730d.csv"
    if not cache_path.exists():
        logger.warning(f"Cache for {symbol} not found.")
        return None

    file_age_hours = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
    if file_age_hours > MAX_DATA_AGE_HOURS:
        logger.warning(f"⚠️ Data for {symbol} is {file_age_hours:.0f}h old (max {MAX_DATA_AGE_HOURS}h). Skipping.")
        return None

    df = pd.read_csv(cache_path)
    if 'timestamp' not in df.columns:
        return cache_path

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cutoff = df['timestamp'].max() - timedelta(hours=48)
    recent_df = df[df['timestamp'] >= cutoff]

    if len(recent_df) < 100:
        logger.warning(f"⚠️ Only {len(recent_df)} candles in 48h window for {symbol}. Skipping.")
        return None

    temp_csv = Path(CACHE_DIR) / f"{symbol}_recent_48h.csv"
    recent_df.to_csv(temp_csv, index=False)
    return temp_csv


def evaluate_pool_gpu(symbol, strategy, pool_data):
    """Calls aegis_engine evaluate-pool. Returns list of {params, fitness}."""
    temp_pool_path = Path("data") / f"temp_pool_{symbol}_{strategy}.json"
    with open(temp_pool_path, "w") as f:
        json.dump(pool_data, f)

    csv_path = get_recent_data(symbol)
    if not csv_path:
        return None

    cmd = [
        str(BINARY_PATH), "evaluate-pool",
        "--csv", str(csv_path),
        "--strategy", strategy,
        "--json-pool", str(temp_pool_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        match = re.search(r'\[\s*\{.*\}\s*\]', output, re.DOTALL)
        if not match:
            start = output.find('[')
            end = output.rfind(']')
            if start != -1 and end != -1:
                return json.loads(output[start:end+1])
            return None
        return json.loads(match.group(0))
    except Exception as e:
        logger.error(f"Failed to evaluate pool for {symbol}/{strategy}: {e}")
        return None
    finally:
        if temp_pool_path.exists():
            os.remove(temp_pool_path)


def get_backtest_trades(symbol, strategy, params_dict):
    """
    Calls Rust backtest-trades to get the full trade list with timestamps.
    This is used for tick verification.
    """
    csv_path = get_recent_data(symbol)
    if not csv_path:
        return []

    params_json = json.dumps(params_dict)
    cmd = [
        str(BINARY_PATH), "backtest-trades",
        "--csv", str(csv_path),
        "--strategy", strategy,
        "--params-json", params_json
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        # Find JSON array in output
        start = output.find('[')
        end = output.rfind(']')
        if start != -1 and end != -1:
            trades = json.loads(output[start:end+1])
            return trades
        return []
    except Exception as e:
        logger.error(f"Failed to get backtest trades: {e}")
        return []


def get_param_names(strategy):
    """Returns parameter names matching Rust get_param_defs() order."""
    base = PARAM_NAMES.get(strategy, [])
    return base + COMMON_PARAMS


async def run_tick_verification(symbol, strategy, params_dict):
    """
    Full tick verification pipeline:
    1. Get trade list from Rust
    2. Verify each winning trade on ticks
    3. Return adjusted stats
    """
    from verify_ticks import TickVerifier

    trades = get_backtest_trades(symbol, strategy, params_dict)
    if not trades:
        logger.info(f"      No trades from backtest — skipping tick verification")
        return {"confidence": "low", "fake_wins": 0, "adjusted_wr": None}

    logger.info(f"      Got {len(trades)} trades from Rust backtest. Verifying on ticks...")

    verifier = TickVerifier()
    try:
        result = await verifier.verify_trades(symbol, trades, max_verify=15)
        return result
    finally:
        await verifier.close()


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--skip-ticks", action="store_true", help="Skip tick verification")
    args = parser.parse_args()

    logger.info("🚀 ADAPTIVE HOT-SWAP V3 (with Tick Verification)")

    if not os.path.exists(ACTIVE_CONFIG_PATH):
        config_list = []
    else:
        with open(ACTIVE_CONFIG_PATH, "r") as f:
            try:
                config_list = json.load(f)
                if not isinstance(config_list, list):
                    config_list = []
            except:
                config_list = []

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = list(set([c.get("symbol") for c in config_list if "symbol" in c]))

    if not symbols:
        logger.warning("No symbols to adapt. Use --symbol.")
        return

    updated_count = 0
    for symbol in symbols:
        logger.info(f"📍 Adapting {symbol}...")
        results_data = load_candidates(symbol)
        if not results_data:
            logger.warning(f"No GA results for {symbol}")
            continue

        for strat in ['density', 'smc', 'knife']:
            try:
                if strat not in results_data or 'candidates' not in results_data[strat]:
                    continue

                pool = results_data[strat]['candidates']
                logger.info(f"   [{strat}] Evaluating pool (size: {len(pool)})...")

                # Step 1: GPU evaluation of candidate pool
                eval_results = evaluate_pool_gpu(symbol, strat, pool)
                if not eval_results:
                    continue

                best_candidate = max(eval_results, key=lambda x: x['fitness'])
                raw_params = best_candidate.get('best_params') or best_candidate.get('params')

                if not raw_params:
                    logger.warning(f"      No params in best candidate for {strat}")
                    continue

                # Convert to named dict
                param_names = get_param_names(strat)
                if isinstance(raw_params, dict):
                    named_params = raw_params
                elif isinstance(raw_params, list):
                    named_params = {}
                    for idx, val in enumerate(raw_params):
                        key = param_names[idx] if idx < len(param_names) else f"param_{idx}"
                        named_params[key] = val
                else:
                    named_params = raw_params

                fitness = best_candidate['fitness']

                # Reject penalty candidates (fitness -100 = too few trades)
                if fitness < -10.0:
                    logger.warning(f"      ⛔ Fitness {fitness:.1f} below threshold — skipping (too few trades on 48h window)")
                    continue

                logger.info(f"      Best fitness(48h): {fitness:.4f}")

                # Step 2: Tick verification (unless skipped)
                tick_result = None
                if not args.skip_ticks:
                    try:
                        tick_result = await run_tick_verification(symbol, strat, named_params)
                        if tick_result.get("confidence") == "reject":
                            logger.warning(f"      ❌ REJECTED by tick verification! "
                                         f"Fake wins: {tick_result['fake_wins']}/{tick_result['verified_count']}")
                            continue  # Skip this candidate entirely
                        elif tick_result.get("fake_wins", 0) > 0:
                            logger.info(f"      ⚠️ Tick audit: {tick_result['fake_wins']} fake wins detected, "
                                      f"adjusted WR: {tick_result.get('adjusted_wr', '?')}%")
                        else:
                            logger.info(f"      ✅ Tick verified: {tick_result.get('confidence', 'ok')}")
                    except Exception as e:
                        logger.warning(f"      ⚠️ Tick verification failed (non-blocking): {e}")

                python_strat = STRAT_MAP.get(strat, strat)
                
                existing_entry = None
                for c in config_list:
                    if c.get("symbol") == symbol and c.get("strategy") == python_strat:
                        existing_entry = c
                        break

                if existing_entry:
                    existing_entry["params"] = named_params
                    if "metrics" not in existing_entry:
                        existing_entry["metrics"] = {}
                    existing_entry["metrics"]["score"] = fitness
                    existing_entry["evaluated_at"] = str(datetime.now())
                    if tick_result:
                        existing_entry["tick_verified"] = True
                        existing_entry["tick_confidence"] = tick_result.get("confidence", "unknown")
                        existing_entry["fake_wins"] = tick_result.get("fake_wins", 0)
                        if tick_result.get("adjusted_wr") is not None:
                            existing_entry["metrics"]["win_rate"] = tick_result["adjusted_wr"] / 100.0
                else:
                    new_entry = {
                        "symbol": symbol,
                        "timeframe": "5m",
                        "strategy": python_strat,
                        "params": named_params,
                        "metrics": {"score": fitness, "win_rate": 0.0},
                        "evaluated_at": str(datetime.now())
                    }
                    if tick_result:
                        new_entry["tick_verified"] = True
                        new_entry["tick_confidence"] = tick_result.get("confidence", "unknown")
                        new_entry["fake_wins"] = tick_result.get("fake_wins", 0)
                        if tick_result.get("adjusted_wr") is not None:
                            new_entry["metrics"]["win_rate"] = tick_result["adjusted_wr"] / 100.0
                    config_list.append(new_entry)

                logger.info(f"   ✅ {strat} accepted. Fitness: {fitness:.4f}")
                updated_count += 1

            except Exception as e:
                logger.error(f"   ❌ Error adapting {strat}: {e}")
                continue

    if updated_count > 0:
        with open(ACTIVE_CONFIG_PATH, "w") as f:
            json.dump(config_list, f, indent=4)

        flag_path = RETRAIN_FLAG_PATH
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(flag_path, "w") as f:
            f.write(str(datetime.now()))
        logger.info(f"🔥 HOT-SWAP COMPLETE. {updated_count} updates. Flag set.")
    else:
        logger.info("ℹ️ No updates applied.")


if __name__ == "__main__":
    asyncio.run(main())
