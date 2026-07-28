"""
AutoCore Engine (core/engine.py)
================================
Runs on a periodic schedule (every N hours). For each symbol in the watch list,
downloads fresh OHLCV data via CCXT and runs ALL registered strategies through
a back-test using their own backtest_logic(). The top-N (symbol, strategy, params)
combos ranked by:

    score = (win_rate * profit_factor) / max(max_drawdown_pct, 0.01)

are written to `data/active_config.json`, which LiveExecutor reads to decide
what to trade.
"""

import asyncio
import gc
import json
import logging
import itertools
import os
import aiohttp
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import ccxt.async_support as ccxt
import pandas as pd
from dotenv import load_dotenv

from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy
from strategies.density import DensityStrategy
from strategies.knife_catcher_ml import KnifeCatcherMLStrategy
from strategies.scalp_mtf import ScalpMTFStrategy
from strategies.funding_rate import FundingRateStrategy

from core.utils import strip_proxies, get_browser_headers, recursive_url_rewrite, is_in_cooldown, set_global_cooldown, get_cooldown_remaining, wait_for_cooldown

load_dotenv()
strip_proxies()

logger = logging.getLogger("AutoCore.Engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────────

WATCH_LIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT",
    "DOGE/USDT", "WIF/USDT", "LINK/USDT", "ADA/USDT", "XRP/USDT",
    "DOT/USDT", "TRX/USDT", "NEAR/USDT", "ATOM/USDT", "OP/USDT",
]

TIMEFRAME_DEFAULT = "5m"
LOOKBACK_BARS   = 2000           # ~7 days of 5-min bars
TOP_N_CONFIGS   = 5
CYCLE_MINUTES   = 30
CONFIG_PATH     = "data/active_config.json"
TESTNET         = os.getenv("TESTNET", "True").lower() == "true"

# Active strategies for LIVE trading (Phase 11: only Rust-backed strategies)
STRATEGIES: Dict[str, Any] = {
    "Ultimate_SMC_Trail": UltimateSMCTrailStrategy(),
    "KnifeCatcher_ML":    KnifeCatcherMLStrategy(),
    "Density":            DensityStrategy(),
    "ScalpMTF":           ScalpMTFStrategy(),
    "FundingRate_MR":     FundingRateStrategy(),
}

# Full registry = same as STRATEGIES (legacy strategies removed in Phase 11 audit)
ALL_STRATEGIES: Dict[str, Any] = {
    **STRATEGIES,
}


# ── Exchange helpers ────────────────────────────────────────────────────────────

async def _create_exchange() -> ccxt.Exchange:
    """Create public exchange for historical data using CCXT defaults."""
    ex = ccxt.binance({
        "enableRateLimit": True,
        'timeout': 15000,
        'headers': get_browser_headers(),
        'trust_env': False,
        "options": {
            "defaultType": "future",
            "fetchMarkets": ["linear"]
        },
    })
    await ex.load_markets()
    return ex


async def _fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    await wait_for_cooldown("Engine")
    try:
        fetch_coro = exchange.fetch_ohlcv(symbol, timeframe, limit=LOOKBACK_BARS)
        raw = await asyncio.wait_for(fetch_coro, timeout=12.0)
        
        if len(raw) < 200:
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.reset_index(drop=True)
    except asyncio.TimeoutError:
        logger.warning(f"[Engine] {symbol} fetch timed out.")
        return None
    except Exception as e:
        if '429' in str(e):
            set_global_cooldown(60)
        logger.warning(f"[Engine] {symbol} fetch error: {e}")
        return None


# ── Backtest core ───────────────────────────────────────────────────────────────

def _simulate(df: pd.DataFrame, strategy, params: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Run one (df, strategy, params) combination and return metrics dict or None."""
    try:
        result = strategy.backtest_logic(df, params)
    except Exception:
        return None

    # Strategies may name the column 'pnl_r' or 'trade_pnl_r'
    pnl_col = None
    for col in ['pnl_r', 'trade_pnl_r']:
        if col in result.columns:
            pnl_col = col
            break
    if pnl_col is None:
        return None

    trades = result[pnl_col].dropna()
    trades = trades[trades != 0]
    if len(trades) < 10:
        return None

    wins   = trades[trades > 0]
    losses = trades[trades < 0]

    win_rate      = len(wins) / len(trades)
    gross_profit  = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss    = float(abs(losses.sum())) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    cumulative = trades.cumsum()
    peak       = cumulative.cummax()
    drawdown   = (peak - cumulative)
    max_dd     = float(drawdown.max())
    max_pk     = float(peak.max()) if float(peak.max()) > 0 else 1.0
    max_dd_pct = max_dd / max_pk

    score = (win_rate * profit_factor) / max(max_dd_pct, 0.01)

    return {
        "win_rate":         round(win_rate, 4),
        "profit_factor":    round(profit_factor, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "total_trades":     int(len(trades)),
        "score":            round(score, 4),
    }


def _expand_params(space: Dict[str, List]) -> List[Dict[str, Any]]:
    keys   = list(space.keys())
    values = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ── Main engine cycle ───────────────────────────────────────────────────────────

async def run_engine_cycle() -> List[Dict]:
    """Full engine cycle: fetch → backtest all combos → rank → write config."""
    logger.info("=" * 60)
    logger.info("  AUTOCORE ENGINE CYCLE STARTED")
    logger.info(f"  {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)

    exchange = await _create_exchange()
    all_results: List[Dict] = []

    total_symbols = len(WATCH_LIST)
    for i, symbol in enumerate(WATCH_LIST):
        logger.info(f"Engine Progress: {i+1}/{total_symbols} | Analyzing {symbol}...")
        
        # Cache for dataframes to avoid redundant fetching
        tf_cache: Dict[str, pd.DataFrame] = {}

        for strat_name, strategy in STRATEGIES.items():
            tf = getattr(strategy, 'default_timeframe', TIMEFRAME_DEFAULT)
            
            if tf not in tf_cache:
                df = await _fetch_ohlcv(exchange, symbol, tf)
                if df is not None:
                    tf_cache[tf] = df
            
            df_original = tf_cache.get(tf)
            if df_original is None:
                continue

            combos      = _expand_params(strategy.get_parameter_space())
            best_score  = -1.0
            best_metrics: Optional[Dict] = None
            best_params:  Optional[Dict] = None

            for params in combos:
                m = _simulate(df_original.copy(), strategy, params)
                if m and m["score"] > best_score:
                    best_score   = m["score"]
                    best_metrics = m
                    best_params  = params

            if best_metrics:
                all_results.append({
                    "symbol":       symbol,
                    "timeframe":    tf,
                    "strategy":     strat_name,
                    "params":       best_params,
                    "metrics":      best_metrics,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                })
                logger.info(
                    f"  {symbol:12s} | {strat_name:20s} | "
                    f"WR={best_metrics['win_rate']*100:.1f}%  "
                    f"PF={best_metrics['profit_factor']:.2f}  "
                    f"Score={best_score:.3f}"
                )
        
        # Phase 11: Free memory after each symbol to prevent OOM at 50+ symbols
        tf_cache.clear()
        gc.collect()

        # Intra-loop delay to avoid 429 during heavy fetching cycles
        await asyncio.sleep(2.0)

    await exchange.close()

    if not all_results:
        logger.warning("[Engine] No valid results — config not updated.")
        return []

    # Sort by score, select top-N with unique symbols
    all_results.sort(key=lambda x: x["metrics"]["score"], reverse=True)
    seen: set = set()
    top: List[Dict] = []
    for r in all_results:
        if r["symbol"] not in seen:
            top.append(r)
            seen.add(r["symbol"])
        if len(top) >= TOP_N_CONFIGS:
            break

    os.makedirs("data", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(top, f, indent=2)

    logger.info(f"\n[Engine] ✅  Top-{len(top)} configs saved to {CONFIG_PATH}")
    for i, c in enumerate(top, 1):
        m = c["metrics"]
        logger.info(
            f"  #{i} {c['symbol']:12s} {c['strategy']:22s} "
            f"WR={m['win_rate']*100:.1f}%  Score={m['score']:.3f}"
        )

    return top


# ── Scheduled loop ──────────────────────────────────────────────────────────────

async def engine_loop():
    """Call run_engine_cycle() immediately, then every CYCLE_MINUTES minutes."""
    while True:
        try:
            await run_engine_cycle()
        except Exception as e:
            logger.error(f"[Engine] Cycle crashed: {e}", exc_info=True)
        logger.info(f"[Engine] Next run in {CYCLE_MINUTES}m...")
        await asyncio.sleep(CYCLE_MINUTES * 60)


# ── Standalone test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_engine_cycle())
