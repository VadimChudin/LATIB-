"""
Phase 25: Python vs Rust Strategy Comparison
=============================================
Runs the same data through both Python backtest_logic() and
Rust backtest-trades, then compares trade counts, WR, and PnL.

This reveals whether WFA (Python-based) can be trusted.

Run: python compare_py_vs_rust.py
     python compare_py_vs_rust.py --symbol BTC_USDT --strategy knife
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import STRAT_MAP, CACHE_DIR, BINARY_PATH
from strategies.knife_catcher import KnifeCatcherStrategy
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy
from strategies.density import DensityStrategy

# Default params per strategy (from GA defaults / active_config)
DEFAULT_PARAMS = {
    "knife": {
        "rsi_oversold": 25, "bb_std": 2.0, "vol_spike_mult": 1.5,
        "tp_rr": 1.0, "sl_atr_mult": 1.0,
    },
    "smc": {
        "swing_length": 5, "fvg_min_atr": 0.3, "ob_min_score": 3,
        "sl_atr_mult": 1.0, "trail_activate_r": 1.0, "trail_atr_mult": 0.5,
    },
    "density": {
        "vol_spike_mult": 2.5, "min_touches": 2, "shakeout_pct": 0.006,
        "tp_rr": 2.0, "sl_atr_mult": 1.0,
    },
}

PYTHON_STRATEGIES = {
    "knife": KnifeCatcherStrategy(),
    "smc": UltimateSMCTrailStrategy(),
    "density": DensityStrategy(),
}


def load_csv(symbol: str, timeframe: str = "5m") -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_{timeframe}_730d.csv"
    if not path.exists():
        print(f"  ❌ CSV not found: {path}")
        return pd.DataFrame()
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(path, dtype=dtypes, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def run_python_backtest(strat_key: str, df: pd.DataFrame, params: dict) -> dict:
    """Run Python backtest_logic and extract trade stats."""
    strat = PYTHON_STRATEGIES.get(strat_key)
    if not strat:
        return {"trades": 0, "wins": 0, "wr": 0, "entries": []}

    try:
        import pandas_ta as ta
        df_copy = df.copy()
        # Pre-compute indicators for knife
        if strat_key == "knife":
            bb_std = params.get('bb_std', 2.0)
            if not any(c.startswith('BBL') for c in df_copy.columns):
                df_copy.ta.bbands(length=20, std=bb_std, append=True)
            if 'RSI_14' not in df_copy.columns:
                df_copy.ta.rsi(length=14, append=True)
            if 'ATRr_14' not in df_copy.columns:
                df_copy.ta.atr(length=14, append=True)

        result = strat.backtest_logic(df_copy, params)
        trades = result[result['trade_pnl_r'] != 0].copy()

        wins = len(trades[trades['trade_pnl_r'] > 0])
        total = len(trades)
        wr = (wins / total * 100) if total > 0 else 0

        entries = []
        for _, row in trades.head(20).iterrows():
            entries.append({
                "idx": int(row.get('entry_idx', 0)),
                "pnl_r": float(row['trade_pnl_r']),
                "dir": row.get('trade_dir', '?'),
            })

        return {"trades": total, "wins": wins, "wr": round(wr, 1), "entries": entries}
    except Exception as e:
        print(f"    ⚠️ Python backtest error: {e}")
        import traceback; traceback.print_exc()
        return {"trades": 0, "wins": 0, "wr": 0, "entries": [], "error": str(e)}


def run_rust_backtest(symbol: str, strat_key: str, params: dict, timeframe: str = "5m") -> dict:
    """Run Rust backtest-trades and extract trade stats."""
    csv_path = CACHE_DIR / f"{symbol}_{timeframe}_730d.csv"
    binary = str(BINARY_PATH)
    if not os.path.exists(binary):
        binary = binary.replace(".exe", "")
    if not os.path.exists(binary):
        return {"trades": 0, "wins": 0, "wr": 0, "entries": [], "error": "binary not found"}

    cmd = [
        binary, "backtest-trades",
        "--csv", str(csv_path),
        "--strategy", strat_key,
        "--params-json", json.dumps(params)
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        output = res.stdout.strip()
        start = output.find('[')
        end = output.rfind(']')
        if start == -1 or end == -1:
            return {"trades": 0, "wins": 0, "wr": 0, "entries": [], "error": "no JSON output"}

        trades = json.loads(output[start:end+1])
        wins = sum(1 for t in trades if t.get("pnl_r", 0) > 0)
        total = len(trades)
        wr = (wins / total * 100) if total > 0 else 0

        entries = []
        for t in trades[:20]:
            entries.append({
                "ts": t.get("entry_ts", "?"),
                "pnl_r": t.get("pnl_r", 0),
                "dir": t.get("direction", "?"),
            })

        return {"trades": total, "wins": wins, "wr": round(wr, 1), "entries": entries}
    except subprocess.TimeoutExpired:
        return {"trades": 0, "wins": 0, "wr": 0, "entries": [], "error": "timeout"}
    except Exception as e:
        return {"trades": 0, "wins": 0, "wr": 0, "entries": [], "error": str(e)}


def compare_strategy(symbol: str, strat_key: str):
    """Compare one strategy on one symbol."""
    params = DEFAULT_PARAMS.get(strat_key, {})
    print(f"\n{'='*60}")
    print(f"  {symbol} | {strat_key.upper()}")
    print(f"  Params: {params}")
    print(f"{'='*60}")

    df = load_csv(symbol)
    if df.empty:
        return None

    print(f"  📊 CSV: {len(df)} candles loaded")

    # Python
    print(f"  🐍 Running Python backtest...")
    py = run_python_backtest(strat_key, df, params)

    # Rust
    print(f"  🦀 Running Rust backtest...")
    rs = run_rust_backtest(symbol, strat_key, params)

    # Compare
    trade_diff = abs(py["trades"] - rs["trades"])
    wr_diff = abs(py["wr"] - rs["wr"])
    match_pct = 0
    if max(py["trades"], rs["trades"]) > 0:
        match_pct = min(py["trades"], rs["trades"]) / max(py["trades"], rs["trades"]) * 100

    verdict = "✅ MATCH" if match_pct > 80 else "⚠️ DIVERGED" if match_pct > 50 else "❌ COMPLETELY DIFFERENT"

    print(f"\n  {'':30s} {'Python':>10s}  {'Rust':>10s}  {'Diff':>10s}")
    print(f"  {'─'*65}")
    print(f"  {'Total Trades':30s} {py['trades']:10d}  {rs['trades']:10d}  {trade_diff:10d}")
    print(f"  {'Wins':30s} {py['wins']:10d}  {rs['wins']:10d}  {abs(py['wins']-rs['wins']):10d}")
    print(f"  {'Win Rate':30s} {py['wr']:9.1f}%  {rs['wr']:9.1f}%  {wr_diff:9.1f}%")
    print(f"  {'Match':30s} {match_pct:9.1f}%")
    print(f"\n  Verdict: {verdict}")

    if py.get("error"):
        print(f"  ⚠️ Python error: {py['error']}")
    if rs.get("error"):
        print(f"  ⚠️ Rust error: {rs['error']}")

    # Show first few trades for manual comparison
    if py["entries"] and rs["entries"]:
        print(f"\n  First 5 Python trades:")
        for e in py["entries"][:5]:
            print(f"    idx={e['idx']:6d} | {e['dir']:5s} | pnl_r={e['pnl_r']:+.2f}")
        print(f"\n  First 5 Rust trades:")
        for e in rs["entries"][:5]:
            print(f"    ts={e['ts']} | {e['dir']:5s} | pnl_r={e['pnl_r']:+.2f}")

    return {
        "symbol": symbol,
        "strategy": strat_key,
        "py_trades": py["trades"],
        "rs_trades": rs["trades"],
        "py_wr": py["wr"],
        "rs_wr": rs["wr"],
        "match_pct": round(match_pct, 1),
        "verdict": verdict,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
    strategies = [args.strategy] if args.strategy else ["knife", "smc", "density"]

    print("🔬 PHASE 25: Python vs Rust Strategy Comparison")
    print(f"   Symbols: {symbols}")
    print(f"   Strategies: {strategies}")

    results = []
    for sym in symbols:
        for strat in strategies:
            r = compare_strategy(sym, strat)
            if r:
                results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Symbol':12s} {'Strategy':10s} {'Py Trades':>10s} {'Rs Trades':>10s} {'Py WR':>8s} {'Rs WR':>8s} {'Match':>8s} {'Verdict'}")
    print(f"  {'─'*80}")
    for r in results:
        print(f"  {r['symbol']:12s} {r['strategy']:10s} {r['py_trades']:10d} {r['rs_trades']:10d} {r['py_wr']:7.1f}% {r['rs_wr']:7.1f}% {r['match_pct']:7.1f}% {r['verdict']}")

    # Final conclusion
    diverged = [r for r in results if "DIFFERENT" in r["verdict"] or "DIVERGED" in r["verdict"]]
    if diverged:
        print(f"\n  🚨 {len(diverged)} strategy/symbol pairs have significant divergences!")
        print(f"     WFA results for these strategies are UNRELIABLE.")
        print(f"     Options: fix Python strategies OR run WFA through Rust.")
    else:
        print(f"\n  ✅ All strategies match within tolerance. WFA is reliable.")


if __name__ == "__main__":
    main()
