#!/usr/bin/env python3
"""
Pipeline: Knife Tick HFT — Multi-Symbol Per-Symbol DE Optimization
===================================================================
For each symbol that has epicenter tick data:
  1. Runs Rust DE optimizer (cargo run -- optimize-ticks)
  2. Saves per-symbol params to data/tick_params/{symbol}.json
  3. Accumulates all trades into data/tick_trades_all.json
  4. Builds active_config.json with per-symbol params
  5. (Optional) Trains ML on all accumulated trades

Usage:
  python pipeline_knife_all.py --generations 50
  python pipeline_knife_all.py --generations 50 --skip-ml
  python pipeline_knife_all.py --symbols SOL_USDT,DOGE_USDT  # specific symbols only

Excludes: BTC, ETH, XAG, XAU, PAXG, BNB (low volatility / not suitable for knife catching)
"""
import os, sys, json, shutil, subprocess, argparse, logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RUST_DIR = os.path.join(BASE_DIR, 'rust_engine')

# Rust outputs (overwritten each run)
TICK_TRADES_PATH = os.path.join(DATA_DIR, 'tick_trades_knifetick.json')
BEST_PARAMS_PATH = os.path.join(DATA_DIR, 'ga_best_tick_params.json')
ACTIVE_CONFIG_PATH = os.path.join(DATA_DIR, 'active_config.json')

# Per-symbol output dirs
PARAMS_DIR = os.path.join(DATA_DIR, 'tick_params')
TRADES_DIR = os.path.join(DATA_DIR, 'tick_trades')
os.makedirs(PARAMS_DIR, exist_ok=True)
os.makedirs(TRADES_DIR, exist_ok=True)

# Symbols to EXCLUDE from knife catching (too low volatility / weird microstructure)
EXCLUDED_SYMBOLS = {
    'BTC_USDT', 'ETH_USDT',       # Majors - dumps too small for knives
    'XAG_USDT', 'XAU_USDT',       # Commodities - different market structure
    'PAXG_USDT',                   # Gold-backed stablecoin
    'BNB_USDT',                    # Exchange token - low vol
    'TSLA_USDT', 'MSTR_USDT',     # Tokenized stocks - different market structure
}


def get_symbols_with_epicenters() -> list:
    """Find all symbols that have epicenter tick data downloaded."""
    epicenters_dir = Path(DATA_DIR) / 'epicenters_ticks'
    if not epicenters_dir.exists():
        return []
    
    symbols = []
    for d in sorted(epicenters_dir.iterdir()):
        if d.is_dir() and d.name not in EXCLUDED_SYMBOLS:
            # Check that it has actual tick CSVs (in LONG/ or SHORT/ or root)
            has_data = False
            for sub in [d / 'LONG', d / 'SHORT', d]:
                if sub.exists() and any(sub.glob('*.csv')):
                    has_data = True
                    break
            if has_data:
                symbols.append(d.name)
    
    return symbols


def run_de_for_symbol(symbol: str, direction: str, generations: int) -> dict | None:
    """Run Rust DE optimizer for a single symbol. Returns results dict or None on failure."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  DE OPTIMIZATION: {symbol}")
    logger.info(f"{'='*60}")
    
    cmd = [
        "cargo", "run", "--release", "--",
        "optimize-ticks",
        "--symbol", symbol,
        "--direction", direction,
        "--generations", str(generations),
    ]
    logger.info(f"  CMD: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=RUST_DIR, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"  ❌ Rust optimizer failed for {symbol}")
        return None
    
    # Check outputs exist
    if not os.path.exists(BEST_PARAMS_PATH) or not os.path.exists(TICK_TRADES_PATH):
        logger.error(f"  ❌ Output files not found for {symbol}")
        return None
    
    # Read results
    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)
    with open(TICK_TRADES_PATH) as f:
        trades = json.load(f)
    
    # Save per-symbol copies
    params_file = os.path.join(PARAMS_DIR, f'{symbol}.json')
    trades_file = os.path.join(TRADES_DIR, f'{symbol}.json')
    shutil.copy2(BEST_PARAMS_PATH, params_file)
    shutil.copy2(TICK_TRADES_PATH, trades_file)
    
    train_wr = best_params.get('train_wr', 0)
    test_wr = best_params.get('test_wr', 0)
    train_trades = best_params.get('train_trades', 0)
    test_trades = best_params.get('test_trades', 0)
    fitness = best_params.get('fitness', 0)
    
    logger.info(f"  ✅ {symbol}: Train WR={train_wr:.1f}% ({train_trades}t) | Test WR={test_wr:.1f}% ({test_trades}t) | Fitness={fitness:.1f}")
    logger.info(f"  📁 Params → {params_file}")
    logger.info(f"  📁 Trades → {trades_file} ({len(trades)} trades)")
    
    return {
        'symbol': symbol,
        'params': best_params.get('params', {}),
        'train_wr': train_wr,
        'test_wr': test_wr,
        'train_trades': train_trades,
        'test_trades': test_trades,
        'fitness': fitness,
        'num_trades_total': len(trades),
    }


def build_active_config(results: list):
    """Build active_config.json with per-symbol params."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  BUILDING ACTIVE CONFIG (per-symbol params)")
    logger.info(f"{'='*60}")
    
    now_str = datetime.now(timezone.utc).isoformat()
    
    # Load existing config, remove old knife_tick entries
    configs = []
    if os.path.exists(ACTIVE_CONFIG_PATH):
        with open(ACTIVE_CONFIG_PATH) as f:
            configs = json.load(f)
    configs = [c for c in configs if c.get('strategy') != 'knife_tick']
    
    # Add per-symbol knife_tick entries
    added = 0
    for r in results:
        if r['test_trades'] < 5:
            logger.warning(f"  ⚠️ Skipping {r['symbol']}: only {r['test_trades']} test trades")
            continue
            
        configs.append({
            'symbol': r['symbol'],
            'timeframe': 'tick',
            'strategy': 'knife_tick',
            'params': r['params'],
            'metrics': {
                'win_rate': r['test_wr'] / 100.0,
                'train_wr': r['train_wr'] / 100.0,
                'total_trades': r['train_trades'] + r['test_trades'],
                'fitness': r['fitness'],
            },
            'evaluated_at': now_str,
        })
        added += 1
    
    with open(ACTIVE_CONFIG_PATH, 'w') as f:
        json.dump(configs, f, indent=4)
    
    logger.info(f"  ✅ Added {added} per-symbol knife_tick entries")
    logger.info(f"  📁 Saved to {ACTIVE_CONFIG_PATH}")


def accumulate_all_trades():
    """Merge all per-symbol trade files into one dataset for ML."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  ACCUMULATING ALL TRADES FOR ML")
    logger.info(f"{'='*60}")
    
    all_trades = []
    for trades_file in sorted(Path(TRADES_DIR).glob('*.json')):
        symbol = trades_file.stem
        with open(trades_file) as f:
            trades = json.load(f)
        # Tag each trade with its symbol
        for t in trades:
            t['symbol'] = symbol
        all_trades.extend(trades)
        logger.info(f"  📊 {symbol}: {len(trades)} trades")
    
    out_path = os.path.join(DATA_DIR, 'tick_trades_all.json')
    with open(out_path, 'w') as f:
        json.dump(all_trades, f)
    
    wins = sum(1 for t in all_trades if t.get('pnl_r', 0) > 0)
    wr = wins / len(all_trades) * 100 if all_trades else 0
    
    logger.info(f"\n  ✅ Total: {len(all_trades)} trades, WR={wr:.1f}%")
    logger.info(f"  📁 Saved to {out_path}")
    return all_trades


def main():
    parser = argparse.ArgumentParser(description="Multi-Symbol Knife Tick Pipeline")
    parser.add_argument('--direction', default='ALL', help='LONG, SHORT, or ALL')
    parser.add_argument('--generations', type=int, default=50, help='DE generations per symbol')
    parser.add_argument('--symbols', type=str, default='', help='Comma-separated symbols (default: all with epicenter data)')
    parser.add_argument('--skip-de', action='store_true', help='Skip DE, use existing per-symbol params')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML training')
    parser.add_argument('--skip-existing', action='store_true', help='Skip symbols that already have params in tick_params/')
    args = parser.parse_args()

    logger.info("🔪 MULTI-SYMBOL KNIFE TICK PIPELINE")
    logger.info(f"   Direction: {args.direction} | Generations: {args.generations}")
    logger.info(f"   Excluded: {', '.join(sorted(EXCLUDED_SYMBOLS))}")
    
    # Determine symbols to process
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = get_symbols_with_epicenters()
    
    # Filter out symbols that already have params
    if args.skip_existing:
        before = len(symbols)
        symbols = [s for s in symbols if not os.path.exists(os.path.join(PARAMS_DIR, f"{s}.json"))]
        logger.info(f"   --skip-existing: {before} → {len(symbols)} (skipped {before - len(symbols)} already optimized)")
    
    logger.info(f"   Symbols to process: {len(symbols)}")
    logger.info("")

    if not symbols:
        logger.error("❌ No symbols with epicenter data found!")
        logger.error("   Run find_epicenters_all.py and download_epicenters_all.py first.")
        return

    # Step 1: Run DE for each symbol
    results = []
    if not args.skip_de:
        for i, sym in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Processing {sym}...")
            result = run_de_for_symbol(sym, args.direction, args.generations)
            if result:
                results.append(result)
    else:
        logger.info("⏭️ Skipping DE (--skip-de). Loading existing per-symbol params...")
        for params_file in sorted(Path(PARAMS_DIR).glob('*.json')):
            symbol = params_file.stem
            if symbol not in symbols:
                continue
            with open(params_file) as f:
                best = json.load(f)
            results.append({
                'symbol': symbol,
                'params': best.get('params', {}),
                'train_wr': best.get('train_wr', 0),
                'test_wr': best.get('test_wr', 0),
                'train_trades': best.get('train_trades', 0),
                'test_trades': best.get('test_trades', 0),
                'fitness': best.get('fitness', 0),
            })

    if not results:
        logger.error("❌ No successful optimizations!")
        return

    # Step 2: Build active_config with per-symbol params
    build_active_config(results)

    # Step 3: Accumulate all trades
    all_trades = accumulate_all_trades()

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info(f"  SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Symbol':<20} {'Train WR':>10} {'Test WR':>10} {'Trades':>8} {'Fitness':>10}")
    logger.info(f"  {'-'*58}")
    for r in sorted(results, key=lambda x: x['test_wr'], reverse=True):
        logger.info(f"  {r['symbol']:<20} {r['train_wr']:>9.1f}% {r['test_wr']:>9.1f}% {r['train_trades']+r['test_trades']:>8} {r['fitness']:>10.1f}")
    
    avg_test_wr = sum(r['test_wr'] for r in results) / len(results) if results else 0
    logger.info(f"\n  Average Test WR: {avg_test_wr:.1f}%")
    logger.info(f"  Symbols optimized: {len(results)}/{len(symbols)}")

    # Step 4: ML Training (optional)
    if not args.skip_ml and all_trades and len(all_trades) >= 100:
        logger.info(f"\n  🧠 ML training available ({len(all_trades)} trades)")
        logger.info(f"  Run: python train_ml_tick_trades.py")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  ✅ PIPELINE COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
