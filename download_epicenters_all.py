#!/usr/bin/env python3
"""
Download Tick Data for All Symbol Epicenters
=============================================
Downloads aggTrades from Binance Vision for each epicenter event.
Processes top N symbols from top_symbols.json.

Usage: python download_epicenters_all.py [--top 50]
"""
import os, json, asyncio, argparse
import pandas as pd
from pathlib import Path
from lazy_tick_loader import LazyTickLoader

OUT_DIR = "data/epicenters_ticks"
WINDOW_BEFORE_MS = 2 * 60 * 1000    # 2 min before
WINDOW_AFTER_MS  = 30 * 60 * 1000   # 30 min after (was 5 — too short for 1.5R TP)

# Skip these — not suitable for knife catching
EXCLUDED_SYMBOLS = {
    'BTC_USDT', 'ETH_USDT',
    'XAG_USDT', 'XAU_USDT',
    'PAXG_USDT', 'BNB_USDT',
    'TSLA_USDT', 'MSTR_USDT',
}


async def download_symbol(loader: LazyTickLoader, symbol: str, events: list, force: bool = False) -> tuple:
    """Download tick data for all epicenters of a symbol. Returns (success, errors)."""
    symbol_pure = symbol.replace("_", "")
    out_symbol_dir = Path(OUT_DIR) / symbol
    success = 0
    errors = 0
    skipped = 0

    for i, ev in enumerate(events):
        ev_ts = ev["ts_ms"]
        direction = ev.get("direction", "LONG")
        direction_dir = out_symbol_dir / direction
        direction_dir.mkdir(parents=True, exist_ok=True)

        out_file = direction_dir / f"{ev_ts}.csv"

        if out_file.exists() and not force:
            skipped += 1
            continue

        start_ms = ev_ts - WINDOW_BEFORE_MS
        end_ms = ev_ts + WINDOW_AFTER_MS

        try:
            df = await loader.load_trade_window(symbol_pure, start_ms, end_ms)
            if not df.empty:
                df_export = df[['timestamp', 'price', 'qty', 'is_buyer_maker']]
                df_export.to_csv(out_file, index=False)
                success += 1
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"    [{i+1}/{len(events)}] {symbol}: {ev_ts} ({len(df_export)} ticks)")
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    [{i+1}/{len(events)}] ERROR {symbol} {ev_ts}: {e}")

    return success, errors, skipped


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=50, help='Number of top symbols to process')
    parser.add_argument('--force', action='store_true', help='Re-download even if file exists (for window size change)')
    args = parser.parse_args()

    symbols_path = Path("data/top_symbols.json")
    with open(symbols_path) as f:
        all_symbols = json.load(f)
    symbols = all_symbols[:args.top]

    epicenters_dir = Path("data/epicenters")

    # Collect all symbols that have epicenter JSON files
    symbol_events = []
    for sym in symbols:
        if sym in EXCLUDED_SYMBOLS:
            continue
        events_file = epicenters_dir / f"{sym}_epicenters.json"
        if events_file.exists():
            with open(events_file) as f:
                events = json.load(f)
            if events:
                symbol_events.append((sym, events))

    total_events = sum(len(e) for _, e in symbol_events)
    print(f"📥 Downloading tick data for {len(symbol_events)} symbols, {total_events} epicenters total")
    print("=" * 60)

    loader = LazyTickLoader()
    grand_success = 0
    grand_errors = 0
    grand_skipped = 0

    for idx, (sym, events) in enumerate(symbol_events):
        print(f"  [{idx+1}/{len(symbol_events)}] {sym}: {len(events)} epicenters...")
        s, e, sk = await download_symbol(loader, sym, events, force=args.force)
        grand_success += s
        grand_errors += e
        grand_skipped += sk
        print(f"    → OK={s} ERR={e} SKIP={sk}")

    await loader.close()

    print(f"\n{'=' * 60}")
    print(f"  ✅ Downloaded: {grand_success}")
    print(f"  ⏭️  Skipped (cached): {grand_skipped}")
    print(f"  ❌ Errors: {grand_errors}")
    print(f"  📁 Saved to {OUT_DIR}/")


if __name__ == '__main__':
    asyncio.run(main())
