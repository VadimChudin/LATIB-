import os
import json
import asyncio
import pandas as pd
from pathlib import Path
from lazy_tick_loader import LazyTickLoader

OUT_DIR = "data/epicenters_ticks"
WINDOW_BEFORE_MS = 2 * 60 * 1000
WINDOW_AFTER_MS = 30 * 60 * 1000   # 30 min after (was 5)

async def download_symbol_epicenters(symbol: str, limit: int = None):
    events_file = Path(f"data/epicenters/{symbol}_epicenters.json")
    if not events_file.exists():
        print(f"❌ '{events_file}' not found.")
        return

    with open(events_file, "r") as f:
        events = json.load(f)

    if limit and limit > 0:
        events = events[-limit:] # Latest events first for testing
        print(f"⚠️ LIMIT SET: processing only the last {limit} events.")

    # Convert to pure format symbol for lazy loader e.g. BTCUSDT
    symbol_pure = symbol.replace("_", "")
    out_symbol_dir = Path(OUT_DIR) / symbol

    loader = LazyTickLoader()
    success = 0
    errors = 0

    print(f"📥 Downloading ticks for {len(events)} epicenters of {symbol}...")
    
    for i, ev in enumerate(events):
        ev_ts = ev["ts_ms"]
        direction = ev.get("direction", "LONG")
        direction_dir = out_symbol_dir / direction
        direction_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file as {timestamp_ms}.csv
        out_file = direction_dir / f"{ev_ts}.csv"
        
        if out_file.exists():
            success += 1
            # print(f"  [{i+1}/{len(events)}] SKIP (exists): {ev_ts}")
            continue

        start_ms = ev_ts - WINDOW_BEFORE_MS
        end_ms = ev_ts + WINDOW_AFTER_MS
        
        try:
            df = await loader.load_trade_window(symbol_pure, start_ms, end_ms)
            if not df.empty:
                # Save purely essential columns for Rust
                # timestamp, price, qty, is_buyer_maker
                df_export = df[['timestamp', 'price', 'qty', 'is_buyer_maker']]
                df_export.to_csv(out_file, index=False)
                success += 1
                if (i+1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{len(events)}] SUCCESS: {ev_ts} ({len(df_export)} ticks)")
            else:
                errors += 1
                print(f"  [{i+1}/{len(events)}] EMPTY: {ev_ts}")
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(events)}] ERROR {ev_ts}: {e}")

    await loader.close()
    print(f"🏁 DONE {symbol}: {success} OK, {errors} ERRORS. Saved to {out_symbol_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTC_USDT")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of epicenters to download")
    args = parser.parse_args()
    
    asyncio.run(download_symbol_epicenters(args.symbol, args.limit))
