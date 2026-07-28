#!/usr/bin/env python3
"""
run_live.py
Launcher for the LAITB 2.0 Rust Live Engine.
Reads the active_config.json to find all symbols currently approved
by the WFA and GA, and passes them to the Rust executable.
"""
import os
import json
import subprocess

CONFIG_PATH = "data/active_config.json"
RUST_DIR = "rust_engine"

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Error: {CONFIG_PATH} not found!")
        print("Please run `python apply_ga_config.py` first to generate the active configuration.")
        return

    print("🔍 Reading active configuration...")
    with open(CONFIG_PATH, "r") as f:
        configs = json.load(f)

    # Extract unique symbols, converting "BTC_USDT" -> "BTCUSDT"
    unique_symbols = set()
    for cfg in configs:
        sym = cfg.get("symbol", "").replace("_", "").replace("/", "")
        if sym:
            unique_symbols.add(sym)

    if not unique_symbols:
        print("❌ No approved symbols found in active_config.json. The bot has nothing to trade!")
        return

    sorted_symbols = sorted(list(unique_symbols))
    symbol_str = ",".join(sorted_symbols)
    
    print(f"✅ Found {len(sorted_symbols)} active symbols to monitor.")
    print(f"🚀 Launching AEGIS Rust Engine (Live Mode)...")
    print("-" * 60)
    
    # Construct the cargo command
    cmd = [
        "cargo", "run", "--release", "--manifest-path", "rust_engine/Cargo.toml", "--bin", "aegis_engine", 
        "--", "live", "--symbols", symbol_str
    ]
    
    try:
        # Run from root so Rust finds data/ correctly
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested. Live Engine stopped.")
    except Exception as e:
        print(f"\n❌ Error launching Rust engine: {e}")

if __name__ == "__main__":
    main()
