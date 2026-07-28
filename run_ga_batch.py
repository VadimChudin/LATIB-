import json
import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fix Windows cp1251 encoding for emoji output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

CACHE_DIR = "data/cache"
GA_RESULTS_DIR = "data/ga_results"
AGGREGATED_RESULTS_PATH = "data/ga_aggregated_results.json"
MAX_WORKERS = 2
TIER_1_COUNT = 20

def run_rust_ga(symbol_name):
    csv_path = Path(CACHE_DIR) / f"{symbol_name}_5m_730d.csv"
    if not csv_path.exists():
        return (symbol_name, False, None, "CSV not found")
    
    output_path = Path(GA_RESULTS_DIR) / f"{symbol_name}.json"
    binary_path = Path("rust_engine/target/release/aegis_engine.exe")
    if not binary_path.exists():
        binary_path = Path("rust_engine/target/release/aegis_engine")

    cmd = [
        str(binary_path), "optimize-all",
        "--csv", str(csv_path),
        "--output", str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=1200, text=True)
        # Extract key metrics from stdout
        summary = ""
        for line in result.stdout.split('\n'):
            if 'Best:' in line or 'fitness' in line.lower() or '🏆' in line:
                summary += line.strip() + " | "
        return (symbol_name, True, str(output_path), summary[:200])
    except subprocess.TimeoutExpired:
        return (symbol_name, False, None, "TIMEOUT after 1200s")
    except subprocess.CalledProcessError as e:
        err = e.stderr[:300] if e.stderr else "no stderr"
        return (symbol_name, False, None, f"Exit {e.returncode}: {err}")
    except Exception as e:
        return (symbol_name, False, None, f"Error: {str(e)[:200]}")

def run_optimization_batch(symbols, batch_name):
    print(f"\n🚀 Launching GA optimization for {batch_name} ({len(symbols)} symbols)...")
    sys.stdout.flush()
    start_time = time.time()
    success_count = 0
    fail_count = 0
    results = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {executor.submit(run_rust_ga, sym): sym for sym in symbols}
        completed = 0
        for future in as_completed(future_to_symbol):
            completed += 1
            symbol_name, success, output_path, info = future.result()
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (len(symbols) - completed)
            
            if success and output_path and os.path.exists(output_path):
                success_count += 1
                try:
                    with open(output_path, "r") as f:
                        results.append({
                            "raw_symbol": symbol_name,
                            "results": json.load(f)
                        })
                except Exception as e:
                    print(f"  ⚠️ Could not read GA output for {symbol_name}: {e}")
                
                print(f"  ✅ [{completed}/{len(symbols)}] {symbol_name} done ({elapsed:.0f}s elapsed, ETA: {eta/60:.1f}min)")
            else:
                fail_count += 1
                print(f"  ❌ [{completed}/{len(symbols)}] {symbol_name} FAILED: {info}")
            sys.stdout.flush()
                
    total_time = time.time() - start_time
    print(f"🏁 {batch_name} COMPLETE: {success_count} ok / {fail_count} fail / {len(symbols)} total | Time: {total_time/60:.1f} min")
    sys.stdout.flush()
    return results

def merge_and_save_results(tier1_res, background_res):
    merged = {}
    for item in background_res + tier1_res:
        merged[item['raw_symbol']] = item
    
    final_list = list(merged.values())
    with open(AGGREGATED_RESULTS_PATH, "w") as f:
        json.dump(final_list, f, indent=4)
        
    return final_list

def main():
    print("🤖 AEGIS BATCH GA ORCHESTRATOR (TIERED MODE)")
    sys.stdout.flush()
    
    print("\n🔨 Pre-compiling Rust Engine (release mode)...")
    sys.stdout.flush()
    try:
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", "rust_engine/Cargo.toml", "--bin", "aegis_engine"],
            check=True
        )
    except Exception:
        print("❌ Rust compilation failed.")
        sys.exit(1)
        
    top_symbols_path = "data/top_symbols.json"
    if os.path.exists(top_symbols_path):
        with open(top_symbols_path, "r") as f:
            all_symbols = json.load(f)
    else:
        print("❌ top_symbols.json not found!")
        sys.exit(1)
        
    os.makedirs(GA_RESULTS_DIR, exist_ok=True)
    
    tier_1_symbols = all_symbols[:TIER_1_COUNT]
    tier_2_symbols = all_symbols[TIER_1_COUNT:]
    
    print(f"\n📊 Total: {len(all_symbols)} symbols | Tier 1: {len(tier_1_symbols)} | Tier 2: {len(tier_2_symbols)}")
    sys.stdout.flush()
    
    # 1. Optimize Tier 1 (Leaders — first 20)
    print(f"\n{'='*60}")
    print(f"  TIER 1: PRIORITY OPTIMIZATION ({len(tier_1_symbols)} symbols)")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    tier1_results = run_optimization_batch(tier_1_symbols, "Tier 1 (Leaders)")
    merge_and_save_results(tier1_results, [])
    
    # Apply Tier 1 immediately
    print("\n📡 Applying Tier 1 parameters...")
    sys.stdout.flush()
    try:
        subprocess.run([sys.executable, "apply_ga_config.py"], check=True, timeout=300)
    except Exception as e:
        print(f"⚠️ Error applying Tier 1 configs: {e}")
    
    if not tier_2_symbols:
        print("\n✅ No Tier 2 symbols. Done.")
        return
    
    # 2. Optimize Tier 2 (background — remaining symbols)
    print(f"\n{'='*60}")
    print(f"  TIER 2: DEEP OPTIMIZATION ({len(tier_2_symbols)} symbols)")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    background_results = []
    chunk_size = 4  # Process 4 at a time for speed
    total_chunks = (len(tier_2_symbols) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(tier_2_symbols), chunk_size):
        chunk_num = i // chunk_size + 1
        chunk = tier_2_symbols[i:i+chunk_size]
        chunk_results = run_optimization_batch(chunk, f"Tier 2 Chunk {chunk_num}/{total_chunks}")
        background_results.extend(chunk_results)
        
        # Merge and apply after each chunk
        merge_and_save_results(tier1_results, background_results)
        try:
            subprocess.run([sys.executable, "apply_ga_config.py"], check=True, timeout=300)
        except Exception as e:
            print(f"⚠️ Error applying chunk configs: {e}")
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"  🏁 FULL GA BATCH COMPLETE")
    print(f"  Tier 1: {len(tier1_results)} results")
    print(f"  Tier 2: {len(background_results)} results")
    print(f"  Total:  {len(tier1_results) + len(background_results)} results")
    print(f"{'='*60}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
