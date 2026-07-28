import subprocess
import os
from pathlib import Path

def debug_ga():
    csv_path = "data/cache/BTC_USDT_5m_730d.csv"
    output_path = "data/ga_results/BTC_USDT_debug.json"
    
    cmd = [
        "cargo", "run", "--release", "--manifest-path", "rust_engine/Cargo.toml", 
        "--bin", "aegis_engine", "--", "optimize-all",
        "--csv", csv_path,
        "--output", output_path
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(f"Exit Code: {result.returncode}")
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    debug_ga()
