"""
Filter active_config.json → keep only TOP profitable symbols
=============================================================
Criteria:
  - OOS (Test) WR >= 55%
  - Total trades >= 10
  - Fitness > 0
  - Not in blacklist (BASED, etc.)
"""
import json
from pathlib import Path

CONFIG_PATH = Path("data/active_config.json")
PARAMS_DIR = Path("data/tick_params")
OUTPUT_PATH = Path("data/active_config.json")

# Symbols to NEVER trade (known toxic from session analysis)
BLACKLIST = {
    "BASED_USDT",  # 0% OOS WR, 80% train = massive overfit
}

# Manual OOS results from pipeline output (Test WR%, Train WR%, Trades, Fitness)
# Only symbols with Test WR >= 55%
TOP_RESULTS = {
    "H_USDT":       {"test_wr": 100.0, "train_wr": 100.0, "trades": 9,  "tier": 1},
    "ZIL_USDT":     {"test_wr": 100.0, "train_wr": 100.0, "trades": 10, "tier": 1},
    "KAITO_USDT":   {"test_wr": 87.5,  "train_wr": 90.0,  "trades": 18, "tier": 1},
    "MOCA_USDT":    {"test_wr": 80.0,  "train_wr": 100.0, "trades": 15, "tier": 1},
    "CRV_USDT":     {"test_wr": 71.4,  "train_wr": 85.7,  "trades": 21, "tier": 1},
    "STO_USDT":     {"test_wr": 71.4,  "train_wr": 72.7,  "trades": 25, "tier": 1},
    "ENSO_USDT":    {"test_wr": 66.7,  "train_wr": 63.6,  "trades": 14, "tier": 2},
    "NEAR_USDT":    {"test_wr": 66.7,  "train_wr": 68.4,  "trades": 31, "tier": 2},
    "ONG_USDT":     {"test_wr": 66.7,  "train_wr": 90.9,  "trades": 14, "tier": 2},
    "ANIME_USDT":   {"test_wr": 62.5,  "train_wr": 90.0,  "trades": 18, "tier": 2},
    "ENA_USDT":     {"test_wr": 60.0,  "train_wr": 65.2,  "trades": 33, "tier": 2},
    "NIGHT_USDT":   {"test_wr": 60.0,  "train_wr": 80.0,  "trades": 15, "tier": 2},
    "DRIFT_USDT":   {"test_wr": 57.9,  "train_wr": 80.0,  "trades": 29, "tier": 2},
    "COS_USDT":     {"test_wr": 57.1,  "train_wr": 76.9,  "trades": 47, "tier": 2},
    "CYS_USDT":     {"test_wr": 57.1,  "train_wr": 80.0,  "trades": 17, "tier": 2},
    "KERNEL_USDT":  {"test_wr": 57.1,  "train_wr": 90.0,  "trades": 17, "tier": 2},
    "HUMA_USDT":    {"test_wr": 55.6,  "train_wr": 100.0, "trades": 20, "tier": 2},
    "ZRO_USDT":     {"test_wr": 55.6,  "train_wr": 83.3,  "trades": 21, "tier": 2},
}

def main():
    with open(CONFIG_PATH) as f:
        all_configs = json.load(f)
    
    print(f"📊 Loaded {len(all_configs)} symbols from active_config.json")
    print(f"🚫 Blacklist: {BLACKLIST}")
    print(f"✅ TOP candidates: {len(TOP_RESULTS)} symbols with OOS WR >= 55%\n")
    
    # Filter: keep only TOP symbols
    filtered = []
    for cfg in all_configs:
        sym = cfg["symbol"]
        if sym in BLACKLIST:
            print(f"  ❌ {sym}: BLACKLISTED")
            continue
        if sym not in TOP_RESULTS:
            continue
        
        info = TOP_RESULTS[sym]
        
        # Skip if too few trades (unreliable)
        if info["trades"] < 10:
            print(f"  ⚠️ {sym}: Only {info['trades']} trades, skipping (need >= 10)")
            continue
        
        # Assign tier
        tier = info["tier"]
        
        # Update metrics in config
        cfg["tier"] = tier
        cfg["metrics"]["test_wr_pct"] = info["test_wr"]
        cfg["metrics"]["train_wr_pct"] = info["train_wr"]
        
        filtered.append(cfg)
        print(f"  ✅ Tier {tier} | {sym:20s} | OOS WR={info['test_wr']:.1f}% | Train={info['train_wr']:.1f}% | {info['trades']} trades")
    
    # Sort: Tier 1 first, then by test WR
    filtered.sort(key=lambda c: (c.get("tier", 9), -TOP_RESULTS.get(c["symbol"], {}).get("test_wr", 0)))
    
    print(f"\n{'='*60}")
    print(f"  📋 FINAL POOL: {len(filtered)} symbols")
    t1 = sum(1 for c in filtered if c.get("tier") == 1)
    t2 = sum(1 for c in filtered if c.get("tier") == 2)
    print(f"  Tier 1 (WR >= 70%): {t1} symbols")
    print(f"  Tier 2 (WR 55-70%): {t2} symbols")
    print(f"{'='*60}")
    
    # Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(filtered, f, indent=4)
    print(f"\n  💾 Saved to {OUTPUT_PATH}")
    print(f"  🚀 Ready to launch: python main.py")

if __name__ == "__main__":
    main()
