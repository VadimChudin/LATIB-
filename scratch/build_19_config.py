"""Build active_config.json with correct format for Rust config_loader"""
import json, os

PROFITABLE = [
    'DOGE_USDT', 'NOM_USDT', '1000PEPE_USDT', 'SOLV_USDT', 'ZIL_USDT',
    'COS_USDT', 'ASTER_USDT', 'FIDA_USDT', 'XAN_USDT', 'WIF_USDT',
    'WLFI_USDT', 'ENJ_USDT', 'HBAR_USDT', 'ADA_USDT', 'VANRY_USDT',
    'BANK_USDT', 'UNI_USDT', 'BARD_USDT', 'FIL_USDT'
]

configs = []
for sym in PROFITABLE:
    pf = f'data/tick_params/{sym}.json'
    if not os.path.exists(pf):
        print(f"  SKIP {sym}")
        continue
    with open(pf) as f:
        p = json.load(f)
    
    config = {
        "symbol": sym,
        "strategy": "knife_tick",
        "timeframe": "1m",
        "tier": 1,
        "leverage": 10,
        "params": p.get("params", {}),
        "metrics": {
            "win_rate": p.get("test_wr", 0) / 100.0,
            "train_wr": p.get("train_wr", 0) / 100.0,
            "total_trades": p.get("test_trades", 0) + p.get("train_trades", 0),
            "score": p.get("fitness", 0),
        }
    }
    configs.append(config)
    print(f"  ✅ {sym}")

with open('data/active_config.json', 'w') as f:
    json.dump(configs, f, indent=2)

print(f"\n  ✅ Wrote {len(configs)} configs (with timeframe field)")
