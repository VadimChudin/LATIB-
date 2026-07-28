"""Add missing profitable symbols to active_config.json from tick_params/"""
import json, os
from datetime import datetime, timezone

PROFITABLE = [
    'POWER_USDT','STO_USDT','ALT_USDT','KERNEL_USDT','BR_USDT',
    'HBAR_USDT','AIOT_USDT','ZIL_USDT','BARD_USDT','ZRO_USDT',
    'CHR_USDT','LIGHT_USDT','HYPE_USDT','H_USDT','NEAR_USDT',
    'BEAT_USDT','GALA_USDT','MAGIC_USDT','DEGO_USDT','AIN_USDT'
]

# Load current config
with open('data/active_config.json') as f:
    configs = json.load(f)

existing = {c['symbol'] for c in configs if c.get('strategy') == 'knife_tick'}
print(f"Currently in config: {len(existing)} symbols: {sorted(existing)}")

# Find missing
missing = [s for s in PROFITABLE if s not in existing]
print(f"Missing: {len(missing)} symbols: {missing}")

now_str = datetime.now(timezone.utc).isoformat()
added = 0

for sym in missing:
    params_path = f'data/tick_params/{sym}.json'
    if not os.path.exists(params_path):
        print(f"  SKIP {sym}: no tick_params file")
        continue
    
    with open(params_path) as f:
        tick_data = json.load(f)
    
    params = tick_data.get('params', {})
    if not params:
        print(f"  SKIP {sym}: empty params")
        continue
    
    configs.append({
        'symbol': sym,
        'timeframe': 'tick',
        'strategy': 'knife_tick',
        'params': params,
        'metrics': {
            'win_rate': tick_data.get('test_wr', 0) / 100.0,
            'train_wr': tick_data.get('train_wr', 0) / 100.0,
            'total_trades': tick_data.get('total_trades', 0),
            'fitness': tick_data.get('test_pnl_r', 0),
        },
        'evaluated_at': now_str,
    })
    added += 1
    print(f"  ADDED {sym}: test_wr={tick_data.get('test_wr',0):.1f}% trades={tick_data.get('total_trades',0)}")

with open('data/active_config.json', 'w') as f:
    json.dump(configs, f, indent=4)

total_knife = sum(1 for c in configs if c.get('strategy') == 'knife_tick')
print(f"\nDone! Added {added} symbols. Total knife_tick in config: {total_knife}")
