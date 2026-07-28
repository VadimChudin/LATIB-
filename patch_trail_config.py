"""Patches active_config.json to fix trail_activate_r and trail_atr_mult in all SwingICT_Trail configs."""
import json

with open('data/active_config.json', 'r') as f:
    configs = json.load(f)

patched = 0
for c in configs:
    params = c.get('params', {})
    # Fix trail params for all strategies that use trailing stops
    if c.get('strategy') in ('SwingICT_Trail', 'Ultimate_SMC_Trail'):
        if params.get('trail_activate_r', 0) < 1.0:
            params['trail_activate_r'] = 1.0
            patched += 1
        if params.get('trail_atr_mult', 0) < 0.5:
            params['trail_atr_mult'] = 0.5
            patched += 1

with open('data/active_config.json', 'w') as f:
    json.dump(configs, f, indent=4)

print(f"Patched {patched} trail params in active_config.json")
for c in configs:
    p = c.get('params', {})
    if 'trail_activate_r' in p:
        print(f"  {c['symbol']:12s} | {c['strategy']:20s} | trail_activate_r={p['trail_activate_r']}, trail_atr_mult={p['trail_atr_mult']}")
