import json

TOP = [
    'ZIL_USDT','KAITO_USDT','MOCA_USDT','CRV_USDT','STO_USDT',
    'ENSO_USDT','NEAR_USDT','ONG_USDT','ANIME_USDT','ENA_USDT',
    'NIGHT_USDT','DRIFT_USDT','COS_USDT','CYS_USDT','KERNEL_USDT',
    'HUMA_USDT','ZRO_USDT'
]

with open('data/active_config.json') as f:
    configs = json.load(f)

for sym in TOP:
    c = next((x for x in configs if x['symbol'] == sym), None)
    if not c:
        continue
    p = c['params']
    m = c['metrics']
    
    win_ms = p.get('window_ms', 0)
    zscore = p.get('min_zscore', 0)
    vol_spike = p.get('min_vol_spike', 0)
    sl = p.get('sl_buffer_pct', 0) * 100
    be = p.get('be_trigger_pct', 0) * 100
    trail = p.get('trail_pct', 0) * 100
    micro = p.get('micro_window_ms', 0)
    absorb = p.get('min_absorption', 0)
    reclaim = p.get('min_reclaim_pct', 0) * 100
    speed = p.get('max_speed_mult', 0)
    base_s = p.get('baseline_window_sec', 0)
    dur_s = p.get('max_absorber_sec', 0)
    cool_s = p.get('rewake_cooldown_sec', 0)
    
    test_wr = m.get('win_rate', 0) * 100
    train_wr = m.get('train_wr', 0) * 100
    trades = m.get('total_trades', 0)
    fitness = m.get('fitness', 0)
    
    print(f"{sym}|{win_ms:.0f}|{zscore:.2f}|{vol_spike:.2f}|{sl:.3f}|{be:.3f}|{trail:.3f}|{micro:.0f}|{absorb:.1f}|{reclaim:.4f}|{speed:.2f}|{base_s:.0f}|{dur_s:.0f}|{cool_s:.0f}|{test_wr:.1f}|{train_wr:.1f}|{trades}|{fitness:.1f}")
