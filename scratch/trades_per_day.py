import json, os
from pathlib import Path

symbols = ['DOGE_USDT','NOM_USDT','SOLV_USDT','ZIL_USDT','COS_USDT','XAN_USDT','WIF_USDT','HBAR_USDT']

total_tpd = 0
total_rpd = 0

print(f"{'Symbol':<18} {'Trades':>7} {'Days':>6} {'T/day':>7} {'PnL_R':>8} {'R/day':>7}")
print("-" * 60)

for sym in symbols:
    pf = f'data/tick_params/{sym}.json'
    if not os.path.exists(pf): continue
    with open(pf) as f:
        p = json.load(f)
    
    test_trades = p.get('test_trades', 0)
    test_pnl = p.get('test_pnl_r', 0)
    
    ep_dir = Path(f'data/epicenters_ticks/{sym}')
    min_ts = 9e18
    max_ts = 0
    for d in ['LONG','SHORT']:
        dd = ep_dir / d
        if dd.exists():
            for f2 in dd.glob('*.csv'):
                ts = int(f2.stem)
                if ts < min_ts: min_ts = ts
                if ts > max_ts: max_ts = ts
    
    days = (max_ts - min_ts) / 1000 / 86400 if max_ts > min_ts else 1
    # Test = 30% of data → test_days = days * 0.3
    test_days = days * 0.3
    tpd = test_trades / test_days if test_days > 0 else 0
    rpd = test_pnl / test_days if test_days > 0 else 0
    
    total_tpd += tpd
    total_rpd += rpd
    
    print(f"  {sym:<16} {test_trades:>7} {test_days:>6.0f} {tpd:>7.2f} {test_pnl:>+8.2f} {rpd:>+7.3f}")

print("-" * 60)
print(f"  {'TOTAL':<16} {'':>7} {'':>6} {total_tpd:>7.2f} {'':>8} {total_rpd:>+7.3f}")
print(f"\n  Expected: ~{total_tpd:.1f} trades/day, ~{total_rpd:+.2f}R/day")
print(f"  At 2% risk on $70: ~${total_rpd * 0.02 * 70:.2f}/day = ~${total_rpd * 0.02 * 70 * 30:.1f}/month")
