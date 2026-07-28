import json, os
from pathlib import Path

print("=== ALL OOS PROFITABLE (min 5 trades) ===\n")
results = []

for f in sorted(os.listdir('data/tick_params')):
    if not f.endswith('.json'): continue
    with open(f'data/tick_params/{f}') as fh:
        p = json.load(fh)
    test_pnl = p.get('test_pnl_r', 0)
    test_wr = p.get('test_wr', 0)
    test_trades = p.get('test_trades', 0)
    train_wr = p.get('train_wr', 0)
    train_pnl = p.get('train_pnl_r', 0)
    gap = abs(train_wr - test_wr)
    sym = f.replace('.json', '')

    if test_pnl > 0 and test_trades >= 5:
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
        test_days = days * 0.3
        tpd = test_trades / test_days if test_days > 0 else 0
        rpd = test_pnl / test_days if test_days > 0 else 0

        strict = "✅" if test_wr > 35 and gap < 20 else "  "
        results.append((sym, test_pnl, test_wr, train_wr, gap, test_trades, tpd, rpd, strict))

results.sort(key=lambda x: x[1], reverse=True)

print(f"{'':>2} {'Symbol':<18} {'PnL':>8} {'WR':>6} {'Gap':>6} {'Trades':>7} {'T/day':>6} {'R/day':>7}")
print("-" * 65)
total_tpd = 0
total_rpd = 0
for r in results:
    sym, pnl, wr, twr, gap, trades, tpd, rpd, strict = r
    print(f"{strict} {sym:<18} {pnl:>+7.2f}R {wr:>5.1f}% {gap:>5.1f}% {trades:>7} {tpd:>6.2f} {rpd:>+7.3f}")
    total_tpd += tpd
    total_rpd += rpd

print("-" * 65)
print(f"   {'TOTAL':<18} {'':>8} {'':>6} {'':>6} {sum(r[5] for r in results):>7} {total_tpd:>6.1f} {total_rpd:>+7.3f}")
print(f"\n   ~{total_tpd:.0f} trades/day, ~{total_rpd:+.2f}R/day")
print(f"   At 2% risk $70: ~${total_rpd * 0.02 * 70:.2f}/day = ~${total_rpd * 0.02 * 70 * 30:.0f}/month")
