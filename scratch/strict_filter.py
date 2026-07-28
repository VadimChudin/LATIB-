import json, os

profitable = []
for f in sorted(os.listdir('data/tick_params')):
    if not f.endswith('.json'):
        continue
    with open(f'data/tick_params/{f}') as fh:
        p = json.load(fh)
    test_pnl = p.get('test_pnl_r', 0)
    test_wr = p.get('test_wr', 0)
    test_trades = p.get('test_trades', 0)
    train_wr = p.get('train_wr', 0)
    gap = abs(train_wr - test_wr)
    sym = f.replace('.json', '')

    # STRICT: OOS profitable + WR > 35% + gap < 20% + min 5 trades
    if test_pnl > 0 and test_wr > 35 and gap < 20 and test_trades >= 5:
        profitable.append({
            'sym': sym,
            'test_pnl': test_pnl,
            'test_wr': test_wr,
            'train_wr': train_wr,
            'gap': gap,
            'trades': test_trades,
        })

print("=== STRICT FILTER: OOS profitable + WR>35% + gap<20% + min 5t ===")
print(f"Passed: {len(profitable)} symbols\n")
for p in sorted(profitable, key=lambda x: x['test_pnl'], reverse=True):
    print(f"  {p['sym']:<20} PnL={p['test_pnl']:>+6.2f}R | WR={p['test_wr']:>5.1f}% | gap={p['gap']:>4.1f}% | trades={p['trades']}")

# RELAXED: just OOS positive
relaxed = []
for f in sorted(os.listdir('data/tick_params')):
    if not f.endswith('.json'):
        continue
    with open(f'data/tick_params/{f}') as fh:
        p = json.load(fh)
    test_pnl = p.get('test_pnl_r', 0)
    test_wr = p.get('test_wr', 0)
    test_trades = p.get('test_trades', 0)
    if test_pnl > 0 and test_trades >= 5:
        relaxed.append(f.replace('.json',''))

print(f"\n=== RELAXED FILTER: OOS positive + min 5 trades ===")
print(f"Passed: {len(relaxed)} symbols")
