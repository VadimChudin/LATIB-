import json

with open('data/active_config.json') as f:
    configs = json.load(f)

kt = [c for c in configs if c.get('strategy') == 'knife_tick']
print(f'Total knife_tick configs: {len(kt)}')
print()

header = f"{'Symbol':<20} {'Test WR':>10} {'Train WR':>10} {'Trades':>8} {'Fitness':>10}"
print(header)
print('-' * 62)

for c in sorted(kt, key=lambda x: x.get('metrics',{}).get('win_rate',0), reverse=True):
    m = c.get('metrics', {})
    twr = m.get('win_rate', 0) * 100
    trwr = m.get('train_wr', 0) * 100
    trades = m.get('total_trades', 0)
    fit = m.get('fitness', 0)
    sym = c['symbol']
    print(f"  {sym:<18} {twr:>9.1f}% {trwr:>9.1f}% {trades:>8} {fit:>10.1f}")

# Summary stats
if kt:
    avg_wr = sum(c.get('metrics',{}).get('win_rate',0) for c in kt) / len(kt) * 100
    print(f"\nAvg Test WR: {avg_wr:.1f}%")
    good = [c for c in kt if c.get('metrics',{}).get('win_rate',0) > 0.40]
    print(f"Symbols with Test WR > 40%: {len(good)}/{len(kt)}")
    
    great = [c for c in kt if c.get('metrics',{}).get('win_rate',0) > 0.50]
    print(f"Symbols with Test WR > 50%: {len(great)}/{len(kt)}")
    
    # Show top 20 by fitness
    print(f"\n{'='*62}")
    print("TOP 20 BY FITNESS:")
    print('='*62)
    for c in sorted(kt, key=lambda x: x.get('metrics',{}).get('fitness',0), reverse=True)[:20]:
        m = c.get('metrics', {})
        twr = m.get('win_rate', 0) * 100
        trwr = m.get('train_wr', 0) * 100
        trades = m.get('total_trades', 0)
        fit = m.get('fitness', 0)
        sym = c['symbol']
        # Check for overfit: train vs test WR gap
        gap = abs(trwr - twr)
        flag = "!!" if gap > 15 else "  "
        print(f"  {flag} {sym:<18} Test={twr:>5.1f}% Train={trwr:>5.1f}% gap={gap:>4.1f}% trades={trades:>4} fit={fit:>7.1f}")

    # Check PnL from tick_params
    print(f"\n{'='*62}")
    print("PNL CHECK (from tick_params):")
    print('='*62)
    import os
    total_test_pnl = 0
    total_train_pnl = 0
    count = 0
    for c in kt:
        sym = c['symbol']
        pf = f'data/tick_params/{sym}.json'
        if os.path.exists(pf):
            with open(pf) as f2:
                p = json.load(f2)
            test_pnl = p.get('test_pnl_r', 0)
            train_pnl = p.get('train_pnl_r', 0)
            test_wr = p.get('test_wr', 0)
            train_wr = p.get('train_wr', 0)
            test_trades = p.get('test_trades', 0)
            total_test_pnl += test_pnl
            total_train_pnl += train_pnl
            count += 1
            if test_pnl > 0:
                print(f"  + {sym:<18} Test: {test_pnl:>+7.2f}R ({test_wr:.1f}% WR, {test_trades}t) | Train: {train_pnl:>+7.2f}R ({train_wr:.1f}%)")

    print(f"\n  TOTAL TEST PnL:  {total_test_pnl:>+8.2f}R across {count} symbols")
    print(f"  TOTAL TRAIN PnL: {total_train_pnl:>+8.2f}R")
