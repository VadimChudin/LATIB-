import json

# New bot started around 2026-04-12T02:44 UTC (05:44 Moscow)
RESTART_TS = "2026-04-12T02:44"

wins = []
losses = []
all_trades = []

with open('data/trade_log.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        t = json.loads(line)
        if t['event'] != 'EXIT': continue
        if t['ts'] < RESTART_TS: continue
        
        pnl = t.get('pnl_r', 0)
        sym = t['symbol']
        reason = t.get('exit_reason', '?')
        dur = t.get('duration_secs', 0)
        
        entry = {
            'ts': t['ts'],
            'sym': sym,
            'dir': t['direction'],
            'pnl_r': pnl,
            'pnl_pct': t.get('pnl_pct', 0),
            'reason': reason,
            'dur': dur
        }
        all_trades.append(entry)
        if pnl > 0:
            wins.append(entry)
        else:
            losses.append(entry)

print(f"=== LIVE TRADES SINCE RESTART ({RESTART_TS}) ===\n")
print(f"  Total: {len(all_trades)} trades")
print(f"  Wins:  {len(wins)}")
print(f"  Losses: {len(losses)}")
print(f"  WR: {len(wins)/len(all_trades)*100:.1f}%" if all_trades else "")

total_pnl = sum(t['pnl_r'] for t in all_trades)
print(f"  Total PnL: {total_pnl:+.2f}R")
print()

print(f"{'Time':>11} {'Symbol':<16} {'Dir':>5} {'PnL_R':>8} {'Exit':>12} {'Dur':>6}")
print("-" * 65)
for t in all_trades:
    ts_short = t['ts'][11:19]
    mark = "✅" if t['pnl_r'] > 0 else "❌"
    print(f"  {ts_short} {t['sym']:<16} {t['dir']:>5} {t['pnl_r']:>+7.2f}R {t['reason']:>12} {t['dur']:>5}s {mark}")

print(f"\n{'='*65}")
print(f"  TOTAL: {total_pnl:>+7.2f}R  |  WR: {len(wins)}/{len(all_trades)}")

# Per symbol
print(f"\n  PER SYMBOL:")
syms = {}
for t in all_trades:
    s = t['sym']
    if s not in syms: syms[s] = {'pnl': 0, 'n': 0, 'w': 0}
    syms[s]['pnl'] += t['pnl_r']
    syms[s]['n'] += 1
    if t['pnl_r'] > 0: syms[s]['w'] += 1

for s, d in sorted(syms.items(), key=lambda x: x[1]['pnl'], reverse=True):
    print(f"    {s:<16} {d['pnl']:>+7.2f}R  {d['w']}/{d['n']} wins")
