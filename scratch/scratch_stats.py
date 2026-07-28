import json
import collections

with open('data/trade_log.jsonl', 'r', encoding='utf-8') as f:
    lines = [x.strip() for x in f.readlines() if x.strip()]

exits = []
for line in lines:
    if "EXIT" in line:
        try:
            parsed = json.loads(line)
            exits.append(parsed)
        except:
            pass

last_50 = exits[-50:]

if not last_50:
    print("WARNING: Not enough trades found.")
    exit(0)

total_pnl = sum(t.get('pnl_pct', 0.0) for t in last_50 if t.get('pnl_pct') is not None)
wins = sum(1 for t in last_50 if t.get('pnl_pct', 0.0) > 0)
win_rate = (wins / len(last_50)) * 100

strat_stats = collections.defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0.0})
for t in last_50:
    s = t.get('strategy', 'unknown')
    strat_stats[s]['count'] += 1
    val = t.get('pnl_pct', 0.0)
    if val > 0:
        strat_stats[s]['wins'] += 1
    strat_stats[s]['pnl'] += val

print(f'=== STATS FOR LAST {len(last_50)} TRADES ===')
print(f'Total Trades: {len(last_50)}')
print(f'Overall Win Rate: {win_rate:.1f}% ({wins} profitable)')
print(f'Total PnL: {total_pnl:.3f}%')

print(f'\nBreakdown by Strategy:')
for s, stats in strat_stats.items():
    if stats['count'] > 0:
        wr = (stats['wins'] / stats['count']) * 100
    else:
        wr = 0.0
    print(f'  - {s.upper():<10} | Trades: {stats["count"]:<2} | Win Rate: {wr:>5.1f}% | PnL: {stats["pnl"]:>7.3f}%')

best = max(last_50, key=lambda t: t.get('pnl_pct', -100))
worst = min(last_50, key=lambda t: t.get('pnl_pct', 100))

print(f'\nBest Trade: {best.get("symbol")} ({best.get("strategy")}): {best.get("pnl_pct", 0):+.3f}% (Reason: {best.get("exit_reason")})')
print(f'Worst Trade: {worst.get("symbol")} ({worst.get("strategy")}): {worst.get("pnl_pct", 0):+.3f}% (Reason: {worst.get("exit_reason")})')
