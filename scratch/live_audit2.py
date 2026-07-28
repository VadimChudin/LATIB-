import json

RESTART = '2026-04-12T07:40'

entries = []
exits = []

with open('data/trade_log.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        if t['ts'] < RESTART:
            continue
        if t['event'] == 'ENTRY':
            entries.append(t)
        elif t['event'] == 'EXIT':
            exits.append(t)

print(f"=== SINCE {RESTART} ===")
print(f"  Entries: {len(entries)}")
print(f"  Exits:  {len(exits)}")
print()

# Show entries with SL/TP distances
print(f"{'Time':>8} {'Symbol':<16} {'Dir':>5} {'Entry':>10} {'SL dist%':>9} {'TP dist%':>9} {'R:R':>5}")
print("-" * 70)

for e in entries[-40:]:
    ts = e['ts'][11:19]
    sym = e['symbol']
    d = e['direction']
    ep = e['entry_price']
    sl = e['sl_price']
    tp = e.get('tp_price', ep)
    
    sl_dist = abs(ep - sl)
    sl_pct = sl_dist / ep * 100 if ep > 0 else 0
    tp_dist = abs(ep - tp)
    tp_pct = tp_dist / ep * 100 if ep > 0 else 0
    rr = tp_dist / sl_dist if sl_dist > 0 else 0
    
    flag = "⚠️" if sl_pct < 0.01 else ""
    print(f"  {ts} {sym:<16} {d:>5} {ep:>10.6f} {sl_pct:>8.4f}% {tp_pct:>8.3f}% {rr:>5.1f} {flag}")

# Show exits with PnL  
print(f"\n{'Time':>8} {'Symbol':<16} {'PnL_R':>8} {'Exit':>12} {'Dur':>6}")
print("-" * 55)
total_pnl = 0
for e in exits[-40:]:
    ts = e['ts'][11:19]
    pnl = e.get('pnl_r', 0)
    total_pnl += pnl
    print(f"  {ts} {e['symbol']:<16} {pnl:>+7.2f}R {e.get('exit_reason','?'):>12} {e.get('duration_secs',0):>5}s")

print(f"\n  Total PnL: {total_pnl:>+.2f}R in {len(exits)} trades")

# Time between entries
if len(entries) >= 2:
    from datetime import datetime
    times = []
    for e in entries:
        t = datetime.fromisoformat(e['ts'].replace('Z', '+00:00'))
        times.append(t)
    gaps = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
    avg_gap = sum(gaps) / len(gaps)
    min_gap = min(gaps)
    print(f"\n  Avg gap between entries: {avg_gap:.0f}s ({avg_gap/60:.1f}m)")
    print(f"  Min gap: {min_gap:.0f}s")
    print(f"  Trade rate: {3600/avg_gap:.1f} trades/hour")
