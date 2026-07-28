import json

trades = []
with open('data/trade_log.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        trades.append(json.loads(line))

# Build paired trades
pairs = []
entry_map = {}
for t in trades:
    if t.get('event') == 'ENTRY':
        entry_map[t['trade_id']] = t
    elif t.get('event') == 'EXIT':
        tid = t['trade_id']
        if tid in entry_map:
            pairs.append((entry_map[tid], t))

# Filter knife_tick only
kt_pairs = [(en, ex) for en, ex in pairs if ex.get('strategy') == 'knife_tick']

total_r = sum(e['pnl_r'] for _, e in kt_pairs)
n = len(kt_pairs)
wins = len([1 for _, e in kt_pairs if e['pnl_r'] > 0])
losses = n - wins

print("=== KNIFE_TICK SUMMARY ===")
print(f"Total trades: {n}")
print(f"Wins: {wins}, Losses: {losses}")
wr = wins / n * 100 if n > 0 else 0
print(f"Win Rate: {wr:.1f}%")
print(f"Total PnL R: {total_r:.2f}")
if n > 0:
    print(f"Avg PnL R: {total_r / n:.3f}")

# Per-reason breakdown
reasons = {}
for _, e in kt_pairs:
    r = e.get('exit_reason', '?')
    if r not in reasons:
        reasons[r] = {'count': 0, 'total_r': 0.0, 'avg_pnl_pct': 0.0}
    reasons[r]['count'] += 1
    reasons[r]['total_r'] += e['pnl_r']
    reasons[r]['avg_pnl_pct'] += e.get('pnl_pct', 0)

print("\n=== BY EXIT REASON ===")
for r, s in sorted(reasons.items()):
    avg_r = s['total_r'] / s['count']
    avg_pct = s['avg_pnl_pct'] / s['count']
    cnt = s['count']
    tr = s['total_r']
    print(f"  {r:15s}: {cnt:3d} trades | Total R: {tr:+8.2f} | Avg R: {avg_r:+.3f} | Avg PnL%: {avg_pct:+.3f}%")

# Per-symbol breakdown
symbols = {}
for en, ex in kt_pairs:
    sym = ex.get('symbol', '?')
    if sym not in symbols:
        symbols[sym] = {'count': 0, 'total_r': 0.0, 'wins': 0, 'risk_dists': [], 'durations': [], 'sl_hits': 0}
    symbols[sym]['count'] += 1
    symbols[sym]['total_r'] += ex['pnl_r']
    if ex['pnl_r'] > 0:
        symbols[sym]['wins'] += 1
    if en.get('risk_dist'):
        symbols[sym]['risk_dists'].append(en['risk_dist'])
    symbols[sym]['durations'].append(ex.get('duration_secs', 0))
    if ex.get('exit_reason') == 'SL':
        symbols[sym]['sl_hits'] += 1

print("\n=== BY SYMBOL ===")
for sym, s in sorted(symbols.items(), key=lambda x: x[1]['total_r']):
    wr_s = s['wins'] / s['count'] * 100 if s['count'] > 0 else 0
    avg_risk = sum(s['risk_dists']) / len(s['risk_dists']) if s['risk_dists'] else 0
    avg_dur = sum(s['durations']) / len(s['durations']) if s['durations'] else 0
    cnt = s['count']
    tr = s['total_r']
    sl = s['sl_hits']
    print(f"  {sym:15s}: {cnt:3d} trades | WR={wr_s:5.1f}% | Total R: {tr:+8.2f} | SL hits: {sl} | Avg risk_dist: {avg_risk:.6f} | Avg dur: {avg_dur:.0f}s")

# ==========================================================================
# KEY ANALYSIS: What DE backtests vs what actually happens live
# ==========================================================================
print("\n" + "=" * 80)
print("=== CRITICAL: SL HITS ANALYSIS (Backtest vs Live Discrepancy) ===")
print("=" * 80)

for en, ex in kt_pairs:
    if ex.get('exit_reason') != 'SL':
        continue
    risk_dist = en.get('risk_dist', 0)
    entry = en.get('entry_price', 0)
    sl = en.get('sl_price', 0)
    exit_p = ex.get('exit_price', 0)
    pnl_pct = ex.get('pnl_pct', 0)
    direction = en.get('direction', '')
    entry_sl_dist_pct = abs(entry - sl) / entry * 100 if entry > 0 else 0
    actual_loss_pct = abs(pnl_pct)
    print(f"  {en['symbol']:15s} {direction:5s} | Entry={entry:.6f} SL={sl:.6f} Exit={exit_p:.6f} | SL dist: {entry_sl_dist_pct:.3f}% | Actual loss: {actual_loss_pct:.3f}% | PnL R: {ex['pnl_r']:+.3f}")

# ==========================================================================
# KEY ANALYSIS #2: SMART_EXIT losses — Brain exits that lose money
# ==========================================================================
print("\n" + "=" * 80)
print("=== SMART_EXIT LOSSES (Brain lost money) ===")
print("=" * 80)

smart_loses = [(en, ex) for en, ex in kt_pairs if ex.get('exit_reason') == 'SMART_EXIT' and ex['pnl_r'] < 0]
for en, ex in smart_loses:
    risk_dist = en.get('risk_dist', 0)
    entry = en.get('entry_price', 0)
    sl = en.get('sl_price', 0)
    exit_p = ex.get('exit_price', 0)
    pnl_pct = ex.get('pnl_pct', 0)
    direction = en.get('direction', '')
    dur = ex.get('duration_secs', 0)
    mfe = ex.get('mfe_pct', 0)
    entry_sl_dist_pct = abs(entry - sl) / entry * 100 if entry > 0 else 0
    print(f"  {en['symbol']:15s} {direction:5s} | dur={dur:4d}s | Entry={entry:.6f} SL={sl:.6f} Exit={exit_p:.6f} | SL dist: {entry_sl_dist_pct:.3f}% | PnL: {pnl_pct:+.3f}% ({ex['pnl_r']:+.3f}R) | MFE: {mfe:.3f}%")

# ==========================================================================
# KEY ANALYSIS #3: Comparing DE SL distance vs live actual risk_dist logged
# ==========================================================================
print("\n" + "=" * 80)
print("=== DE CONFIG sl_buffer_pct vs LIVE risk_dist ===")
print("=" * 80)

# Load active config
with open('data/active_config.json', 'r') as f:
    configs = json.load(f)

config_map = {}
for c in configs:
    sym = c['symbol'].replace('_', '/')
    params = c.get('params', {})
    sl_pct = params.get('sl_buffer_pct', params.get('sl_pct', 0))
    be_pct = params.get('be_trigger_pct', 0)
    trail = params.get('trail_pct', 0)
    config_map[sym] = {'sl_pct': sl_pct, 'be_pct': be_pct, 'trail': trail}

for en, ex in kt_pairs[-30:]:  # Last 30 trades
    sym = en.get('symbol', '?')
    entry = en.get('entry_price', 0)
    sl = en.get('sl_price', 0)
    risk_dist = en.get('risk_dist', 0)
    
    live_sl_pct = abs(entry - sl) / entry * 100 if entry > 0 else 0
    de_sl_pct = config_map.get(sym, {}).get('sl_pct', 0) * 100
    
    flag = "!! MISMATCH" if abs(live_sl_pct - de_sl_pct) > 0.05 else "   OK"
    reason = ex.get('exit_reason','?')
    dur = ex.get('duration_secs',0)
    print(f"  {flag} {sym:15s} | DE sl_pct={de_sl_pct:.3f}% | LIVE SL dist={live_sl_pct:.3f}% | risk_dist={risk_dist:.6f} | dur={dur}s | reason={reason}")
