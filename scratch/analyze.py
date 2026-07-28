import json
import collections
import statistics

entries = {}
trades = []

with open(r'd:\LAITB 2.0\data\trade_log.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        try:
            d = json.loads(line)
            if d.get("strategy") != "knife_tick": continue
            tid = d["trade_id"]
            if d["event"] == "ENTRY":
                entries[tid] = d
            elif d["event"] == "EXIT" and tid in entries:
                trade = {**entries[tid], **d}
                trades.append(trade)
        except Exception:
            pass

# Take only the last 35 trades
trades = trades[-35:]

winners = [t for t in trades if t.get("pnl_r", 0) > 0.3 or t.get("exit_reason") == "SMART_EXIT"]
losers = [t for t in trades if t.get("pnl_r", 0) < -0.8]

def get_normalized_metrics(t):
    cvd = t.get("cvd_delta", 0)
    imb = t.get("imbalance_ratio", 1)
    if t["direction"] == "LONG":
        return -cvd, (1/imb if imb > 0 else 1)
    else:
        return cvd, imb

def print_stats(group, name):
    if not group: return
    metrics = [get_normalized_metrics(t) for t in group]
    cvd = [m[0] for m in metrics]
    imb = [m[1] for m in metrics]
    speed = [t.get("tape_speed", 0) for t in group if t.get("tape_speed") is not None]
    mfe = [t.get("mfe_pct", 0) for t in group if t.get("mfe_pct") is not None]
    dur = [t.get("duration_secs", 0) for t in group if t.get("duration_secs") is not None]

    print(f"\n--- {name} (Count: {len(group)}) ---")
    print(f"CVD Entry Agression: {statistics.mean(cvd):.3f} (Med {statistics.median(cvd):.3f})")
    print(f"Imbalance Agression: {statistics.mean(imb):.1f}x (Med {statistics.median(imb):.1f}x)")
    print(f"Tape Speed (t/s):    {statistics.mean(speed):.1f}   (Median: {statistics.median(speed):.1f})")
    print(f"Max Favorable %:     {statistics.mean(mfe):.3f}%")
    print(f"Duration Secs:       {statistics.mean(dur):.0f}s")

print(f"Total analyzed recent trades: {len(trades)}")
print_stats(winners, "WINNERS (Good Bounce >0.3R)")
print_stats(losers, "LOSERS (Full SL Hit)")

# Print specific list of Losers to see exactly what symbol/speed they had
print("\n--- LOSERS DETAILS ---")
for t in losers:
    m = get_normalized_metrics(t)
    print(f"{t['symbol']} {t['direction']:>5} | CVD: {m[0]:.3f} | Spd: {t.get('tape_speed',0):.1f} | Dur: {t.get('duration_secs',0):>3}s")
