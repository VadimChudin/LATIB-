import json

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
        except: pass

# Full session from compile time
session = [t for t in trades if t.get("ts","") >= "2026-04-09T03:30:00Z"]

wins = [t for t in session if t.get("pnl_r",0) > 0.01]
losses = [t for t in session if t.get("pnl_r",0) < -0.01]
be = [t for t in session if -0.01 <= t.get("pnl_r",0) <= 0.01]

total_win_r = sum(t["pnl_r"] for t in wins)
total_loss_r = sum(t["pnl_r"] for t in losses)

print(f"=== FULL SESSION: {len(session)} trades ===")
print(f"WINS: {len(wins)} | LOSSES: {len(losses)} | BE: {len(be)}")
print(f"Total Win R: {total_win_r:+.2f}R | Total Loss R: {total_loss_r:+.2f}R | NET: {total_win_r+total_loss_r:+.2f}R")
print(f"Avg Win: {total_win_r/max(1,len(wins)):+.2f}R | Avg Loss: {total_loss_r/max(1,len(losses)):+.2f}R")

# Per symbol
from collections import defaultdict
sym_stats = defaultdict(lambda: {"w":0,"l":0,"be":0,"pnl":0.0,"full_sl":0})
for t in session:
    s = t["symbol"]
    r = t.get("pnl_r",0)
    sym_stats[s]["pnl"] += r
    if r > 0.01: sym_stats[s]["w"] += 1
    elif r < -0.01: sym_stats[s]["l"] += 1
    else: sym_stats[s]["be"] += 1
    if r < -0.9: sym_stats[s]["full_sl"] += 1

print("\n=== PER SYMBOL ===")
print(f"{'Symbol':12} {'W':>3} {'L':>3} {'BE':>3} {'PnL':>8} {'FullSL':>6}")
for sym in sorted(sym_stats, key=lambda s: sym_stats[s]["pnl"]):
    d = sym_stats[sym]
    print(f"{sym:12} {d['w']:3} {d['l']:3} {d['be']:3} {d['pnl']:+8.2f}R {d['full_sl']:6}")

# Loss categories
micro = [t for t in losses if t["pnl_r"] >= -0.20]
medium = [t for t in losses if -0.80 < t["pnl_r"] < -0.20]
heavy = [t for t in losses if t["pnl_r"] <= -0.80]

print(f"\n=== LOSS BREAKDOWN ===")
print(f"Micro (-0.01 to -0.20R): {len(micro)} trades, total: {sum(t['pnl_r'] for t in micro):+.2f}R")
print(f"Medium (-0.20 to -0.80R): {len(medium)} trades, total: {sum(t['pnl_r'] for t in medium):+.2f}R")
print(f"Heavy (< -0.80R / full SL): {len(heavy)} trades, total: {sum(t['pnl_r'] for t in heavy):+.2f}R")

# MFE analysis
print(f"\n=== MFE ANALYSIS ===")
dead_entries = [t for t in losses if t.get("mfe_pct",0) < 0.20]
bounced_losses = [t for t in losses if t.get("mfe_pct",0) >= 0.20]
print(f"Dead entries (MFE < 0.20%, no bounce): {len(dead_entries)} / {len(losses)} losses")
print(f"  -> Total damage: {sum(t['pnl_r'] for t in dead_entries):+.2f}R")
print(f"Bounced but lost (MFE >= 0.20%, had a chance): {len(bounced_losses)} / {len(losses)} losses")
print(f"  -> Total damage: {sum(t['pnl_r'] for t in bounced_losses):+.2f}R")
print(f"  -> These could be saved by better trailing/Dynamic Targets")
