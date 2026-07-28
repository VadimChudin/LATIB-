import json
import numpy as np

# Load logs
trades = []
with open('data/trade_log.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if '"EXIT"' in line and '"knife_tick"' in line:
            try:
                trades.append(json.loads(line))
            except:
                pass

# Take the last 55 trades
trades = trades[-55:]

if not trades:
    print("No knife trades found.")
    exit(0)

wins = [t for t in trades if t.get('pnl_pct', 0) > 0]
losses = [t for t in trades if t.get('pnl_pct', 0) <= 0]

# MFE/MAE analysis
mfe_all = [t.get('mfe_pct', 0) for t in trades if t.get('mfe_pct') is not None]
mae_all = [t.get('mae_pct', 0) for t in trades if t.get('mae_pct') is not None]
mfe_losses = [t.get('mfe_pct', 0) for t in losses if t.get('mfe_pct') is not None]

# Exit reasons
reasons = {}
for t in trades:
    r = t.get('exit_reason', 'UNKNOWN')
    reasons[r] = reasons.get(r, 0) + 1

# Durations
durations_wins = [t.get('duration_secs', 0) for t in wins]
durations_losses = [t.get('duration_secs', 0) for t in losses]

print(f"--- KNIFE TICK ANALYSIS ({len(trades)} TRADES) ---")
print(f"Wr: {len(wins)} / {len(trades)} ({len(wins)/len(trades)*100:.1f}%)")
print(f"Reasons: {reasons}")
print(f"Avg Duration (Wins): {np.mean(durations_wins) if durations_wins else 0:.1f}s")
print(f"Avg Duration (Losses): {np.mean(durations_losses) if durations_losses else 0:.1f}s")
print(f"Avg MFE (All): {np.mean(mfe_all):.3f}% | Avg MAE (All): {np.mean(mae_all):.3f}%")
print(f"Avg MFE on Losses (Missed TP?): {np.mean(mfe_losses):.3f}%")

# Count how many losses had an MFE > 0.15% before hitting SL
missed_tps = sum(1 for m in mfe_losses if m > 0.15)
print(f"Losses with MFE > 0.15%: {missed_tps} / {len(losses)}")
