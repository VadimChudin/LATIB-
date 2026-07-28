import json
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

# Get the trades since compilation (from ~12:00 UTC = 15:00 user time)
session_trades = [t for t in trades if getattr(t, "ts", t.get("ts", "")) > "2026-04-09T12:00:00Z"]

losses = [t for t in session_trades if t.get("pnl_r", 0) <= 0]
wins = [t for t in session_trades if t.get("pnl_r", 0) > 0]

print(f"--- CURRENT SESSION ({len(session_trades)} trades) ---")
print(f"Wins: {len(wins)} | Losses: {len(losses)}")

print("\n[LOSSES DETAILED]")
for t in losses:
    pnl = t.get("pnl_r", 0)
    mfe = t.get("mfe_pct", 0)
    spd = t.get("tape_speed", 0)
    exit_rc = t.get("exit_reason", "")
    print(f"{t['ts'][11:19]} {t['symbol']:10} {t['direction']:5} | PnL: {pnl:+6.2f}R | MFE: +{mfe:5.3f}% | Spd: {spd:5.1f} | Dur: {t.get('duration_secs',0):4}s | {exit_rc}")

print("\n[WINS DETAILED]")
for t in wins:
    pnl = t.get("pnl_r", 0)
    mfe = t.get("mfe_pct", 0)
    spd = t.get("tape_speed", 0)
    exit_rc = t.get("exit_reason", "")
    print(f"{t['ts'][11:19]} {t['symbol']:10} {t['direction']:5} | PnL: {pnl:+6.2f}R | MFE: +{mfe:5.3f}% | Spd: {spd:5.1f} | Dur: {t.get('duration_secs',0):4}s | {exit_rc}")

