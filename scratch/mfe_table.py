import json
import datetime

trades = []
with open('data/trade_log.jsonl', 'r') as f:
    for line in f:
        if not line.strip(): continue
        try:
            t = json.loads(line)
            if t.get('exit_reason'): trades.append(t)
        except: pass

recent_trades = trades[-14:]

print("| Time (UTC) | Symbol | Dir | Duration | MFE (%) | MAE (%) | Exit Reason | PnL (R) |")
print("|---|---|---|---|---|---|---|---|")

for t in recent_trades:
    ts_str = t.get('ts', '')
    if ts_str:
        try:
            ts_dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            time_str = ts_dt.strftime("%H:%M:%S")
        except:
            time_str = ts_str[:8]
    else: time_str = "---"
        
    symbol = t.get('symbol', '')
    direction = t.get('direction', '')
    dur = t.get('duration_secs', 0)
    
    # Raw values are already percentages (computed in Rust as `ratio * 100.0`)
    mfe = t.get('mfe_pct', 0.0) 
    mae = t.get('mae_pct', 0.0)
    
    reason = t.get('exit_reason', '')
    pnl = t.get('pnl_r', 0.0)
    
    # Only print MAE/MFE up to 3 decimals
    print(f"| {time_str} | {symbol} | {direction} | {dur}s | {mfe:.3f}% | {mae:.3f}% | {reason} | {pnl:.2f}R |")
