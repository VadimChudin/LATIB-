import os
import json

log_path = os.path.join('d:\\', 'LAITB 2.0', 'data', 'trade_log.jsonl')
out_path = os.path.join(os.environ.get('APPDATA', 'C:\\Users\\вадим\\AppData\\Roaming').replace('Roaming', '.gemini\\antigravity\\brain\\d31068fc-e008-405f-aba1-58f54f50e8de\\artifacts'), 'last_trades.md')

if not os.path.exists(log_path):
    print("Log not found")
else:
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    recent_knifes = []
    
    # scan from end
    for line in reversed(lines):
        try:
            data = json.loads(line.strip())
            # collect only knife_tick
            if "knife_tick" in data.get('strategy', '').lower() or "knifetick" in data.get('strategy', '').lower():
                recent_knifes.append(data)
                if len(recent_knifes) >= 10:
                    break
        except:
            pass

    recent_knifes.reverse()
    
    with open('d:\\LAITB 2.0\\data\\recent_knife_trades_report.txt', 'w', encoding='utf-8') as f:
        f.write("# Последние трейды Ножа:\n")
        for trade in recent_knifes:
            f.write(json.dumps(trade, indent=2) + "\n")
