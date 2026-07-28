
import json
from datetime import datetime, timedelta

def calc():
    now = datetime(2026, 3, 15, 23, 1)
    window = now - timedelta(hours=24)
    trades = 0
    pnl = 0.0
    wins = 0
    with open("data/trade_log.jsonl", "r") as f:
        for line in f:
            if '"event":"EXIT"' in line:
                data = json.loads(line)
                ts = datetime.strptime(data['ts'], "%Y-%m-%dT%H:%M:%SZ")
                if ts >= window:
                    trades += 1
                    val = data.get('pnl_usd', 0.0)
                    pnl += val
                    if val > 0:
                        wins += 1
    print(f"Total Trades (24h): {trades}")
    print(f"Realized PnL (24h): ${pnl:.2f}")
    if trades > 0:
        print(f"Win Rate: {wins/trades*100:.1f}%")

if __name__ == "__main__":
    calc()
