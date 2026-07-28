import json
import pandas as pd

with open('data/journal.json', 'r') as f:
    trades = json.load(f)

df = pd.DataFrame(trades)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter tonight only (after bot restart at ~03:17 March 10)
tonight = df[df['timestamp'] >= '2026-03-10 03:00:00']
print(f"Total trades tonight: {len(tonight)}")

if len(tonight) > 0:
    print(f"Time range: {tonight['timestamp'].min()} -> {tonight['timestamp'].max()}")
    print()
    
    for strat in tonight['strategy'].unique():
        s = tonight[tonight['strategy'] == strat]
        print(f"--- {strat} ---")
        print(f"  Signals: {len(s)}")
        print(f"  Symbols: {', '.join(s['symbol'].unique())}")
        print(f"  Directions: {s['direction'].value_counts().to_dict()}")
        print(f"  Avg ML Prob: {s['ml_prob'].mean():.3f}")
        print(f"  ML Range: {s['ml_prob'].min():.3f} - {s['ml_prob'].max():.3f}")
        print()
else:
    print("No trades found tonight!")
    print(f"Last trade in journal: {df['timestamp'].max()}")
    print(f"Total entries in journal: {len(df)}")
    print(f"\nLast 5 trades:")
    for _, row in df.tail(5).iterrows():
        print(f"  {row['timestamp']} | {row['symbol']} | {row['strategy']} | {row['direction']}")
