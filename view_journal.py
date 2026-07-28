import json
import os
import pandas as pd

JOURNAL_PATH = 'data/journal.json'

def main():
    if not os.path.exists(JOURNAL_PATH):
        print(f"Journal not found at {JOURNAL_PATH}")
        return
        
    try:
        with open(JOURNAL_PATH, 'r') as f:
            data = json.load(f)
            
        print(f"{"="*50}")
        print("📊 AEGIS JOURNAL.JSON RESULTS")
        print(f"{"="*50}")
        print(f"Total Entries Recorded: {len(data)}")
        
        if len(data) == 0:
            print("Journal is empty.")
            return

        # Check for exit identifiers
        closed_trades = [t for t in data if 'exit_price' in t or 'pnl_usd' in t or 'status' in t and t['status'] == 'CLOSED']
        print(f"Entries with Exit Data (Closed Trades): {len(closed_trades)}")
        
        if len(closed_trades) > 0:
            df = pd.DataFrame(closed_trades)
            if 'pnl_usd' in df.columns:
                pnl = df['pnl_usd'].sum()
                print(f"Total Realized PnL (from JSON): ${pnl:.2f}")
                
            print("\n🕒 Last 5 Closed Trades from JSON:")
            print(df.tail(5).to_string(index=False))
        else:
            print("\nNo closed trades found in the JSON file. All entries look like 'OPEN' logs.")
            
            print("\n🕒 Last 3 Open Entries:")
            df_open = pd.DataFrame(data[-3:])
            print(df_open.to_string(index=False))
            
    except Exception as e:
        print(f"❌ Error reading journal: {e}")

if __name__ == '__main__':
    main()
