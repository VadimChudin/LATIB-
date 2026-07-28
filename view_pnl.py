import sqlite3
import pandas as pd
import os

DB_PATH = 'data/autocore.db'

def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return
        
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query('SELECT * FROM trades', conn)
        
        print(f"{"="*50}")
        print("📊 AEGIS PAPER TRADING RESULTS")
        print(f"{"="*50}")
        print(f"Total Trades Recorded: {len(df)}")
        
        if len(df) == 0:
            print("No trades found in the database yet. The bot hasn't opened any positions.")
            return

        open_trades = df[df.status == "OPEN"]
        print(f"Currently Open Trades: {len(open_trades)}")
        
        closed = df[df.status == "CLOSED"]
        print(f"Closed Trades: {len(closed)}")
        
        if len(closed) > 0:
            pnl = closed.pnl_usd.sum()
            win_rate = len(closed[closed.pnl_usd > 0]) / len(closed) * 100
            print(f"Total Realized PnL: ${pnl:.2f}")
            print(f"Win Rate: {win_rate:.1f}%")
            
            print("\n🕒 Last 10 Trades:")
            columns_to_show = ['id', 'symbol', 'strategy', 'direction', 'entry_time', 'pnl_usd', 'status']
            # Only pick columns that exist
            existing_cols = [c for c in columns_to_show if c in df.columns]
            print(df[existing_cols].tail(10).to_string(index=False))
        else:
            print("\nNo closed trades yet to calculate PnL.")
            
    except Exception as e:
        print(f"❌ Error reading database: {e}")

if __name__ == '__main__':
    main()
