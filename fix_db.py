import sqlite3

db_path = 'data/autocore.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

columns = [info[1] for info in c.execute('PRAGMA table_info(trades)').fetchall()]

required_columns = {
    'exit_reason': 'TEXT',
    'pnl_r': 'REAL',
    'duration_minutes': 'REAL',
    'order_id': 'TEXT',
    'sl_order_id': 'TEXT',
    'tp_order_id': 'TEXT',
    'timeframe': 'TEXT'
}

for col_name, col_type in required_columns.items():
    if col_name not in columns:
        try:
            c.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
            print(f"Added missing column: {col_name}")
        except Exception as e:
            print(f"Error adding {col_name}: {e}")

conn.commit()
print("\n--- Последние 20 сделок ---")
try:
    c.execute("SELECT timestamp, symbol, direction, strategy, entry_price, pnl_pct, exit_reason, status FROM trades ORDER BY id DESC LIMIT 20")
    for row in c.fetchall():
        print(f"{row[0][:19]} | {row[1]:<9} | {row[2]:<5} | {str(row[5])+'%':<7} | {row[6]} ({row[7]}) | Strat: {row[3]}")
except Exception as e:
    print("Error querying:", e)

conn.close()
