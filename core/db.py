import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger("Aegis.DB")

class CortexDB:
    """
    Unified Database Manager for Aegis System V2.0.
    Consolidates 'core/db_manager.py' and 'data/database.py' into a single source of truth.
    """
    
    def __init__(self, db_path: str = "data/autocore.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initializes the SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Table: Trades (Consolidated)
        # Status can be 'OPEN', 'CLOSED', 'CANCELLED'
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            timeframe TEXT,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            sl_price REAL NOT NULL,
            tp_price REAL,
            size REAL NOT NULL,
            leverage REAL NOT NULL,
            risk_usd REAL NOT NULL,
            ml_prob REAL NOT NULL,
            status TEXT DEFAULT 'OPEN',
            pnl_usd REAL,
            pnl_pct REAL,
            pnl_r REAL,
            exit_reason TEXT,
            duration_minutes REAL,
            params_json TEXT,
            order_id TEXT, -- Binance Order ID for tracking
            sl_order_id TEXT,
            tp_order_id TEXT
        )''')
        
        # Table: Backtest Runs (from data/database.py)
        c.execute('''CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            best_strategy TEXT,
            best_timeframe TEXT,
            win_rate REAL,
            profit_factor REAL,
            max_drawdown REAL,
            custom_metric REAL,
            params_json TEXT
        )''')
        
        # Table: System Logs (for audit)
        c.execute('''CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            message TEXT,
            component TEXT
        )''')
        
        conn.commit()
        conn.close()
        logger.info(f"CortexDB initialized at {self.db_path}")

    def log_trade_open(self, symbol: str, strategy: str, direction: str, 
                      entry: float, sl: float, tp: float, size: float, 
                      lev: float, risk: float, ml_prob: float, 
                      timeframe: str = '5m', params: dict = None,
                      order_id: str = None) -> int:
        """Logs a newly opened trade."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            ts = datetime.utcnow().isoformat()
            
            c.execute('''INSERT INTO trades (
                timestamp, symbol, strategy, timeframe, direction, 
                entry_price, sl_price, tp_price, size, leverage, 
                risk_usd, ml_prob, params_json, order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (ts, symbol, strategy, timeframe, direction, entry, sl, tp, 
             size, lev, risk, ml_prob, json.dumps(params or {}), order_id))
            
            conn.commit()
            trade_id = c.lastrowid
            conn.close()
            return trade_id
        except Exception as e:
            logger.error(f"Failed to log trade open: {e}")
            return None

    def close_trade(self, trade_id: int, exit_price: float, pnl_usd: float, 
                    pnl_pct: float, pnl_r: float = 0.0, reason: str = 'TP/SL'):
        """Updates a trade record with exit data and calculates duration."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Fetch start time to calculate duration
            c.execute("SELECT timestamp FROM trades WHERE id=?", (trade_id,))
            row = c.fetchone()
            duration = 0
            if row:
                start_ts = datetime.fromisoformat(row[0])
                duration = (datetime.utcnow() - start_ts).total_seconds() / 60
            
            c.execute('''UPDATE trades SET 
                status='CLOSED', exit_price=?, pnl_usd=?, pnl_pct=?, 
                pnl_r=?, exit_reason=?, duration_minutes=?
                WHERE id=?''', 
                (exit_price, pnl_usd, pnl_pct, pnl_r, reason, duration, trade_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return False

    def get_open_trades(self, symbol: str = None) -> List[Dict]:
        """Returns all currently open trades."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            if symbol:
                c.execute("SELECT * FROM trades WHERE status='OPEN' AND symbol=?", (symbol,))
            else:
                c.execute("SELECT * FROM trades WHERE status='OPEN'")
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Failed to fetch open trades: {e}")
            return []

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Returns last N trades (open or closed)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Failed to fetch recent trades: {e}")
            return []

    def get_last_closed_trade_id(self) -> int:
        """Returns the ID of the most recently closed trade."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT MAX(id) FROM trades WHERE status != 'OPEN'")
            result = c.fetchone()
            conn.close()
            return result[0] or 0
        except Exception:
            return 0

    def get_new_closed_trades(self, since_id: int) -> List[Dict]:
        """Fetches trades that reached a final status since a certain ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM trades WHERE status != 'OPEN' AND id > ?", (since_id,))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows
        except Exception:
            return []

    def get_total_pnl(self) -> float:
        """Calculates total realized PnL from all closed trades."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT SUM(pnl_usd) FROM trades WHERE status != 'OPEN'")
            result = c.fetchone()
            conn.close()
            return result[0] or 0.0
        except Exception:
            return 0.0

    def get_performance_stats(self) -> dict:
        """Returns aggregated performance stats (Win Rate, Total PnL, Win count)."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM trades WHERE status != 'OPEN'")
            total_closed = c.fetchone()[0] or 0
            
            c.execute("SELECT COUNT(*) FROM trades WHERE status != 'OPEN' AND pnl_usd > 0")
            wins = c.fetchone()[0] or 0
            
            c.execute("SELECT SUM(pnl_usd) FROM trades WHERE status != 'OPEN'")
            pnl = c.fetchone()[0] or 0.0
            
            conn.close()
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
            
            return {
                "total_trades": total_closed,
                "winning_trades": wins,
                "win_rate": win_rate,
                "total_pnl": pnl
            }
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {"total_trades": 0, "winning_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}

    def get_performance_stats(self) -> dict:
        """Returns aggregated performance stats (Win Rate, Total PnL, Win count)."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM trades WHERE status != 'OPEN'")
            total_closed = c.fetchone()[0] or 0
            
            c.execute("SELECT COUNT(*) FROM trades WHERE status != 'OPEN' AND pnl_usd > 0")
            wins = c.fetchone()[0] or 0
            
            c.execute("SELECT SUM(pnl_usd) FROM trades WHERE status != 'OPEN'")
            pnl = c.fetchone()[0] or 0.0
            
            conn.close()
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
            
            return {
                "total_trades": total_closed,
                "winning_trades": wins,
                "win_rate": win_rate,
                "total_pnl": pnl
            }
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {"total_trades": 0, "winning_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}

    def update_trade_order_ids(self, trade_id: int, sl_id: str = None, tp_id: str = None):
        """Stores Binance Order IDs for active trade tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            if sl_id:
                c.execute("UPDATE trades SET sl_order_id=? WHERE id=?", (sl_id, trade_id))
            if tp_id:
                c.execute("UPDATE trades SET tp_order_id=? WHERE id=?", (tp_id, trade_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update order IDs for trade {trade_id}: {e}")
