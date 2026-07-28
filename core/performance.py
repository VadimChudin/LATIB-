"""
Performance Monitor (core/performance.py)
==========================================
Monitors the live trading account and enforces risk circuit-breakers:

  - Loss Streak Guard : if N consecutive losses → alert + pause new entries
  - Daily Drawdown Hard Stop : if equity drawdown > MAX_DD_PCT → halt all trading
  - Periodic re-evaluation trigger : after every RE_EVAL_TRADES trades → notifies Engine

Designed to be called by LiveExecutor after every trade result is recorded,
or run as a companion async task that polls the DB.
"""

import asyncio
import logging
import sqlite3
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger("AutoCore.Performance")

# ── Config ──────────────────────────────────────────────────────────────────────

MAX_LOSS_STREAK   = 5        # Consecutive losses before pausing
MAX_DD_PCT        = 0.20     # 20% account drawdown → hard stop
RE_EVAL_TRADES    = 20       # Trigger Engine re-scan after every N trades
POLL_INTERVAL_SEC = 120      # Reduced frequency (2 min) to save API weight


from core.db import CortexDB

class PerformanceMonitor:
    """
    Stateful performance guard. Should be instantiated once and held by the Executor.
    Call `record_trade(pnl_usd, equity)` after every trade result is recorded,
    or run as a companion async task that polls the DB.
    """

    def __init__(
        self,
        db_path: str = "data/autocore.db",
        max_streak: int = MAX_LOSS_STREAK,
        max_dd_pct: float = MAX_DD_PCT,
        re_eval_after: int = RE_EVAL_TRADES,
    ):
        self.db = CortexDB(db_path)
        self.max_streak   = max_streak
        self.max_dd_pct   = max_dd_pct
        self.re_eval_after = re_eval_after

        self._loss_streak: int     = 0
        self._peak_equity: float   = 0.0
        self._total_trades: int    = 0
        self._halted: bool         = False       # Hard-stop flag
        self._paused: bool         = False       # Soft-stop (loss streak)

        self._alert_callback = None              # Optional async callback (e.g. Telegram)

    @property
    def trading_allowed(self) -> bool:
        """True unless halted by drawdown or paused by loss streak."""
        return not self._halted and not self._paused

    def record_trade(self, pnl_usd: float, current_equity: float):
        """Records a completed trade, updating peaks, drawdowns, and loss streaks."""
        self._total_trades += 1

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if pnl_usd < 0:
            self._loss_streak += 1
            if self._loss_streak >= self.max_streak and not self._paused:
                self._paused = True
                msg = f"⚠️ Loss Streak limit reached ({self.max_streak}). Trading PAUSED."
                logger.warning(msg)
                self._fire_alert(msg)
        else:
            self._loss_streak = 0
            if self._paused:
                self._paused = False
                msg = "✅ Win logged. Loss Streak reset. Trading RESUMED."
                logger.info(msg)
                self._fire_alert(msg)

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self.max_dd_pct and not self._halted:
                self._halted = True
                msg = f"🛑 HARD STOP: DRAWDOWN LIMIT MET ({drawdown*100:.1f}%). Trading HALTED."
                logger.error(msg)
                self._fire_alert(msg)

    # ... (skipping unchanged methods) ...

    async def polling_loop(self, executor=None):
        """
        Background loop that polls the SQLite DB and checks performance metrics.
        Can optionally pull equity from the executor instance.
        """
        logger.info("[Perf] Performance monitor polling loop started.")
        last_trade_id = self.db.get_last_closed_trade_id()

        while True:
            await asyncio.sleep(POLL_INTERVAL_SEC)

            new_trades = self.db.get_new_closed_trades(last_trade_id)
            for trade in new_trades:
                pnl_usd = trade.get("pnl_usd", 0.0) or 0.0
                equity  = 0.0

                if executor is not None:
                    try:
                        equity = await executor.get_equity()
                    except Exception:
                        pass

                if equity <= 0:
                    equity = self.db.get_total_pnl() + 1000.0

                self.record_trade(pnl_usd, equity)
                last_trade_id = max(last_trade_id, trade.get("id", 0))

    def _get_cumulative_pnl(self) -> float:
        return self.db.get_total_pnl()

    def _fire_alert(self, message: str):
        """Non-blocking fire-and-forget of the alert callback."""
        if self._alert_callback:
            try:
                asyncio.create_task(self._alert_callback(message))
            except RuntimeError:
                pass
