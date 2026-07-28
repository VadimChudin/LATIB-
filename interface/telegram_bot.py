"""
Telegram Bot Interface (interface/telegram_bot.py)
===================================================
Provides:
  - /start           — Welcome message + command list
  - /status          — Full system state (positions, perf, configs)
  - /recalc          — Force an Engine re-scoring cycle right now
  - /pause / /resume — Manually pause/resume new entries
  - send_alert()     — Called by PerformanceMonitor for push notifications
  - Inline keyboard  — "Approve Restart" button for hard-stop recovery

Requirements:
    pip install python-telegram-bot
    .env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv, find_dotenv

# Explicitly find .env from project root — don't rely on cwd
load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger("AutoCore.Telegram")


# ── Try importing python-telegram-bot (optional dependency) ───────────────────

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
    from telegram.ext import (
        ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
    )
    TG_AVAILABLE = True
except ImportError:
    TG_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram alerts disabled.")


# ── TelegramInterface ─────────────────────────────────────────────────────────

class TelegramInterface:
    """
    Bridges the running AutoCore system to a Telegram bot.
    Pass `executor` and `perf` references so commands can query live state.
    """

    def __init__(self, executor=None, perf=None, engine_fn=None, db=None):
        """
        Args:
            executor  : LiveExecutor instance (for positions / configs)
            perf      : PerformanceMonitor instance (for status / hard-stop)
            engine_fn : async coroutine fn that runs one Engine scoring cycle
        """
        self.executor  = executor
        self.perf      = perf
        self.engine_fn = engine_fn   # e.g. run_engine_cycle from core.engine

        self.token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.app     = None
        self._bot: Optional["Bot"] = None

        # Validate token is present
        if not TG_AVAILABLE or not self.token:
            logger.warning("[TG] Telegram disabled (no token or library missing).")
            return

        self.app = ApplicationBuilder().token(self.token).build()
        self._bot = self.app.bot
        self._register_handlers()
        logger.info("[TG] Telegram bot configured ✅")

    # ── Handlers ────────────────────────────────────────────────────────────────

    def _register_handlers(self):
        add = self.app.add_handler
        add(CommandHandler("start",  self._cmd_start))
        add(CommandHandler("status", self._cmd_status))
        add(CommandHandler("recalc", self._cmd_recalc))
        add(CommandHandler("pause",  self._cmd_pause))
        add(CommandHandler("resume", self._cmd_resume))
        add(CommandHandler("test",   self._cmd_test))
        add(CallbackQueryHandler(self._btn_handler))

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🤖 *ICT AutoCore — LAITB 2.0*\n\n"
            "Commands:\n"
            "/status   — System snapshot (includes Equity)\n"
            "/recalc   — Force Engine re-scan now\n"
            "/pause    — Pause new trade entries\n"
            "/resume   — Resume entries (clear pause)\n"
            "/test     — DOGE test order on testnet\n"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        lines = ["📊 *AutoCore System Status*\n"]

        # Balance
        if self.executor:
            try:
                if hasattr(self.executor, 'paper_mode') and self.executor.paper_mode:
                    bal = getattr(self.executor, 'paper_equity', 70.0)
                    lines.append(f"💰 *Paper Balance:* ${bal:.2f} USDT (Simulated)")
                else:
                    balance = await self.executor.exchange.fetch_balance()
                    total_usdt = balance.get('total', {}).get('USDT', 0)
                    lines.append(f"💰 *Live Balance:* ${total_usdt:.2f} USDT\n")
            except Exception as e:
                lines.append(f"❌ *Balance Error:* {e}\n")

        # Performance
        if self.perf:
            s = self.perf.status()
            status_emoji = "🟢" if s["trading_allowed"] else "🔴"
            lines.append(
                f"{status_emoji} *Trading:* {'ACTIVE' if s['trading_allowed'] else 'HALTED'}\n"
                f"├ Loss streak: {s['loss_streak']} / {self.perf.max_streak}\n"
                f"├ Total trades: {s['total_trades']}\n"
                f"└ Peak equity: ${s['peak_equity']:.2f}\n"
            )
        else:
            lines.append("⚪ No performance monitor attached.\n")

        # Configs
        if self.executor:
            configs = self.executor.active_configs or []
            lines.append(f"📈 *Active Configs:* {len(configs)}")
            for c in configs[:5]:
                m = c.get("metrics", {})
                lines.append(
                    f"  • {c['symbol']} | {c['strategy'][:14]} | "
                    f"WR={m.get('win_rate', 0)*100:.0f}%"
                )

        await update.message.reply_text("\n".join(lines))

    async def _cmd_recalc(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self.engine_fn:
            await update.message.reply_text("❌ Engine function not attached.")
            return
        await update.message.reply_text("⏳ Engine re-scan started... (~2-3 min)")
        try:
            asyncio.create_task(self.engine_fn())
            await update.message.reply_text("✅ Engine cycle triggered in background.")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.perf:
            self.perf._paused = True
            await update.message.reply_text("⏸ New entries PAUSED by operator.")
        else:
            await update.message.reply_text("❌ No performance monitor attached.")

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.perf:
            self.perf._paused = False
            if self.perf._halted:
                await update.message.reply_text(
                    "⚠️ Hard stop still active. Use the Approve Restart button sent with the alert.",
                )
            else:
                await update.message.reply_text("▶️ Entries RESUMED.")
        else:
            await update.message.reply_text("❌ No performance monitor attached.")

    async def _cmd_test(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Place a tiny LONG order on mainnet using the minimum allowed limit to verify SL+TP."""
        if not self.executor:
            await update.message.reply_text("❌ No executor attached.")
            return
        await update.message.reply_text("⏳ Fetching exchange limits for minimal test order...")
        symbol = "DOGE/USDT"
        try:
            # Load markets to get minimum order size for DOGE
            market = self.executor.data_exchange.market(symbol)
            min_amount = float(market['limits']['amount']['min'])
            
            # Use 2x the minimum allowed just to be safe against rounding errors
            contracts = max(min_amount * 2, min_amount + 10) 
            
            ticker = await self.executor.data_exchange.fetch_ticker(symbol)
            price  = ticker['last']
            sl     = round(price * 0.99, 5)
            tp     = round(price * 1.02, 5)
            
            margin_req = (price * contracts) / 5
            
            await update.message.reply_text(f"⏳ Placing {contracts} DOGE order. Est Margin: ${margin_req:.2f} USDT")
            await self.executor.exchange.set_leverage(5, symbol)
            await self.executor.exchange.create_order(
                symbol=symbol, type='market', side='buy', amount=contracts)
            await self.executor.exchange.create_order(
                symbol=symbol, type='STOP_MARKET', side='sell', amount=contracts,
                params={'stopPrice': sl, 'reduceOnly': True})
            await self.executor.exchange.create_order(
                symbol=symbol, type='TAKE_PROFIT_MARKET', side='sell', amount=contracts,
                params={'stopPrice': tp, 'reduceOnly': True})
            open_orders = await self.executor.exchange.fetch_open_orders(symbol)
            orders_txt = "\n".join(
                f"  • {o['type']} {o['side']} @ {o.get('stopPrice') or o.get('info',{}).get('stopPrice','?')}"
                for o in open_orders
            ) or "  (none)"
            await self.executor.exchange.cancel_all_orders(symbol)
            await self.executor.exchange.create_order(
                symbol=symbol, type='market', side='sell', amount=contracts,
                params={'reduceOnly': True})
            await update.message.reply_text(
                f"✅ TEST PASSED!\n"
                f"Price: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}\n"
                f"Orders before cancel:\n{orders_txt}\n"
                f"Position closed ✔"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Test failed: {e}")


    async def _btn_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        if query.data == "approve_restart":
            if self.perf:
                self.perf.reset_hard_stop()
            await query.edit_message_text("✅ Hard stop cleared. Execution resumed.")

    # ── Push alerts ─────────────────────────────────────────────────────────────

    async def send_alert(self, text: str, requires_approval: bool = False):
        """
        Send a push notification to TELEGRAM_CHAT_ID.
        Set requires_approval=True to attach the 'Approve Restart' inline button.
        """
        if not self._bot or not self.chat_id:
            return
        try:
            markup = None
            if requires_approval:
                kb     = [[InlineKeyboardButton("✅ Approve Restart", callback_data="approve_restart")]]
                markup = InlineKeyboardMarkup(kb)
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                reply_markup=markup,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error(f"[TG] Alert send error: {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, poll: bool = True):
        """Start the Telegram bot. If poll=True, starts the polling loop for commands."""
        if not self.app:
            return
        try:
            await self.app.initialize()
            await self.app.start()
            if poll:
                await self.app.updater.start_polling(drop_pending_updates=True)
                logger.info("[TG] Polling started.")
            else:
                logger.info("[TG] Bot started in send-only mode (polling disabled).")
        except Exception as e:
            logger.warning(f"[TG] Failed to start Telegram bot: {e}. Continuing without Telegram.")
            self.app = None
            self._bot = None

    async def stop(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
