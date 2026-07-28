import os

# Strip proxy environment variables to prevent websockets/aiohttp from 
# attempting to use unsupported SOCKS schemes (socks://127.0.0.1:...).
for proxy_var in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

import asyncio
import logging
from dotenv import load_dotenv
import json

from core.engine import run_engine_cycle
from core.executor import LiveExecutor
from core.performance import PerformanceMonitor
from interface.telegram_bot import TelegramInterface
from core.db import CortexDB

from core.utils import strip_proxies

# Load environment configuration
load_dotenv()
strip_proxies()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("AutoCore.Main")

async def engine_loop(tg: TelegramInterface):
    """
    DISABLED: The old engine_loop ran run_engine_cycle() every 30 minutes,
    which backtested ALL strategies on recent data and OVERWROTE active_config.json.
    This destroyed our validated GA+WFA configs with random results.
    
    The proper pipeline is: master_pipeline.py (GA → WFA → apply_ga_config → ML Retrain)
    which runs on the ml_retrain_loop schedule (every 48 hours).
    """
    while True:
        await asyncio.sleep(30 * 60)  # Sleep 30 min, do nothing
        logger.info("⏳ Engine heartbeat. Config managed by Master Pipeline (48h cycle).")

async def ml_retrain_loop(tg: TelegramInterface):
    """Triggers the ML retraining process every 2 days (Auto-Pilot ML)."""
    # Wait a bit on startup before the first run (e.g., 1 hour), unless we want to trigger immediately.
    # We will trigger the first run after 2 days by default to not overwhelm the startup.
    RETRAIN_INTERVAL_DAYS = 999  # DISABLED: was 2. Set back to 2 to re-enable master pipeline
    while True:
        await asyncio.sleep(RETRAIN_INTERVAL_DAYS * 24 * 60 * 60)
        logger.info("📅 [CRON] Initiating 48-hour Rolling Retrain of Triple-AI Models...")
        await tg.send_alert(f"🤖 [Auto-Pilot] 48-hour Master Pipeline started in background. The executor will Hot-Swap dynamically when finished.")
        try:
            # Fire and forget subprocess
            import subprocess
            import sys
            subprocess.Popen(
                [sys.executable, "master_pipeline.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.error(f"Failed to spawn cron_retrain.py: {e}")

async def adaptive_hotswap_loop(tg: TelegramInterface):
    """Triggers the GPU-based Adaptive Hot-Swap every 24 hours."""
    while True:
        await asyncio.sleep(999 * 24 * 60 * 60) # DISABLED: was 24h. Set back to (24 * 60 * 60) to re-enable
        logger.info("🕒 [CRON] Initiating Daily GPU Adaptive Hot-Swap...")
        await tg.send_alert("🧬 [Adaptive] Running daily GPU parameter fine-tuning...")
        try:
            import subprocess
            import sys
            subprocess.Popen(
                [sys.executable, "adaptive_hotswap.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.error(f"Failed to spawn adaptive_hotswap.py: {e}")

async def main():
    logger.info("Starting ICT AutoCore System (LAITB 2.0)...")
    
    # Init DB (Shared instance)
    db = CortexDB()
    
    # Init Components
    performance = PerformanceMonitor()
    
    # Check if we should use the Rust engine or Python executor
    USE_RUST_ENGINE = os.environ.get("USE_RUST_ENGINE", "true").lower() in ("true", "1", "yes")
    
    if USE_RUST_ENGINE:
        logger.info("🦀 Using RUST Live Engine (Phase 6)")
    else:
        logger.info("🐍 Using Python Live Executor (legacy)")
    
    # SHARE Strategy instances (still needed for Telegram commands & initial config)
    from core.engine import STRATEGIES
    shared_strategies = STRATEGIES
    
    executor = LiveExecutor(config_path="data/active_config.json", db=db, strategies=shared_strategies)
    
    # Init Interface
    tg = TelegramInterface(executor=executor, perf=performance, db=db)
    executor._tg_bot = tg
    
    rust_process = None
    
    try:
        # Start Telegram background task
        await tg.start(poll=False)
        await tg.send_alert("🤖 System Startup Initiated...")

        if tg.chat_id and tg._bot:
            try:
                # Manually invoke the status logic and send it to the chat on startup
                # so the user can see the balance without needing to type /status
                class DummyMessage:
                    async def reply_text(self, text, parse_mode=None):
                        await tg._bot.send_message(chat_id=tg.chat_id, text=text, parse_mode=parse_mode)
                
                class DummyUpdate:
                    message = DummyMessage()
                
                await tg._cmd_status(update=DummyUpdate(), ctx=None)
            except Exception as e:
                logger.error(f"Failed to auto-send status on startup: {e}")
        
        # Initial Engine run if no config exists
        if not os.path.exists("data/active_config.json"):
            configs = await run_engine_cycle()
            if configs and len(configs) > 0:
                os.makedirs("data", exist_ok=True)
                with open("data/active_config.json", "w") as f:
                    import json
                    json.dump(configs, f, indent=4)
            else:
                logger.warning("Initial engine run returned no configs. Using hardcoded defaults in Executor.")
                
        # Schedule repeating tasks
        engine_task = asyncio.create_task(engine_loop(tg))
        retrain_task = asyncio.create_task(ml_retrain_loop(tg))
        hotswap_task = asyncio.create_task(adaptive_hotswap_loop(tg))
        
        if USE_RUST_ENGINE:
            # ═══════════════════════════════════════════════════════
            # RUST LIVE ENGINE: Spawn as subprocess
            # ═══════════════════════════════════════════════════════
            import subprocess
            import sys
            
            rust_binary = os.path.join("rust_engine", "target", "release", "aegis_engine")
            if sys.platform == "win32":
                rust_binary += ".exe"
            
            if not os.path.exists(rust_binary):
                logger.error(f"❌ Rust binary not found at {rust_binary}")
                logger.error("   Build it first: cargo build --release --manifest-path rust_engine/Cargo.toml")
                await tg.send_alert("❌ Rust engine binary not found! Falling back to Python executor.")
                # Fallback to Python
                logger.info("Starting Python Real-time Execution Loop (fallback)...")
                await executor.start_websocket_streams()
            else:
                # Pass env vars to Rust
                rust_env = os.environ.copy()
                rust_env["PAPER_MODE"] = os.environ.get("PAPER_MODE", "true")
                rust_env["RUST_LOG"] = "info"
                
                logger.info(f"🚀 Launching Rust Live Engine: {rust_binary}")
                await tg.send_alert("🦀 Rust Live Engine launched! Paper mode active.")
                
                rust_process = subprocess.Popen(
                    [rust_binary, "live"],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    env=rust_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                
                # Stream Rust output to Python logger
                async def stream_rust_logs():
                    loop = asyncio.get_event_loop()
                    while rust_process and rust_process.poll() is None:
                        line = await loop.run_in_executor(None, rust_process.stdout.readline)
                        if line:
                            log_line = line.decode('utf-8', errors='replace').strip()
                            if log_line:
                                logger.info(f"[RUST] {log_line}")
                        else:
                            await asyncio.sleep(0.1)
                    
                    # Process exited
                    exit_code = rust_process.returncode if rust_process else -1
                    logger.warning(f"🦀 Rust engine exited with code {exit_code}")
                    await tg.send_alert(f"⚠️ Rust engine stopped (exit code {exit_code}). Restart manually.")
                
                rust_log_task = asyncio.create_task(stream_rust_logs())
                
                # Listen to Rust IPC events and forward to Telegram
                async def rust_ipc_listener():
                    # Wait for Rust to open port 9090
                    reader, writer = None, None
                    for _ in range(15):
                        try:
                            reader, writer = await asyncio.open_connection('127.0.0.1', 9090)
                            logger.info("[IPC] 🔗 Connected to Rust Engine on port 9090")
                            break
                        except ConnectionRefusedError:
                            await asyncio.sleep(2)
                    
                    if not reader:
                        logger.warning("[IPC] ⚠️ Could not connect to Rust Engine on 9090.")
                        return

                    while True:
                        try:
                            line = await reader.readline()
                            if not line:
                                break
                            
                            event = json.loads(line.decode('utf-8', errors='replace'))
                            ev_type = event.get('event')
                            
                            if ev_type == 'trade_opened':
                                sym = event.get('symbol', '?')
                                d = event.get('direction', '?')
                                strat = event.get('strategy', '?')
                                ep = event.get('entry_price', 0)
                                sl = event.get('sl_price', 0)
                                
                                emoji = "🟢" if d == "LONG" else "🔴"
                                await tg.send_alert(
                                    f"🚨 *NEW TRADE: {sym}*\n"
                                    f"├ Strategy: {strat}\n"
                                    f"├ Direction: {emoji} {d}\n"
                                    f"├ Entry: {ep:.5f}\n"
                                    f"└ SL: {sl:.5f}"
                                )
                            
                            elif ev_type == 'trade_closed':
                                sym = event.get('symbol', '?')
                                d = event.get('direction', '?')
                                r = event.get('pnl_r', 0.0)
                                p_pct = event.get('pnl_pct', 0.0)
                                reason = event.get('reason', '?')
                                
                                emoji = "✅" if p_pct > 0 else "❌"
                                await tg.send_alert(
                                    f"{emoji} *TRADE CLOSED: {sym}*\n"
                                    f"├ Direction: {d}\n"
                                    f"├ PnL: {p_pct:+.2f}%\n"
                                    f"├ R-Multiple: {r:+.2f} R\n"
                                    f"└ Reason: {reason}"
                                )
                                
                                # Update Paper Equity and Database for Statistics
                                try:
                                    if hasattr(executor, 'paper_mode') and getattr(executor, 'paper_mode', False):
                                        current_eq = getattr(executor, 'paper_equity', 70.0)
                                        risk_usd = current_eq * 0.02
                                        pnl_usd = risk_usd * r
                                        executor.paper_equity = current_eq + pnl_usd
                                        
                                        # Log to DB so /status sees the history
                                        t_id = db.log_trade_open(sym, "rust_hft", d, 0, 0, 0, 0, 1, risk_usd, 0.5)
                                        if t_id:
                                            db.close_trade(t_id, 0, pnl_usd, p_pct, r, reason)
                                            if performance:
                                                performance.record_trade(pnl_usd, executor.paper_equity)
                                except Exception as e:
                                    logger.error(f"Failed to update equity/db: {e}")

                                # Show updated stats / balance directly after trade
                                if tg.chat_id and tg._bot:
                                    try:
                                        class DummyMessage:
                                            async def reply_text(self, text, parse_mode=None):
                                                await tg._bot.send_message(chat_id=tg.chat_id, text=text, parse_mode=parse_mode)
                                        class DummyUpdate:
                                            message = DummyMessage()
                                        await tg._cmd_status(update=DummyUpdate(), ctx=None)
                                    except Exception:
                                        pass
                                
                        except Exception as e:
                            logger.error(f"[IPC] Error reading event: {e}")
                            await asyncio.sleep(1)

                ipc_task = asyncio.create_task(rust_ipc_listener())
                
                # Keep Python alive for Telegram and retrain
                await asyncio.gather(engine_task, retrain_task, hotswap_task, rust_log_task, ipc_task)
        else:
            # ═══════════════════════════════════════════════════════
            # PYTHON EXECUTOR: Legacy mode
            # ═══════════════════════════════════════════════════════
            logger.info("Starting Python Real-time Execution Loop...")
            await executor.start_websocket_streams()
        
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested.")
    except Exception as e:
        logger.fatal(f"Unhandled system error: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        
        # Kill Rust process if running
        if rust_process and rust_process.poll() is None:
            logger.info("Terminating Rust engine...")
            rust_process.terminate()
            try:
                rust_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rust_process.kill()
        
        # Cancel background tasks
        if 'engine_task' in locals():
            engine_task.cancel()
            try: await engine_task
            except asyncio.CancelledError: pass
        if 'retrain_task' in locals():
            retrain_task.cancel()
            try: await retrain_task
            except asyncio.CancelledError: pass
            
        # Stop Telegram bot
        try:
            await tg.stop()
            logger.info("Telegram interface stopped.")
        except Exception as e:
            logger.error(f"Error stopping Telegram: {e}")
            
        # Close exchange session
        try:
            if hasattr(executor, 'exchange'):
                await executor.exchange.close()
                logger.info("Binance session closed.")
        except Exception as e:
            logger.error(f"Error closing exchange: {e}")

if __name__ == "__main__":
    asyncio.run(main())

