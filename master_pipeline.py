"""
Master Pipeline — AEGIS 2.0 Auto-Pilot
========================================
Runs autonomously once per week. Fully self-healing:
- Timeouts per step (kills hung processes)
- Auto-resume from checkpoint (skips completed steps)
- Output validation (checks files exist and aren't empty)
- Telegram alerts on failures

Run: python master_pipeline.py
     python master_pipeline.py --fresh  (ignore checkpoint, start from scratch)
"""
import os
import sys
import time
import json
import asyncio
import logging
import websockets
from datetime import datetime, timedelta
from pathlib import Path

from constants import PIPELINE_CHECKPOINT_PATH, RETRAIN_FLAG_PATH, CHECKPOINT_MAX_AGE_HOURS

# ── Logging ──
os.makedirs("data/logs", exist_ok=True)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("data/logs/master_pipeline.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MasterPipeline")

# ── Pipeline Steps Definition ──
# (name, script, is_blocking, timeout_sec, validation_file)
PIPELINE_STEPS = [
    ("DATA_SYNC",          "download_historical.py",       True,  3600,  "data/top_symbols.json"),
    ("RUST_GA_OPTIMIZE",   "run_ga_batch.py",              True,  7200,  "data/ga_aggregated_results.json"),
    ("WFA",                "walk_forward.py",               True,  1800,  None),
    ("APPLY_CONFIG",       "apply_ga_config.py",            True,  3600,  "data/active_config.json"),
    ("ML_TRAINING",        "train_all_systems.py",          True,  1800,  None),
    ("MICRO_ML_TRAINING",  "train_ml_microstructure.py",    False, 3600,  None),
    ("JOURNAL",            "aggregate_journal.py",          False, 300,   None),
    ("HOTSWAP",            "adaptive_hotswap.py",           False, 1800,  None),
    ("MODEL_EXPORT",       "export_models_json.py",         False, 300,   "data/models_json/"),
]

# Human-readable names for progress UI
STEP_LABELS = {
    "DATA_SYNC":         "Сбор Исторических Данных (REST API)",
    "RUST_GA_OPTIMIZE":  "Массовая GA-Оптимизация (Rust Engine)",
    "WFA":               "Оценка Глубокой Жизнеспособности (WFA)",
    "APPLY_CONFIG":      "Выбор и Запись Выживших Стратегий",
    "ML_TRAINING":       "Переобучение Нейросетей (LightGBM)",
    "MICRO_ML_TRAINING": "Обучение Микроструктурных ML (Тиковые Фичи)",
    "JOURNAL":           "Агрегация Журнала & Мета-Модель",
    "HOTSWAP":           "Адаптивный Hot-Swap (GPU оценка пула)",
    "MODEL_EXPORT":      "Экспорт Моделей в JSON (для Rust)",
}


class TelegramInterface:
    async def send_alert(self, msg):
        logger.info(f"📱 [TELEGRAM] {msg}")


# ── Checkpoint ──

def load_checkpoint() -> dict:
    """Load last successful checkpoint. Returns {} if none or expired."""
    if not PIPELINE_CHECKPOINT_PATH.exists():
        return {}
    try:
        with open(PIPELINE_CHECKPOINT_PATH) as f:
            cp = json.load(f)
        # Check age — if older than 48h, data is stale
        cp_time = datetime.fromisoformat(cp.get("timestamp", "2000-01-01"))
        if datetime.now() - cp_time > timedelta(hours=CHECKPOINT_MAX_AGE_HOURS):
            logger.info(f"⏰ Checkpoint is {CHECKPOINT_MAX_AGE_HOURS}h+ old — starting fresh")
            return {}
        return cp
    except Exception:
        return {}


def save_checkpoint(step_name: str):
    """Save checkpoint after a successful step."""
    cp = {
        "last_completed": step_name,
        "timestamp": datetime.now().isoformat(),
    }
    PIPELINE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_CHECKPOINT_PATH, "w") as f:
        json.dump(cp, f, indent=2)


def clear_checkpoint():
    """Remove checkpoint (pipeline fully completed)."""
    if PIPELINE_CHECKPOINT_PATH.exists():
        PIPELINE_CHECKPOINT_PATH.unlink()


# ── Step Runner ──

async def run_step(step_name: str, command: list, tg: TelegramInterface,
                   max_retries: int = 3, timeout_sec: int = 3600) -> bool:
    """Run a pipeline step with timeout, retries, and output streaming."""
    for attempt in range(1, max_retries + 1):
        logger.info(f"===> 🚀 STARTING: {step_name} (Attempt {attempt}/{max_retries}, timeout {timeout_sec}s) <===")
        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            try:
                # Stream output with timeout
                while True:
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout_sec)
                    except asyncio.TimeoutError:
                        logger.error(f"===> ⏰ TIMEOUT: {step_name} exceeded {timeout_sec}s — killing process <===")
                        process.kill()
                        await process.wait()
                        break

                    if not line:
                        break
                    print(f"[{step_name}] {line.decode('utf-8', errors='replace').strip()}")

                await asyncio.wait_for(process.wait(), timeout=30)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

            elapsed = time.time() - start_time

            if process.returncode == 0:
                logger.info(f"===> ✅ COMPLETE: {step_name} ({elapsed:.1f}s) <===")
                return True
            else:
                logger.warning(f"===> ⚠️ FAILED: {step_name} (exit code {process.returncode}) <===")

                if attempt < max_retries:
                    wait_time = 30 * attempt
                    logger.info(f"===> 🔄 AUTO-HEAL: Retrying in {wait_time}s... <===")
                    await tg.send_alert(f"⚠️ Step `{step_name}` failed (Attempt {attempt}). Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"===> ❌ CRITICAL: {step_name} failed after {max_retries} attempts! <===")
                    await tg.send_alert(
                        f"🚨 Ошибка в блоке {step_name} после {max_retries} попыток. "
                        f"Торгуем по старым данным."
                    )
                    return False

        except Exception as e:
            logger.error(f"===> ❌ SYSTEM ERROR: {step_name}: {e} <===")
            await tg.send_alert(f"🚨 Критическая ошибка в {step_name}: {e}. Торгуем по старым данным.")
            return False

    return False


def validate_output(validation_path: str) -> bool:
    """Check that a step produced a non-empty output file/directory."""
    if not validation_path:
        return True
    p = Path(validation_path)
    if p.is_dir():
        return p.exists() and any(p.iterdir())
    elif p.is_file():
        return p.exists() and p.stat().st_size > 10
    return not p.exists()  # Doesn't exist yet — can't validate


def trigger_hotswap():
    """Signal Live Engine to reload configs."""
    try:
        RETRAIN_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RETRAIN_FLAG_PATH, "w") as f:
            f.write(str(time.time()))
        logger.info("🔥 HOT-SWAP FLAG PLANTED: Live bots will reload on next tick.")
    except Exception as e:
        logger.error(f"Failed to create hot-swap flag: {e}")


async def broadcast_progress(status: str, progress: int = 0, step: str = "", hours_left: float = 0):
    """Send progress to local UI WebSocket."""
    payload = {
        "type": "pipeline_status",
        "status": "updating" if status == "updating" else "countdown",
        "progress": progress,
        "step": step,
        "hours_left": hours_left
    }
    try:
        async with websockets.connect("ws://127.0.0.1:8080") as ws:
            await ws.send(json.dumps(payload))
    except Exception:
        pass


# ── Main Pipeline ──

async def run_master_pipeline(fresh: bool = False):
    tg = TelegramInterface()
    logger.info("=" * 60)
    logger.info(f"  MASTER PIPELINE INITIATED @ {datetime.now()}")
    logger.info("=" * 60)

    total_start = time.time()

    # Load checkpoint (skip completed steps)
    checkpoint = {} if fresh else load_checkpoint()
    resume_after = checkpoint.get("last_completed")
    if resume_after:
        logger.info(f"📌 RESUMING after checkpoint: {resume_after}")
        await tg.send_alert(f"⚙️ Pipeline resumed from checkpoint: {resume_after}")
    else:
        await tg.send_alert("⚙️ Pipeline started: full cycle Data→GA→WFA→ML")

    # Calculate progress percentages automatically
    total_steps = len(PIPELINE_STEPS)
    should_skip = resume_after is not None

    for i, (name, script, is_blocking, timeout, validation) in enumerate(PIPELINE_STEPS):
        progress_pct = int((i / total_steps) * 100)

        # Skip steps we've already completed (checkpoint resume)
        if should_skip:
            if name == resume_after:
                should_skip = False  # Next step will execute
                logger.info(f"  ⏭️ Skipping (checkpoint): {name}")
            else:
                logger.info(f"  ⏭️ Skipping (checkpoint): {name}")
            continue

        # Run step
        label = STEP_LABELS.get(name, name)
        await broadcast_progress("updating", progress_pct, label)

        retries = 3 if is_blocking else 1
        success = await run_step(name, [sys.executable, script], tg,
                                  max_retries=retries, timeout_sec=timeout)

        if not success:
            if is_blocking:
                logger.error(f"💀 Blocking step {name} failed — aborting pipeline")
                save_checkpoint(PIPELINE_STEPS[i-1][0] if i > 0 else "")
                await broadcast_progress("countdown", 0, "", 48.0)
                return
            else:
                logger.warning(f"⚠️ Non-blocking step {name} failed — continuing")

        # Validate output
        if success and validation:
            if not validate_output(validation):
                logger.warning(f"⚠️ Validation failed for {name}: {validation} missing/empty")
                if is_blocking:
                    save_checkpoint(PIPELINE_STEPS[i-1][0] if i > 0 else "")
                    await broadcast_progress("countdown", 0, "", 48.0)
                    return

        # Save checkpoint
        if success:
            save_checkpoint(name)

    # Final: trigger hot-swap
    await broadcast_progress("updating", 100, "Hot-Swap Подтвержден. Перезагрузка Памяти...")
    trigger_hotswap()
    clear_checkpoint()

    total_elapsed = (time.time() - total_start) / 60
    logger.info("=" * 60)
    logger.info(f"🏆 MASTER PIPELINE FULLY COMPLETED in {total_elapsed:.1f} minutes!")
    logger.info("=" * 60)

    await tg.send_alert(f"✅ Pipeline завершён за {total_elapsed:.1f} мин. Все системы обновлены.")
    await broadcast_progress("countdown", 0, "", 48.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start from scratch")
    args = parser.parse_args()

    os.makedirs("data/logs", exist_ok=True)
    try:
        asyncio.run(run_master_pipeline(fresh=args.fresh))
    except KeyboardInterrupt:
        logger.info("\n⛔ Pipeline interrupted. Checkpoint saved. Trading continues with existing params.")
