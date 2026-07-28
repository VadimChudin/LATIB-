"""
Standalone Rolling Retrain Script (Auto-Pilot ML)
=================================================
Runs in the background, pulls fresh data for all active symbols 
(plus BTC Gravity), retrains the ML models sequentially, and sets 
a HOT-SWAP flag when finished so LiveExecutor can reload memory 
without dropping WebSockets.
"""
import os
import sys
import json
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CronRetrain")

ACTIVE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'data', 'active_config.json')
HOTSWAP_FLAG_PATH = os.path.join(os.path.dirname(__file__), 'data', 'models', 'retrain_flag.txt')

TRAINING_SCRIPTS = [
    "train_ml_smc.py",
    "train_ml_swingict.py",
    "train_ml_orb.py",
    "train_ml_knife.py",
    "train_ml_vwap.py",
    "train_ml_ttm.py"
]

def main():
    logger.info("=== 🏃 BACKGROUND ROLLING RETRAIN STARTED ===")
    start_t = time.time()
    
    # 1. Download Latest Data
    logger.info("1. Booting up Data Downloader...")
    try:
        # Run standard downloader (we assume it handles incremental properly based on active_config)
        # Note: In a true production environment, we'd pass specific symbols to speed this up.
        # For LAITB 2.0, download_historical.py reads symbols from a config or fetches default.
        dl_process = subprocess.run([sys.executable, "download_historical.py"], capture_output=True)
        if dl_process.returncode != 0:
            err_msg = dl_process.stderr.decode('utf-8', errors='replace')[-200:]
            logger.warning(f"Downloader warned/failed: {err_msg}")
        else:
            logger.info("Data download complete.")
    except Exception as e:
        logger.error(f"Failed to run data downloader: {e}")
        return # If we can't get fresh data, don't retrain

    # 2. Run Training Scripts Sequentially
    logger.info("2. Launching ML Training Pipeline...")
    for script in TRAINING_SCRIPTS:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if not os.path.exists(script_path):
            logger.warning(f"Script missing, skipping: {script}")
            continue
            
        logger.info(f" -> Training {script}...")
        try:
            res = subprocess.run([sys.executable, script_path], capture_output=True)
            if res.returncode == 0:
                logger.info(f"    ✅ {script} completed successfully.")
            else:
                err_msg = res.stderr.decode('utf-8', errors='replace')[-200:] if res.stderr else res.stdout.decode('utf-8', errors='replace')[-200:]
                logger.error(f"    ❌ {script} failed! Error: {err_msg}")
                # We could abort the whole flow, but let's try to train the others
        except Exception as e:
            logger.error(f"Error launching {script}: {e}")

    # 3. Emit Hot-Swap Flag
    elapsed = (time.time() - start_t) / 60
    logger.info(f"3. All training complete in {elapsed:.1f} minutes! Emitting Hot-Swap Flag.")
    
    os.makedirs(os.path.dirname(HOTSWAP_FLAG_PATH), exist_ok=True)
    with open(HOTSWAP_FLAG_PATH, "w") as f:
        f.write(f"Updated at {time.time()}")
        
    logger.info("=== 🏁 RETRAIN PROCESS FINISHED ===")

if __name__ == "__main__":
    main()
