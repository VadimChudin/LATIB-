import subprocess
import sys
import os
import time

def run_script(script_name):
    print(f"\n" + "="*60)
    print(f"🚀 STARTING RETRAINING: {script_name}")
    print("="*60 + "\n")
    
    start_time = time.time()
    try:
        # Use the same python executable
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=False)
        duration = time.time() - start_time
        print(f"\n✅ {script_name} COMPLETED in {duration:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} FAILED with error: {e}")
        return False
    return True

def main():
    scripts = [
        "train_ml_smc.py",
        "train_ml_knife.py",
        "train_ml_scalping.py",
        "train_ml_fundingrate.py",
        "train_ml_density.py",
    ]
    
    print("🧠 AEGIS SYSTEM: MASTER ML RETRAINING PIPELINE")
    print(f"Detected {len(scripts)} training systems.")
    
    success_count = 0
    for script in scripts:
        if not os.path.exists(script):
            print(f"⚠️ Warning: {script} not found. Skipping.")
            continue
            
        if run_script(script):
            success_count += 1
            
    print("\n" + "#"*60)
    print(f"🏁 MASTER PIPELINE FINISHED: {success_count}/{len(scripts)} SUCCESSFUL")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
