import json
from stable_baselines3 import PPO
from train_rl_knife import export_ppo_weights_to_json
import sys

# Force stdout to utf-8 just in case, but using plain text
sys.stdout.reconfigure(encoding='utf-8')

def main():
    best_model_path = "data/models/best_model.zip"
    out_file = "data/models/knife_ppo_weights.json"
    
    print(f"Loading interrupted model from {best_model_path}...")
    try:
        model = PPO.load(best_model_path)
        export_ppo_weights_to_json(model, out_file)
        print(f"Successfully extracted brains to {out_file}!")
        print("You can now run the Rust backtester: cargo run --release --bin backtest")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()
