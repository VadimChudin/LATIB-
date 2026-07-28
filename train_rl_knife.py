import os
import json
import glob
import random
import torch
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from density_breakout.rl_env_knife import KnifeContinuousEnv


def export_ppo_weights_to_json(model, out_file: str):
    """
    Extracts the PyTorch weights of the trained PPO policy network
    and saves them out as JSON so that Rust can run them locally
    using simple matrix multiplication! (Blazing fast for HFT)
    """
    policy = model.policy
    state_dict = policy.state_dict()
    
    weights_json = {
        "architecture": "MlpPolicy",
        "actor": {}
    }
    
    for k, v in state_dict.items():
        if "mlp_extractor.policy_net" in k or "action_net" in k:
            weights_json["actor"][k] = v.cpu().numpy().tolist()
            
    with open(out_file, 'w') as f:
        json.dump(weights_json, f, indent=2)
        
    print(f"✅ Extracted PyTorch weights to {out_file} for Rust Engine!")


def split_trajectories(src_dir: str, train_dir: str, test_dir: str, train_ratio: float = 0.5):
    """Split trajectory CSVs into train/test folders (50/50 by default)."""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    all_files = glob.glob(os.path.join(src_dir, "*.csv"))
    random.seed(42)  # Reproducible split
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    print(f"   Total trajectories: {len(all_files)}")
    print(f"   Train: {len(train_files)} | Test: {len(test_files)}")
    
    # Symlink files instead of copying to save disk space
    for f in train_files:
        dst = os.path.join(train_dir, os.path.basename(f))
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(f), dst)
            except OSError:
                # Fallback to copy if symlinks not supported
                shutil.copy2(f, dst)
    
    for f in test_files:
        dst = os.path.join(test_dir, os.path.basename(f))
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(f), dst)
            except OSError:
                shutil.copy2(f, dst)
    
    return len(train_files), len(test_files)


def main():
    print("=" * 60)
    print("  RL Exit-Manager Training (PPO) — 50/50 Train/Test Split")
    print("=" * 60)
    
    src_dir = "data/rl_trajectories"
    train_dir = "data/rl_train"
    test_dir = "data/rl_test"
    
    print("\n1. Splitting trajectories 50/50...")
    n_train, n_test = split_trajectories(src_dir, train_dir, test_dir, train_ratio=0.5)
    
    train_kwargs = {"trajectories_dir": train_dir, "max_steps": 1200}
    test_kwargs = {"trajectories_dir": test_dir, "max_steps": 1200}
    
    print("\n2. Creating Vectorized Environments...")
    print(f"   Train env: {n_train} trajectories")
    print(f"   Test env:  {n_test} trajectories")
    vec_env = make_vec_env(KnifeContinuousEnv, n_envs=4, env_kwargs=train_kwargs)
    
    print("\n3. Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,        # Higher entropy to encourage exploration
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
    )
    
    # Eval callback uses TEST set only
    eval_env = make_vec_env(KnifeContinuousEnv, n_envs=1, env_kwargs=test_kwargs)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='data/models/',
        log_path='data/logs/', 
        eval_freq=10000,       # Eval every 10k steps
        n_eval_episodes=20,    # Run 20 test episodes per eval
        deterministic=True, 
        render=False
    )
    
    print("\n4. Training Model (50M timesteps, Ctrl+C to stop early)...")
    print("   Eval runs on HELD-OUT test set every 10k steps.")
    print("   Best model auto-saved to data/models/best_model.zip\n")
    
    try:
        model.learn(total_timesteps=50_000_000, callback=eval_callback, progress_bar=False)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user. Saving current model...")
    
    print("\n5. Saving final model...")
    model.save("data/models/knife_ppo_final")
    
    print("\n6. Exporting weights to JSON for Rust Engine...")
    export_ppo_weights_to_json(model, "data/models/knife_ppo_weights.json")

    print("\n✅ Training Complete!")
    print(f"   Train set: {n_train} trajectories")
    print(f"   Test set:  {n_test} trajectories")
    print("   Run `python extract_best.py` to get best checkpoint weights.")


if __name__ == "__main__":
    main()
