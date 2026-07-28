"""
Phase 30.4: Train PPO Agent for Density Breakout
==================================================
Trains a PPO agent on labeled density episodes using Stable-Baselines3.
Exports the trained model to ONNX for use in Rust live engine.

Usage:
  python density_breakout/train_rl.py
  python density_breakout/train_rl.py --timesteps 500000 --eval
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# ── Paths ──
LABELED_DIR = "density_breakout/data/episodes_labeled"
MODEL_DIR = Path("density_breakout/data/rl_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_ppo(
    total_timesteps: int = 200_000,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int = 42,
    eval_episodes: int = 100,
):
    """Train PPO on density breakout episodes."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("Run: pip install stable-baselines3[extra] gymnasium")
        return None

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from density_breakout.rl_env import DensityBreakoutEnv

    # Check if we have data
    labeled_path = Path(LABELED_DIR)
    labeled_files = list(labeled_path.glob("*_labeled.json"))
    if not labeled_files:
        print(f"ERROR: No labeled episodes in {LABELED_DIR}")
        print("Run the pipeline first:")
        print("  1. python density_breakout/find_density_episodes.py --all-symbols --days 30")
        print("  2. python density_breakout/download_episodes.py --all-symbols")
        print("  3. python density_breakout/label_episodes.py --all-symbols")
        return None

    # Count total episodes
    total_episodes = 0
    label_counts = {}
    for f in labeled_files:
        with open(f) as fh:
            data = json.load(fh)
            for ep in data:
                if ep.get("label") and ep.get("features"):
                    total_episodes += 1
                    l = ep["label"]
                    label_counts[l] = label_counts.get(l, 0) + 1

    print(f"="*60)
    print(f"Density Breakout PPO Training")
    print(f"="*60)
    print(f"Episodes: {total_episodes}")
    for k, v in sorted(label_counts.items()):
        print(f"  {k}: {v} ({v/total_episodes*100:.0f}%)")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"LR: {learning_rate}, Batch: {batch_size}, Epochs: {n_epochs}")
    print()

    if total_episodes < 50:
        print("WARNING: Less than 50 episodes. Model may overfit.")
        print("Consider collecting more data first.")

    # Create environment
    env = DummyVecEnv([lambda: DensityBreakoutEnv(labeled_dir=LABELED_DIR)])

    # Create eval environment
    eval_env = DummyVecEnv([lambda: DensityBreakoutEnv(labeled_dir=LABELED_DIR)])

    # Train PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        seed=seed,
        policy_kwargs={
            "net_arch": [128, 64, 32],  # 3-layer MLP
        },
    )

    print("Training PPO...")
    model.learn(
        total_timesteps=total_timesteps,
    )

    # Save model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    model_path = MODEL_DIR / f"density_ppo_{timestamp}"
    model.save(str(model_path))
    print(f"\nModel saved: {model_path}")

    # Evaluate
    print(f"\nEvaluating on {eval_episodes} episodes...")
    eval_results = evaluate_model(model, eval_env, eval_episodes)

    # Export to ONNX
    onnx_path = export_to_onnx(model, MODEL_DIR / f"density_ppo_{timestamp}.onnx")

    return model, eval_results


def evaluate_model(model, env, n_episodes: int = 100) -> dict:
    """Evaluate trained model on episodes."""
    results = {
        "total": 0,
        "correct_entries": 0,
        "wrong_entries": 0,
        "correct_skips": 0,
        "missed_opportunities": 0,
        "total_reward": 0.0,
        "actions": {0: 0, 1: 0, 2: 0, 3: 0},
        "label_accuracy": {},
    }

    for i in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

            # Extract info from vectorized env
            if info and len(info) > 0 and isinstance(info[0], dict):
                last_info = info[0]

            results["actions"][int(action[0])] = results["actions"].get(int(action[0]), 0) + 1

        results["total"] += 1
        results["total_reward"] += episode_reward

        # Classify result
        action_taken = last_info.get("action", 0)
        label = last_info.get("label", "UNKNOWN")
        side = last_info.get("side", "resistance")

        if action_taken in (1, 2):  # Entered a trade
            if episode_reward > 0:
                results["correct_entries"] += 1
            else:
                results["wrong_entries"] += 1
        elif action_taken == 3:  # Skipped
            if label in ("FAKE", "REJECTED"):
                results["correct_skips"] += 1
            else:
                results["missed_opportunities"] += 1

    # Print summary
    total = results["total"]
    print(f"\nEvaluation Results ({total} episodes):")
    print(f"  Total reward: {results['total_reward']:.4f}")
    print(f"  Avg reward/episode: {results['total_reward']/max(total,1):.4f}")
    print(f"  Correct entries: {results['correct_entries']}")
    print(f"  Wrong entries: {results['wrong_entries']}")
    print(f"  Correct skips: {results['correct_skips']}")
    print(f"  Missed opportunities: {results['missed_opportunities']}")
    print(f"  Actions: {results['actions']}")

    win_rate = (
        results["correct_entries"] /
        max(results["correct_entries"] + results["wrong_entries"], 1)
    )
    print(f"  Win Rate: {win_rate*100:.1f}%")

    return results


def export_to_onnx(model, onnx_path: Path) -> Optional[Path]:
    """Export trained SB3 model to ONNX for Rust inference."""
    try:
        import torch
        from stable_baselines3.common.policies import ActorCriticPolicy

        # Get the policy network
        policy = model.policy

        # Create dummy input matching observation space
        obs_size = model.observation_space.shape[0]
        dummy_input = torch.randn(1, obs_size)

        # Export actor network only (we just need action predictions)
        class PolicyWrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.features_extractor = policy.features_extractor
                self.mlp_extractor = policy.mlp_extractor
                self.action_net = policy.action_net

            def forward(self, x):
                features = self.features_extractor(x)
                latent_pi, _ = self.mlp_extractor(features)
                action_logits = self.action_net(latent_pi)
                return action_logits

        wrapper = PolicyWrapper(policy)
        wrapper.eval()

        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_path),
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
            opset_version=11,
        )

        print(f"ONNX model exported: {onnx_path} ({onnx_path.stat().st_size // 1024}KB)")
        return onnx_path

    except Exception as e:
        print(f"WARNING: ONNX export failed: {e}")
        print("Model saved as .zip only. ONNX export requires torch and onnx packages.")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 30.4: Train PPO for Density Breakout")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    args = parser.parse_args()

    result = train_ppo(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch,
        seed=args.seed,
    )

    if result:
        model, eval_results = result
        # Save eval results
        eval_path = MODEL_DIR / "eval_results.json"
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        serializable = json.loads(json.dumps(eval_results, default=convert))
        with open(eval_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nEval results saved: {eval_path}")
