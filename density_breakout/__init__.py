"""
Density Breakout Strategy — Phase 30
======================================
Pipeline:
  1. find_density_episodes.py — Find consolidation zones near S/R levels
  2. download_episodes.py    — Download hybrid data (1m + ticks + L2)
  3. label_episodes.py       — Label outcomes (BROKE/FAKE/REJECTED)
  4. rl_env.py + train_rl.py — Train PPO agent, export ONNX
"""
