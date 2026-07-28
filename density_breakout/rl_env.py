"""
Phase 30.4: RL Environment for Density Breakout
=================================================
Gym environment where a PPO agent learns to decide:
  WAIT / ENTER_LONG / ENTER_SHORT / SKIP

Each episode = one density consolidation zone near a S/R level.
Agent observes features extracted from body candles, tick data, and L2 book depth.
Reward = PnL-based (price_move × direction - fees).

Usage:
  from density_breakout.rl_env import DensityBreakoutEnv
  env = DensityBreakoutEnv(labeled_dir="density_breakout/data/episodes_labeled")
"""

import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ── Feature names (must match label_episodes.py output) ──
FEATURE_KEYS = [
    # Body features (from 1m consolidation candles)
    "box_width_pct",           # 0: Width of consolidation box
    "duration_mins",           # 1: How long price consolidated
    "episode_touches",         # 2: Times price touched the level
    "volume_trend",            # 3: Volume trend slope (climbing = pressure)
    "level_touches",           # 4: Historical touches on this S/R level
    "body_volume_slope",       # 5: Volume acceleration in body
    "price_return_kurtosis",   # 6: Fat tails = big move coming
    "volume_autocorrelation",  # 7: Volume serial correlation
    "poc_to_level_dist",       # 8: POC pushed toward wall?
    "squeeze_ratio",           # 9: Range compression (< 1 = squeezing)
    "taker_ratio",             # 10: Buy-side aggression at level

    # Head features (from tick data near level)
    "tick_speed_accel",        # 11: Tape speeding up at level?
    "trade_size_entropy",      # 12: Low = algo, high = crowd
    "buy_sell_clustering",     # 13: Serial buy/sell runs (HFT pattern)

    # Book depth features (from L2 snapshots)
    "wall_size_usd",           # 14: Size of wall at level ($)
    "wall_eaten_pct",          # 15: How much wall has been eaten
    "book_depth_behind",       # 16: Liquidity behind the wall ($)
    "book_imbalance",          # 17: Bid-ask imbalance near level
]

N_FEATURES = len(FEATURE_KEYS)

# ── Actions ──
ACTION_WAIT = 0
ACTION_ENTER_LONG = 1
ACTION_ENTER_SHORT = 2
ACTION_SKIP = 3

# ── Reward config ──
FEES_PCT = 0.0004       # 0.04% taker fee × 2 (entry + exit)
REWARD_SKIP = -0.005    # High penalty for skipping a good setup (BROKE)
REWARD_CORRECT_SKIP = 0.0   # ZERO reward for doing nothing (fixes skip-only policy)
MAX_STEPS_PER_EPISODE = 10  # Steps before forced decision


class DensityBreakoutEnv(gym.Env):
    """
    Gym environment for density breakout strategy.
    
    One episode = one historical density zone.
    Agent decides: enter long/short, wait, or skip.
    Reward based on actual price outcome.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        labeled_dir: str = "density_breakout/data/episodes_labeled",
        max_steps: int = MAX_STEPS_PER_EPISODE,
    ):
        super().__init__()

        self.max_steps = max_steps

        # Load all labeled episodes
        self.episodes = self._load_episodes(labeled_dir)
        if not self.episodes:
            raise ValueError(f"No labeled episodes found in {labeled_dir}")

        print(f"DensityBreakoutEnv: loaded {len(self.episodes)} episodes")

        # Action space: WAIT, ENTER_LONG, ENTER_SHORT, SKIP
        self.action_space = spaces.Discrete(4)

        # Observation: features + step counter + side encoding
        # side: 0 = support, 1 = resistance
        # step_pct: how far we are in the episode (0-1)
        obs_dim = N_FEATURES + 2  # features + side + step_pct
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self.current_episode_idx = 0
        self.current_step = 0
        self.current_episode = None

    def _load_episodes(self, labeled_dir: str) -> List[Dict]:
        """Load all labeled episodes from JSON files."""
        episodes = []
        labeled_path = Path(labeled_dir)

        if not labeled_path.exists():
            return episodes

        for json_file in sorted(labeled_path.glob("*_labeled.json")):
            with open(json_file) as f:
                data = json.load(f)
                for ep in data:
                    if ep.get("label") and ep.get("features"):
                        episodes.append(ep)

        return episodes

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current episode features."""
        ep = self.current_episode
        features = ep.get("features", {})

        obs = np.zeros(N_FEATURES + 2, dtype=np.float32)

        # Fill feature values
        for i, key in enumerate(FEATURE_KEYS):
            val = features.get(key, 0.0)
            if val is None or np.isnan(val) or np.isinf(val):
                val = 0.0
            obs[i] = val

        # Side encoding: 0 = support, 1 = resistance
        obs[N_FEATURES] = 1.0 if ep.get("side") == "resistance" else 0.0

        # Step progress (0-1)
        obs[N_FEATURES + 1] = self.current_step / self.max_steps

        return obs

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to prevent exploding gradients."""
        # Clip extreme values
        obs = np.clip(obs, -100.0, 100.0)

        # Normalize specific features that can have large ranges
        # wall_size_usd (idx 14): log-scale
        if obs[14] > 0:
            obs[14] = np.log1p(obs[14]) / 15.0  # log($1M) ≈ 14
        # book_depth_behind (idx 16): log-scale
        if obs[16] > 0:
            obs[16] = np.log1p(obs[16]) / 15.0
        # duration_mins (idx 1): normalize to 0-1 (max ~480 min)
        obs[1] = obs[1] / 480.0

        return obs

    def reset(self, *, seed=None, options=None):
        """Reset to a new random episode."""
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Pick next episode (sequential for reproducibility, random for training)
        if options and options.get("sequential"):
            self.current_episode_idx = (self.current_episode_idx + 1) % len(self.episodes)
        else:
            self.current_episode_idx = np.random.randint(len(self.episodes))

        self.current_episode = self.episodes[self.current_episode_idx]
        self.current_step = 0

        obs = self._get_observation()
        obs = self._normalize_obs(obs)

        return obs, {}

    def step(self, action: int):
        """
        Execute one step.
        
        Actions:
          0 = WAIT (observe more, get same obs next step)
          1 = ENTER_LONG 
          2 = ENTER_SHORT
          3 = SKIP (done, small penalty or reward)
        """
        self.current_step += 1
        ep = self.current_episode
        label = ep.get("label", "REJECTED")
        side = ep.get("side", "resistance")

        terminated = False
        truncated = False
        reward = 0.0

        if action == ACTION_WAIT:
            # Keep observing — small time penalty
            reward = -0.0005
            if self.current_step >= self.max_steps:
                # Forced to decide — treat as skip
                reward = REWARD_SKIP
                truncated = True

        elif action == ACTION_SKIP:
            # Skip this setup
            if label in ("FAKE", "REJECTED"):
                reward = REWARD_CORRECT_SKIP  # Good: avoided bad trade
            else:
                reward = REWARD_SKIP  # Missed opportunity
            terminated = True

        elif action == ACTION_ENTER_LONG:
            reward = self._calculate_trade_reward("LONG", label, side)
            terminated = True

        elif action == ACTION_ENTER_SHORT:
            reward = self._calculate_trade_reward("SHORT", label, side)
            terminated = True

        obs = self._get_observation()
        obs = self._normalize_obs(obs)

        info = {
            "label": label,
            "action": action,
            "side": side,
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def _calculate_trade_reward(
        self, direction: str, label: str, side: str
    ) -> float:
        """
        Calculate PnL-based reward for a trade.
        
        Logic:
          - BROKE at resistance + LONG = good (breakout up)
          - BROKE at support + SHORT = good (breakout down)
          - REJECTED at resistance + SHORT = good (bounce down)
          - REJECTED at support + LONG = good (bounce up)
          - FAKE = loss (got faked out)
          - Wrong direction = loss
        """
        ep = self.current_episode
        box_width = ep.get("features", {}).get("box_width_pct", 0.005)

        if label == "BROKE":
            # Breakout happened
            if side == "resistance" and direction == "LONG":
                # Correct: long breakout above resistance
                reward = 0.02 - FEES_PCT  # ~2% move - fees
            elif side == "support" and direction == "SHORT":
                # Correct: short breakout below support
                reward = 0.02 - FEES_PCT
            else:
                # Wrong direction on breakout
                reward = -(0.01 + FEES_PCT)  # SL hit + fees

        elif label == "REJECTED":
            # Level held
            if side == "resistance" and direction == "SHORT":
                # Correct: short rejection from resistance (fade the level)
                reward = 0.01 - FEES_PCT
            elif side == "support" and direction == "LONG":
                # Correct: long bounce from support
                reward = 0.01 - FEES_PCT
            else:
                # Wrong: tried to trade breakout but level held
                reward = -(box_width + FEES_PCT)  # Box width loss + fees

        elif label == "FAKE":
            # Fakeout — everyone loses
            reward = -(0.005 + FEES_PCT)  # Modest loss on fakeout

        return reward


def make_env(labeled_dir: str = "density_breakout/data/episodes_labeled"):
    """Create the environment (for SB3 compatibility)."""
    return DensityBreakoutEnv(labeled_dir=labeled_dir)
