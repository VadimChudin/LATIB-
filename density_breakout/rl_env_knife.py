import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class KnifeContinuousEnv(gym.Env):
    """
    Exit-Manager RL Environment for Knife Strategy.
    
    Agent is FORCED into a position at episode start.
    Its only job: decide WHEN to exit.
    
    Feature vector (15 features):
      [0] normalized_absorption  - from CSV
      [1] normalized_tps         - from CSV
      [2] reclaim_pct * 1000     - from CSV
      [3] cvd_divergence / 1000  - from CSV
      [4] move_pct * 1000        - from CSV
      [5] direction (+1/-1)      - from CSV
      [6] micro_volume / 1000    - from CSV
      [7] time_in_trade          - INJECTED (0.0 → 1.0, normalized by max_steps)
      [8] unrealized_pnl         - INJECTED (pnl_pct * 100, clipped [-10, 10])
      [9-14] reserved (zeros)
    """
    
    BURN_IN_STEPS = 10  # Agent cannot exit during first N steps
    
    def __init__(self, trajectories_dir: str, max_steps: int = 1200):
        super().__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.num_features = 15
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32)
        
        self.trajectories_dir = trajectories_dir
        self.max_steps = max_steps
        
        import glob
        import os
        self.trajectory_files = glob.glob(os.path.join(self.trajectories_dir, "*.csv"))
        if not self.trajectory_files:
            print(f"Warning: No trajectories found in {self.trajectories_dir}")
        self.current_trajectory = None
        self.current_step = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.file_idx = 0
        
        # Risk configs
        self.fee_rate = 0.0004       # 0.04% taker fee per side
        self.max_drawdown = 0.015    # 1.5% stop loss
        
    def _get_obs(self):
        """Returns the current feature vector with dynamic features injected."""
        if self.current_trajectory is None or self.current_step >= len(self.current_trajectory):
            return np.zeros(self.num_features, dtype=np.float32)
            
        row = self.current_trajectory.iloc[self.current_step]
        # Get raw features from CSV (indices 2..17 → 15 features)
        features = row.values[2 : 2 + self.num_features].astype(np.float32)
        
        # Sanitize base features
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        features = np.clip(features, -100.0, 100.0)
        
        # === INJECT DYNAMIC FEATURES ===
        # [7] time_in_trade: normalized 0.0 → 1.0
        features[7] = min(self.current_step / max(self.max_steps, 1), 1.0)
        
        # [8] unrealized_pnl: percentage * 100, clipped
        current_price = row['price'] if 'price' in row.index else row.iloc[1]
        if self.entry_price > 0 and self.position_size != 0:
            if self.position_size > 0:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price
            features[8] = np.clip(pnl_pct * 100.0, -10.0, 10.0)
        else:
            features[8] = 0.0
        
        return features
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._load_next_trajectory()
        
        self.current_step = 0
        
        # PRE-ENTER POSITION BASED ON TRAJECTORY DIRECTION
        if self.current_trajectory is not None and not self.current_trajectory.empty:
            # Read direction from raw CSV data (index 7 = direction bias)
            raw_dir = self.current_trajectory.iloc[0].values[7]  # 0=ts, 1=price, 2-6=features, 7=direction
            self.entry_price = self.current_trajectory.iloc[0].values[1]  # price column
        else:
            raw_dir = 1.0
            self.entry_price = 0.0
        
        if raw_dir > 0:
            self.position_size = 1.0   # Forced Long
        else:
            self.position_size = -1.0  # Forced Short
            
        obs = self._get_obs()
        return obs, {}
        
    def step(self, action):
        """
        Exit-Manager Step:
        - First BURN_IN_STEPS: exit blocked, agent just observes drift
        - After burn-in: Long exits when action <= -0.1, Short exits when action >= 0.1
        - Stop-loss only checked AFTER burn-in
        """
        action_val = float(action[0])
        action_val = max(min(action_val, 1.0), -1.0)
        
        if self.current_trajectory is None or self.current_step >= len(self.current_trajectory) - 1:
            return self._get_obs(), 0.0, True, False, {}
            
        row = self.current_trajectory.iloc[self.current_step]
        current_price = row.values[1]  # price column by index
        step_reward = 0.0
        done = False
        
        in_burn_in = self.current_step < self.BURN_IN_STEPS
        
        # === EXIT CHECK (only after burn-in) ===
        if not in_burn_in and self.position_size != 0:
            agent_wants_out = False
            if self.position_size > 0.5 and action_val <= -0.1:
                agent_wants_out = True
            elif self.position_size < -0.5 and action_val >= 0.1:
                agent_wants_out = True
            
            if agent_wants_out:
                if self.position_size > 0:
                    pnl_pct = (current_price - self.entry_price) / max(self.entry_price, 1e-8)
                else:
                    pnl_pct = (self.entry_price - current_price) / max(self.entry_price, 1e-8)
                step_reward = (pnl_pct * 100.0) - (self.fee_rate * 2.0 * 100.0)
                self.position_size = 0.0
                done = True
        
        # === HOLD REWARD (drift) ===
        if not done and self.position_size != 0:
            if self.current_step + 1 < len(self.current_trajectory):
                next_price = self.current_trajectory.iloc[self.current_step + 1].values[1]
                drift = next_price - current_price
                if self.position_size > 0:
                    step_reward += (drift / max(current_price, 1e-8)) * 100.0
                else:
                    step_reward += -(drift / max(current_price, 1e-8)) * 100.0
            
            # Stop-loss: ONLY after burn-in to let agent learn from full trajectories
            if not in_burn_in and self.entry_price > 0:
                if self.position_size > 0:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                else:
                    pnl_pct = (self.entry_price - current_price) / self.entry_price
                if pnl_pct <= -self.max_drawdown:
                    step_reward -= 1.0  # Hard penalty for hitting stop
                    self.position_size = 0.0
                    done = True
        
        self.current_step += 1
        if self.current_step >= len(self.current_trajectory) - 1 or self.current_step >= self.max_steps:
            done = True
        
        # Force close at end of episode
        if done and self.position_size != 0 and self.entry_price > 0:
            if self.position_size > 0:
                pnl_pct = (current_price - self.entry_price) / max(self.entry_price, 1e-8)
            else:
                pnl_pct = (self.entry_price - current_price) / max(self.entry_price, 1e-8)
            step_reward += (pnl_pct * 100.0) - (self.fee_rate * 2.0 * 100.0)
            self.position_size = 0.0
            
        # Sanitize reward
        step_reward = float(np.clip(np.nan_to_num(step_reward), -10.0, 10.0))
        
        obs = self._get_obs()
        return obs, step_reward, done, False, {}

    def _load_next_trajectory(self):
        """Loads a random trajectory from exported CSVs."""
        import random
        if not self.trajectory_files:
            return
            
        file_path = random.choice(self.trajectory_files)
        try:
            self.current_trajectory = pd.read_csv(file_path, header=0)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            self.current_trajectory = None

if __name__ == "__main__":
    env = KnifeContinuousEnv(trajectories_dir="data/rl_trajectories")
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Initial obs:", obs)
    print(f"  time_in_trade = {obs[7]:.4f}")
    print(f"  unrealized_pnl = {obs[8]:.4f}")
    
    score = 0
    done = False
    steps = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        steps += 1
        
    print(f"\nEpisode finished after {steps} steps! Score: {score:.4f}%")
