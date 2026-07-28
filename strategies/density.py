import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy, Signal

class DensityStrategy(BaseStrategy):
    """
    Python implementation of the Density Breakout strategy (mimics density.rs).
    Used for Walk-Forward Analysis and Robustness testing.
    """
    def __init__(self):
        super().__init__(name="Density", default_timeframe="5m")

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            "vol_spike_mult": [2.0, 2.5, 3.0],
            "min_touches": [2, 3],
            "shakeout_pct": [0.005, 0.006, 0.01],
            "tp_rr": [1.5, 2.0, 3.0],
            "sl_atr_mult": [1.0, 1.5]
        }

    def generate_signal(self, df: pd.DataFrame, current_idx: int, params: Dict[str, Any]) -> Optional[Signal]:
        # Minimal real-time logic for Density
        if current_idx < 200: return None
        
        vol_spike_mult = params.get('vol_spike_mult', 2.5)
        min_touches = params.get('min_touches', 2)
        
        curr = df.iloc[current_idx]
        vol_ma = df['volume'].iloc[current_idx-20:current_idx].mean()
        
        if curr['volume'] > vol_ma * vol_spike_mult:
            # Simple placeholder for real-time signal
            return Signal(
                direction="LONG",
                entry_price=curr['close'],
                sl_price=curr['close'] * 0.99,
                features=[],
                confidence=0.7
            )
    def backtest_logic(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        if len(df) < 200:
            return df

        vol_spike_mult = params.get('vol_spike_mult', 2.5)
        min_touches    = int(params.get('min_touches', 2))
        shakeout_pct   = params.get('shakeout_pct', 0.006)
        tp_rr          = params.get('tp_rr', 2.0)
        sl_atr_mult    = params.get('sl_atr_mult', 1.0)
        
        # Initialize output columns
        df['trade_pnl_r'] = 0.0
        df['entry_idx'] = 0
        df['entry_p'] = 0.0
        df['exit_p'] = 0.0
        df['sl_p'] = 0.0
        df['trade_dir'] = ""
        
        # Calculate Volume MA
        df['vol_ma'] = df['volume'].rolling(20).mean()
        
        # Placeholder for resistance detection (Vectorized where possible, but level find is complex)
        # To keep it performant in WFA, we use a rolling window approach
        
        cooldown = 0
        for i in range(200, len(df)):
            if cooldown > 0:
                cooldown -= 1
                continue
                
            curr = df.iloc[i]
            window = df.iloc[i-200:i]
            
            # 1. Find Proxy Wall (Resistance)
            res_level = self._find_proxy_wall(window, min_touches)
            if res_level is None:
                continue
                
            # 2. Stop-Hunt Defense (Shakeout)
            recent = df.iloc[i-15:i]
            is_shaken_out = self._detect_shakeout(recent, res_level, shakeout_pct)
            
            # 3. Breakout Logic
            is_vol_spike = curr['volume'] > curr['vol_ma'] * vol_spike_mult
            body_range = abs(curr['high'] - curr['low'])
            if body_range == 0: body_range = 0.000001
            body_ratio = abs(curr['close'] - curr['open']) / body_range
            is_full_body = body_ratio > 0.7 and curr['close'] > curr['open']
            
            if curr['close'] > res_level and is_vol_spike and is_full_body:
                # Entry!
                sl_dist = curr['close'] * 0.005 * sl_atr_mult
                sl_price = curr['close'] - sl_dist
                tp_price = curr['close'] + (sl_dist * tp_rr)
                
                # Simple future scan for exit (mimics backtest.rs simulate_trade)
                pnl, exit_idx = self._simulate_exit(df, i, "LONG", curr['close'], sl_price, tp_price)
                
                if exit_idx > 0:
                    df.at[df.index[exit_idx], 'trade_pnl_r'] = pnl
                    df.at[df.index[exit_idx], 'entry_idx'] = i
                    df.at[df.index[exit_idx], 'entry_p'] = curr['close']
                    df.at[df.index[exit_idx], 'exit_p'] = tp_price if pnl > 0 else sl_price
                    df.at[df.index[exit_idx], 'sl_p'] = sl_price
                    df.at[df.index[exit_idx], 'trade_dir'] = "LONG"
                    cooldown = 10
                
        return df

    def _find_proxy_wall(self, window, min_touches):
        highs = window['high'].values
        last_close = window['close'].iloc[-1]
        
        # Simple clustering: find if any high price has neighbors within 0.05%
        # We'll use a fast approach: sort and count
        sorted_highs = np.sort(highs)
        for j in range(len(sorted_highs) - min_touches + 1):
            base = sorted_highs[j]
            if base <= last_close: continue
            
            count = 1
            for k in range(j + 1, len(sorted_highs)):
                if (sorted_highs[k] - base) / base < 0.0005:
                    count += 1
                else:
                    break
            
            if count >= min_touches:
                return base
        return None

    def _detect_shakeout(self, recent, level, depth_pct):
        lows = recent['low'].values
        closes = recent['close'].values
        
        dipped = False
        for low, close in zip(lows, closes):
            dist = (level - low) / level
            if dist > depth_pct:
                dipped = True
            if dipped and close > level * 0.995:
                return True
        return False

    def _simulate_exit(self, df, entry_idx, direction, entry_price, sl_price, tp_price):
        risk = abs(entry_price - sl_price)
        if risk == 0: return 0.0, -1
        
        for j in range(entry_idx + 1, min(entry_idx + 500, len(df))):
            curr = df.iloc[j]
            if direction == "LONG":
                if curr['low'] <= sl_price:
                    return -1.0, j
                if curr['high'] >= tp_price:
                    return (tp_price - entry_price) / risk, j
        return 0.0, -1
