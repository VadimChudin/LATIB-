import pandas as pd
import numpy as np
import ta
import uuid
import json
from typing import Dict, Any, List, Optional

from strategies.base_strategy import BaseStrategy, Signal


class ScalpMTFStrategy(BaseStrategy):
    """
    1m MTF Scalping Strategy
    ------------------------
    Focuses on micro-breakouts on the 1m chart, filtered by the trend of
    a higher timeframe (represented by slower EMAs).
    
    In the Live Rust Engine, this will be combined with Order Book Imbalance
    and Trade Delta / CVD. In Python (for ML training), we rely on 
    the technicals to generate candidate setups.
    """
    
    def __init__(self):
        super().__init__(name="ScalpMTF", default_timeframe="1m")
        
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            "fast_ema": [9, 13],
            "slow_ema": [50, 100],
            "rsi_thresh": [25, 30, 70, 75],
            "tp_rr": [0.8, 1.0, 1.5]
        }
        
    def backtest_logic(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        df = df.copy()
        
        fast_ema = int(params.get('fast_ema', 9))
        slow_ema = int(params.get('slow_ema', 50))
        rsi_thresh = int(params.get('rsi_thresh', 30))
        tp_rr = float(params.get('tp_rr', 1.0))
        
        # Fixed SL for scalping (tight, e.g., 0.5%)
        # But we use ATR to adapt to current volatility
        atr_period = 14
        if len(df) < atr_period:
            df['signal'] = 0
            df['trade_pnl_r'] = 0.0
            return df
            
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_period)
        
        # Trend EMAs
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=fast_ema)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=slow_ema)
        
        # Momentum
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Micro-Breakout indicator (Donchian-like or close > prev highs)
        df['highest_high_5'] = df['high'].rolling(5).max().shift(1)
        df['lowest_low_5'] = df['low'].rolling(5).min().shift(1)
        
        # Signals initialization
        df['signal'] = 0       # 1=Long, -1=Short
        df['entry_price'] = 0.0
        df['sl_price'] = 0.0
        df['tp_price'] = 0.0
        df['trade_pnl_r'] = 0.0
        df['entry_idx'] = 0
        df['entry_p'] = 0.0
        df['exit_p'] = 0.0
        df['sl_p'] = 0.0
        df['trade_dir_str'] = ""
        
        # We drop NaNs to avoid iterating over invalid rows
        valid_idx = df.dropna().index
        
        in_trade = False
        trade_dir = 0
        entry_p = 0.0
        sl_p = 0.0
        tp_p = 0.0
        entry_bar = 0
        
        # SL/TP logic based on R:R
        # For a 1m scalp, ATR multiplier is typically smaller
        sl_atr_mult = 1.0 
        
        # We iterate over valid rows to simulate trades
        for i in valid_idx:
            curr = df.loc[i]
            
            if in_trade:
                # Check SL
                if trade_dir == 1 and curr['low'] <= sl_p:
                    df.at[i, 'trade_pnl_r'] = -1.0
                    df.at[i, 'entry_idx'] = entry_bar
                    df.at[i, 'entry_p'] = entry_p
                    df.at[i, 'exit_p'] = sl_p
                    df.at[i, 'sl_p'] = sl_p
                    df.at[i, 'trade_dir'] = "LONG"
                    in_trade = False
                elif trade_dir == -1 and curr['high'] >= sl_p:
                    df.at[i, 'trade_pnl_r'] = -1.0
                    df.at[i, 'entry_idx'] = entry_bar
                    df.at[i, 'entry_p'] = entry_p
                    df.at[i, 'exit_p'] = sl_p
                    df.at[i, 'sl_p'] = sl_p
                    df.at[i, 'trade_dir_str'] = "SHORT"
                    in_trade = False
                # Check TP
                elif trade_dir == 1 and curr['high'] >= tp_p:
                    df.at[i, 'trade_pnl_r'] = tp_rr
                    df.at[i, 'entry_idx'] = entry_bar
                    df.at[i, 'entry_p'] = entry_p
                    df.at[i, 'exit_p'] = tp_p
                    df.at[i, 'sl_p'] = sl_p
                    df.at[i, 'trade_dir_str'] = "LONG"
                    in_trade = False
                elif trade_dir == -1 and curr['low'] <= tp_p:
                    df.at[i, 'trade_pnl_r'] = tp_rr
                    df.at[i, 'entry_idx'] = entry_bar
                    df.at[i, 'entry_p'] = entry_p
                    df.at[i, 'exit_p'] = tp_p
                    df.at[i, 'sl_p'] = sl_p
                    df.at[i, 'trade_dir_str'] = "SHORT"
                    in_trade = False
                    
            if not in_trade:
                # ENTRY LOGIC
                trend_up = curr['ema_fast'] > curr['ema_slow']
                trend_down = curr['ema_fast'] < curr['ema_slow']
                
                # Long: Trend up + RSI pullback + micro-breakout
                if rsi_thresh < 50:
                    rsi_ok = curr['rsi'] < rsi_thresh
                    if i - 1 in df.index:
                        rsi_ok = rsi_ok or df.loc[i-1, 'rsi'] < rsi_thresh
                    long_cond = trend_up and rsi_ok and (curr['close'] > curr['highest_high_5'])
                    if long_cond:
                        in_trade = True
                        trade_dir = 1
                        entry_p = curr['close']
                        sl_p = entry_p - (curr['atr'] * sl_atr_mult)
                        tp_p = entry_p + (curr['atr'] * sl_atr_mult * tp_rr)
                        entry_bar = i
                        
                        df.at[i, 'signal'] = 1
                        df.at[i, 'entry_price'] = entry_p
                        df.at[i, 'sl_price'] = sl_p
                        df.at[i, 'tp_price'] = tp_p
                        continue
                
                # Short: Trend down + RSI overbought + breakdown
                if rsi_thresh > 50:
                    rsi_ok = curr['rsi'] > rsi_thresh
                    if i - 1 in df.index:
                        rsi_ok = rsi_ok or df.loc[i-1, 'rsi'] > rsi_thresh
                    short_cond = trend_down and rsi_ok and (curr['close'] < curr['lowest_low_5'])
                    if short_cond:
                        in_trade = True
                        trade_dir = -1
                        entry_p = curr['close']
                        sl_p = entry_p + (curr['atr'] * sl_atr_mult)
                        tp_p = entry_p - (curr['atr'] * sl_atr_mult * tp_rr)
                        entry_bar = i
                        
                        df.at[i, 'signal'] = -1
                        df.at[i, 'entry_price'] = entry_p
                        df.at[i, 'sl_price'] = sl_p
                        df.at[i, 'tp_price'] = tp_p
                        
        return df

    def generate_signal(self, df, current_idx, params):
        """Real-time signal generation for ScalpMTF."""
        if current_idx < 205:
            return None
        
        fast_ema = int(params.get('fast_ema', 9))
        slow_ema = int(params.get('slow_ema', 50))
        rsi_thresh = int(params.get('rsi_thresh', 30))
        tp_rr = float(params.get('tp_rr', 1.0))
        sl_atr_mult = 1.0
        
        curr = df.iloc[current_idx]
        prev = df.iloc[current_idx - 1]
        
        # Need pre-computed indicators
        if 'ema_fast' not in df.columns:
            return None
        
        trend_up = curr['ema_fast'] > curr['ema_slow']
        trend_down = curr['ema_fast'] < curr['ema_slow']
        atr = curr.get('atr', 0)
        if atr <= 0:
            return None
        
        price = curr['close']
        
        # Long signal
        if rsi_thresh < 50 and trend_up:
            if (curr['rsi'] < rsi_thresh or prev['rsi'] < rsi_thresh) and curr['close'] > curr.get('highest_high_5', 0):
                sl = price - (atr * sl_atr_mult)
                tp = price + (atr * sl_atr_mult * tp_rr)
                from strategies.base_strategy import Signal
                return Signal(direction='LONG', entry_price=price, sl_price=sl, tp_price=tp, confidence=0.5, features=[])
        
        # Short signal
        if rsi_thresh > 50 and trend_down:
            if (curr['rsi'] > rsi_thresh or prev['rsi'] > rsi_thresh) and curr['close'] < curr.get('lowest_low_5', float('inf')):
                sl = price + (atr * sl_atr_mult)
                tp = price - (atr * sl_atr_mult * tp_rr)
                from strategies.base_strategy import Signal
                return Signal(direction='SHORT', entry_price=price, sl_price=sl, tp_price=tp, confidence=0.5, features=[])
        
        return None

    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for ML training."""
        f_df = pd.DataFrame(index=df.index)
        
        # Core momentum features
        f_df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        f_df['rsi_5'] = ta.momentum.rsi(df['close'], window=5)
        
        # Trend features
        f_df['ema_9_dist'] = (df['close'] - ta.trend.ema_indicator(df['close'], window=9)) / df['close']
        f_df['ema_50_dist'] = (df['close'] - ta.trend.ema_indicator(df['close'], window=50)) / df['close']
        f_df['ema_200_dist'] = (df['close'] - ta.trend.ema_indicator(df['close'], window=200)) / df['close']
        
        # Volatility / Range features
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        f_df['atr_ratio'] = atr / df['close']
        
        # Scalping specific: Micro momentum (bar closures)
        f_df['body_size'] = abs(df['close'] - df['open']) / atr
        f_df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / atr
        f_df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / atr
        
        # Volume features
        volume_sma = df['volume'].rolling(20).mean()
        f_df['volume_surge'] = df['volume'] / volume_sma.replace(0, np.nan)
        
        # New Micro Features for Scalping
        f_df['volatility_zscore'] = (f_df['atr_ratio'] - f_df['atr_ratio'].rolling(100).mean()) / f_df['atr_ratio'].rolling(100).std()
        f_df['micro_trend'] = (df['close'] - df['close'].rolling(5).mean()) / atr.replace(0, np.nan)
        f_df['tightness'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / atr.replace(0, np.nan)

        # Time of day (hours, minutes) - crucial for scalping sessions
        if pd.api.types.is_datetime64_any_dtype(df.index):
            f_df['hour'] = df.index.hour
            f_df['minute'] = df.index.minute
        elif 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            f_df['hour'] = ts.dt.hour
            f_df['minute'] = ts.dt.minute
            
        return f_df.fillna(0)
