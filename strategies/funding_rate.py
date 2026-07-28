import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional
from strategies.base_strategy import BaseStrategy, Signal

class FundingRateStrategy(BaseStrategy):
    """
    Funding Rate Mean Reversion Strategy (Python)
    -------------------------------------------
    Identifies overcrowded trades via extreme funding rates and 
    technical overextension. Generates candidates for ML filtering.
    """

    def __init__(self):
        super().__init__(name="FundingRate_MR", default_timeframe="5m")

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            "fr_long_thresh": [0.01, 0.03, 0.05],
            "fr_short_thresh": [0.01, 0.03, 0.05],
            "sl_atr_mult": [1.5],
            "tp_rr": [2.0],
            "trail_activate_r": [1.0],
            "trail_atr_mult": [0.5]
        }

    def generate_signal(self, df: pd.DataFrame, current_idx: int, params: Dict[str, Any]) -> Optional[Signal]:
        if current_idx < 210: return None
        curr = df.iloc[current_idx]
        long_thresh = float(params.get('fr_long_thresh', 0.03))
        short_thresh = float(params.get('fr_short_thresh', 0.05))
        fr = curr.get('funding_rate', 0.0)

        if fr < -long_thresh / 100.0:
            return Signal("LONG", curr['close'], curr['close'] * 0.99, [], 0.7)
        elif fr > short_thresh / 100.0:
            return Signal("SHORT", curr['close'], curr['close'] * 1.01, [], 0.7)
        return None

    def backtest_logic(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        df = df.copy()
        
        long_thresh = float(params.get('fr_long_thresh', 0.03))
        short_thresh = float(params.get('fr_short_thresh', 0.05))
        tp_rr = float(params.get('tp_rr', 2.0))
        sl_atr_mult = float(params.get('sl_atr_mult', 1.5))
        
        # In Python backtest, we might not have real FR in the CSV.
        # We simulate it via RSI + Volatility correlation if missing.
        if 'funding_rate' not in df.columns:
            rsi = ta.momentum.rsi(df['close'], window=14)
            vol_sma = df['volume'].rolling(20).mean()
            vol_fac = (df['volume'] / vol_sma).clip(0, 3)
            df['funding_rate'] = ((rsi - 50) / 50.0) * vol_fac * 0.05
            
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        
        df['signal'] = 0
        df['trade_pnl_r'] = 0.0
        df['entry_idx'] = 0
        df['entry_p'] = 0.0
        df['exit_p'] = 0.0
        df['sl_p'] = 0.0
        df['trade_dir'] = ""
        
        in_trade = False
        trade_dir = 0
        entry_p = 0.0
        sl_p = 0.0
        tp_p = 0.0
        
        for i in range(210, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            if in_trade:
                # Simple TP/SL for backtest logic (trailing is handled in Rust/Position Manager)
                if trade_dir == 1:
                    if curr['low'] <= sl_p:
                        df.at[df.index[i], 'trade_pnl_r'] = -1.0
                        in_trade = False
                    elif curr['high'] >= tp_p:
                        df.at[df.index[i], 'trade_pnl_r'] = tp_rr
                        df.at[df.index[i], 'entry_idx'] = entry_bar
                        df.at[df.index[i], 'entry_p'] = entry_p
                        df.at[df.index[i], 'exit_p'] = tp_p
                        df.at[df.index[i], 'sl_p'] = sl_p
                        df.at[df.index[i], 'trade_dir'] = "LONG"
                        in_trade = False
                else:
                    if curr['high'] >= sl_p:
                        df.at[df.index[i], 'trade_pnl_r'] = -1.0
                        in_trade = False
                    elif curr['low'] <= tp_p:
                        df.at[df.index[i], 'trade_pnl_r'] = tp_rr
                        df.at[df.index[i], 'entry_idx'] = entry_bar
                        df.at[df.index[i], 'entry_p'] = entry_p
                        df.at[df.index[i], 'exit_p'] = tp_p
                        df.at[df.index[i], 'sl_p'] = sl_p
                        df.at[df.index[i], 'trade_dir'] = "SHORT"
                        in_trade = False
                continue

            # FR expressed in % (e.g. 0.03 in params means 0.0003 real rate)
            fr = curr['funding_rate']
            
            # LONG: Extreme negative funding
            if fr < -long_thresh / 100.0:
                if curr['close'] > curr['open'] and curr['close'] > curr['ema_200'] * 0.98:
                    in_trade = True
                    trade_dir = 1
                    entry_p = curr['close']
                    sl_p = entry_p - (curr['atr'] * sl_atr_mult)
                    tp_p = entry_p + (curr['atr'] * sl_atr_mult * tp_rr)
                    entry_bar = i
                    df.at[df.index[i], 'signal'] = 1

            # SHORT: Extreme positive funding
            elif fr > short_thresh / 100.0:
                if curr['close'] < curr['open'] and curr['close'] < curr['ema_200'] * 1.02:
                    in_trade = True
                    trade_dir = -1
                    entry_p = curr['close']
                    sl_p = entry_p + (curr['atr'] * sl_atr_mult)
                    tp_p = entry_p - (curr['atr'] * sl_atr_mult * tp_rr)
                    entry_bar = i
                    df.at[df.index[i], 'signal'] = -1

        return df

    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f_df = pd.DataFrame(index=df.index)
        
        # Funding Context
        if 'funding_rate' in df.columns:
            f_df['fr_value'] = df['funding_rate']
            f_df['fr_change'] = df['funding_rate'].diff()
        else:
            f_df['fr_value'] = 0
            f_df['fr_change'] = 0
            
        # Technical Context
        f_df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        ema200 = ta.trend.ema_indicator(df['close'], window=200)
        f_df['ema_dist'] = (df['close'] - ema200) / ema200
        
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        f_df['atr_ratio'] = atr / df['close']
        
        # Bar physics
        f_df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 0.001)
        f_df['wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, 0.001)
        
        return f_df.fillna(0)
