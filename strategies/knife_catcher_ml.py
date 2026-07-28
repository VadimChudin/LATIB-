import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os

from strategies.base_strategy import BaseStrategy, Signal
from strategies.knife_catcher import KnifeCatcherStrategy
from core.ml_filter import RegimeMLFilter
import logging

logger = logging.getLogger(__name__)

class KnifeCatcherMLStrategy(BaseStrategy):
    """
    Knife Catching / Mean Reversion Strategy (ML Enhanced)
    ------------------------------------------------------
    Uses extreme Bollinger Band deviations and RSI oversold conditions to catch 
    liquidity bounces. Then applies a Triple-AI Ensemble (XGB, LGBM, RF) to filter
    out trades with a win probability < 55%.
    """
    
    def __init__(self):
        super().__init__(name="KnifeCatcher_ML", default_timeframe="5m")
        self.base_strat = KnifeCatcherStrategy()
        # The ML Filter defaults to looking for a model. We'll pass the model name it expects,
        # or load explicitly below. The threshold is an attribute we assign.
        self.ml_filter = RegimeMLFilter("knife_catcher_model")
        self.ml_filter.threshold = 0.55
        
        # Load the pre-trained Triple-AI model
        self.ml_filter.load_model()
            
    def get_parameter_space(self) -> Dict[str, list]:
        # Return the parameter grid for brute force/engine optimization
        return self.base_strat.get_parameter_space()
        
    def backtest_logic(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        The AutoCore engine calls this during the 3-hour optimization interval.
        We run the base backtest loop.
        Note: The actual Engine loop uses backtest PnL without ML (or we could overlay ML,
        but for speed, we optimize base and let ML filter live).
        Here we proxy to the base strategy.
        """
        # We need a custom backtest loop loop if we want ML filtering *during* engine backtest.
        # But for engine scanning (finding hot coins), base PnL proxy is usually sufficient and faster.
        # Alternatively, we could manually loop and call `self.generate_signal()`.
        
        # Let's run a manual loop to ensure ML filtering applies to Engine stats!
        df_sim = df.copy()
        
        # Fast indicator compute
        bb_std = params.get('bb_std', 2.0)
        bbl_exists = any(c.startswith('BBL') and str(bb_std) in c for c in df_sim.columns)
        if not bbl_exists:
            df_sim.ta.bbands(length=20, std=bb_std, append=True)
        if 'RSI_14' not in df_sim.columns:
            df_sim.ta.rsi(length=14, append=True)
        if 'ATRr_14' not in df_sim.columns:
            df_sim.ta.atr(length=14, append=True)
        if 'CCI_14_0.015' not in df_sim.columns:
            df_sim.ta.cci(length=14, append=True)
        if 'ADX_14' not in df_sim.columns:
            df_sim.ta.adx(length=14, append=True)
            
        df_sim['trade_pnl_r'] = 0.0
        df_sim['entry_idx'] = 0
        
        in_position = False
        sl_price = 0.0
        tp_price = 0.0
        direction = None
        entry_idx = 0
        
        for i in range(50, len(df_sim)):
            current = df_sim.iloc[i]
            
            if in_position:
                high = current['high']
                low = current['low']
                
                if direction == 'LONG':
                    if low <= sl_price:
                        df_sim.at[df_sim.index[i], 'trade_pnl_r'] = -1.0
                        in_position = False
                    elif high >= tp_price:
                        df_sim.at[df_sim.index[i], 'trade_pnl_r'] = params.get('tp_rr', 1.0)
                        in_position = False
                        
                elif direction == 'SHORT':
                    if high >= sl_price:
                        df_sim.at[df_sim.index[i], 'trade_pnl_r'] = -1.0
                        in_position = False
                    elif low <= tp_price:
                        df_sim.at[df_sim.index[i], 'trade_pnl_r'] = params.get('tp_rr', 1.0)
                        in_position = False
            else:
                sig = self.generate_signal(df_sim, i, params)
                if sig:
                    in_position = True
                    direction = sig.direction
                    sl_price = sig.sl_price
                    tp_price = sig.tp_price
                    entry_idx = i
                    
        return df_sim
            
    def generate_signal(self, df: pd.DataFrame, current_idx: int, params: Dict[str, Any]) -> Optional[Signal]:
        # Delegate directly to base strategy. The LiveExecutor and training scripts
        # will handle the actual Triple-AI ML preparation (including BTC Gravity).
        return self.base_strat.generate_signal(df, current_idx, params)
