from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class Signal:
    """Standardized signal object returned by all strategies"""
    direction: str       # 'LONG' or 'SHORT'
    entry_price: float   # Target entry price (usually current market close)
    sl_price: float      # Stop Loss price
    features: list       # Raw feature array for ML/Analytics (optional)
    confidence: float    # Internal confidence or AI prediction (0.0 to 1.0)
    tp_price: Optional[float] = None # Optional TP (if strategy returns fixed TP instead of calculated via RR)

class BaseStrategy(ABC):
    """
    Abstract Base Class for all ICT AutoCore Strategies.
    Every new strategy placed in the `strategies/` folder MUST inherit from this.
    """
    
    def __init__(self, name: str, default_timeframe: str = '15m'):
        self.name = name
        self.default_timeframe = default_timeframe
        self.btc_df: Optional[pd.DataFrame] = None
        
    def set_btc_context(self, btc_df: pd.DataFrame):
        """Allows the executor to inject macro BTC data into the strategy context."""
        self.btc_df = btc_df
        
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Returns the parameter grid for the backtester.
        Example:
        return {
            'sl_atr_mult': [1.0, 1.5, 2.0],
            'tp_rr': [2.0, 3.0, 4.0],
            'trailing_stop': [False, True]
        }
        """
        pass

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, current_idx: int, params: Dict[str, Any]) -> Optional[Signal]:
        """
        Real-time logic for signal generation.
        Args:
            df (pd.DataFrame): OHLCV data with pre-calculated indicators.
            current_idx (int): Current row index being evaluated.
            params (Dict): Specific parameters chosen by the Engine for this evaluation.
        Returns:
            Signal object if a setup is found, else None.
        """
        pass

    @abstractmethod
    def backtest_logic(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Vectorized or fast loop simulation for the Backtest Engine.
        Args:
            df (pd.DataFrame): Historical data.
            params (Dict): Current parameter set.
        Returns:
            pd.DataFrame: A copy of df with added columns: ['signal', 'entry', 'sl', 'tp']
        """
        pass
