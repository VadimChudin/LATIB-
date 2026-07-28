import pandas as pd
import numpy as np

class StatArbStrategy:
    """
    Statistical Arbitrage (Pairs Trading) Strategy
    
    Operates on two highly correlated assets (A and B).
    Calculates the spread, its rolling mean, and standard deviation to generate a Z-score.
    
    Signals:
    - Z_SCORE > entry_z: Short A, Long B (expecting spread to mean-revert down)
    - Z_SCORE < -entry_z: Long A, Short B (expecting spread to mean-revert up)
    - |Z_SCORE| < exit_z: Close both legs (take profit on mean reversion)
    """
    
    def __init__(self, lookback_bars: int = 100, entry_z: float = 2.0, exit_z: float = 0.0, sl_z: float = 4.0):
        self.lookback_bars = lookback_bars
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.sl_z = sl_z
        self.name = "stat_arb"
        
    def generate_signals(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the z-score of the spread and evaluates entry/exit signals.
        Returns a DataFrame identical in length to df_a with signal columns for the executor.
        """
        # Ensure identical timestamps
        # Ensure identical timestamps (and drop index to prevent ambiguity)
        df_a = df_a.copy().reset_index(drop=True)
        df_b = df_b.copy().reset_index(drop=True)
        df_a['timestamp'] = pd.to_datetime(df_a['timestamp'])
        df_b['timestamp'] = pd.to_datetime(df_b['timestamp'])
        
        # Merge on timestamp to align the series perfectly
        merged = pd.merge(df_a[['timestamp', 'close']], df_b[['timestamp', 'close']], 
                          on='timestamp', suffixes=('_A', '_B'), how='inner')
        
        # Calculate Log Spread: log(Price A) - log(Price B)
        # We use log spread to normalize percentage changes
        merged['log_spread'] = np.log(merged['close_A']) - np.log(merged['close_B'])
        
        # Rolling Mean and Std Dev of Spread
        merged['spread_mean'] = merged['log_spread'].rolling(window=self.lookback_bars).mean()
        merged['spread_std'] = merged['log_spread'].rolling(window=self.lookback_bars).std()
        
        # Calculate Z-Score: (Current Spread - Mean) / Std Dev
        merged['z_score'] = (merged['log_spread'] - merged['spread_mean']) / merged['spread_std']
        
        # Initialize signals
        merged['signal'] = 0 # 0=Hold/Wait, 1=Short A/Long B, -1=Long A/Short B
        merged['exit_signal'] = False
        merged['stop_loss'] = False
        
        # Evaluate Entry conditions
        # signal == 1: Spread is too high. A is overvalued relative to B. Short A, Long B.
        merged.loc[merged['z_score'] > self.entry_z, 'signal'] = 1
        
        # signal == -1: Spread is too low. A is undervalued relative to B. Long A, Short B.
        merged.loc[merged['z_score'] < -self.entry_z, 'signal'] = -1
        
        # Evaluate Exit conditions
        # Mean Reversion Achieved (Profit target hit)
        merged.loc[merged['z_score'].abs() <= self.exit_z, 'exit_signal'] = True
        
        # Stop Loss Hit (Spread continues diverging against us)
        merged.loc[merged['z_score'].abs() >= self.sl_z, 'stop_loss'] = True
        
        # Map signals back to original df_a structure for executor
        result_df = df_a.copy()
        result_df = pd.merge(result_df, merged[['timestamp', 'z_score', 'signal', 'exit_signal', 'stop_loss']], on='timestamp', how='left')
        
        return result_df
