import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from strategies.base_strategy import BaseStrategy, Signal
import pandas_ta as ta

class KnifeCatcherStrategy(BaseStrategy):
    """
    Knife Catching / Mean Reversion Strategy
    ----------------------------------------
    Catches aggressive liquidity dumps or FOMO spikes by looking for extreme deviations 
    from the moving average, confirmed by RSI and Volume Spikes (indicating a liquidity wall).
    
    Timeframe: Best on 5m or 15m.
    """
    
    def __init__(self):
        super().__init__(name="KnifeCatcher", default_timeframe="5m")
        
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            'rsi_oversold': [20, 25, 30],
            'bb_std': [2.5, 3.0, 3.5], # Extreme standard deviation for Bollinger
            'vol_spike_mult': [2.0, 3.0, 4.0],
            'tp_rr': [1.0, 1.5, 2.0],
            'sl_atr_mult': [1.0, 1.5]
        }
        
    def generate_signal(self, df: pd.DataFrame, current_idx: int, params: Dict[str, Any]) -> Optional[Signal]:
        if current_idx < 210:
            return None
            
        rsi_oversold = params.get('rsi_oversold', 25)
        bb_std = params.get('bb_std', 3.0)
        vol_spike_mult = params.get('vol_spike_mult', 3.0)
        tp_rr = params.get('tp_rr', 1.5)
        sl_atr_mult = params.get('sl_atr_mult', 1.0)
        
        if 'RSI_14' not in df.columns:
            return None
            
        current = df.iloc[current_idx]
        
        # Bollinger Band columns
        cache_key = f"BB_COLS_{bb_std}"
        if not hasattr(self, '_col_cache'): self._col_cache = {}
        
        if cache_key not in self._col_cache:
            self._col_cache[cache_key] = {
                'lower': next((c for c in df.columns if c.startswith('BBL') and str(bb_std) in c), None),
                'upper': next((c for c in df.columns if c.startswith('BBU') and str(bb_std) in c), None)
            }
            
        cols = self._col_cache[cache_key]
        bb_lower_col = cols['lower']
        bb_upper_col = cols['upper']
        
        if bb_lower_col is None or bb_lower_col not in df.columns:
            return None
            
        rsi = current['RSI_14']
        vol = current['volume']
        vol_sma = df['volume'].iloc[current_idx-20:current_idx].mean()
        is_vol_spike = vol > (vol_sma * vol_spike_mult)
        
        atr = current.get('ATRr_14', df['high'].iloc[current_idx-14:current_idx].max() - df['low'].iloc[current_idx-14:current_idx].min())
        
        # ═══════════════════════════════════════════════════════════════
        # STRENGTHENED FILTERS (added to improve WR from ~45% to ~55%+)
        # ═══════════════════════════════════════════════════════════════
        
        # Filter 1: EMA200 Trend — only catch dips in UPTREND
        # This single filter prevents fighting strong downtrends
        ema200 = df['close'].iloc[current_idx-200:current_idx].mean()
        
        # Filter 2: RSI Divergence Check
        # Price lower low + RSI higher low = bullish divergence (reversal signal)
        def check_rsi_divergence(direction):
            lookback = 10
            if current_idx < lookback:
                return False
            rsi_vals = df['RSI_14'].iloc[current_idx-lookback:current_idx+1]
            price_vals = df['close'].iloc[current_idx-lookback:current_idx+1]
            if rsi_vals.isna().any():
                return True  # Skip check if data missing
            
            if direction == 'LONG':
                # Bullish div: price lower low, RSI higher low
                price_min_idx = price_vals.idxmin()
                if price_vals.iloc[-1] <= price_vals.min() * 1.005:  # Near low
                    rsi_at_low = rsi_vals.loc[price_min_idx] if price_min_idx in rsi_vals.index else rsi_vals.min()
                    if rsi_vals.iloc[-1] >= rsi_at_low:  # RSI not making new low
                        return True
            else:
                # Bearish div: price higher high, RSI lower high
                price_max_idx = price_vals.idxmax()
                if price_vals.iloc[-1] >= price_vals.max() * 0.995:
                    rsi_at_high = rsi_vals.loc[price_max_idx] if price_max_idx in rsi_vals.index else rsi_vals.max()
                    if rsi_vals.iloc[-1] <= rsi_at_high:
                        return True
            return False
        
        # Filter 3: Double-touch confirmation
        # Require at least 2 touches of BB extreme within last 10 bars
        def check_double_touch(bb_col, direction):
            lookback = 10
            touches = 0
            for j in range(max(0, current_idx - lookback), current_idx + 1):
                row = df.iloc[j]
                if direction == 'LONG' and row['low'] < row.get(bb_col, row['low']):
                    touches += 1
                elif direction == 'SHORT' and row['high'] > row.get(bb_col, row['high']):
                    touches += 1
            return touches >= 2
        
        # --- LONG SIGNAL (Catching dips in uptrend) ---
        if (current['low'] < current[bb_lower_col] 
            and rsi < rsi_oversold 
            and is_vol_spike
            and current['close'] > ema200          # Filter 1: Only in uptrend
            and check_rsi_divergence('LONG')       # Filter 2: RSI divergence
            and check_double_touch(bb_lower_col, 'LONG')  # Filter 3: Double touch
        ):
            entry_price = current['close']
            sl_price = current['low'] - (atr * sl_atr_mult)
            risk = entry_price - sl_price
            if risk > 0:
                tp_price = entry_price + (risk * tp_rr)
                return Signal(
                    direction='LONG', entry_price=entry_price,
                    sl_price=sl_price, tp_price=tp_price,
                    confidence=0.75, features=['dip_buy', 'rsi_div']
                )
                    
        # --- SHORT SIGNAL (Fading FOMO in downtrend) ---
        rsi_overbought = 100 - rsi_oversold
        if (current['high'] > current[bb_upper_col] 
            and rsi > rsi_overbought 
            and is_vol_spike
            and current['close'] < ema200          # Filter 1: Only in downtrend
            and check_rsi_divergence('SHORT')      # Filter 2: RSI divergence  
            and check_double_touch(bb_upper_col, 'SHORT')  # Filter 3: Double touch
        ):
            entry_price = current['close']
            sl_price = current['high'] + (atr * sl_atr_mult)
            risk = sl_price - entry_price
            if risk > 0:
                tp_price = entry_price - (risk * tp_rr)
                return Signal(
                    direction='SHORT', entry_price=entry_price,
                    sl_price=sl_price, tp_price=tp_price,
                    confidence=0.75, features=['fomo_fade', 'rsi_div']
                )
                    
        return None
        
    def backtest_logic(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Custom fast backtester for the Knife Catcher strategy.
        Pre-calculates indicators using pandas-ta for speed.
        """
        df_sim = df.copy()
        
        # Pre-calculate required indicators if missing
        bb_std = params.get('bb_std', 3.0)
        
        # We need an intelligent check to see if the column exists due to pandas-ta float formatting
        bbl_exists = any(c.startswith('BBL') and str(bb_std) in c for c in df_sim.columns)
        
        # Only compute if missing to save massive RAM in grid search
        if not bbl_exists:
            df_sim.ta.bbands(length=20, std=bb_std, append=True)
        if 'RSI_14' not in df_sim.columns:
            df_sim.ta.rsi(length=14, append=True)
        if 'EMA_20' not in df_sim.columns:
            df_sim.ta.ema(length=20, append=True)
        if 'ATRr_14' not in df_sim.columns:
            df_sim.ta.atr(length=14, append=True)
        
        df_sim['trade_pnl_r'] = 0.0
        df_sim['entry_idx'] = 0
        df_sim['entry_p'] = 0.0
        df_sim['exit_p'] = 0.0
        df_sim['sl_p'] = 0.0
        df_sim['trade_dir'] = ""
        
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        sl_price = 0.0
        tp_price = 0.0
        direction = None
        
        h_arr = df_sim['high'].values
        l_arr = df_sim['low'].values
        c_arr = df_sim['close'].values
        v_arr = df_sim['volume'].values

        rsi_arr = df_sim['RSI_14'].values if 'RSI_14' in df_sim.columns else np.full(len(c_arr), 50.0)
        atr_arr = df_sim['ATRr_14'].values if 'ATRr_14' in df_sim.columns else np.zeros(len(c_arr))
        
        bbl_col = next((c for c in df_sim.columns if c.startswith('BBL') and str(bb_std) in c), None)
        bbu_col = next((c for c in df_sim.columns if c.startswith('BBU') and str(bb_std) in c), None)
        bbl_arr = df_sim[bbl_col].values if bbl_col else np.zeros(len(c_arr))
        bbu_arr = df_sim[bbu_col].values if bbu_col else np.zeros(len(c_arr))
        
        ema200 = df_sim['close'].rolling(200).mean().values
        vol_sma_arr = df_sim['volume'].rolling(20).mean().shift(1).values
        
        rsi_oversold = params.get('rsi_oversold', 25)
        rsi_overbought = 100 - rsi_oversold
        vol_spike_mult = params.get('vol_spike_mult', 3.0)
        tp_rr = params.get('tp_rr', 1.5)
        sl_atr_mult = params.get('sl_atr_mult', 1.0)
        
        for i in range(200, len(df_sim)):
            if in_position:
                high = h_arr[i]
                low = l_arr[i]
                
                if direction == 'LONG':
                    if low <= sl_price:
                        idx_now = df_sim.index[i]
                        df_sim.at[idx_now, 'trade_pnl_r'] = -1.0
                        df_sim.at[idx_now, 'entry_idx'] = entry_idx
                        df_sim.at[idx_now, 'entry_p'] = entry_price
                        df_sim.at[idx_now, 'exit_p'] = sl_price
                        df_sim.at[idx_now, 'sl_p'] = sl_price
                        df_sim.at[idx_now, 'trade_dir'] = "LONG"
                        in_position = False
                    elif high >= tp_price:
                        idx_now = df_sim.index[i]
                        df_sim.at[idx_now, 'trade_pnl_r'] = tp_rr
                        df_sim.at[idx_now, 'entry_idx'] = entry_idx
                        df_sim.at[idx_now, 'entry_p'] = entry_price
                        df_sim.at[idx_now, 'exit_p'] = tp_price
                        df_sim.at[idx_now, 'sl_p'] = sl_price
                        df_sim.at[idx_now, 'trade_dir'] = "LONG"
                        in_position = False
                        
                elif direction == 'SHORT':
                    if high >= sl_price:
                        idx_now = df_sim.index[i]
                        df_sim.at[idx_now, 'trade_pnl_r'] = -1.0
                        df_sim.at[idx_now, 'entry_idx'] = entry_idx
                        df_sim.at[idx_now, 'entry_p'] = entry_price
                        df_sim.at[idx_now, 'exit_p'] = sl_price
                        df_sim.at[idx_now, 'sl_p'] = sl_price
                        df_sim.at[idx_now, 'trade_dir'] = "SHORT"
                        in_position = False
                    elif low <= tp_price:
                        idx_now = df_sim.index[i]
                        df_sim.at[idx_now, 'trade_pnl_r'] = tp_rr
                        df_sim.at[idx_now, 'entry_idx'] = entry_idx
                        df_sim.at[idx_now, 'entry_p'] = entry_price
                        df_sim.at[idx_now, 'exit_p'] = tp_price
                        df_sim.at[idx_now, 'sl_p'] = sl_price
                        df_sim.at[idx_now, 'trade_dir'] = "SHORT"
                        in_position = False
            else:
                # FAST NUMPY LOGIC
                c_price = c_arr[i]
                l_price = l_arr[i]
                h_price = h_arr[i]
                rsi = rsi_arr[i]
                bbl = bbl_arr[i]
                bbu = bbu_arr[i]
                vol_sma = vol_sma_arr[i]
                
                if pd.isna(bbl) or pd.isna(ema200[i]):
                    continue
                    
                is_vol_spike = v_arr[i] > (vol_sma * vol_spike_mult)
                
                # --- LONG SIGNAL ---
                if l_price < bbl and rsi < rsi_oversold and is_vol_spike and c_price > ema200[i]:
                    # RSI Div
                    rsi_slice = rsi_arr[i-10:i+1]
                    price_slice = c_arr[i-10:i+1]
                    p_min_idx = np.argmin(price_slice)
                    if price_slice[-1] <= price_slice[p_min_idx] * 1.005:
                        rsi_at_low = rsi_slice[p_min_idx]
                        if rsi_slice[-1] >= rsi_at_low:
                            # Double Touch
                            touches = np.sum(l_arr[i-10:i+1] < bbl_arr[i-10:i+1])
                            if touches >= 2:
                                sl_price = l_price - (atr_arr[i] * sl_atr_mult)
                                risk = c_price - sl_price
                                if risk > 0:
                                    in_position = True
                                    direction = 'LONG'
                                    entry_price = c_price
                                    tp_price = entry_price + (risk * tp_rr)
                                    entry_idx = i
                                    continue
                                    
                # --- SHORT SIGNAL ---
                if h_price > bbu and rsi > rsi_overbought and is_vol_spike and c_price < ema200[i]:
                    # RSI Div
                    rsi_slice = rsi_arr[i-10:i+1]
                    price_slice = c_arr[i-10:i+1]
                    p_max_idx = np.argmax(price_slice)
                    if price_slice[-1] >= price_slice[p_max_idx] * 0.995:
                        rsi_at_high = rsi_slice[p_max_idx]
                        if rsi_slice[-1] <= rsi_at_high:
                            # Double Touch
                            touches = np.sum(h_arr[i-10:i+1] > bbu_arr[i-10:i+1])
                            if touches >= 2:
                                sl_price = h_price + (atr_arr[i] * sl_atr_mult)
                                risk = sl_price - c_price
                                if risk > 0:
                                    in_position = True
                                    direction = 'SHORT'
                                    entry_price = c_price
                                    tp_price = entry_price - (risk * tp_rr)
                                    entry_idx = i
                    
        return df_sim
