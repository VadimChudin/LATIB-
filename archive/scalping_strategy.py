import pandas as pd
from typing import Dict, List, Any, Optional
from strategies.base_strategy import BaseStrategy, Signal

class ScalpingStrategy(BaseStrategy):
    """
    1m-5m Scalping bounce from EMA8
    """
    
    def __init__(self):
        super().__init__(name="Scalping", default_timeframe="5m")
        
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            'ema_fast': [8],
            'ema_slow': [21],
            'tp_rr': [1.5],
            'sl_atr_mult': [0.3, 0.5]
        }
        
    def generate_signal(self, df: pd.DataFrame, current_idx: int, params: Dict[str, Any]) -> Optional[Signal]:
        if current_idx < 30:
            return None
            
        ema_fast_len = params.get('ema_fast', 8)
        ema_slow_len = params.get('ema_slow', 21)
        tp_rr = params.get('tp_rr', 1.5)
        sl_atr_mult = params.get('sl_atr_mult', 0.3)
        
        import pandas_ta as ta
        current = df.iloc[current_idx]
        prev = df.iloc[current_idx - 1]
        
        # Calculate indicators 
        ema_fast = ta.ema(df['close'], length=ema_fast_len).iloc[current_idx]
        ema_slow = ta.ema(df['close'], length=ema_slow_len).iloc[current_idx]
        atr = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[current_idx]
        
        price = current['close']
        
        # Pattern: Bounce from EMA8 in trend direction (Above EMA21)
        if price > ema_slow:
            # LONG Bounce
            if prev['low'] <= ema_fast * 1.002 and current['close'] > current['open']:
                if current['close'] > ema_fast:
                    sl = min(prev['low'], price - atr * sl_atr_mult)
                    risk = price - sl
                    if risk > 0:
                        tp = price + risk * tp_rr
                        return Signal(direction='LONG', entry_price=price, sl_price=sl, tp_price=tp, confidence=0.65, features=[])
                        
        elif price < ema_slow:
            # SHORT Bounce
            if prev['high'] >= ema_fast * 0.998 and current['close'] < current['open']:
                if current['close'] < ema_fast:
                    sl = max(prev['high'], price + atr * sl_atr_mult)
                    risk = sl - price
                    if risk > 0:
                        tp = price - risk * tp_rr
                        return Signal(direction='SHORT', entry_price=price, sl_price=sl, tp_price=tp, confidence=0.65, features=[])
                        
        return None
        
    def backtest_logic(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df_sim = df.copy()
        df_sim['trade_pnl_r'] = 0.0
        df_sim['entry_idx'] = 0
        
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        sl_price = 0.0
        tp_price = 0.0
        direction = None
        risk = 0.0
        
        ema_fast_len = params.get('ema_fast', 8)
        ema_slow_len = params.get('ema_slow', 21)
        
        # Memory-efficient EMA
        df_sim['ema_fast'] = df_sim['close'].ewm(span=ema_fast_len, adjust=False).mean()
        df_sim['ema_slow'] = df_sim['close'].ewm(span=ema_slow_len, adjust=False).mean()
        
        # Memory-efficient ATR calculation 
        tr1 = df_sim['high'] - df_sim['low']
        tr2 = (df_sim['high'] - df_sim['close'].shift(1)).abs()
        tr3 = (df_sim['low'] - df_sim['close'].shift(1)).abs()
        df_sim['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_sim['atr'] = df_sim['tr'].rolling(window=14).mean()
        
        for i in range(30, len(df_sim)):
            if not in_position:
                current = df_sim.iloc[i]
                prev = df_sim.iloc[i-1]
                
                price = current['close']
                ema_fast = current['ema_fast']
                ema_slow = current['ema_slow']
                atr = current['atr']
                
                if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(atr): continue
                
                if price > ema_slow:
                    if prev['low'] <= ema_fast * 1.002 and current['close'] > current['open'] and current['close'] > ema_fast:
                        sl = min(prev['low'], price - atr * params.get('sl_atr_mult', 0.3))
                        risk = price - sl
                        if risk > 0:
                            tp = price + risk * params.get('tp_rr', 1.5)
                            in_position = True
                            direction = 'LONG'
                            entry_price = price
                            sl_price = sl
                            tp_price = tp
                            entry_idx = i
                elif price < ema_slow:
                    if prev['high'] >= ema_fast * 0.998 and current['close'] < current['open'] and current['close'] < ema_fast:
                        sl = max(prev['high'], price + atr * params.get('sl_atr_mult', 0.3))
                        risk = sl - price
                        if risk > 0:
                            tp = price - risk * params.get('tp_rr', 1.5)
                            in_position = True
                            direction = 'SHORT'
                            entry_price = price
                            sl_price = sl
                            tp_price = tp
                            entry_idx = i
            else:
                current = df_sim.iloc[i]
                high = current['high']
                low = current['low']
                
                if direction == 'LONG':
                    if low <= sl_price:
                        df_sim.at[df_sim.index[i], 'trade_pnl_r'] = -1.0
                        df_sim.at[df_sim.index[i], 'entry_idx'] = entry_idx
                        in_position = False
                    elif high >= tp_price:
                        df_sim.at[df_sim.index[i], 'trade_pnl_r'] = params.get('tp_rr', 1.5)
                        df_sim.at[df_sim.index[i], 'entry_idx'] = entry_idx
                        in_position = False
                elif direction == 'SHORT':
                    if high >= sl_price:
                        df_sim.at[df_sim.index[i], 'trade_pnl_r'] = -1.0
                        df_sim.at[df_sim.index[i], 'entry_idx'] = entry_idx
                        in_position = False
                    elif low <= tp_price:
                        df_sim.at[df_sim.index[i], 'trade_pnl_r'] = params.get('tp_rr', 1.5)
                        df_sim.at[df_sim.index[i], 'entry_idx'] = entry_idx
                        in_position = False
                        
        return df_sim
