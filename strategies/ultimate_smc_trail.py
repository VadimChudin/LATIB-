"""
Ultimate SMC Trailing Strategy
==============================
Version of Ultimate SMC that uses an ATR trailing stop to capture
large runner trades instead of a fixed Risk:Reward. This strategy
is designed to be filtered by the Regime-Adaptive ML model.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Any, Optional
from strategies.base_strategy import BaseStrategy, Signal

class UltimateSMCTrailStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="Ultimate_SMC_Trail", default_timeframe="5m")

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            'swing_length':     [5],          
            'fvg_min_atr':      [0.3],        
            'ob_min_score':     [2, 3],
            'sl_atr_mult':      [1.0],        
            'trail_activate_r': [1.0], 
            'trail_atr_mult':   [0.5],
        }

    # ════════════════════════════════════════════════════════════════════════
    # PURE PANDAS INDICATOR FUNCTIONS
    # ════════════════════════════════════════════════════════════════════════

    def _calc_atr(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Wilder's Smoothing (RMA) to match Rust engine exactly
        return tr.ewm(alpha=1/length, adjust=False).mean()

    def _calc_ema(self, series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    def _find_swing_highs_lows(self, df: pd.DataFrame, swing_length: int) -> pd.DataFrame:
        n = swing_length
        highs = df['high']
        lows = df['low']
        is_swing_high = pd.Series(False, index=df.index)
        is_swing_low  = pd.Series(False, index=df.index)

    def _find_swing_highs_lows(self, df: pd.DataFrame, swing_length: int) -> pd.DataFrame:
        n = swing_length
        highs = df['high']
        lows = df['low']

        rolling_max = highs.rolling(window=2*n+1, center=True).max()
        rolling_min = lows.rolling(window=2*n+1, center=True).min()

        is_swing_high = (highs == rolling_max)
        is_swing_low = (lows == rolling_min)

        result = pd.DataFrame(index=df.index)
        result['swing_type'] = np.where(is_swing_high, 1, np.where(is_swing_low, -1, 0))
        result['level'] = np.where(is_swing_high, highs, np.where(is_swing_low, lows, np.nan))

        return result

    def _find_fvgs(self, df: pd.DataFrame, atr: pd.Series, fvg_min_atr: float) -> pd.DataFrame:
        high = df['high']
        low = df['low']
        close = df['close']
        open_p = df['open']
        
        high_shift_2 = high.shift(2)
        low_shift_2 = low.shift(2)
        close_shift_1 = close.shift(1)
        open_shift_1 = open_p.shift(1)
        
        bull_gap = low - high_shift_2
        bull_cond = (bull_gap > atr * fvg_min_atr) & (close_shift_1 > open_shift_1)
        
        bear_gap = low_shift_2 - high
        bear_cond = (bear_gap > atr * fvg_min_atr) & (close_shift_1 < open_shift_1)
        
        res_df = pd.DataFrame(index=df.index)
        res_df['bull_fvg_top'] = np.where(bull_cond, low, np.nan)
        res_df['bull_fvg_bot'] = np.where(bull_cond, high_shift_2, np.nan)
        res_df['bear_fvg_top'] = np.where(bear_cond, low_shift_2, np.nan)
        res_df['bear_fvg_bot'] = np.where(bear_cond, high, np.nan)

        return res_df

    def _detect_bos(self, df: pd.DataFrame, swings: pd.DataFrame, swing_length: int) -> pd.Series:
        # Confirmation lag: we only know the swing exists AFTER swing_length bars
        # So we shift the confirmed levels by swing_length
        last_high = swings['level'].where(swings['swing_type'] == 1).ffill().shift(swing_length)
        last_low = swings['level'].where(swings['swing_type'] == -1).ffill().shift(swing_length)
        
        bos = pd.Series(0, index=df.index)
        bos.loc[(df['close'] > last_high) & (~last_high.isna())] = 1
        bos.loc[(df['close'] < last_low) & (~last_low.isna())] = -1

        return bos

    def _score_ob(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, ops: np.ndarray, vols: np.ndarray,
                  i: int, direction: str, fvg_zone_top: float, fvg_zone_bot: float, fvg_vol: float,
                  atr_val: float, swing_types: np.ndarray, swing_levels: np.ndarray, bos_vals: np.ndarray, swing_len: int) -> int:
        score = 0

        if i >= 3:
            impulse_candle = abs(closes[i-1] - ops[i-1])
            if impulse_candle > atr_val * 0.8:
                score += 1

        is_mitigated = False
        for k in range(max(0, i-20), i):
            if fvg_zone_bot <= closes[k] <= fvg_zone_top:
                is_mitigated = True
                break
        if not is_mitigated:
            score += 1

        if direction == 'LONG':
            recent_st = swing_types[max(0, i-30):i]
            recent_sl = swing_levels[max(0, i-30):i]
            # Swings must be confirmed (j + swing_len <= i)
            for s_idx in range(len(recent_st)):
                abs_idx = max(0, i-30) + s_idx
                if abs_idx + swing_len <= i and recent_st[s_idx] == -1:
                    lvl = recent_sl[s_idx]
                    if (lows[max(0,i-10):i] < lvl).any():
                        score += 1
                        break
        else:
            recent_st = swing_types[max(0, i-30):i]
            recent_sl = swing_levels[max(0, i-30):i]
            for s_idx in range(len(recent_st)):
                abs_idx = max(0, i-30) + s_idx
                if abs_idx + swing_len <= i and recent_st[s_idx] == 1:
                    lvl = recent_sl[s_idx]
                    if (highs[max(0,i-10):i] > lvl).any():
                        score += 1
                        break

        if direction == 'LONG' and lows[i] <= fvg_zone_top and closes[i] > ops[i]:
            score += 1
        elif direction == 'SHORT' and highs[i] >= fvg_zone_bot and closes[i] < ops[i]:
            score += 1

        zone_size = fvg_zone_top - fvg_zone_bot
        wicks_in_zone = 0
        for k in range(max(0, i-10), i):
            upper_wick = highs[k] - max(closes[k], ops[k])
            lower_wick = min(closes[k], ops[k]) - lows[k]
            if upper_wick > zone_size or lower_wick > zone_size:
                wicks_in_zone += 1
        if wicks_in_zone <= 1:
            score += 1

        # 7. Volume-weighted FVG (Institutional move)
        if i >= 20:
            avg_vol = np.mean(vols[i-20:i])
            if avg_vol > 0 and fvg_vol > avg_vol * 2.0:
                score += 1

        return score

    # ════════════════════════════════════════════════════════════════════════
    # BACKTEST LOGIC (WITH TRAILING STOP)
    # ════════════════════════════════════════════════════════════════════════

    def backtest_logic(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df_sim = df.copy().reset_index(drop=True)
        df_sim['trade_pnl_r'] = 0.0
        df_sim['entry_idx'] = 0
        df_sim['entry_p'] = 0.0
        df_sim['exit_p'] = 0.0
        df_sim['sl_p'] = 0.0
        df_sim['tp_p'] = 0.0
        df_sim['trade_dir'] = ""

        swing_len    = params.get('swing_length', 5)
        fvg_min_atr  = params.get('fvg_min_atr', 0.3)
        ob_min_score = params.get('ob_min_score', 3)
        sl_atr_mult  = params.get('sl_atr_mult', 1.0)
        
        trail_act_r  = params.get('trail_activate_r', 1.0)
        trail_atr    = params.get('trail_atr_mult', 0.5)

        atr    = self._calc_atr(df_sim, 14)
        ema200 = self._calc_ema(df_sim['close'], 200)
        swings = self._find_swing_highs_lows(df_sim, swing_len)
        fvgs   = self._find_fvgs(df_sim, atr, fvg_min_atr)
        bos    = self._detect_bos(df_sim, swings, swing_len)

        # ADX Chop Filter
        if 'ADX_14' not in df_sim.columns:
            df_sim.ta.adx(length=14, append=True)

        # Kill Zones (Disabled for higher frequency)
        if 'timestamp' in df_sim.columns:
            hour = pd.to_datetime(df_sim['timestamp']).dt.hour
        else:
            hour = pd.Series(14, index=df_sim.index)
        in_kz = pd.Series(True, index=df_sim.index) # Allow all hours

        # Numpy arrays for massive speedup
        c_vals = df_sim['close'].values
        h_vals = df_sim['high'].values
        l_vals = df_sim['low'].values
        o_vals = df_sim['open'].values
        
        adx_vals = df_sim['ADX_14'].values if 'ADX_14' in df_sim.columns else np.full(len(df_sim), 25.0)
        in_kz_vals = in_kz.values
        
        e200_vals  = ema200.values
        atr_vals   = atr.values
        vol_vals   = df_sim['volume'].values
        
        # Swings can sometimes return empty DataFrames, so ensure strict 1D NumPy arrays
        if swings is not None and not swings.empty and 'swing_type' in swings.columns:
            st_vals = swings['swing_type'].values
            sl_vals = swings['level'].values
        else:
            st_vals = np.zeros(len(df_sim))
            sl_vals = np.zeros(len(df_sim))
            
        bos_vals = bos.values
        
        bull_top_vals = fvgs['bull_fvg_top'].values
        bull_bot_vals = fvgs['bull_fvg_bot'].values
        bear_top_vals = fvgs['bear_fvg_top'].values
        bear_bot_vals = fvgs['bear_fvg_bot'].values
        
        # We need FVG volumes too
        bull_fvg_vol = np.where(~np.isnan(bull_top_vals), df_sim['volume'].shift(1).values, 0.0)
        bear_fvg_vol = np.where(~np.isnan(bear_top_vals), df_sim['volume'].shift(1).values, 0.0)

        in_position   = False
        direction     = None
        entry_price   = 0.0
        sl_price      = 0.0
        entry_idx     = 0
        trail_sl      = 0.0
        trail_active  = False
        init_risk     = 0.0
        entry_atr     = 0.0
        start_idx     = max(200, swing_len * 2 + 5)

        for i in range(start_idx, len(df_sim) - 1):
            if not in_position:
                if not in_kz_vals[i]:
                    continue

                adx_val = adx_vals[i]
                if pd.isna(adx_val) or adx_val < 20:
                    continue

                price  = c_vals[i]
                e200   = e200_vals[i]
                atr_v  = atr_vals[i]

                if pd.isna(e200) or pd.isna(atr_v) or atr_v == 0:
                    continue

                if price > e200:
                    for j in range(1, 15):
                        k = i - j
                        if k < start_idx: break
                        top = bull_top_vals[k]
                        bot = bull_bot_vals[k]
                        if pd.isna(top): continue

                        score = self._score_ob(c_vals, h_vals, l_vals, o_vals, vol_vals, i, 'LONG', top, bot, bull_fvg_vol[k], atr_v, st_vals, sl_vals, bos_vals, swing_len)
                        if score >= ob_min_score:
                            sl = l_vals[i] - atr_v * sl_atr_mult
                            risk = price - sl
                            if risk > 0:
                                entry_price  = price
                                sl_price     = sl
                                in_position  = True
                                direction    = 'LONG'
                                init_risk    = risk
                                entry_atr    = atr_v
                                trail_active = False
                                trail_sl     = sl
                                entry_idx    = i
                                break

                elif price < e200:
                    for j in range(1, 15):
                        k = i - j
                        if k < start_idx: break
                        top = bear_top_vals[k]
                        bot = bear_bot_vals[k]
                        if pd.isna(top): continue

                        score = self._score_ob(c_vals, h_vals, l_vals, o_vals, vol_vals, i, 'SHORT', top, bot, bear_fvg_vol[k], atr_v, st_vals, sl_vals, bos_vals, swing_len)
                        if score >= ob_min_score:
                            sl = h_vals[i] + atr_v * sl_atr_mult
                            risk = sl - price
                            if risk > 0:
                                entry_price  = price
                                sl_price     = sl
                                in_position  = True
                                direction    = 'SHORT'
                                init_risk    = risk
                                entry_atr    = atr_v
                                trail_active = False
                                trail_sl     = sl
                                entry_idx    = i
                                break
            else:
                high = h_vals[i]
                low  = l_vals[i]
                current_atr = atr_vals[i] if not pd.isna(atr_vals[i]) else entry_atr

                if direction == 'LONG':
                    if high >= entry_price + trail_act_r * init_risk:
                        trail_active = True
                    if trail_active:
                        new_trail = high - current_atr * trail_atr
                        if new_trail > trail_sl:
                            trail_sl = new_trail
                    
                    active_sl = trail_sl if trail_active else sl_price

                    if low <= active_sl:
                        pnl = (active_sl - entry_price) / init_risk
                        df_sim.at[i, 'trade_pnl_r'] = round(pnl, 3)
                        df_sim.at[i, 'entry_idx'] = entry_idx
                        df_sim.at[i, 'entry_p'] = entry_price
                        df_sim.at[i, 'exit_p'] = active_sl
                        df_sim.at[i, 'sl_p'] = sl_price
                        df_sim.at[i, 'trade_dir'] = "LONG"
                        in_position = False

                elif direction == 'SHORT':
                    if low <= entry_price - trail_act_r * init_risk:
                        trail_active = True
                    if trail_active:
                        new_trail = low + current_atr * trail_atr
                        if new_trail < trail_sl or trail_sl == sl_price:
                            trail_sl = new_trail
                    
                    active_sl = trail_sl if trail_active else sl_price

                    if high >= active_sl:
                        pnl = (entry_price - active_sl) / init_risk
                        df_sim.at[i, 'trade_pnl_r'] = round(pnl, 3)
                        df_sim.at[i, 'entry_idx'] = entry_idx
                        df_sim.at[i, 'entry_p'] = entry_price
                        df_sim.at[i, 'exit_p'] = active_sl
                        df_sim.at[i, 'sl_p'] = sl_price
                        df_sim.at[i, 'trade_dir'] = "SHORT"
                        in_position = False

        # df_sim index was reset, put it back if needed, but in backtests we just need trade_pnl_r matched 
        # to the same length. Return with original index to match input df.
        df_sim.index = df.index
        return df_sim

    # ════════════════════════════════════════════════════════════════════════
    # LIVE SIGNAL GENERATION
    # ════════════════════════════════════════════════════════════════════════

    def generate_signal(self, df: pd.DataFrame, current_idx: int, params: Dict[str, Any]) -> Optional[Signal]:
        if current_idx < 210:
            return None

        swing_len    = params.get('swing_length', 5)
        fvg_min_atr  = params.get('fvg_min_atr', 0.3)
        ob_min_score = params.get('ob_min_score', 3)
        sl_atr_mult  = params.get('sl_atr_mult', 1.0)

        window = df.iloc[max(0, current_idx - 250): current_idx + 1].copy().reset_index(drop=True)

        atr    = self._calc_atr(window, 14)
        ema200 = self._calc_ema(window['close'], 200)
        swings = self._find_swing_highs_lows(window, swing_len)
        fvgs   = self._find_fvgs(window, atr, fvg_min_atr)
        bos    = self._detect_bos(window, swings, swing_len)

        # ADX Chop Filter
        if 'ADX_14' not in window.columns:
            window.ta.adx(length=14, append=True)

        i = len(window) - 1
        price  = window['close'].iloc[i]
        e200   = ema200.iloc[i]
        atr_v  = atr.iloc[i]
        adx_val = window['ADX_14'].iloc[i] if 'ADX_14' in window.columns else 25

        if pd.isna(e200) or pd.isna(atr_v) or atr_v == 0 or pd.isna(adx_val) or adx_val < 20:
            return None

        if 'timestamp' in window.columns:
            hour = pd.to_datetime(window['timestamp'].iloc[i]).hour

        if price > e200:
            for j in range(1, 15):
                k = i - j
                if k < 0: break
                top = fvgs['bull_fvg_top'].iloc[k]
                bot = fvgs['bull_fvg_bot'].iloc[k]
                if pd.isna(top): continue

                fvg_vol = window['volume'].iloc[k-1] if k > 0 else 0.0
                score = self._score_ob(window['close'].values, window['high'].values, window['low'].values, window['open'].values, window['volume'].values, i, 'LONG', top, bot, fvg_vol, atr_v, swings['swing_type'].values, swings['level'].values, bos.values, swing_len)
                if score >= ob_min_score:
                    sl   = window['low'].iloc[i] - atr_v * sl_atr_mult
                    risk = price - sl
                    if risk > 0:
                        tp = price + risk * 5.0 # Dummy high TP
                        return Signal(direction='LONG', entry_price=price,
                                      sl_price=sl, tp_price=tp,
                                      confidence=0.6 + 0.05 * score, features=[f'score:{score}', 'trail'])

        elif price < e200:
            for j in range(1, 15):
                k = i - j
                if k < 0: break
                top = fvgs['bear_fvg_top'].iloc[k]
                bot = fvgs['bear_fvg_bot'].iloc[k]
                if pd.isna(top): continue

                fvg_vol = window['volume'].iloc[k-1] if k > 0 else 0.0
                score = self._score_ob(window['close'].values, window['high'].values, window['low'].values, window['open'].values, window['volume'].values, i, 'SHORT', top, bot, fvg_vol, atr_v, swings['swing_type'].values, swings['level'].values, bos.values, swing_len)
                if score >= ob_min_score:
                    sl   = window['high'].iloc[i] + atr_v * sl_atr_mult
                    risk = sl - price
                    if risk > 0:
                        tp = price - risk * 5.0
                        return Signal(direction='SHORT', entry_price=price,
                                      sl_price=sl, tp_price=tp,
                                      confidence=0.6 + 0.05 * score, features=[f'score:{score}', 'trail'])

        return None
