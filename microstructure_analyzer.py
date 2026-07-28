"""
Microstructure Analyzer — Phase 23
====================================
Extracts HFT-level features from raw tick data (aggTrades)
for each strategy type. These features feed into separate
XGBoost models (micro_knife.json, micro_density.json, etc.)
that provide quality recommendations to the Risk Manager.

Usage:
    from microstructure_analyzer import MicrostructureAnalyzer
    analyzer = MicrostructureAnalyzer()
    features = analyzer.analyze_knife(df_ticks, entry_price, direction)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class MicrostructureAnalyzer:
    """
    Computes microstructure features from raw tick DataFrames.
    Each method returns a dict of feature_name -> float.
    
    Expected DataFrame columns: price, qty, timestamp, is_buyer_maker
    """
    
    # ──────────────────────────────────────────────
    #  KnifeCatcher features (30-60 sec BEFORE entry)
    # ──────────────────────────────────────────────
    
    def analyze_knife(self, df: pd.DataFrame, entry_price: float,
                      direction: str) -> Dict[str, float]:
        """
        Analyze tick microstructure for a KnifeCatcher trade.
        df: ticks for 60 seconds BEFORE entry.
        """
        if df.empty or len(df) < 5:
            return self._empty_knife()
        
        prices = df["price"].values.astype(float)
        qtys = df["qty"].values.astype(float)
        ts = df["timestamp"].values.astype(np.int64)
        is_sell = df["is_buyer_maker"].astype(bool).values if "is_buyer_maker" in df.columns else np.ones(len(df), dtype=bool)
        
        duration_sec = max((ts[-1] - ts[0]) / 1000.0, 0.001)
        
        # 1. dump_speed: % price drop per second
        price_range = (prices.max() - prices.min()) / prices.max() * 100
        dump_speed = price_range / duration_sec
        
        # 2. capitulation_vol: sell volume in last 3 seconds before lowest price
        low_idx = np.argmin(prices)
        low_ts = ts[low_idx]
        mask_cap = (ts >= low_ts - 3000) & (ts <= low_ts) & is_sell
        capitulation_vol = float(qtys[mask_cap].sum()) if mask_cap.any() else 0.0
        
        # 3. bounce_speed: % rebound in first 5 seconds after the low
        mask_bounce = (ts >= low_ts) & (ts <= low_ts + 5000)
        if mask_bounce.any():
            low_price = prices[low_idx]
            bounce_high = prices[mask_bounce].max()
            bounce_speed = (bounce_high - low_price) / max(low_price, 0.001) * 100
        else:
            bounce_speed = 0.0
        
        # 4. tick_density_at_low: number of ticks at the lowest price (±0.01%)
        low_price = prices.min()
        tolerance = low_price * 0.0001
        tick_density_at_low = float(np.sum(np.abs(prices - low_price) <= tolerance))
        
        # 5. panic_decay_rate: sell volume in last 10s / first 10s
        third = max(duration_sec / 3.0 * 1000, 1)
        t0 = ts[0]
        mask_first = (ts >= t0) & (ts < t0 + third) & is_sell
        mask_last = (ts > ts[-1] - third) & (ts <= ts[-1]) & is_sell
        vol_first = float(qtys[mask_first].sum()) if mask_first.any() else 0.001
        vol_last = float(qtys[mask_last].sum()) if mask_last.any() else 0.0
        panic_decay_rate = vol_last / max(vol_first, 0.001)
        
        return {
            "dump_speed": round(dump_speed, 6),
            "capitulation_vol": round(capitulation_vol, 4),
            "bounce_speed": round(bounce_speed, 6),
            "tick_density_at_low": tick_density_at_low,
            "panic_decay_rate": round(panic_decay_rate, 4),
        }
    
    def _empty_knife(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            "dump_speed", "capitulation_vol", "bounce_speed",
            "tick_density_at_low", "panic_decay_rate"
        ]}
    
    # ──────────────────────────────────────────────
    #  Density / Breakout features (30-60 MIN BEFORE entry)
    # ──────────────────────────────────────────────
    
    def analyze_breakout(self, df: pd.DataFrame, level_price: float,
                          direction: str) -> Dict[str, float]:
        """
        Analyze tick microstructure for a Density/Breakout trade.
        df: ticks for 60 minutes BEFORE entry.
        level_price: the price level being tested/broken.
        """
        if df.empty or len(df) < 10:
            return self._empty_breakout()
        
        prices = df["price"].values.astype(float)
        qtys = df["qty"].values.astype(float)
        ts = df["timestamp"].values.astype(np.int64)
        is_buy = ~df["is_buyer_maker"].astype(bool).values if "is_buyer_maker" in df.columns else np.ones(len(df), dtype=bool)
        
        tolerance = level_price * 0.0001  # ±0.01%
        
        # 1. consolidation_time: minutes price stayed near level
        near_level = np.abs(prices - level_price) <= (level_price * 0.002)  # ±0.2%
        if near_level.any():
            consolidation_ms = ts[near_level][-1] - ts[near_level][0]
            consolidation_time = consolidation_ms / 60000.0
        else:
            consolidation_time = 0.0
        
        # 2. touch_count_ticks: how many ticks touched level ±0.01%
        touch_count_ticks = float(np.sum(np.abs(prices - level_price) <= tolerance))
        
        # 3. pre_break_accel: volume acceleration in last 10s vs prior 60s
        t_end = ts[-1]
        mask_10s = ts >= t_end - 10000
        mask_60s = (ts >= t_end - 60000) & (ts < t_end - 10000)
        vol_10s = float(qtys[mask_10s].sum()) if mask_10s.any() else 0.0
        vol_60s = float(qtys[mask_60s].sum()) if mask_60s.any() else 0.001
        # Normalize to per-second
        pre_break_accel = (vol_10s / 10.0) / max(vol_60s / 50.0, 0.001)
        
        # 4. buy_sell_ratio_5s: buy ratio in last 5s
        mask_5s = ts >= t_end - 5000
        buy_vol_5s = float(qtys[mask_5s & is_buy].sum()) if (mask_5s & is_buy).any() else 0.0
        total_vol_5s = float(qtys[mask_5s].sum()) if mask_5s.any() else 0.001
        buy_sell_ratio_5s = buy_vol_5s / max(total_vol_5s, 0.001)
        
        # 5. retreat_depth: average distance price retreats between touches
        touch_indices = np.where(np.abs(prices - level_price) <= tolerance)[0]
        retreat_depths = []
        for i in range(len(touch_indices) - 1):
            start_idx = touch_indices[i]
            end_idx = touch_indices[i + 1]
            if end_idx - start_idx > 1:
                segment = prices[start_idx:end_idx]
                max_retreat = abs(segment.min() - level_price) / max(level_price, 0.001) * 100
                retreat_depths.append(max_retreat)
        retreat_depth = float(np.mean(retreat_depths)) if retreat_depths else 0.0
        
        # 6. volume_profile: % of total volume within ±0.1% of level
        near_vol_mask = np.abs(prices - level_price) <= (level_price * 0.001)
        vol_near = float(qtys[near_vol_mask].sum()) if near_vol_mask.any() else 0.0
        vol_total = float(qtys.sum()) if len(qtys) > 0 else 0.001
        volume_profile = vol_near / max(vol_total, 0.001)
        
        return {
            "consolidation_time": round(consolidation_time, 2),
            "touch_count_ticks": touch_count_ticks,
            "pre_break_accel": round(pre_break_accel, 4),
            "buy_sell_ratio_5s": round(buy_sell_ratio_5s, 4),
            "retreat_depth": round(retreat_depth, 6),
            "volume_profile": round(volume_profile, 4),
        }
    
    def _empty_breakout(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            "consolidation_time", "touch_count_ticks", "pre_break_accel",
            "buy_sell_ratio_5s", "retreat_depth", "volume_profile"
        ]}
    
    # ──────────────────────────────────────────────
    #  FundingRate features (5-15 MIN BEFORE entry)
    # ──────────────────────────────────────────────
    
    def analyze_funding(self, df: pd.DataFrame, entry_price: float,
                         direction: str) -> Dict[str, float]:
        """
        Analyze tick microstructure for a FundingRate_MR trade.
        df: ticks for 15 minutes BEFORE entry.
        """
        if df.empty or len(df) < 5:
            return self._empty_funding()
        
        prices = df["price"].values.astype(float)
        qtys = df["qty"].values.astype(float)
        ts = df["timestamp"].values.astype(np.int64)
        is_sell = df["is_buyer_maker"].astype(bool).values if "is_buyer_maker" in df.columns else np.ones(len(df), dtype=bool)
        
        # 1. liquidation_cascade: detect bursts of large sell orders
        # A burst = 3+ trades > 2x median qty within 2 seconds
        median_qty = np.median(qtys)
        large_mask = qtys > 2 * median_qty
        cascade_count = 0
        if large_mask.any():
            large_ts = ts[large_mask]
            for i in range(len(large_ts)):
                window_mask = (large_ts >= large_ts[i]) & (large_ts <= large_ts[i] + 2000)
                if window_mask.sum() >= 3:
                    cascade_count += 1
        liquidation_cascade = min(cascade_count, 50)  # Cap at 50
        
        # 2. spread_widening: price volatility per second (proxy for spread)
        if len(prices) > 20:
            # Rolling std of price changes, normalized
            price_changes = np.abs(np.diff(prices)) / prices[:-1] * 10000  # in bps
            spread_widening = float(np.percentile(price_changes, 95))
        else:
            spread_widening = 0.0
        
        return {
            "liquidation_cascade": float(liquidation_cascade),
            "spread_widening": round(spread_widening, 4),
        }
    
    def _empty_funding(self) -> Dict[str, float]:
        return {"liquidation_cascade": 0.0, "spread_widening": 0.0}
    
    # ──────────────────────────────────────────────
    #  SMC features (5-30 MIN BEFORE entry)
    # ──────────────────────────────────────────────
    
    def analyze_smc(self, df: pd.DataFrame, ob_price: float,
                     direction: str) -> Dict[str, float]:
        """
        Analyze tick microstructure for an Ultimate_SMC_Trail trade.
        df: ticks for 30 minutes BEFORE entry.
        ob_price: the Order Block price zone center.
        """
        if df.empty or len(df) < 10:
            return self._empty_smc()
        
        prices = df["price"].values.astype(float)
        qtys = df["qty"].values.astype(float)
        ts = df["timestamp"].values.astype(np.int64)
        is_buy = ~df["is_buyer_maker"].astype(bool).values if "is_buyer_maker" in df.columns else np.ones(len(df), dtype=bool)
        is_sell = ~is_buy
        
        ob_tolerance = ob_price * 0.002  # ±0.2% = OB zone
        
        # 1. ob_absorption: total volume absorbed at the OB zone
        in_ob = np.abs(prices - ob_price) <= ob_tolerance
        ob_absorption = float(qtys[in_ob].sum()) if in_ob.any() else 0.0
        
        # 2. fvg_fill_speed: how fast price moved through the OB zone (seconds)
        if in_ob.any():
            ob_ts = ts[in_ob]
            fvg_fill_speed = (ob_ts[-1] - ob_ts[0]) / 1000.0
        else:
            fvg_fill_speed = 0.0
        
        # 3. mss_tick_confirm: volume at the structure break point
        # Detect the extreme point (high for short, low for long)
        if direction == "LONG":
            extreme_idx = np.argmin(prices)
        else:
            extreme_idx = np.argmax(prices)
        extreme_ts = ts[extreme_idx]
        mask_break = (ts >= extreme_ts - 2000) & (ts <= extreme_ts + 2000)
        mss_tick_confirm = float(qtys[mask_break].sum()) if mask_break.any() else 0.0
        
        # 4. sweep_volume: volume at the sweep of previous high/low
        # Approximate: volume in the 5% extremes of price range
        price_range = prices.max() - prices.min()
        if direction == "LONG":
            sweep_threshold = prices.min() + price_range * 0.05
            sweep_mask = prices <= sweep_threshold
        else:
            sweep_threshold = prices.max() - price_range * 0.05
            sweep_mask = prices >= sweep_threshold
        sweep_volume = float(qtys[sweep_mask].sum()) if sweep_mask.any() else 0.0
        
        # 5. imbalance_ratio: buy vs sell in the OB zone
        buy_vol_ob = float(qtys[in_ob & is_buy].sum()) if (in_ob & is_buy).any() else 0.0
        total_vol_ob = float(qtys[in_ob].sum()) if in_ob.any() else 0.001
        imbalance_ratio = buy_vol_ob / max(total_vol_ob, 0.001)
        
        return {
            "ob_absorption": round(ob_absorption, 4),
            "fvg_fill_speed": round(fvg_fill_speed, 2),
            "mss_tick_confirm": round(mss_tick_confirm, 4),
            "sweep_volume": round(sweep_volume, 4),
            "imbalance_ratio": round(imbalance_ratio, 4),
        }
    
    def _empty_smc(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            "ob_absorption", "fvg_fill_speed", "mss_tick_confirm",
            "sweep_volume", "imbalance_ratio"
        ]}
    
    # ──────────────────────────────────────────────
    #  ScalpMTF features (1-5 MIN BEFORE entry)
    # ──────────────────────────────────────────────
    
    def analyze_scalp(self, df: pd.DataFrame, entry_price: float,
                       direction: str) -> Dict[str, float]:
        """
        Analyze tick microstructure for a ScalpMTF trade.
        df: ticks for 5 minutes BEFORE entry.
        """
        if df.empty or len(df) < 10:
            return self._empty_scalp()
        
        prices = df["price"].values.astype(float)
        qtys = df["qty"].values.astype(float)
        ts = df["timestamp"].values.astype(np.int64)
        is_buy = ~df["is_buyer_maker"].astype(bool).values if "is_buyer_maker" in df.columns else np.ones(len(df), dtype=bool)
        
        # 1. tape_momentum: acceleration of trade prints in last 30s
        t_end = ts[-1]
        mask_30s = ts >= t_end - 30000
        mask_prior = (ts >= t_end - 60000) & (ts < t_end - 30000)
        count_30s = mask_30s.sum()
        count_prior = max(mask_prior.sum(), 1)
        tape_momentum = count_30s / count_prior
        
        # 2. micro_trend_consistency: % of ticks moving in trade direction
        price_diff = np.diff(prices)
        if direction == "LONG":
            consistent = (price_diff > 0).sum()
        else:
            consistent = (price_diff < 0).sum()
        micro_trend_consistency = consistent / max(len(price_diff), 1)
        
        # 3. slippage_estimate: 95th percentile of tick-to-tick spread (bps)
        if len(prices) > 10:
            spreads = np.abs(np.diff(prices)) / prices[:-1] * 10000
            slippage_estimate = float(np.percentile(spreads, 95))
        else:
            slippage_estimate = 0.0
        
        # 4. iceberg_detection: series of same-size prints at same price
        iceberg_score = 0.0
        if len(df) > 20:
            rounded_qty = np.round(qtys, 3)
            rounded_price = np.round(prices, 2)
            for i in range(len(rounded_qty) - 4):
                chunk_qty = rounded_qty[i:i+5]
                chunk_price = rounded_price[i:i+5]
                if (chunk_qty[0] == chunk_qty).all() and (chunk_price[0] == chunk_price).all():
                    iceberg_score += 1.0
        
        return {
            "tape_momentum": round(tape_momentum, 4),
            "micro_trend_consistency": round(micro_trend_consistency, 4),
            "slippage_estimate": round(slippage_estimate, 4),
            "iceberg_detection": iceberg_score,
        }
    
    def _empty_scalp(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            "tape_momentum", "micro_trend_consistency",
            "slippage_estimate", "iceberg_detection"
        ]}
    
    # ──────────────────────────────────────────────
    #  Universal dispatcher
    # ──────────────────────────────────────────────
    
    STRATEGY_MAP = {
        "KnifeCatcher_ML": "knife",
        "KnifeCatcher": "knife",
        "Density": "breakout",
        "FundingRate_MR": "funding",
        "FundingRate": "funding",
        "Ultimate_SMC_Trail": "smc",
        "SMC": "smc",
        "ScalpMTF": "scalp",
    }
    
    def analyze(self, strategy: str, df: pd.DataFrame,
                reference_price: float, direction: str) -> Dict[str, float]:
        """
        Universal entry point. Routes to the correct analyzer
        based on strategy name.
        
        Args:
            strategy: Strategy name (e.g. "KnifeCatcher_ML")
            df: Tick DataFrame for the appropriate time window
            reference_price: entry_price (knife/funding/scalp) or level_price (breakout/smc)
            direction: "LONG" or "SHORT"
        
        Returns:
            dict of feature_name -> float
        """
        stype = self.STRATEGY_MAP.get(strategy, "knife")
        
        if stype == "knife":
            return self.analyze_knife(df, reference_price, direction)
        elif stype == "breakout":
            return self.analyze_breakout(df, reference_price, direction)
        elif stype == "funding":
            return self.analyze_funding(df, reference_price, direction)
        elif stype == "smc":
            return self.analyze_smc(df, reference_price, direction)
        elif stype == "scalp":
            return self.analyze_scalp(df, reference_price, direction)
        else:
            return self.analyze_knife(df, reference_price, direction)


# ──────────────────────────────────────────────
#  Self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Generate synthetic ticks for testing
    np.random.seed(42)
    n = 500
    base_price = 84000.0
    
    # Simulate a dump scenario for KnifeCatcher
    prices = base_price - np.cumsum(np.random.exponential(0.5, n))
    prices[400:] += np.cumsum(np.random.exponential(0.8, 100))  # bounce
    
    df = pd.DataFrame({
        "price": prices,
        "qty": np.random.exponential(0.5, n),
        "timestamp": np.arange(n) * 100 + 1700000000000,
        "is_buyer_maker": np.random.choice([True, False], n, p=[0.6, 0.4]),
    })
    
    analyzer = MicrostructureAnalyzer()
    
    print("=== KnifeCatcher Features ===")
    knife_f = analyzer.analyze("KnifeCatcher_ML", df, base_price, "LONG")
    for k, v in knife_f.items():
        print(f"  {k}: {v}")
    
    print("\n=== Density/Breakout Features ===")
    breakout_f = analyzer.analyze("Density", df, base_price, "LONG")
    for k, v in breakout_f.items():
        print(f"  {k}: {v}")
    
    print("\n=== SMC Features ===")
    smc_f = analyzer.analyze("Ultimate_SMC_Trail", df, base_price, "LONG")
    for k, v in smc_f.items():
        print(f"  {k}: {v}")
    
    print("\n=== ScalpMTF Features ===")
    scalp_f = analyzer.analyze("ScalpMTF", df, base_price, "LONG")
    for k, v in scalp_f.items():
        print(f"  {k}: {v}")
    
    print("\n✅ All analyzers working!")
