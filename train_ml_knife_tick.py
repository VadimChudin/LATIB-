"""
ML Training Script — Knife Tick (Tick-Level)
=============================================
Correct training pipeline for the HFT knife_tick strategy:
1. Loads symbols + optimized params from active_config.json
2. Evaluates epicenters ONE-BY-ONE (streaming, no MemoryError)
3. Maps epicenter timestamps → 5m candle indices → extracts 29 ML features
4. Trains Triple-AI Ensemble (XGB + LGBM + RF) → knife_catcher_model.joblib
5. Auto-exports to JSON for Rust inference engine

Run:  python train_ml_knife_tick.py
"""

import os
import sys
import json
import glob
import logging
import gc
import numpy as np
import pandas as pd
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.ml_filter import RegimeMLFilter

# ═══════════════════════════════════════════════════════════
# 1. STREAMING EPICENTER LOADER (no MemoryError)
# ═══════════════════════════════════════════════════════════

def iter_epicenter_paths(symbol: str):
    """Yield (ts_ms, direction, csv_path) without loading tick data."""
    base_dir = os.path.join(BASE_DIR, 'data', 'epicenters_ticks', symbol)
    entries = []
    
    for direction in ['LONG', 'SHORT']:
        dir_path = os.path.join(base_dir, direction)
        if not os.path.exists(dir_path):
            continue
        for csv_path in glob.glob(os.path.join(dir_path, '*.csv')):
            stem = os.path.splitext(os.path.basename(csv_path))[0]
            try:
                ts_ms = int(stem)
                entries.append((ts_ms, direction, csv_path))
            except ValueError:
                continue
    
    entries.sort(key=lambda x: x[0])
    return entries


def load_ticks_numpy(path: str):
    """Load tick CSV into numpy arrays for minimal memory."""
    ts_list = []
    price_list = []
    qty_list = []
    ibm_list = []
    
    try:
        with open(path, 'r') as f:
            f.readline()  # skip header
            for line in f:
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        ts_list.append(int(parts[0]))
                        price_list.append(float(parts[1]))
                        qty_list.append(float(parts[2]))
                        ibm_list.append(parts[3].strip().lower() == 'true')
                    except (ValueError, IndexError):
                        continue
    except Exception:
        return None
    
    if not ts_list:
        return None
    
    # Sort by timestamp
    order = np.argsort(ts_list)
    return {
        'ts_ms': np.array(ts_list, dtype=np.int64)[order],
        'price': np.array(price_list, dtype=np.float32)[order],
        'qty': np.array(qty_list, dtype=np.float32)[order],
        'ibm': np.array(ibm_list, dtype=np.bool_)[order],
    }


# ═══════════════════════════════════════════════════════════
# 2. TICK-LEVEL EPICENTER EVALUATOR (numpy-optimized)
# ═══════════════════════════════════════════════════════════

def evaluate_epicenter(ts_ms_arr, price_arr, qty_arr, ibm_arr, direction: str, params: list) -> dict:
    """
    Python port of strategies/knife_tick.rs::evaluate_epicenter
    Uses numpy arrays directly — no dict overhead, no slice copies.
    """
    n = len(ts_ms_arr)
    if n < 3 or len(params) < 5:
        return None
    
    window_ms = int(params[0])
    min_drop_pct = params[1]
    tp_pct = params[3]
    sl_pct = params[4]
    
    micro_window_ms = int(params[7]) if len(params) > 7 else 500
    min_micro_delta_mult = params[8] if len(params) > 8 else 0.0
    min_size_mult = params[9] if len(params) > 9 else 1.0
    max_speed_mult = params[10] if len(params) > 10 else 10.0
    
    # ═══ BASELINE: first 60 seconds ═══
    start_ts = ts_ms_arr[0]
    baseline_end = start_ts + 60_000
    baseline_mask = ts_ms_arr <= baseline_end
    baseline_count = int(np.sum(baseline_mask))
    if baseline_count > 0:
        baseline_volume = float(np.sum(price_arr[baseline_mask] * qty_arr[baseline_mask]))
    else:
        baseline_volume = 0.0
    
    baseline_volume = max(baseline_volume, 50_000.0)
    baseline_count = max(baseline_count, 100)
    baseline_avg_size = baseline_volume / baseline_count
    baseline_tps = baseline_count / 60.0
    
    # ═══ SCAN FOR ENTRY ═══
    win_start = 0  # sliding window left pointer
    entry_idx = -1
    
    for i in range(n):
        # Advance window start
        while win_start < i and ts_ms_arr[i] - ts_ms_arr[win_start] > window_ms:
            win_start += 1
        
        if win_start >= i:
            continue
        
        first_price = price_arr[win_start]
        cur_price = price_arr[i]
        
        if first_price <= 0:
            continue
        
        if direction == "LONG":
            price_move = (first_price - cur_price) / first_price
        else:
            price_move = (cur_price - first_price) / first_price
        
        if price_move >= min_drop_pct:
            # ═══ MICRO-WINDOW: 3 atomic signals ═══
            target_micro_ts = ts_ms_arr[i] - micro_window_ms
            
            # Find micro window bounds
            micro_mask = (ts_ms_arr[win_start:i+1] >= target_micro_ts)
            micro_slice = slice(win_start + int(np.argmax(micro_mask)), i + 1)
            
            micro_prices = price_arr[micro_slice]
            micro_qtys = qty_arr[micro_slice]
            micro_ibms = ibm_arr[micro_slice]
            micro_trades = len(micro_prices)
            
            if micro_trades == 0:
                micro_trades = 1
            
            quote_qtys = micro_prices * micro_qtys
            micro_volume = float(np.sum(quote_qtys))
            
            # Delta: sellers negative, buyers positive
            signs = np.where(micro_ibms, -1.0, 1.0)
            micro_delta = float(np.sum(quote_qtys * signs))
            
            micro_avg_size = micro_volume / micro_trades
            micro_seconds = micro_window_ms / 1000.0
            micro_tps = micro_trades / max(micro_seconds, 0.001)
            
            # Signal 1: DELTA
            delta_threshold = min_micro_delta_mult * baseline_avg_size * baseline_tps * micro_seconds
            delta_ok = (micro_delta >= delta_threshold) if direction == "LONG" else (micro_delta <= -delta_threshold)
            
            # Signal 2: SIZE
            size_ok = micro_avg_size >= baseline_avg_size * min_size_mult
            
            # Signal 3: SPEED
            speed_ok = micro_tps <= baseline_tps * max_speed_mult
            
            if delta_ok and size_ok and speed_ok:
                entry_idx = i
                break
    
    if entry_idx < 0 or entry_idx + 1 >= n:
        return None
    
    # ═══ EXECUTE TRADE (index-based, no slice copy) ═══
    taker_fee = 0.0005
    
    entry_price_raw = float(price_arr[entry_idx + 1])
    entry_price = entry_price_raw * (1.0 + taker_fee) if direction == "LONG" else entry_price_raw * (1.0 - taker_fee)
    
    be_trigger_pct = params[5] if len(params) > 5 else 0.003
    trail_pct = params[6] if len(params) > 6 else 0.002
    
    if direction == "LONG":
        sl_price = entry_price * (1.0 - sl_pct)
        tp_price = entry_price * (1.0 + tp_pct)
        be_trigger_price = entry_price * (1.0 + be_trigger_pct)
    else:
        sl_price = entry_price * (1.0 + sl_pct)
        tp_price = entry_price * (1.0 - tp_pct)
        be_trigger_price = entry_price * (1.0 - be_trigger_pct)
    
    is_breakeven = False
    best_price = entry_price
    risk = abs(entry_price - sl_price)
    if risk < 1e-10:
        risk = entry_price * 0.0005  # fallback min risk
    exit_price = entry_price
    pnl_r = 0.0
    
    # Iterate remaining ticks by index (NO slice copy)
    for j in range(entry_idx + 2, n):
        p_raw = float(price_arr[j])
        p = p_raw * (1.0 - taker_fee) if direction == "LONG" else p_raw * (1.0 + taker_fee)
        
        if direction == "LONG":
            if p > best_price:
                best_price = p
            if not is_breakeven and p >= be_trigger_price:
                sl_price = entry_price
                is_breakeven = True
            trailing_sl = best_price * (1.0 - trail_pct)
            current_sl = max(sl_price, trailing_sl)
            
            if p <= current_sl:
                exit_price = p
                pnl_r = (exit_price - entry_price) / risk
                break
        else:
            if p < best_price:
                best_price = p
            if not is_breakeven and p <= be_trigger_price:
                sl_price = entry_price
                is_breakeven = True
            trailing_sl = best_price * (1.0 + trail_pct)
            current_sl = min(sl_price, trailing_sl)
            
            if p >= current_sl:
                exit_price = p
                pnl_r = (entry_price - exit_price) / risk
                break
    
    # Time runs out → exit at market
    if pnl_r == 0.0 and exit_price == entry_price:
        last_p_raw = float(price_arr[-1])
        exit_price = last_p_raw * (1.0 - taker_fee) if direction == "LONG" else last_p_raw * (1.0 + taker_fee)
        pnl_r = ((exit_price - entry_price) / risk) if direction == "LONG" else ((entry_price - exit_price) / risk)
    
    return {
        'pnl_r': pnl_r,
        'direction': direction,
    }


# ═══════════════════════════════════════════════════════════
# 3. CANDLE-LEVEL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════

def load_candles(symbol: str) -> pd.DataFrame:
    """Load 5m candle CSV for a symbol."""
    path = os.path.join(BASE_DIR, 'data', 'cache', f'{symbol}_5m_730d.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Cast to float32 to save RAM
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df


def map_ts_to_candle_idx(ts_ms: int, df_index) -> int:
    """Map epicenter timestamp (ms) to nearest 5m candle index."""
    ep_time = pd.Timestamp(ts_ms, unit='ms', tz='UTC')
    if df_index.tz is None:
        ep_time = ep_time.tz_localize(None)
    idx = df_index.searchsorted(ep_time, side='right') - 1
    return max(0, min(idx, len(df_index) - 1))


# ═══════════════════════════════════════════════════════════
# 4. MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("  ML TRAINING: Knife Tick (Tick-Level Pipeline)")
    logger.info("  Symbols: from active_config.json")
    logger.info("  Data: tick epicenters + 5m candles")
    logger.info("=" * 60)
    
    # 1. Load active config
    config_path = os.path.join(BASE_DIR, 'data', 'active_config.json')
    with open(config_path) as f:
        configs = json.load(f)
    
    configs = [c for c in configs if c.get('strategy') == 'knife_tick']
    logger.info(f"\n📋 Loaded {len(configs)} knife_tick configs\n")
    
    # Pre-load BTC for Gravity Correlation
    btc_df = load_candles("BTC_USDT")
    if btc_df is not None:
        btc_df['btc_ema_50'] = btc_df['close'].rolling(50).mean()
        btc_df['btc_ema_200'] = btc_df['close'].rolling(200).mean()
        btc_df['btc_atr'] = btc_df['high'].rolling(14).max() - btc_df['low'].rolling(14).min()
        logger.info(f"  ✅ BTC loaded: {len(btc_df)} candles for Gravity Correlation\n")
    else:
        logger.warning("  ⚠️ BTC_USDT not found, BTC features will be zero\n")
    
    ml_filter = RegimeMLFilter(model_name="knife_catcher_model")
    all_features = []
    all_labels = []
    total_wins = 0
    total_losses = 0
    
    for config in configs:
        symbol = config['symbol'].replace('/', '_')
        params_dict = config.get('params', {})
        
        # Build params vector (same order as config_loader.rs knifetick)
        params = [
            params_dict.get('window_ms', 2000),
            params_dict.get('min_drop_pct', 0.1),
            params_dict.get('unused', 0.0),
            params_dict.get('tp_pct', 5.0),
            params_dict.get('sl_pct', 1.0),
            params_dict.get('be_trigger_pct', 0.8),
            params_dict.get('trail_pct', 0.4),
            params_dict.get('micro_window_ms', 50),
            params_dict.get('min_micro_delta_mult', -2.0),
            params_dict.get('min_size_mult', 0.5),
            params_dict.get('max_speed_mult', 3.0),
        ]
        
        logger.info(f"🔄 {symbol} | drop={params[1]:.3f} tp={params[3]:.2f}% sl={params[4]:.2f}%")
        
        # 2a. Get epicenter file list (no tick data loaded yet)
        ep_entries = iter_epicenter_paths(symbol)
        if not ep_entries:
            logger.warning(f"  ⚠️ No epicenters for {symbol}, skipping")
            continue
        
        # 2b. Evaluate epicenters ONE-BY-ONE (streaming)
        trade_results = []  # (ts_ms, pnl_r, direction)
        
        for ts_ms, direction, csv_path in ep_entries:
            ticks = load_ticks_numpy(csv_path)
            if ticks is None:
                continue
            
            result = evaluate_epicenter(
                ticks['ts_ms'], ticks['price'], ticks['qty'], ticks['ibm'],
                direction, params
            )
            
            # Free tick data immediately
            del ticks
            
            if result is not None:
                trade_results.append((ts_ms, result['pnl_r'], result['direction']))
        
        gc.collect()
        
        if len(trade_results) < 5:
            logger.warning(f"  ⚠️ {symbol}: only {len(trade_results)} trades from {len(ep_entries)} epicenters, skipping")
            continue
        
        wins = sum(1 for _, pnl, _ in trade_results if pnl > 0)
        losses = len(trade_results) - wins
        total_wins += wins
        total_losses += losses
        wr = wins / len(trade_results) * 100
        avg_r = np.mean([pnl for _, pnl, _ in trade_results])
        logger.info(f"  📊 {len(trade_results)}/{len(ep_entries)} ep → {wins}W/{losses}L WR={wr:.1f}% avgR={avg_r:+.2f}")
        
        # 2c. Load candle data for feature extraction
        df = load_candles(symbol)
        if df is None:
            logger.warning(f"  ⚠️ No candle data for {symbol}, skipping")
            continue
        
        # Compute TA indicators
        try:
            import pandas_ta as ta
            df.ta.rsi(length=14, append=True)
            df.ta.adx(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.stoch(append=True)
            try: df.ta.cci(length=20, append=True)
            except: pass
            try: df.ta.mfi(length=14, append=True)
            except: pass
        except Exception as e:
            logger.warning(f"  TA calc error: {e}")
        
        # 2d. Map epicenter timestamps to candle indices
        trade_indices = []
        trade_labels = []
        
        for ts_ms, pnl_r, direction in trade_results:
            candle_idx = map_ts_to_candle_idx(ts_ms, df.index)
            if candle_idx >= 200:  # need 200 bars lookback
                trade_indices.append(candle_idx)
                trade_labels.append(1 if pnl_r > 0 else 0)
        
        if len(trade_indices) < 5:
            logger.warning(f"  ⚠️ {symbol}: only {len(trade_indices)} mappable trades, skipping")
            del df
            gc.collect()
            continue
        
        # 2e. Extract features (mirrors Rust ml_inference.rs)
        features_df = ml_filter.prepare_features(df, trade_indices, btc_df=btc_df)
        labels = np.array(trade_labels)
        
        if len(features_df) != len(labels):
            min_len = min(len(features_df), len(labels))
            features_df = features_df.iloc[:min_len]
            labels = labels[:min_len]
        
        if len(features_df) > 0:
            all_features.append(features_df)
            all_labels.append(pd.Series(labels))
            logger.info(f"  ✅ {len(features_df)} feature vectors extracted")
        
        del df, trade_results
        gc.collect()
    
    if not all_features:
        logger.error("\n❌ No valid training data from any symbol!")
        return
    
    # 3. Pool all data
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = pd.concat(all_labels, ignore_index=True)
    
    # Free partial data
    del all_features, all_labels
    gc.collect()
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  POOLED DATASET: {len(combined_features)} trades")
    logger.info(f"  Wins: {total_wins} | Losses: {total_losses} | WR: {combined_labels.mean():.1%}")
    logger.info(f"{'=' * 60}")
    
    # 4. Train Triple-AI Ensemble
    logger.info("\n🧠 Training Triple-AI Ensemble (XGB + LGBM + RF)...")
    ml_filter_fresh = RegimeMLFilter(model_name="knife_catcher_model")
    metrics = ml_filter_fresh.train(combined_features, combined_labels)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ✅ TRAINING COMPLETE")
    logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.1%}")
    logger.info(f"  Precision: {metrics.get('precision', 0):.1%}")
    logger.info(f"  Model: {ml_filter_fresh.model_path}")
    logger.info(f"{'=' * 60}")
    
    # 5. Auto-export to JSON for Rust engine
    export_script = os.path.join(BASE_DIR, 'export_models_json.py')
    if os.path.exists(export_script):
        logger.info("\n📦 Exporting models to JSON for Rust engine...")
        os.system(f'python "{export_script}"')
    else:
        logger.warning(f"\n⚠️ export_models_json.py not found, skip JSON export")
    
    logger.info("\n🎯 Done! Restart 'python main.py' to use the new models.")


if __name__ == '__main__':
    main()
