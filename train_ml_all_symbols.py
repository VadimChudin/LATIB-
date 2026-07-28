#!/usr/bin/env python3
"""
train_ml_all_symbols.py — Train ML Ensemble on ALL 100 instruments
==================================================================
FAST version: Uses vectorized numpy for feature extraction.
Pre-computes ALL features per symbol as arrays, then samples at trade indices.

Usage:
  python train_ml_all_symbols.py
  python train_ml_all_symbols.py --max-trades-per-symbol 500
"""
import os, sys, json, math, argparse, logging, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
MODELS_DIR = os.path.join(DATA_DIR, 'models')


# ─── Vectorized indicator computation ────────────────────────────────────────

def vec_rsi(closes, period=14):
    """Vectorized RSI for entire array."""
    n = len(closes)
    rsi = np.full(n, 50.0)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    if n < period + 2:
        return rsi
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def vec_ema(values, period):
    """Vectorized EMA."""
    n = len(values)
    ema = np.zeros(n)
    mult = 2.0 / (period + 1.0)
    ema[0] = values[0]
    for i in range(1, n):
        ema[i] = (values[i] - ema[i - 1]) * mult + ema[i - 1]
    return ema


def vec_atr(highs, lows, closes, period=14):
    """Vectorized ATR."""
    n = len(closes)
    atr = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    if n > period:
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def vec_stoch(highs, lows, closes, period=14):
    """Vectorized Stochastic."""
    n = len(closes)
    stoch = np.full(n, 50.0)
    for i in range(period, n):
        h14 = np.max(highs[i - period:i + 1])
        l14 = np.min(lows[i - period:i + 1])
        if h14 - l14 > 0:
            stoch[i] = (closes[i] - l14) / (h14 - l14) * 100.0
    return stoch


def vec_hurst_fast(closes, window=100):
    """Fast approximate Hurst via rolling std ratio (much faster than R/S)."""
    n = len(closes)
    hurst = np.full(n, 0.5)
    if n < window:
        return hurst
    log_returns = np.diff(np.log(np.clip(closes, 1e-10, None)))
    # Use ratio of std at different lags as crude Hurst proxy
    for i in range(window, n - 1):
        chunk = log_returns[i - window + 1:i + 1]
        if len(chunk) < 20:
            continue
        std1 = np.std(chunk)
        # Std of differences at lag 4
        lagged = chunk[4:] - chunk[:-4]
        std4 = np.std(lagged) if len(lagged) > 4 else std1
        if std1 > 0 and std4 > 0:
            # Hurst ≈ log(std_ratio) / log(lag_ratio)
            h = np.log(std4 / std1) / np.log(4.0) if std4 / std1 > 0 else 0.5
            hurst[i + 1] = np.clip(h, 0.0, 1.0)
    return hurst


def precompute_features(closes, highs, lows, volumes, opens, n):
    """
    Pre-compute ALL 26 features as arrays for the entire symbol.
    Returns dict of feature_name -> np.array[n].
    """
    rsi = vec_rsi(closes, 14)
    atr = vec_atr(highs, lows, closes, 14)
    ema12 = vec_ema(closes, 12)
    ema26 = vec_ema(closes, 26)
    ema10 = pd.Series(closes).rolling(10, min_periods=1).mean().values
    ema20 = pd.Series(closes).rolling(20, min_periods=1).mean().values
    ema50 = pd.Series(closes).rolling(50, min_periods=1).mean().values
    ema200 = pd.Series(closes).rolling(200, min_periods=1).mean().values
    vol_sma = pd.Series(volumes).rolling(20, min_periods=1).mean().values
    stoch = vec_stoch(highs, lows, closes, 14)
    hurst = vec_hurst_fast(closes, 100)
    macd = ema12 - ema26
    
    # BB
    bb_mean = pd.Series(closes).rolling(20, min_periods=1).mean().values
    bb_std = pd.Series(closes).rolling(20, min_periods=1).std().values
    bb_std = np.nan_to_num(bb_std, nan=0.0)
    
    # Rolling range
    high_10 = pd.Series(highs).rolling(10, min_periods=1).max().values
    low_10 = pd.Series(lows).rolling(10, min_periods=1).min().values
    
    feats = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        feats['hurst'] = hurst
        feats['atr_pct'] = np.where(closes > 0, atr / closes * 100.0, 0.0)
        feats['vol_ratio'] = np.where(vol_sma > 0, volumes / vol_sma, 1.0)
        feats['hour'] = np.full(n, 12.0)
        feats['rsi'] = rsi
        
        # ADX (simplified: use trend strength from DI diff)
        # Full ADX is expensive per bar; approximate with |EMA20 slope| / ATR
        ema20_shift = np.roll(ema20, 1); ema20_shift[0] = ema20[0]
        feats['adx'] = np.clip(np.where(atr > 0, np.abs(ema20 - ema20_shift) / atr * 1000.0, 25.0), 0, 100)
        
        feats['macd_hist'] = macd
        feats['stoch'] = stoch
        
        # CCI (simplified)
        tp = (highs + lows + closes) / 3.0
        tp_sma = pd.Series(tp).rolling(20, min_periods=1).mean().values
        tp_md = pd.Series(tp).rolling(20, min_periods=1).std().values * 0.7979  # std→mad conversion
        tp_md = np.nan_to_num(tp_md, nan=1.0)
        feats['cci'] = np.where(tp_md > 0, (tp - tp_sma) / (0.015 * tp_md), 0.0)
        
        # MFI (simplified: use volume-weighted RSI proxy)
        mf_pos = np.where(np.diff(tp, prepend=tp[0]) > 0, tp * volumes, 0.0)
        mf_neg = np.where(np.diff(tp, prepend=tp[0]) <= 0, tp * volumes, 0.0)
        mf_pos_sum = pd.Series(mf_pos).rolling(14, min_periods=1).sum().values
        mf_neg_sum = pd.Series(mf_neg).rolling(14, min_periods=1).sum().values
        feats['mfi'] = np.where(mf_neg_sum > 0, 100.0 - 100.0 / (1.0 + mf_pos_sum / mf_neg_sum), 50.0)
        
        # Rolling lag features
        rsi_shift5 = np.roll(rsi, 5); rsi_shift5[:5] = rsi[:5]
        feats['rsi_change_5'] = rsi - rsi_shift5
        
        vol_shift5 = np.roll(volumes, 5); vol_shift5[:5] = volumes[:5]
        feats['vol_change_5'] = np.where(vol_shift5 > 0, volumes / vol_shift5 - 1.0, 0.0)
        
        atr_shift5 = np.roll(atr, 5); atr_shift5[:5] = atr[:5]
        feats['atr_change_5'] = np.where(atr_shift5 > 0, atr / atr_shift5 - 1.0, 0.0)
        
        macd_shift5 = np.roll(macd, 5); macd_shift5[:5] = macd[:5]
        feats['macd_slope'] = macd - macd_shift5
        
        rng = high_10 - low_10
        feats['close_pos'] = np.where(rng > 0, (closes - low_10) / rng, 0.5)
        feats['ema_slope'] = np.where(ema20 > 0, (ema10 - ema20) / ema20 * 100.0, 0.0)
        feats['dist_to_ema'] = np.where(ema50 > 0, (closes - ema50) / ema50 * 100.0, 0.0)
        feats['htf_trend'] = np.where(ema200 > 0, (ema50 - ema200) / ema200 * 100.0, 0.0)
        feats['stoch_dist_50'] = np.abs(stoch - 50.0)
        feats['bb_pos'] = np.where(bb_std > 0, (closes - bb_mean) / (2.0 * bb_std) + 0.5, 0.5)
        feats['bb_width'] = np.where(closes > 0, bb_std * 4.0 / closes * 100.0, 0.0)
        
        opens_shift11 = np.roll(opens, 11); opens_shift11[:11] = opens[:11]
        feats['h1_direction'] = np.where(closes > opens_shift11, 1.0, -1.0)
        feats['funding_rate'] = np.zeros(n)
    
    return feats

# Feature order matching Rust ml_inference.rs
FEATURE_ORDER = [
    'hurst', 'atr_pct', 'vol_ratio', 'hour', 'rsi', 'adx', 'macd_hist',
    'stoch', 'cci', 'mfi', 'rsi_change_5', 'vol_change_5', 'atr_change_5',
    'macd_slope', 'close_pos', 'ema_slope', 'dist_to_ema', 'htf_trend',
    'stoch_dist_50', 'bb_pos', 'bb_width', 'h1_direction', 'funding_rate',
    'btc_trend', 'btc_vol', 'btc_dump'
]


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ML on all symbols")
    parser.add_argument('--max-trades-per-symbol', type=int, default=500, help='Max trades to sample per symbol')
    parser.add_argument('--min-trades', type=int, default=30, help='Min trades per symbol to include')
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  🧠 MULTI-SYMBOL ML TRAINING (Vectorized)")
    logger.info("=" * 60)

    # Load params
    params_path = os.path.join(DATA_DIR, 'ga_best_tick_params.json')
    with open(params_path) as f:
        best = json.load(f)
    params = best['params']
    min_drop_pct = params.get('min_drop_pct', 0.001)
    tp_pct = params.get('tp_pct', 0.03)
    sl_pct = params.get('sl_pct', 0.003)
    logger.info(f"  📋 Params: min_drop={min_drop_pct:.4f} tp={tp_pct:.4f} sl={sl_pct:.4f}")

    # Load symbols
    symbols_path = os.path.join(DATA_DIR, 'top_symbols.json')
    with open(symbols_path) as f:
        symbols = json.load(f)
    logger.info(f"  📊 Symbols: {len(symbols)}")

    # Load BTC for gravity
    btc_path = os.path.join(CACHE_DIR, 'BTC_USDT_5m_730d.csv')
    btc_df = pd.read_csv(btc_path, engine='c', low_memory=False)
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df.set_index('timestamp', inplace=True)
    btc_c = btc_df['close'].values
    btc_ema50 = pd.Series(btc_c).rolling(50, min_periods=1).mean().values
    btc_ema200 = pd.Series(btc_c).rolling(200, min_periods=1).mean().values
    btc_atr = vec_atr(btc_df['high'].values, btc_df['low'].values, btc_c, 14)
    btc_opens = btc_df['open'].values
    
    btc_trend = np.where(btc_ema200 > 0, (btc_ema50 - btc_ema200) / btc_ema200 * 100.0, 0.0)
    btc_vol = np.where(btc_c > 0, btc_atr / btc_c * 100.0, 0.0)
    btc_opens_shift2 = np.roll(btc_opens, 2); btc_opens_shift2[:2] = btc_opens[:2]
    btc_dump = np.where(btc_opens_shift2 > 0, (btc_c - btc_opens_shift2) / btc_opens_shift2 * 100.0, 0.0)
    logger.info(f"  ₿ BTC candles: {len(btc_df)}")

    # Process all symbols
    all_X = []
    all_y = []
    sym_stats = []

    for sym_i, sym in enumerate(symbols):
        csv_name = f"{sym}_5m_730d.csv"
        csv_path = os.path.join(CACHE_DIR, csv_name)
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, engine='c', low_memory=False)
        except Exception:
            continue

        n = len(df)
        if n < 250:
            continue

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        closes = df['close'].values.astype(float)
        opens = df['open'].values.astype(float)
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        volumes = df['volume'].values.astype(float)

        # ── Vectorized strategy ────────────────────────────────────────
        rsi = vec_rsi(closes, 14)
        
        is_red = closes < opens
        is_green = closes > opens
        drop_pct = np.where(opens > 0, (opens - closes) / opens, 0.0)
        rise_pct = np.where(opens > 0, (closes - opens) / opens, 0.0)
        
        # LONG epicenters: RSI > 50 + red candle + drop >= threshold
        long_mask = (rsi > 50.0) & is_red & (drop_pct >= min_drop_pct)
        # SHORT epicenters: RSI < 50 + green candle + rise >= threshold
        short_mask = (rsi < 50.0) & is_green & (rise_pct >= min_drop_pct)
        
        # Skip first 200 candles (need lookback) and last 50 (need exit)
        long_mask[:200] = False; long_mask[-50:] = False
        short_mask[:200] = False; short_mask[-50:] = False
        
        long_indices = np.where(long_mask)[0]
        short_indices = np.where(short_mask)[0]
        
        # Sample if too many
        if len(long_indices) > args.max_trades_per_symbol // 2:
            long_indices = np.random.choice(long_indices, args.max_trades_per_symbol // 2, replace=False)
        if len(short_indices) > args.max_trades_per_symbol // 2:
            short_indices = np.random.choice(short_indices, args.max_trades_per_symbol // 2, replace=False)
        
        total_signals = len(long_indices) + len(short_indices)
        if total_signals < 5:
            continue

        # ── Pre-compute features for entire symbol ──────────────────────
        feats = precompute_features(closes, highs, lows, volumes, opens, n)
        
        # Align BTC gravity via vectorized searchsorted
        btc_t = np.zeros(n)
        btc_v = np.zeros(n)
        btc_d = np.zeros(n)
        try:
            sym_ts = pd.to_datetime(df['timestamp'].values)
            sym_ts_naive = sym_ts.tz_localize(None) if hasattr(sym_ts, 'tz') and sym_ts.tz else sym_ts
            bi_arr = np.clip(btc_df.index.searchsorted(sym_ts_naive), 0, len(btc_trend) - 1)
            btc_t = btc_trend[bi_arr]
            btc_v = btc_vol[bi_arr]
            btc_d = btc_dump[bi_arr]
        except Exception:
            pass

        feats['btc_trend'] = btc_t
        feats['btc_vol'] = btc_v
        feats['btc_dump'] = btc_d

        # ── Build feature matrix at trade indices ───────────────────────
        sym_X = []
        sym_y = []
        
        for idx in long_indices:
            entry_price = closes[idx]
            sl = entry_price * (1.0 - sl_pct)
            tp = entry_price * (1.0 + tp_pct)
            # Simulate exit
            is_win = False
            for j in range(idx + 1, min(idx + 50, n)):
                if lows[j] <= sl:
                    break
                if highs[j] >= tp:
                    is_win = True
                    break
            
            row = [feats[f][idx] for f in FEATURE_ORDER]
            sym_X.append(row)
            sym_y.append(1 if is_win else 0)
        
        for idx in short_indices:
            entry_price = closes[idx]
            sl = entry_price * (1.0 + sl_pct)
            tp = entry_price * (1.0 - tp_pct)
            is_win = False
            for j in range(idx + 1, min(idx + 50, n)):
                if highs[j] >= sl:
                    break
                if lows[j] <= tp:
                    is_win = True
                    break
            
            row = [feats[f][idx] for f in FEATURE_ORDER]
            sym_X.append(row)
            sym_y.append(1 if is_win else 0)

        if len(sym_X) >= args.min_trades:
            all_X.extend(sym_X)
            all_y.extend(sym_y)
            wins = sum(sym_y)
            wr = wins / len(sym_y) * 100
            sym_stats.append((sym, len(sym_y), wr))
            logger.info(f"  ✅ {sym:20s}: {len(sym_y):5d} trades, WR={wr:.1f}%")

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  📊 Feature extraction done in {elapsed:.1f}s")
    logger.info(f"  📊 Combined: {len(all_X)} trades from {len(sym_stats)} symbols")
    
    if not all_y:
        logger.error("  ❌ No trades generated!")
        return
    
    total_wins = sum(all_y)
    total_wr = total_wins / len(all_y) * 100
    logger.info(f"  📊 Overall WR: {total_wr:.1f}% ({total_wins}W / {len(all_y) - total_wins}L)")

    if len(all_X) < 100:
        logger.error("  ❌ Too few trades for training!")
        return

    # ─── Train ML Ensemble ─────────────────────────────────────────────────────
    logger.info(f"\n  🧠 Training ML Ensemble on {len(all_X)} samples...")

    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.preprocessing import StandardScaler
    import joblib

    X = np.array(all_X, dtype=np.float64)
    y = np.array(all_y)
    
    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    clf_rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        scale_pos_weight=1.2, random_state=42, tree_method='hist',
        device='cuda'
    )
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        class_weight='balanced', random_state=42,
        device_type='gpu', verbose=-1
    )

    clf = VotingClassifier(
        estimators=[('rf', clf_rf), ('xgb', clf_xgb), ('lgb', clf_lgb)],
        voting='soft', n_jobs=None
    )

    logger.info("  ⏳ Training... (RF + XGB + LGBM)")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    logger.info(f"  ✅ Accuracy: {acc:.1%}")
    logger.info(f"  ✅ Precision: {prec:.1%}")

    # Top features
    rf_model = clf.estimators_[0]
    importances = sorted(zip(FEATURE_ORDER, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
    logger.info(f"  🏆 Top 5: {[(n, f'{v:.3f}') for n, v in importances[:5]]}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'knife_catcher_model.joblib')
    joblib.dump(clf, model_path)
    logger.info(f"  💾 Model saved: {model_path}")

    # Export to Rust JSON
    logger.info("\n  📦 Exporting to Rust JSON...")
    os.system(f'python "{os.path.join(BASE_DIR, "export_models_json.py")}"')

    total_time = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ✅ TRAINING COMPLETE in {total_time:.1f}s")
    logger.info(f"     {len(all_X)} trades | {len(sym_stats)} symbols | Acc={acc:.1%} | Prec={prec:.1%}")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
