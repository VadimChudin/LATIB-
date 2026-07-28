"""
Walk-Forward Analysis (WFA) — Rust-Powered
=============================================
Validates strategy robustness by simulating real trading across time windows.

Method:
  - Sliding window: train ML on 6 months, test on next 1 month
  - Backtests are run through RUST (same code as GA) → no Python/Rust divergence
  - Produces Filtered WR per window over entire 2-year period

If Filtered WR is stable across windows → strategy is robust.
If it swings 30-70% → overfitting.

Run: python walk_forward.py
     python walk_forward.py --strategy smc
"""
import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import logging
import warnings
import tempfile
from datetime import timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import BINARY_PATH, CACHE_DIR, STRAT_MAP, ACTIVE_CONFIG_PATH
from core.ml_filter import RegimeMLFilter

# ── Config ──
SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "AVAX_USDT"]
TRAIN_MONTHS = 6
TEST_MONTHS = 1
STEP_MONTHS = 1

# Strategy configs — params from active_config or defaults
STRATEGY_CONFIGS = {
    "knife": {
        "python_name": "KnifeCatcher_ML",
        "model_name": "knife_catcher_model",
        "default_params": {
            "score_threshold": 4.0, "price_vol_weight": 1.0, "flow_weight": 1.0,
            "tech_weight": 1.0, "pattern_weight": 1.0, "lookback_bars": 20.0,
            "cum_delta_bars": 5.0, "min_red_candles": 2.0, "tp_rr": 1.5, "sl_atr_mult": 1.5,
        },
    },
    "smc": {
        "python_name": "Ultimate_SMC_Trail",
        "model_name": "ultimate_smc_trail_model",
        "default_params": {
            "swing_length": 5, "fvg_min_atr": 0.3, "ob_min_score": 3,
            "sl_atr_mult": 1.0, "trail_activate_r": 1.0, "trail_atr_mult": 0.5,
        },
    },
    "density": {
        "python_name": "Density",
        "model_name": "density_model",
        "default_params": {
            "vol_spike_mult": 2.5, "min_touches": 2, "shakeout_pct": 0.006,
            "tp_rr": 2.0, "sl_atr_mult": 1.0,
        },
    },
    "fundingrate": {
        "python_name": "FundingRate_MR",
        "model_name": "fundingrate_model",
        "default_params": {
            "fr_long_thresh": 0.03, "fr_short_thresh": 0.05,
            "sl_atr_mult": 1.5, "trail_activate_r": 1.0,
            "trail_atr_mult": 0.5, "cooldown_bars": 6,
        },
    },
}


def load_params_from_config(python_name: str, default_params: dict) -> dict:
    """Load best params from active_config.json."""
    try:
        if ACTIVE_CONFIG_PATH.exists():
            with open(ACTIVE_CONFIG_PATH) as f:
                configs = json.load(f)
            for c in configs:
                if c.get("strategy") == python_name:
                    return c.get("params", default_params)
    except Exception:
        pass
    return default_params


def load_csv(symbol: str, timeframe: str = "5m") -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_{timeframe}_730d.csv"
    if not path.exists():
        return None
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(path, dtype=dtypes, engine='c', low_memory=False, encoding_errors='replace')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def rust_backtest_trades(csv_path: str, rust_strat: str, params: dict) -> list:
    """Run Rust backtest-trades on a CSV file. Returns list of trade dicts."""
    binary = str(BINARY_PATH)
    if not os.path.exists(binary):
        binary = binary.replace(".exe", "")
    if not os.path.exists(binary):
        return []

    cmd = [
        binary, "backtest-trades",
        "--csv", str(csv_path),
        "--strategy", rust_strat,
        "--params-json", json.dumps(params)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        output = res.stdout.strip()
        start = output.find('[')
        end = output.rfind(']')
        if start != -1 and end != -1:
            return json.loads(output[start:end+1])
    except Exception as e:
        logger.warning(f"    Rust backtest error: {e}")
    return []


def extract_trades_from_rust(df: pd.DataFrame, rust_strat: str, params: dict) -> tuple:
    """
    Save df slice to temp CSV → run Rust backtest → parse results.
    Returns (trade_indices, labels, pnl_list) or (None, None, None).
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
        temp_path = f.name
        df.to_csv(f, index=False)

    try:
        trades = rust_backtest_trades(temp_path, rust_strat, params)
    finally:
        os.unlink(temp_path)

    if len(trades) < 5:
        return None, None, None

    # Parse trade results
    trade_indices = []
    labels = []
    pnl_list = []

    for t in trades:
        idx = t.get("entry_idx", 0)
        pnl_r = t.get("pnl_r", 0.0)
        if pnl_r == 0.0:
            continue  # Skip open/unclosed trades
        trade_indices.append(idx)
        labels.append(1 if pnl_r > 0 else 0)
        pnl_list.append(pnl_r)

    if len(trade_indices) < 5:
        return None, None, None

    return trade_indices, np.array(labels), pnl_list


def walk_forward_one_strategy(rust_strat: str, config: dict) -> pd.DataFrame:
    """Run WFA for one strategy across all symbols using Rust backtests."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  WFA: {config['python_name']} (via Rust '{rust_strat}')")
    logger.info(f"  Train: {TRAIN_MONTHS}mo | Test: {TEST_MONTHS}mo | Step: {STEP_MONTHS}mo")
    logger.info(f"{'='*60}")

    params = load_params_from_config(config["python_name"], config["default_params"])
    logger.info(f"  Params: {params}")

    # Load all symbols
    all_dfs = {}
    for symbol in SYMBOLS:
        df = load_csv(symbol)
        if df is not None:
            all_dfs[symbol] = df
            logger.info(f"  ✅ {symbol}: {len(df)} candles")

    if not all_dfs:
        logger.error("  ❌ No data loaded!")
        return pd.DataFrame()

    first_df = list(all_dfs.values())[0]
    data_start = first_df['timestamp'].min()
    data_end = first_df['timestamp'].max()
    logger.info(f"  📅 Data range: {data_start.date()} → {data_end.date()}")

    train_delta = pd.DateOffset(months=TRAIN_MONTHS)
    test_delta = pd.DateOffset(months=TEST_MONTHS)
    step_delta = pd.DateOffset(months=STEP_MONTHS)

    window_start = data_start
    results = []
    window_num = 0

    while window_start + train_delta + test_delta <= data_end:
        window_num += 1
        train_end = window_start + train_delta
        test_end = train_end + test_delta

        train_features_all = []
        train_labels_all = []
        test_features_all = []
        test_labels_all = []
        train_trades_total = 0
        test_trades_total = 0

        eval_ml = RegimeMLFilter(model_name=config["model_name"])

        for symbol, df in all_dfs.items():
            train_df = df[(df['timestamp'] >= window_start) & (df['timestamp'] < train_end)].reset_index(drop=True)
            test_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < test_end)].reset_index(drop=True)

            if len(train_df) < 200 or len(test_df) < 50:
                continue

            # Train period — via Rust
            train_indices, train_labels, _ = extract_trades_from_rust(train_df, rust_strat, params)
            if train_indices is not None:
                train_feats = eval_ml.prepare_features(train_df, train_indices)
                if len(train_feats) > 0:
                    if len(train_feats) != len(train_labels):
                        valid_indices = train_feats['index'].values
                        valid_mask = [idx in valid_indices for idx in train_indices]
                        train_labels = train_labels[valid_mask]
                    train_features_all.append(train_feats)
                    train_labels_all.append(pd.Series(train_labels))
                    train_trades_total += len(train_labels)

            # Test period — via Rust
            test_indices, test_labels, _ = extract_trades_from_rust(test_df, rust_strat, params)
            if test_indices is not None:
                test_feats = eval_ml.prepare_features(test_df, test_indices)
                if len(test_feats) > 0:
                    if len(test_feats) != len(test_labels):
                        valid_indices = test_feats['index'].values
                        valid_mask = [idx in valid_indices for idx in test_indices]
                        test_labels = test_labels[valid_mask]
                    test_features_all.append(test_feats)
                    test_labels_all.append(pd.Series(test_labels))
                    test_trades_total += len(test_labels)

        if not train_features_all or not test_features_all:
            window_start += step_delta
            continue

        X_train = pd.concat(train_features_all, ignore_index=True)
        y_train = pd.concat(train_labels_all, ignore_index=True)
        X_test = pd.concat(test_features_all, ignore_index=True)
        y_test = pd.concat(test_labels_all, ignore_index=True)

        ml = RegimeMLFilter(model_name=f"wfa_temp_{rust_strat}")
        X_train_clean = X_train.drop(columns=['index']) if 'index' in X_train.columns else X_train
        X_test_clean = X_test.drop(columns=['index']) if 'index' in X_test.columns else X_test

        try:
            X_train_scaled = ml.scaler.fit_transform(X_train_clean)
            ml.clf.fit(X_train_scaled, y_train)

            X_test_scaled = ml.scaler.transform(X_test_clean)
            y_pred = ml.clf.predict(X_test_scaled)
            y_prob = ml.clf.predict_proba(X_test_scaled)[:, 1]

            from sklearn.metrics import accuracy_score, precision_score
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)

            threshold = 0.55
            filtered_mask = y_prob >= threshold
            if filtered_mask.sum() > 0:
                filtered_wr = y_test[filtered_mask].mean()
                filtered_trades = filtered_mask.sum()
            else:
                filtered_wr = 0.0
                filtered_trades = 0

            raw_wr = y_test.mean()

            result = {
                'window': window_num,
                'train_start': window_start.strftime('%Y-%m'),
                'train_end': train_end.strftime('%Y-%m'),
                'test_period': train_end.strftime('%Y-%m'),
                'train_trades': train_trades_total,
                'test_trades': test_trades_total,
                'raw_wr': round(raw_wr * 100, 1),
                'ml_accuracy': round(acc * 100, 1),
                'ml_precision': round(prec * 100, 1),
                'filtered_wr': round(filtered_wr * 100, 1),
                'filtered_trades': int(filtered_trades),
            }
            results.append(result)

            status = "✅" if filtered_wr >= 0.55 else "⚠️" if filtered_wr >= 0.50 else "❌"
            logger.info(
                f"  {status} Window #{window_num} [{train_end.strftime('%Y-%m')}] "
                f"RawWR={raw_wr:.1%} | ML_Acc={acc:.1%} | "
                f"FilteredWR={filtered_wr:.1%} ({filtered_trades}/{test_trades_total} trades)"
            )
        except Exception as e:
            logger.error(f"  ❌ Window #{window_num} failed: {e}")

        window_start += step_delta

    if not results:
        logger.error("  ❌ No valid windows!")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    logger.info(f"\n{'─'*60}")
    logger.info(f"  SUMMARY: {config['python_name']}")
    logger.info(f"{'─'*60}")
    logger.info(f"  Windows tested: {len(results)}")
    logger.info(f"  Avg Raw WR:      {df_results['raw_wr'].mean():.1f}%")
    logger.info(f"  Avg ML Accuracy: {df_results['ml_accuracy'].mean():.1f}%")
    logger.info(f"  Avg Filtered WR: {df_results['filtered_wr'].mean():.1f}% ← KEY METRIC")
    logger.info(f"  Filtered WR StdDev: {df_results['filtered_wr'].std():.1f}%")
    logger.info(f"  Min Filtered WR: {df_results['filtered_wr'].min():.1f}%")
    logger.info(f"  Max Filtered WR: {df_results['filtered_wr'].max():.1f}%")

    good_windows = (df_results['filtered_wr'] >= 55).sum()
    logger.info(f"  Profitable windows (≥55%): {good_windows}/{len(results)} ({good_windows/len(results):.0%})")

    return df_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default=None, help="Run only one strategy: knife, smc, density, fundingrate")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  WALK-FORWARD ANALYSIS (WFA) — Rust-Powered")
    logger.info(f"  Train: {TRAIN_MONTHS}mo → Test: {TEST_MONTHS}mo → Step: {STEP_MONTHS}mo")
    logger.info("=" * 60)

    if args.strategy:
        strategies = {args.strategy: STRATEGY_CONFIGS[args.strategy]}
    else:
        strategies = STRATEGY_CONFIGS

    all_results = {}
    for rust_strat, config in strategies.items():
        try:
            df_results = walk_forward_one_strategy(rust_strat, config)
            if len(df_results) > 0:
                all_results[rust_strat] = df_results
        except Exception as e:
            logger.error(f"  ❌ {rust_strat} FAILED: {e}")

    # Save results
    os.makedirs("data", exist_ok=True)
    for strat, df_r in all_results.items():
        path = f"data/wfa_{strat}.csv"
        df_r.to_csv(path, index=False, encoding='utf-8')
        logger.info(f"  📁 Saved: {path}")

    # Final comparison
    logger.info(f"\n{'#'*60}")
    logger.info("  FINAL WFA COMPARISON")
    logger.info(f"{'#'*60}")
    for strat, df_r in all_results.items():
        avg_fwr = df_r['filtered_wr'].mean()
        std_fwr = df_r['filtered_wr'].std()
        status = "🟢 ROBUST" if avg_fwr >= 55 and std_fwr < 10 else "🟡 MARGINAL" if avg_fwr >= 50 else "🔴 OVERFIT"
        logger.info(f"  {status} {strat:15s} ({STRATEGY_CONFIGS[strat]['python_name']:25s})  AvgWR={avg_fwr:.1f}% ±{std_fwr:.1f}%")


if __name__ == '__main__':
    main()
