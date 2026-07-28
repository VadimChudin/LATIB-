"""
ML Training Script — Microstructure Models (Phase 23)
======================================================
For each strategy, loads backtest trades from Rust, downloads ticks
via Binance Vision, extracts HFT features via microstructure_analyzer,
and trains a multi-class XGBoost model (STRONG_WIN / WEAK_WIN / LOSS).

Produces: data/models_json/micro_{strategy}.json

Run: python train_ml_microstructure.py
     python train_ml_microstructure.py --strategy knife
"""

import os
import sys
import json
import asyncio
import subprocess
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microstructure_analyzer import MicrostructureAnalyzer
from lazy_tick_loader import LazyTickLoader
from constants import CACHE_DIR, MODELS_JSON_DIR, BINARY_PATH, ACTIVE_CONFIG_PATH, TOP_SYMBOLS_PATH

MODELS_DIR = MODELS_JSON_DIR

# Strategy configs: name -> (rust_name, tick_window_before_ms, python_name)
STRATEGY_CONFIGS = {
    "knife": {
        "rust_name": "knife",
        "python_name": "KnifeCatcher_ML",
        "tick_window_ms": 60 * 1000,           # 60 seconds before entry
        "analyzer_method": "knife",
    },
    "density": {
        "rust_name": "density",
        "python_name": "Density",
        "tick_window_ms": 60 * 60 * 1000,      # 60 minutes before entry
        "analyzer_method": "breakout",
    },
    "smc": {
        "rust_name": "smc",
        "python_name": "Ultimate_SMC_Trail",
        "tick_window_ms": 30 * 60 * 1000,      # 30 minutes before entry
        "analyzer_method": "smc",
    },
    "fundingrate": {
        "rust_name": "fundingrate",
        "python_name": "FundingRate_MR",
        "tick_window_ms": 15 * 60 * 1000,      # 15 minutes before entry
        "analyzer_method": "funding",
    },
    "scalpmtf": {
        "rust_name": "scalpmtf",
        "python_name": "ScalpMTF",
        "tick_window_ms": 5 * 60 * 1000,       # 5 minutes before entry
        "analyzer_method": "scalp",
    },
}


def get_params_for_strategy(python_name: str) -> dict:
    """Load best params from active_config.json."""
    config_path = ACTIVE_CONFIG_PATH
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            configs = json.load(f)
        for c in configs:
            if c.get("strategy") == python_name:
                return c.get("params", {})
    except Exception:
        pass
    return {}


def get_backtest_trades(symbol: str, rust_strat: str, params: dict, timeframe="5m") -> list:
    """Run Rust backtest-trades to get list of trades with timestamps."""
    sym = symbol.replace("/", "_")
    csv_path = CACHE_DIR / f"{sym}_{timeframe}_730d.csv"
    if not csv_path.exists():
        return []

    binary = str(BINARY_PATH)
    if not os.path.exists(binary):
        binary = binary.replace(".exe", "")
    if not os.path.exists(binary):
        return []

    cmd = [
        str(binary), "backtest-trades",
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
        logger.warning(f"  ⚠️ Rust backtest failed for {symbol}: {e}")
    return []


def classify_trade(pnl_r: float) -> int:
    """Multi-class label: 0=LOSS, 1=WEAK_WIN, 2=STRONG_WIN."""
    if pnl_r <= 0:
        return 0  # LOSS
    elif pnl_r < 1.5:
        return 1  # WEAK_WIN
    else:
        return 2  # STRONG_WIN


async def extract_features_for_strategy(
    strat_key: str,
    symbols: list,
    max_trades_per_symbol: int = 150
) -> tuple:
    """
    For each symbol: run backtest → get trades → download ticks → extract features.
    Returns (features_df, labels_array).
    """
    cfg = STRATEGY_CONFIGS[strat_key]
    analyzer = MicrostructureAnalyzer()
    loader = LazyTickLoader()
    
    params = get_params_for_strategy(cfg["python_name"])
    if not params:
        logger.warning(f"  ⚠️ No params found for {cfg['python_name']}, using empty params")
    
    all_features = []
    all_labels = []
    processed_symbols = 0
    
    for sym_raw in symbols:
        sym = sym_raw.replace("_USDT", "/USDT").replace("_", "/") if "/" not in sym_raw else sym_raw
        sym_clean = sym.replace("/", "_")
        
        trades = get_backtest_trades(sym, cfg["rust_name"], params)
        if len(trades) < 10:
            continue
        
        # Limit trades per symbol to avoid data imbalance
        if len(trades) > max_trades_per_symbol:
            import random
            trades = random.sample(trades, max_trades_per_symbol)
        
        logger.info(f"  📊 {sym}: {len(trades)} trades → extracting tick features...")
        
        features_for_sym = []
        labels_for_sym = []
        errors = 0
        
        for trade in trades:
            try:
                entry_ts = pd.Timestamp(trade["entry_ts"])
                entry_ms = int(entry_ts.timestamp() * 1000)
                entry_price = float(trade["entry_price"])
                direction = trade.get("direction", "LONG")
                pnl_r = float(trade.get("pnl_r", 0))
                
                # Download ticks for window BEFORE entry
                tick_start = entry_ms - cfg["tick_window_ms"]
                tick_end = entry_ms
                
                df_ticks = await loader.load_trade_window(sym_clean, tick_start, tick_end)
                
                if df_ticks.empty or len(df_ticks) < 5:
                    errors += 1
                    continue
                
                # Extract microstructure features
                features = analyzer.analyze(cfg["python_name"], df_ticks, entry_price, direction)
                
                features_for_sym.append(features)
                labels_for_sym.append(classify_trade(pnl_r))
                
            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning(f"    ⚠️ Trade error: {e}")
                continue
        
        if features_for_sym:
            all_features.extend(features_for_sym)
            all_labels.extend(labels_for_sym)
            processed_symbols += 1
            n_strong = sum(1 for l in labels_for_sym if l == 2)
            n_weak = sum(1 for l in labels_for_sym if l == 1)
            n_loss = sum(1 for l in labels_for_sym if l == 0)
            logger.info(f"    ✅ {len(features_for_sym)} features extracted (S:{n_strong} W:{n_weak} L:{n_loss}, errors:{errors})")
    
    await loader.close()
    
    if not all_features:
        return pd.DataFrame(), np.array([])
    
    features_df = pd.DataFrame(all_features)
    labels_arr = np.array(all_labels)
    
    logger.info(f"\n  📦 Total: {len(features_df)} samples from {processed_symbols} symbols")
    logger.info(f"     STRONG_WIN: {(labels_arr == 2).sum()}")
    logger.info(f"     WEAK_WIN:   {(labels_arr == 1).sum()}")
    logger.info(f"     LOSS:       {(labels_arr == 0).sum()}")
    
    return features_df, labels_arr


def train_xgboost_multiclass(features_df: pd.DataFrame, labels: np.ndarray, model_name: str):
    """Train XGBoost multi-class model and save to JSON."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("❌ xgboost not installed. Run: pip install xgboost")
        return None
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Drop NaN/inf
    mask = features_df.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    features_df = features_df[mask].reset_index(drop=True)
    labels = labels[mask.values]
    
    if len(features_df) < 50:
        logger.warning(f"  ⚠️ Only {len(features_df)} samples, too few for training")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    target_names = ["LOSS", "WEAK_WIN", "STRONG_WIN"]
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    logger.info(f"\n{report}")
    
    # Feature importance
    importance = dict(zip(features_df.columns, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("  📊 Feature Importance:")
    for fname, imp in sorted_imp[:10]:
        logger.info(f"    {fname:30s} {imp:.4f}")
    
    # Save model as JSON (readable by Rust)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"micro_{model_name}.json"
    model.save_model(str(model_path))
    logger.info(f"\n  💾 Model saved: {model_path}")
    
    # Also save feature names for Rust inference
    meta_path = MODELS_DIR / f"micro_{model_name}_meta.json"
    meta = {
        "features": list(features_df.columns),
        "classes": target_names,
        "n_samples": len(features_df),
        "trained_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  💾 Meta saved: {meta_path}")
    
    return model


async def train_strategy(strat_key: str, symbols: list):
    """Full pipeline for one strategy."""
    cfg = STRATEGY_CONFIGS[strat_key]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  MICROSTRUCTURE ML: {cfg['python_name']} ({strat_key})")
    logger.info(f"  Tick window: {cfg['tick_window_ms'] / 1000:.0f}s before entry")
    logger.info(f"{'=' * 60}")
    
    features_df, labels = await extract_features_for_strategy(strat_key, symbols)
    
    if features_df.empty:
        logger.warning(f"  ❌ No data extracted for {strat_key}")
        return
    
    train_xgboost_multiclass(features_df, labels, strat_key)


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default=None,
                       help="Train only this strategy: knife, density, smc, fundingrate, scalpmtf")
    args = parser.parse_args()
    
    # Load symbols
    if TOP_SYMBOLS_PATH.exists():
        with open(TOP_SYMBOLS_PATH) as f:
            symbols = json.load(f)[:50]  # Use top 50 for training
    else:
        symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "AVAX_USDT"]
    
    logger.info(f"🔬 MICROSTRUCTURE ML TRAINING (Phase 23)")
    logger.info(f"   Symbols: {len(symbols)}")
    logger.info(f"   Labels: STRONG_WIN (≥1.5R) / WEAK_WIN (0..1.5R) / LOSS (≤0)")
    
    if args.strategy:
        if args.strategy not in STRATEGY_CONFIGS:
            logger.error(f"❌ Unknown strategy: {args.strategy}")
            logger.info(f"   Available: {list(STRATEGY_CONFIGS.keys())}")
            return
        await train_strategy(args.strategy, symbols)
    else:
        for strat_key in STRATEGY_CONFIGS:
            await train_strategy(strat_key, symbols)
    
    logger.info("\n✅ MICROSTRUCTURE ML TRAINING COMPLETE!")


if __name__ == "__main__":
    asyncio.run(main())
