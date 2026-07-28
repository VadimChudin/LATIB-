"""
Machine Learning Regime-Adaptive Filter
=======================================
Combines Hurst Exponent (for market regime detection) with Random Forest
classification to filter out low-probability trades.

Market Regimes (via Hurst Exponent):
  - H > 0.55: Persistent (Trending) — good for trend-following / breakout
  - H < 0.45: Anti-persistent (Mean-reverting) — good for oscillators / reversion
  - 0.45 <= H <= 0.55: Random Walk (Choppy) — usually best to stay out
"""
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import joblib

try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import lightgbm as lgb
except ImportError:
    logging.error("scikit-learn, xgboost, or lightgbm is required for ML Filter. Run: pip install scikit-learn numpy pandas xgboost lightgbm")

import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

logger = logging.getLogger(__name__)


def calculate_hurst(ts: np.ndarray, max_lag: int = 20) -> float:
    """
    Computes the Hurst Exponent of a time series.
    H < 0.5: Mean-reverting
    H = 0.5: Random Walk
    H > 0.5: Trending
    """
    if len(ts) < max_lag:
        return 0.5
    lags = np.arange(2, max_lag)
    tau = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        diff = ts[lag:] - ts[:-lag]
        tau[i] = np.std(diff) + 1e-8
        
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


class RegimeMLFilter:
    def __init__(self, model_name: str = "default_rf_model"):
        self.model_name = model_name
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'models', f"{self.model_name}.joblib"
        )
        self.clf = None
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.feature_importances = None
        self.min_samples = 50
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.load_model()

    def load_model(self) -> bool:
        """Loads trained model from disk if it exists."""
        if os.path.exists(self.model_path):
            try:
                self.clf = joblib.load(self.model_path)
                self.is_fitted = True
                logger.info(f"Loaded ML model from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
        
        
        # We will build an Ensemble "Jury" using 3 different architectures:
        # 1. Random Forest (Robust against noise, avoids overfitting)
        # 2. XGBoost (Aggressive gradient boosting for non-linear patterns)
        # 3. LightGBM (Leaf-wise gradient boosting, excellent for tabular data)
        self.clf1 = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.clf2 = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=1.2, # Slight bias towards finding winning trades
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        self.clf3 = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            device_type='gpu',
            verbose=-1
        )
        
        # Hard voting: A trade is approved (1) only if the majority of models say 1.
        # Soft voting: Averages the probabilities of all 3 models. We use 'soft' 
        # so we can still apply our custom probability threshold (e.g., > 60%).
        self.clf = VotingClassifier(
            estimators=[('rf', self.clf1), ('xgb', self.clf2), ('lgb', self.clf3)],
            voting='soft',
            n_jobs=None  # Keep None on Windows to avoid GPU multiprocess serialization crash
        )
        self.is_fitted = False
        return False

    def save_model(self):
        """Saves current model to disk."""
        if self.is_fitted and self.clf is not None:
            joblib.dump(self.clf, self.model_path)
            logger.info(f"Saved ML model to {self.model_path}")
            
    def prepare_features(self, df: pd.DataFrame, trade_indices: List[int], btc_df: pd.DataFrame = None) -> pd.DataFrame:
        """Calculates features for specified indices where a trade was identified."""
        import pandas_ta as ta
        features_list = []
        
        # Pre-calculate rolling metrics
        df_feats = df.copy()
        df_feats['atr'] = df_feats['high'].rolling(14).max() - df_feats['low'].rolling(14).min()
        df_feats['vol_sma'] = df_feats['volume'].rolling(20).mean()
        
        # Precompute channel sizes & SMAs (Vectorized, O(1) extraction)
        df_feats['rolling_range_10'] = df_feats['high'].rolling(10).max() - df_feats['low'].rolling(10).min()
        df_feats['rolling_min_10'] = df_feats['low'].rolling(10).min()
        df_feats['sma_10'] = df_feats['close'].rolling(10).mean()
        df_feats['sma_20'] = df_feats['close'].rolling(20).mean()
        df_feats['sma_50'] = df_feats['close'].rolling(50).mean()
        df_feats['sma_200'] = df_feats['close'].rolling(200).mean()
        
        # Pre-calculate BTC metrics if provided
        if btc_df is not None and not btc_df.empty:
            btc_df = btc_df.copy()
            btc_df['btc_ema_50'] = btc_df['close'].rolling(50).mean()
            btc_df['btc_ema_200'] = btc_df['close'].rolling(200).mean()
            btc_df['btc_atr'] = btc_df['high'].rolling(14).max() - btc_df['low'].rolling(14).min()
        
        # New Deep Features
        # Using a broad try/except to prevent obscure pandas_ta math errors from crashing WFA
        try:
            # Drop trailing NaNs so pandas_ta doesn't divide by zero or break CCI mad() ranges
            clean_df = df_feats.dropna(subset=['high', 'low', 'close']).copy()
            if len(clean_df) >= 30:
                clean_df.ta.rsi(length=14, append=True)
                clean_df.ta.adx(length=14, append=True)
                clean_df.ta.macd(fast=12, slow=26, signal=9, append=True)
                clean_df.ta.bbands(length=20, std=2, append=True)
                clean_df.ta.stoch(append=True)
                # CCI can crash natively inside numpy if std variance is 0
                try: clean_df.ta.cci(length=20, append=True)
                except Exception: pass
                try: clean_df.ta.mfi(length=14, append=True)
                except Exception: pass
                
                # Merge back to df_feats to preserve index lengths
                for col in clean_df.columns:
                    if col not in df_feats.columns:
                        df_feats[col] = clean_df[col]
                        
        except Exception as e:
            logger.warning(f"Feature calculation error (likely short dataframe or TA failure): {e}")
        
        closes = df_feats['close'].values
        
        # Pre-filter trade indices
        valid_indices = [idx for idx in trade_indices if idx >= 100]
        if not valid_indices:
            return pd.DataFrame()
            
        v_idx = np.array(valid_indices)
        
        # Fast extraction using numpy arrays directly
        # -- Basic price/volume --
        c = df_feats['close'].values
        h = df_feats['high'].values
        l = df_feats['low'].values
        v = df_feats['volume'].values
        o = df_feats['open'].values
        
        # -- TA features --
        atr = df_feats['atr'].values if 'atr' in df_feats else np.zeros_like(c)
        vol_sma = df_feats['vol_sma'].values if 'vol_sma' in df_feats else np.ones_like(c)
        rsi = df_feats['RSI_14'].values if 'RSI_14' in df_feats else np.full_like(c, 50.0)
        adx = df_feats['ADX_14'].values if 'ADX_14' in df_feats else np.full_like(c, 20.0)
        macd_h = df_feats['MACDh_12_26_9'].values if 'MACDh_12_26_9' in df_feats else np.zeros_like(c)
        stoch_k = df_feats['STOCHk_14_3_3'].values if 'STOCHk_14_3_3' in df_feats else np.full_like(c, 50.0)
        cci = df_feats['CCI_20_0.015'].values if 'CCI_20_0.015' in df_feats else np.zeros_like(c)
        mfi = df_feats['MFI_14'].values if 'MFI_14' in df_feats else np.full_like(c, 50.0)
        
        # -- Extracted Vector MAs --
        rr_arr = df_feats['rolling_range_10'].values
        rmin_arr = df_feats['rolling_min_10'].values
        sma10_arr = df_feats['sma_10'].values
        sma20_arr = df_feats['sma_20'].values
        sma50_arr = df_feats['sma_50'].values
        sma200_arr = df_feats['sma_200'].values
        
        # Use simple dictionary of arrays to build the dataframe instantly
        with np.errstate(divide='ignore', invalid='ignore'):
            features_dict = {
                'index': v_idx,
                # Feature 1: Hurst - still needs loop due to polyfit, but now 1000x faster via raw numpy slicing
                'hurst': np.array([calculate_hurst(closes[idx-100:idx]) if idx>=100 else 0.5 for idx in v_idx]),
                # Feature 2: Volatility Regime
                'atr_pct': np.where(c[v_idx] > 0, (atr[v_idx] / c[v_idx]) * 100, 0),
                # Feature 3: Volume Anomaly
                'vol_ratio': np.where(vol_sma[v_idx] > 0, v[v_idx] / vol_sma[v_idx], 1.0),
                # Feature 4: Session Hour
                'hour': pd.to_datetime(df_feats.index[v_idx]).hour if hasattr(df_feats.index, 'hour') else np.zeros(len(v_idx)),
                # New TA Features
                'rsi': np.nan_to_num(rsi[v_idx], nan=50.0),
                'adx': np.nan_to_num(adx[v_idx], nan=20.0),
                'macd_hist': np.nan_to_num(macd_h[v_idx], nan=0.0),
                'stoch': np.nan_to_num(stoch_k[v_idx], nan=50.0),
                'cci': np.nan_to_num(cci[v_idx], nan=0.0),
                'mfi': np.nan_to_num(mfi[v_idx], nan=50.0)
            }
            
            # B1: Rolling Lag Features
            features_dict['rsi_change_5'] = np.nan_to_num(rsi[v_idx] - rsi[np.maximum(0, v_idx-5)])
            
            vol_5 = v[np.maximum(0, v_idx-5)]
            features_dict['vol_change_5'] = np.where(vol_5 > 0, (v[v_idx] / vol_5) - 1, 0)
            
            atr_5 = atr[np.maximum(0, v_idx-5)]
            features_dict['atr_change_5'] = np.where(atr_5 > 0, (atr[v_idx] / atr_5) - 1, 0)
            
            features_dict['macd_slope'] = np.nan_to_num(macd_h[v_idx] - macd_h[np.maximum(0, v_idx-5)])
            
            # Rolling Max/Min for channels (VECTORIZED)
            rolling_range = np.nan_to_num(rr_arr[v_idx], nan=0.0)
            rolling_min = np.nan_to_num(rmin_arr[v_idx], nan=l[v_idx])
            features_dict['close_pos_in_range'] = np.where(rolling_range > 0, (c[v_idx] - rolling_min) / rolling_range, 0.5)

            # MAs (VECTORIZED)
            ema_10 = np.nan_to_num(sma10_arr[v_idx], nan=c[v_idx])
            ema_20 = np.nan_to_num(sma20_arr[v_idx], nan=c[v_idx])
            ema_50 = np.nan_to_num(sma50_arr[v_idx], nan=c[v_idx])
            ema_200 = np.nan_to_num(sma200_arr[v_idx], nan=c[v_idx])
            
            features_dict['ema_slope'] = np.where(ema_20 > 0, (ema_10 - ema_20) / ema_20 * 100, 0)
            features_dict['dist_to_ema'] = np.where(ema_50 > 0, (c[v_idx] - ema_50) / ema_50 * 100, 0)
        features_dict['htf_trend'] = np.where(ema_200 > 0, (ema_50 - ema_200) / ema_200 * 100, 0)
        features_dict['stoch_dist_50'] = np.abs(np.nan_to_num(stoch_k[v_idx], nan=50.0) - 50.0)

        # Bollinger Bands
        up_bb = df_feats['BBU_20_2.0'].values if 'BBU_20_2.0' in df_feats else np.zeros_like(c)
        dn_bb = df_feats['BBL_20_2.0'].values if 'BBL_20_2.0' in df_feats else np.zeros_like(c)
        bb_pos = df_feats['BBP_20_2.0'].values if 'BBP_20_2.0' in df_feats else np.full_like(c, 0.5)
        features_dict['bb_pos'] = np.nan_to_num(bb_pos[v_idx], nan=0.5)
        
        bb_widths = np.where(~np.isnan(up_bb[v_idx]), (up_bb[v_idx] - dn_bb[v_idx]), 0)
        features_dict['bb_width'] = np.where(c[v_idx] > 0, bb_widths / c[v_idx] * 100, 0)
        
        # H1 Direction
        h1_open = o[np.maximum(0, v_idx-11)]
        features_dict['h1_direction'] = np.where(c[v_idx] > h1_open, 1.0, -1.0)
        
        fr = df_feats['funding_rate'].values if 'funding_rate' in df_feats else np.zeros_like(c)
        features_dict['funding_rate'] = np.nan_to_num(fr[v_idx], nan=0.0)
        
        # BTC Macro Env
        btc_trend = np.zeros(len(v_idx))
        btc_vol = np.zeros(len(v_idx))
        btc_dump = np.zeros(len(v_idx))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            if btc_df is not None and not btc_df.empty:
                timestamps = df_feats.index[v_idx]
                try:
                    btc_indices = btc_df.index.get_indexer(timestamps, method='pad')
                    valid_btc_mask = btc_indices >= 0
                    
                    v_btc_idx = btc_indices[valid_btc_mask]
                    btc_c = btc_df['close'].values[v_btc_idx]
                    btc_ema_50 = btc_df['btc_ema_50'].values[v_btc_idx]
                    btc_ema_200 = btc_df['btc_ema_200'].values[v_btc_idx]
                    btc_atr_arr = btc_df['btc_atr'].values[v_btc_idx]
                    
                    btc_trend[valid_btc_mask] = np.where(btc_ema_200 > 0, (btc_ema_50 - btc_ema_200) / btc_ema_200 * 100, 0)
                    btc_vol[valid_btc_mask] = np.where(btc_c > 0, btc_atr_arr / btc_c * 100, 0)
                    
                    # BTC Dump (since 3 candles ago)
                    btc_o_past = btc_df['open'].values[np.maximum(0, v_btc_idx-2)]
                    btc_dump[valid_btc_mask] = np.where(btc_o_past > 0, (btc_c - btc_o_past) / btc_o_past * 100, 0)
                except Exception as e:
                    logger.warning(f"BTC Gravity error: {e}")
                
        features_dict['btc_trend'] = btc_trend
        features_dict['btc_volatility_pct'] = btc_vol
        features_dict['btc_dump_3c'] = btc_dump
        
        return pd.DataFrame(features_dict)

    def train(self, features_df: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
        """
        Trains the Random Forest model on historical trade features and their outcomes.
        labels: 1 for winning trade, 0 for losing trade.
        """
        if len(features_df) < self.min_samples:
            logger.warning(f"Not enough samples to train ML model (need >= {self.min_samples})")
            return {'accuracy': 0.0, 'precision': 0.0}
            
        # Drop the index column if it exists in features
        X = features_df.drop(columns=['index']) if 'index' in features_df.columns else features_df
        y = labels
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.clf.fit(X_train, y_train)
        self.is_fitted = True
        self.save_model()
        
        # Calculate metrics
        y_pred = self.clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # Precision: When model says 'Trade', how often is it right?
        prec = precision_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"ML Model Trained. Accuracy: {acc:.2%}, Precision (Trade Quality): {prec:.2%}")
        
        # Check if model has feature importances (VotingClassifier doesn't directly expose them)
        if hasattr(self.clf, 'estimators_'):
            # Approximate feature importance by averaging the Random Forest importance
            rf_model = self.clf.estimators_[0]
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                feature_names = list(features_df.drop('index', axis=1).columns)
                self.feature_importances = sorted(
                    zip(feature_names, importances), 
                    key=lambda x: x[1], reverse=True
                )
                logger.info(f"Top 3 ML Features (via RF sub-model): {self.feature_importances[:3]}")
        
        return {'accuracy': acc, 'precision': prec}

    def predict(self, features_df: pd.DataFrame) -> Tuple[float, bool]:
        """
        Standard prediction interface used by strategies.
        Returns: (probability, is_approved)
        """
        if not self.is_fitted or self.clf is None:
            return 0.5, False
        
        # Ensure we only use the columns the model was trained on
        # If the strategy passes more columns, we might need to filter.
        # But usually features_df matches what was given to train.
        try:
            probs = self.clf.predict_proba(features_df)
            win_prob = float(probs[0][1])
            is_approved = win_prob >= getattr(self, 'threshold', 0.5)
            return win_prob, is_approved
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5, False

    def predict_probability(self, features: Dict[str, float], strategy_name: str = None) -> float:
        """
        Returns the probability (0.0 to 1.0) that a trade will be successful.
        If model isn't trained, assumes 0.5 (neutral).
        """
        if not self.is_fitted or self.clf is None:
            return 0.5
            
        # Convert dict to single-row DataFrame matching the training columns
        X = pd.DataFrame([features])
        
        # Output probabilities for classes [0, 1] — we want the probability of 1 (win)
        try:
            probs = self.clf.predict_proba(X)
            win_prob = probs[0][1]
            
            # ── Model Drift Penalty ───────────────────────────────────────
            if strategy_name:
                try:
                    from core.db import CortexDB
                    db = CortexDB("data/autocore.db")
                    with db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT pnl_usd FROM trades 
                            WHERE status = 'CLOSED' AND strategy = ?
                            ORDER BY exit_time DESC LIMIT 10
                        ''', (strategy_name,))
                        rows = cursor.fetchall()
                        if len(rows) >= 5: # Need at least 5 recent trades to judge
                            wins = sum(1 for r in rows if r[0] > 0)
                            win_rate = wins / len(rows)
                            if win_rate < 0.40:
                                penalty = 0.70 # 30% confidence penalty
                                logger.warning(f"📉 ML DRIFT DETECTED [{strategy_name}]: Recent WR {win_rate*100:.0f}%. Penalizing confidence {win_prob:.2f} -> {win_prob * penalty:.2f}")
                                win_prob *= penalty
                except Exception as e:
                    logger.debug(f"ML Drift Monitor error: {e}")
                    
            return float(win_prob)
        except:
            return 0.5
