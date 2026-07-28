"""
HMM Regime Detection
====================
Uses Hidden Markov Model to classify market into 3 regimes:
- Bull (trending up)
- Bear (trending down)  
- Chop (sideways/low vol)

Strategies can use regime to filter trades:
- SMC/SwingICT: only trade in trending regimes
- Knife Catcher: only trade in chop→reverse regimes
- ORB: any regime with sufficient volatility
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. Using simplified regime detection. pip install hmmlearn")


class RegimeDetector:
    """3-state HMM regime detector"""
    
    def __init__(self, n_regimes: int = 3, lookback: int = 100):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = None
        self.regime_map = {}  # HMM state → human label
    
    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """Fit HMM and predict regime for each bar.
        
        Returns Series with values: 'bull', 'bear', 'chop'
        """
        if not HMM_AVAILABLE:
            return self._simplified_regime(df)
        
        try:
            return self._hmm_regime(df)
        except Exception as e:
            logger.warning(f"HMM failed ({e}), using simplified regime")
            return self._simplified_regime(df)
    
    def _hmm_regime(self, df: pd.DataFrame) -> pd.Series:
        """Full HMM-based regime detection"""
        close = df['close'].values.astype(float)
        
        # Features for HMM: returns + volatility
        returns = np.diff(np.log(close))
        vol = pd.Series(returns).rolling(20).std().values
        
        # Stack features (skip NaN)
        valid = ~np.isnan(vol)
        features = np.column_stack([returns[valid], vol[valid]])
        
        # Prevent degenerate covariance matrices in completely flat markets
        features += np.random.normal(0, 1e-6, features.shape)
        
        # Fit HMM
        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            min_covar=1e-4  # Fixes "covars must be symmetric" error on low volatility
        )
        model.fit(features)
        hidden_states = model.predict(features)
        
        # Map states to regimes by mean return
        state_returns = {}
        for s in range(self.n_regimes):
            mask = hidden_states == s
            state_returns[s] = features[mask, 0].mean() if mask.sum() > 0 else 0.0
        
        sorted_states = sorted(state_returns.keys(), key=lambda s: state_returns[s])
        regime_map = {
            sorted_states[0]: 'bear',
            sorted_states[1]: 'chop',
            sorted_states[-1]: 'bull',
        }
        
        # Build full series (pad first values)
        regimes = pd.Series(['chop'] * len(df), index=df.index)
        valid_idx = np.where(valid)[0]
        for i, state in enumerate(hidden_states):
            # +1 because returns shifts by 1
            orig_idx = valid_idx[i] + 1
            if orig_idx < len(df):
                regimes.iloc[orig_idx] = regime_map[state]
        
        self.model = model
        self.regime_map = regime_map
        logger.info(f"HMM Regime: bull={sum(regimes=='bull')}, bear={sum(regimes=='bear')}, chop={sum(regimes=='chop')}")
        
        return regimes
    
    def _simplified_regime(self, df: pd.DataFrame) -> pd.Series:
        """Fallback: ADX + DMI + EMA based regime detection"""
        regimes = pd.Series('chop', index=df.index)
        
        try:
            import pandas_ta as ta
            # Calculate ADX (trend strength) and DMI (direction)
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            
            if adx_df is not None and not adx_df.empty:
                adx = adx_df.iloc[:, 0]  # First col is ADX line
                pdi = adx_df.iloc[:, 1]  # Second col is +DI
                ndi = adx_df.iloc[:, 2]  # Third col is -DI
                
                # ADX > 25 means a strong trend is present
                # +DI > -DI means the trend is upward
                bull_mask = (adx >= 25) & (pdi > ndi)
                bear_mask = (adx >= 25) & (ndi > pdi)
                
                regimes[bull_mask] = 'bull'
                regimes[bear_mask] = 'bear'
                return regimes
        except Exception as e:
            logger.warning(f"ADX calculation failed ({e}), using basic EMA fallback.")
            pass
            
        # Absolute basic fallback if pandas_ta is not available or fails
        close = df['close'].astype(float)
        ema_fast = close.ewm(span=20).mean()
        ema_slow = close.ewm(span=50).mean()
        atr = (df['high'] - df['low']).rolling(14).mean()
        atr_pct = atr / close * 100
        
        bull_mask = (ema_fast > ema_slow) & (atr_pct > 0.3)
        bear_mask = (ema_fast < ema_slow) & (atr_pct > 0.3)
        
        regimes[bull_mask] = 'bull'
        regimes[bear_mask] = 'bear'
        return regimes
    
    def should_trade(self, regime: str, strategy: str) -> bool:
        """Check if strategy should trade in given regime"""
        rules = {
            'ultimate_smc_trail': ['bull', 'bear'],     # Trend strategies
            'swingict_trail':     ['bull', 'bear'],
            'knife_catcher':      ['chop'],              # Mean reversion
            'knifecatcher_ml':    ['chop'],
            'orb_strategy':       ['bull', 'bear', 'chop'],  # Works everywhere
            'ml_orb':             ['bull', 'bear', 'chop'],
            'vwap_squeeze':       ['chop', 'bull'],      # Mean reversion off VWAP bands
            'ttm_squeeze':        ['bull', 'bear'],      # Momentum squeeze breakout
        }
        # Normalize: 'SwingICT_Trail' → 'swingict_trail'
        key = strategy.lower()
        allowed = rules.get(key, ['bull', 'bear', 'chop'])
        return regime in allowed
