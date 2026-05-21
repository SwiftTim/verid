"""
Feature Engine: Market Microstructure + Regime-Aware Features
Upgraded with: Entropy, Velocity/Acceleration, Regime Labels,
               Bollinger Bands, Trend Strength, Reversal Score
"""

import pandas as pd
import numpy as np
from typing import Optional
from .config import ROLLING_WINDOWS, MAX_FEATURES

try:
    from scipy.stats import entropy as scipy_entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class FeatureEngine:
    """
    Generates high-signal features for synthetic index prediction.

    Phase 1 Upgrades:
    - Entropy (randomness detector → SKIP when high)
    - Velocity + Acceleration (tick rate of change)
    - Bollinger Bands + BB Position (mean reversion signal)
    - Trend Strength (momentum quality filter)
    - Reversal Score (Z-score + RSI combined)
    - Volatility Regime Label (0=low, 1=medium, 2=high)
    - Market Regime Label (0=range, 1=trend, 2=high_vol)
    - Price Z-Score (overbought/oversold)
    """

    def __init__(self):
        self.feature_names = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ─────────────────────────────────────────────
        # CORE
        # ─────────────────────────────────────────────
        df['tick_diff'] = df['quote'].diff()
        df['direction'] = (df['tick_diff'] > 0).astype(int)

        # ─────────────────────────────────────────────
        # VOLATILITY REGIME
        # ─────────────────────────────────────────────
        for w in [5, 10, 20]:
            df[f'vol_{w}'] = df['tick_diff'].rolling(w).std()

        df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-8)
        df['vol_zscore'] = (
            (df['vol_10'] - df['vol_10'].rolling(50).mean()) /
            (df['vol_10'].rolling(50).std() + 1e-8)
        )

        # ─────────────────────────────────────────────
        # MOMENTUM (multiple timeframes)
        # ─────────────────────────────────────────────
        for w in [3, 5, 10, 20]:
            df[f'mom_{w}'] = df['quote'].pct_change(w)

        # ─────────────────────────────────────────────
        # VELOCITY + ACCELERATION  ← NEW
        # ─────────────────────────────────────────────
        df['velocity']     = df['tick_diff'].rolling(3).mean()
        df['acceleration'] = df['velocity'].diff()

        # ─────────────────────────────────────────────
        # MEAN REVERSION SIGNALS
        # ─────────────────────────────────────────────
        df['ma_5']  = df['quote'].rolling(5).mean()
        df['ma_20'] = df['quote'].rolling(20).mean()
        df['ma_cross']       = df['ma_5'] - df['ma_20']
        df['dist_from_ma20'] = (df['quote'] - df['ma_20']) / (df['vol_20'] + 1e-8)

        # Bollinger Bands  ← NEW
        df['bb_upper']    = df['ma_20'] + 2 * df['vol_20']
        df['bb_lower']    = df['ma_20'] - 2 * df['vol_20']
        df['bb_position'] = (
            (df['quote'] - df['bb_lower']) /
            (df['bb_upper'] - df['bb_lower'] + 1e-8)
        )  # 0 = at lower band, 1 = at upper band

        # ─────────────────────────────────────────────
        # STREAK & RUNS
        # ─────────────────────────────────────────────
        direction_change = (df['direction'] != df['direction'].shift(1)).astype(int)
        run_id = direction_change.cumsum()
        df['streak'] = df.groupby(run_id).cumcount() + 1
        df['streak'] = df['streak'] * (df['direction'] * 2 - 1)

        # ─────────────────────────────────────────────
        # AUTOCORRELATION
        # ─────────────────────────────────────────────
        df['autocorr_1'] = df['tick_diff'].rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 2 else 0, raw=False
        )
        df['autocorr_2'] = df['tick_diff'].rolling(20).apply(
            lambda x: x.autocorr(lag=2) if len(x) > 2 else 0, raw=False
        )

        # ─────────────────────────────────────────────
        # RSI
        # ─────────────────────────────────────────────
        delta = df['tick_diff']
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-8)
        df['rsi_14']  = 100 - (100 / (1 + rs))
        df['rsi_norm'] = (df['rsi_14'] - 50) / 50

        # ─────────────────────────────────────────────
        # HIGHER-ORDER STATS
        # ─────────────────────────────────────────────
        df['skew_10'] = df['tick_diff'].rolling(10).skew()
        df['kurt_10'] = df['tick_diff'].rolling(10).kurt()

        # ─────────────────────────────────────────────
        # TICK SIZE ANALYSIS
        # ─────────────────────────────────────────────
        df['abs_diff']      = df['tick_diff'].abs()
        df['rel_tick_size'] = df['abs_diff'] / (df['abs_diff'].rolling(20).mean() + 1e-8)
        df['big_move']      = (df['rel_tick_size'] > 2.0).astype(int)

        # ─────────────────────────────────────────────
        # PRESSURE / ORDER FLOW
        # ─────────────────────────────────────────────
        df['buy_pressure']      = df['direction'].rolling(10).mean()
        df['buy_pressure_fast'] = df['direction'].rolling(5).mean()
        df['pressure_diff']     = df['buy_pressure_fast'] - df['buy_pressure']

        # ─────────────────────────────────────────────
        # PRICE Z-SCORE  ← NEW
        # ─────────────────────────────────────────────
        rolling_mean = df['quote'].rolling(20).mean()
        rolling_std  = df['quote'].rolling(20).std()
        df['zscore_20'] = (df['quote'] - rolling_mean) / (rolling_std + 1e-8)

        # ─────────────────────────────────────────────
        # TREND STRENGTH  ← NEW
        # ─────────────────────────────────────────────
        df['trend_strength'] = df['mom_5'].abs() / (df['vol_5'] + 1e-8)

        # ─────────────────────────────────────────────
        # REVERSAL SCORE  ← NEW
        # ─────────────────────────────────────────────
        df['reversal_score'] = (df['rsi_norm'] * -1) + (df['zscore_20'] * -1)

        # ─────────────────────────────────────────────
        # ENTROPY FEATURE  ← NEW (most important filter)
        # ─────────────────────────────────────────────
        def rolling_entropy(series, window=20):
            result = []
            for i in range(len(series)):
                if i < window:
                    result.append(np.nan)
                    continue
                chunk = series.iloc[i - window:i].dropna()
                if len(chunk) < 5:
                    result.append(np.nan)
                    continue
                hist, _ = np.histogram(chunk, bins=10)
                if SCIPY_AVAILABLE:
                    result.append(scipy_entropy(hist + 1e-8))
                else:
                    # Manual Shannon entropy fallback
                    p = hist / (hist.sum() + 1e-8)
                    result.append(-np.sum(p * np.log(p + 1e-8)))
            return pd.Series(result, index=series.index)

        df['entropy_20'] = rolling_entropy(df['tick_diff'], 20)

        # ── HURST EXPONENT (trend vs random walk detector) ──
        def hurst_exponent(series, max_lag=20):
            """H > 0.5 = trending, H < 0.5 = mean-reverting, H ≈ 0.5 = random"""
            lags = range(2, max_lag)
            tau = []
            for lag in lags:
                tau.append(np.std(series[lag:] - series[:-lag]))
            poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return poly[0]  # slope = Hurst exponent

        df['hurst'] = df['quote'].rolling(50).apply(
            lambda x: hurst_exponent(x.values) if len(x) == 50 else 0.5,
            raw=False
        )
        # Only trade when H deviates from random (0.5)
        df['regime_edge'] = (df['hurst'] - 0.5).abs()  # higher = more predictable

        # ─────────────────────────────────────────────
        # VOLATILITY REGIME LABEL  ← NEW (0=low, 1=med, 2=high)
        # ─────────────────────────────────────────────
        vol_abs = df['vol_10'].abs()
        # use quantile-based bins to avoid scale dependence
        q33 = vol_abs.rolling(100, min_periods=20).quantile(0.33)
        q66 = vol_abs.rolling(100, min_periods=20).quantile(0.66)
        df['vol_regime'] = 1  # default medium
        df.loc[vol_abs <= q33, 'vol_regime'] = 0  # low
        df.loc[vol_abs >= q66, 'vol_regime'] = 2  # high
        df['tradeable'] = (df['vol_regime'] < 2).astype(int)

        # ─────────────────────────────────────────────
        # MARKET REGIME LABEL  ← NEW (0=range, 1=trend, 2=high_vol)
        # ─────────────────────────────────────────────
        def classify_regime(row):
            if row['vol_ratio'] > 1.5:
                return 2   # high volatility
            if abs(row['mom_10']) > 0.003:
                return 1   # trending
            return 0       # ranging

        df['market_regime'] = df.apply(classify_regime, axis=1)

        # ─────────────────────────────────────────────
        # NORMALIZE (rolling z-score, no lookahead bias)
        # ─────────────────────────────────────────────
        skip_norm = {
            'timestamp', 'quote', 'direction', 'tick_diff',
            'streak', 'big_move', 'tradeable', 'market_regime', 'vol_regime',
            'entropy_20', 'vol_regime', 'market_regime'
        }
        
        # Adaptive normalization window
        norm_window = min(200, len(df) // 3)
        if norm_window < 10: norm_window = 10

        for col in df.columns:
            if col not in skip_norm and df[col].dtype in [np.float64, np.float32, np.int64]:
                roll_mean = df[col].rolling(norm_window, min_periods=5).mean()
                roll_std  = df[col].rolling(norm_window, min_periods=5).std()
                df[col] = (df[col] - roll_mean) / (roll_std + 1e-8)

        # Fill NaNs with 0 instead of dropping rows
        # This ensures we don't lose the entire dataset if one feature is noisy
        df = df.fillna(0)
        
        # FINAL ENSURANCE: All columns must be numeric for LSTM/Tree
        # (Fixes Keras ValueError: Invalid dtype: object)
        for col in df.columns:
            if col not in ['timestamp']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        
        df = df.fillna(0)

        self.feature_names = [
            c for c in df.columns
            if c not in ['timestamp', 'quote', 'direction', 'tick_diff']
        ]
        return df

    def get_feature_importance_proxy(self, df: pd.DataFrame) -> dict:
        if 'direction' not in df.columns:
            return {}
        importance = {}
        for col in self.feature_names:
            if col in df.columns:
                corr = abs(df[col].corr(df['direction']))
                importance[col] = corr
        # Only return features with signal > 1%
        filtered = {k: v for k, v in importance.items() if v > 0.01}
        return dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True))

    def detect_drift(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> dict:
        drift_metrics = {}
        for col in self.feature_names:
            if col in df_old.columns and col in df_new.columns:
                mean_old = df_old[col].mean()
                mean_new = df_new[col].mean()
                std_old  = df_old[col].std()
                drift = abs(mean_new - mean_old) / (std_old + 1e-8)
                drift_metrics[col] = drift
        return drift_metrics

    def validate_features(self, df: pd.DataFrame) -> dict:
        issues = []
        for col in self.feature_names:
            if col in df.columns and df[col].nunique() == 1:
                issues.append(f'{col} is constant')
        numeric_cols = (
            df[self.feature_names].select_dtypes(include=[np.number]).columns
        )
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr = (corr_matrix > 0.95) & (corr_matrix < 1.0)
            if high_corr.any().any():
                issues.append('High feature correlation detected')
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'feature_count': len(self.feature_names)
        }
