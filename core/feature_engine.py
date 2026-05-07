"""
Feature Engine: Short-Memory Features for Single-Tick Prediction
Optimized for drift adaptation and minimal overfitting
"""

import pandas as pd
import numpy as np
from typing import Optional
from .config import ROLLING_WINDOWS, MAX_FEATURES


class FeatureEngine:
    """
    Generates minimal, high-signal features
    
    Design Philosophy:
    - Short memory windows (3-10 ticks)
    - Avoid over-engineering
    - Focus on microstructure patterns
    """
    
    def __init__(self):
        self.feature_names = []
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate robust features for synthetic indices
        """
        df = df.copy()
        
        # --- CORE ---
        df['tick_diff'] = df['quote'].diff()
        df['direction'] = (df['tick_diff'] > 0).astype(int)
        
        # --- VOLATILITY REGIME (most predictive on synthetics) ---
        for w in [5, 10, 20]:
            df[f'vol_{w}'] = df['tick_diff'].rolling(w).std()
        
        df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-8)  # regime shift detector
        df['vol_zscore'] = (df['vol_10'] - df['vol_10'].rolling(50).mean()) / (df['vol_10'].rolling(50).std() + 1e-8)
        
        # --- MOMENTUM (multiple timeframes) ---
        for w in [3, 5, 10, 20]:
            df[f'mom_{w}'] = df['quote'].pct_change(w)
        
        # --- MEAN REVERSION SIGNALS ---
        df['ma_5'] = df['quote'].rolling(5).mean()
        df['ma_20'] = df['quote'].rolling(20).mean()
        df['ma_cross'] = df['ma_5'] - df['ma_20']           # golden/death cross
        df['dist_from_ma20'] = (df['quote'] - df['ma_20']) / (df['vol_20'] + 1e-8)
        
        # --- STREAK & RUNS (key for pseudo-random series) ---
        # Consecutive same-direction ticks
        direction_change = (df['direction'] != df['direction'].shift(1)).astype(int)
        run_id = direction_change.cumsum()
        df['streak'] = df.groupby(run_id).cumcount() + 1
        df['streak'] = df['streak'] * (df['direction'] * 2 - 1)  # signed
        
        # --- AUTOCORRELATION FEATURES (pseudo-random series have weak -ve autocorr) ---
        df['autocorr_1'] = df['tick_diff'].rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 2 else 0, raw=False
        )
        df['autocorr_2'] = df['tick_diff'].rolling(20).apply(
            lambda x: x.autocorr(lag=2) if len(x) > 2 else 0, raw=False
        )
        
        # --- RSI (overbought/oversold) ---
        delta = df['tick_diff']
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = (df['rsi_14'] - 50) / 50   # normalize to -1..1
        
        # --- HIGHER-ORDER STATS ---
        df['skew_10'] = df['tick_diff'].rolling(10).skew()
        df['kurt_10'] = df['tick_diff'].rolling(10).kurt()
        
        # --- TICK SIZE ANALYSIS ---
        df['abs_diff'] = df['tick_diff'].abs()
        df['rel_tick_size'] = df['abs_diff'] / (df['abs_diff'].rolling(20).mean() + 1e-8)
        df['big_move'] = (df['rel_tick_size'] > 2.0).astype(int)  # outlier ticks
        
        # --- UP/DOWN RATIO (pressure) ---
        df['buy_pressure'] = df['direction'].rolling(10).mean()
        df['buy_pressure_fast'] = df['direction'].rolling(5).mean()
        df['pressure_diff'] = df['buy_pressure_fast'] - df['buy_pressure']
        
        # --- NORMALIZE (rolling z-score, NOT global - avoids lookahead bias) ---
        skip_norm = {'timestamp', 'quote', 'direction', 'tick_diff', 
                    'streak', 'big_move', 'direction'}
        for col in df.columns:
            if col not in skip_norm and df[col].dtype in [np.float64, np.int64]:
                roll_mean = df[col].rolling(200, min_periods=20).mean()
                roll_std  = df[col].rolling(200, min_periods=20).std()
                df[col] = (df[col] - roll_mean) / (roll_std + 1e-8)
        
        df = df.dropna()
        self.feature_names = [c for c in df.columns 
                            if c not in ['timestamp', 'quote', 'direction', 'tick_diff']]
        return df
    
    def _calculate_streak(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate consecutive up/down streaks
        
        Returns:
            Series with streak values (positive=up, negative=down)
        """
        if 'direction' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Group consecutive directions
        direction_changes = (df['direction'] != df['direction'].shift()).cumsum()
        
        # Count streak length
        streak = df.groupby(direction_changes).cumcount() + 1
        
        # Make negative for down streaks
        streak = streak * (df['direction'] * 2 - 1)
        
        return streak
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using rolling statistics
        
        Note: We use rolling normalization (not global) to adapt to drift
        """
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'quote', 'direction', 'tick_diff', 'streak']]
        
        for col in feature_cols:
            # Rolling z-score normalization
            rolling_mean = df[col].rolling(100, min_periods=10).mean()
            rolling_std = df[col].rolling(100, min_periods=10).std()
            
            df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        return df
    
    def extract_tree_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features specifically for Decision Tree
        
        Trees work better with:
        - Categorical features
        - Binned continuous features
        - Interaction terms
        """
        tree_df = df.copy()
        
        # Bin volatility into regimes
        tree_df['volatility_regime'] = pd.cut(
            df['volatility'], 
            bins=3, 
            labels=['low', 'medium', 'high']
        )
        
        # Bin streak
        tree_df['streak_category'] = pd.cut(
            df['streak'], 
            bins=[-np.inf, -3, -1, 1, 3, np.inf],
            labels=['strong_down', 'weak_down', 'neutral', 'weak_up', 'strong_up']
        )
        
        # Momentum direction
        tree_df['momentum_direction'] = (df['momentum_3'] > 0).astype(int)
        
        return tree_df
    
    def get_feature_importance_proxy(self, df: pd.DataFrame) -> dict:
        """
        Calculate simple feature importance proxy
        (correlation with target)
        """
        if 'direction' not in df.columns:
            return {}
        
        importance = {}
        
        for col in self.feature_names:
            if col in df.columns:
                # Absolute correlation with target
                corr = abs(df[col].corr(df['direction']))
                importance[col] = corr
        
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def detect_drift(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> dict:
        """
        Detect feature drift between two periods
        
        Returns:
            Dict with drift metrics
        """
        drift_metrics = {}
        
        for col in self.feature_names:
            if col in df_old.columns and col in df_new.columns:
                # Compare distributions using mean and std
                mean_old = df_old[col].mean()
                mean_new = df_new[col].mean()
                std_old = df_old[col].std()
                std_new = df_new[col].std()
                
                # Normalized drift
                drift = abs(mean_new - mean_old) / (std_old + 1e-8)
                drift_metrics[col] = drift
        
        return drift_metrics
    
    def validate_features(self, df: pd.DataFrame) -> dict:
        """
        Validate feature quality
        
        Returns:
            Dict with validation results
        """
        issues = []
        
        # Check for constant features
        for col in self.feature_names:
            if col in df.columns and df[col].nunique() == 1:
                issues.append(f'{col} is constant')
        
        # Check for high correlation (multicollinearity)
        numeric_cols = df[self.feature_names].select_dtypes(include=[np.number]).columns
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
