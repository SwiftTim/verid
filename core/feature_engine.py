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
        Generate all features
        
        Input: DataFrame with ['timestamp', 'quote']
        Output: DataFrame with engineered features
        """
        df = df.copy()
        
        # ============ CORE FEATURES ============
        
        # 1. Tick difference (primary signal)
        df['tick_diff'] = df['quote'].diff()
        
        # 2. Direction (target variable)
        df['direction'] = (df['tick_diff'] > 0).astype(int)
        
        # ============ ROLLING STATISTICS ============
        
        # 3-4. Rolling means
        for window in ROLLING_WINDOWS['mean']:
            df[f'rolling_mean_{window}'] = df['quote'].rolling(window).mean()
        
        # 5-6. Rolling standard deviations (volatility proxy)
        for window in ROLLING_WINDOWS['std']:
            df[f'rolling_std_{window}'] = df['tick_diff'].rolling(window).std()
        
        # 7-8. Momentum (price change over window)
        for window in ROLLING_WINDOWS['momentum']:
            df[f'momentum_{window}'] = df['quote'] - df['quote'].shift(window)
        
        # ============ MICROSTRUCTURE FEATURES ============
        
        # 9. Volatility (10-tick window)
        df['volatility'] = df['tick_diff'].rolling(10).std()
        
        # 10. Streak detection
        df['streak'] = self._calculate_streak(df)
        
        # 11. Mean reversion signal
        df['mean_reversion'] = (df['quote'] - df['rolling_mean_5']) / (df['rolling_std_5'] + 1e-8)
        
        # 12. Tick acceleration (2nd derivative)
        df['acceleration'] = df['tick_diff'].diff()
        
        # 13. Up/Down ratio (last 10 ticks)
        df['up_down_ratio'] = df['direction'].rolling(10).mean()
        
        # ============ NORMALIZATION ============
        
        # Normalize features to prevent scale issues
        df = self._normalize_features(df)
        
        # ============ CLEANUP ============
        
        # Drop NaN rows (from rolling windows)
        df = df.dropna()
        
        # Store feature names
        self.feature_names = [col for col in df.columns 
                             if col not in ['timestamp', 'quote', 'direction', 'tick_diff']]
        
        # Safety check: limit feature count
        if len(self.feature_names) > MAX_FEATURES:
            print(f"⚠️ Warning: {len(self.feature_names)} features exceeds MAX_FEATURES ({MAX_FEATURES})")
        
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
