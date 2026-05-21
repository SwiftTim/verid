"""
Data Engine: Streaming Buffer + Sequence Generation
Optimized for live retraining with time-based splits
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple, Optional
from .config import BUFFER_SIZE, SEQUENCE_LENGTH


class StreamingBuffer:
    """
    Rolling buffer for tick data
    Maintains fixed-size window for adaptive learning
    """
    
    def __init__(self, max_size: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_tick(self, tick: dict):
        """
        Add new tick to buffer
        
        Args:
            tick: {'timestamp': int, 'quote': float, 'symbol': str}
        """
        self.buffer.append(tick)
    
    def add_ticks(self, ticks: list):
        """Add multiple ticks at once"""
        self.buffer.extend(ticks)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame"""
        if len(self.buffer) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.buffer)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def size(self) -> int:
        """Current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int = SEQUENCE_LENGTH + 1) -> bool:
        """Check if buffer has enough data"""
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear buffer (use with caution)"""
        self.buffer.clear()


class DataEngine:
    """
    Core data processing engine
    - Creates sequences for LSTM
    - Performs time-based splits (NO SHUFFLE)
    - Prevents data leakage
    """
    
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.buffer = StreamingBuffer()
    
    def create_sequences(self, df, lookahead=5, min_move=0.003):
        """
        Only label ticks where price moves MEANINGFULLY.
        Skip ambiguous ticks entirely.
        """
        feature_cols = [col for col in df.columns
                        if col not in ['timestamp', 'quote', 'direction', 'tick_diff']]
        X, y = [], []

        for i in range(len(df) - self.sequence_length - lookahead):
            seq = df.iloc[i:i + self.sequence_length][feature_cols].values

            # Net price move over next N ticks
            entry  = df['quote'].iloc[i + self.sequence_length]
            exit_p = df['quote'].iloc[i + self.sequence_length + lookahead]
            net_move = (exit_p - entry) / entry

            # SKIP ambiguous ticks — only label clear moves
            if abs(net_move) < min_move:
                continue

            target = 1 if net_move > 0 else 0
            X.append(seq)
            y.append(target)

        return np.array(X), np.array(y)

    
    def time_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        split_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Time-based train/test split (NO SHUFFLE)
        
        Critical: Maintains temporal order to prevent leakage
        """
        if len(X) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        split_idx = int(len(X) * split_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test

    def walk_forward_split(self, X, y, n_splits=5):
        """
        Walk-forward (time-series) cross-validation.
        MUCH more reliable accuracy estimate than single split.
        """
        n = len(X)
        min_train = n // (n_splits + 1)
        splits = []
        
        for i in range(1, n_splits + 1):
            train_end = min_train * i
            test_end  = min_train * (i + 1)
            
            if test_end > n:
                break
            
            splits.append((
                X[:train_end], X[train_end:test_end],
                y[:train_end], y[train_end:test_end]
            ))
        
        return splits
    
    def prepare_for_prediction(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepare most recent sequence for prediction
        
        Returns:
            X: (1, sequence_length, n_features) or None if insufficient data
        """
        if len(df) < self.sequence_length:
            return None
        
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'quote', 'direction', 'tick_diff']]
        
        # Take last sequence_length rows
        seq = df.iloc[-self.sequence_length:][feature_cols].values
        
        # Reshape for LSTM: (1, sequence_length, n_features)
        return seq.reshape(1, self.sequence_length, -1)
    
    def get_latest_features(self, df: pd.DataFrame) -> dict:
        """
        Extract latest features for RL state
        
        Returns:
            Dict with: volatility, streak, recent_direction
        """
        if len(df) < 10:
            return {
                'volatility': 0.0,
                'streak': 0,
                'recent_direction': 0.5
            }
        
        latest = df.iloc[-10:]
        
        return {
            'volatility': latest['tick_diff'].std() if 'tick_diff' in df.columns else 0.0,
            'streak': self._calculate_streak(df),
            'recent_direction': latest['direction'].mean() if 'direction' in df.columns else 0.5
        }
    
    def _calculate_streak(self, df: pd.DataFrame) -> int:
        """Calculate current win/loss streak"""
        if 'direction' not in df.columns or len(df) < 2:
            return 0
        
        directions = df['direction'].values[-10:]  # Last 10 ticks
        
        if len(directions) < 2:
            return 0
        
        streak = 1
        last_dir = directions[-1]
        
        for i in range(len(directions) - 2, -1, -1):
            if directions[i] == last_dir:
                streak += 1
            else:
                break
        
        return streak if last_dir == 1 else -streak
    
    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Check data quality
        
        Returns:
            Dict with quality metrics
        """
        if len(df) == 0:
            return {'valid': False, 'reason': 'Empty dataframe'}
        
        issues = []
        
        # Check for NaN values
        if df.isnull().any().any():
            issues.append('Contains NaN values')
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number]).values).any():
            issues.append('Contains infinite values')
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns and df['timestamp'].duplicated().any():
            issues.append('Duplicate timestamps detected')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'size': len(df)
        }
