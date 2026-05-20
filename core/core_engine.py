"""
Core Engine: Main Orchestrator
Coordinates all subsystems for live prediction
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import time

from .data_engine import DataEngine, StreamingBuffer
from .feature_engine import FeatureEngine
from .models import LSTMEngine, TreeEngine, EnsembleEngine, RegimeFilter
from .risk_engine import RiskEngine
from .config import (
    RETRAIN_CONFIG,
    MONITOR_CONFIG,
    SEQUENCE_LENGTH,
    TRAIN_TEST_SPLIT
)


class HybridEngine:
    """
    Main prediction engine orchestrator
    
    Workflow:
    1. Receive tick stream
    2. Generate features
    3. Create sequences
    4. Predict (LSTM + Tree)
    5. Ensemble fusion
    6. RL execution decision
    7. Risk approval
    8. Execute or skip
    9. Log result
    10. Periodic retraining
    
    Optimized for:
    - Single tick prediction
    - Live retraining
    - Google Colab deployment
    """
    
    def __init__(self, verbose: bool = True):
        # Core components
        self.data_engine = DataEngine()
        self.feature_engine = FeatureEngine()
        self.lstm_engine = LSTMEngine()
        self.tree_engine = TreeEngine()
        self.ensemble_engine = EnsembleEngine()
        self.q_agent = RegimeFilter()
        self.risk_engine = RiskEngine()
        
        # State
        self.is_trained = False
        self.tick_count = 0
        self.prediction_count = 0
        self.last_retrain_tick = 0
        self.verbose = verbose
        
        # Performance tracking
        self.performance_log = []
    
    def add_tick(self, tick: dict):
        """
        Add new tick to buffer
        
        Args:
            tick: {'timestamp': int, 'quote': float, 'symbol': str}
        """
        self.data_engine.buffer.add_tick(tick)
        self.tick_count += 1
        
        if self.verbose and self.tick_count % 100 == 0:
            print(f"📊 Ticks received: {self.tick_count}")
    
    def initial_train(self) -> dict:
        """
        Initial training on buffered data
        
        Call this once you have enough data
        
        Returns:
            Training results dict
        """
        if self.tick_count < RETRAIN_CONFIG['min_buffer_size']:
            raise ValueError(
                f"Insufficient data. Need {RETRAIN_CONFIG['min_buffer_size']}, "
                f"have {self.tick_count}"
            )
        
        print("🚀 Starting initial training...")
        start_time = time.time()
        
        # 1. Get data
        df = self.data_engine.buffer.get_dataframe()
        
        # 2. Generate features
        df = self.feature_engine.transform(df)
        
        # 3. Create sequences
        X, y = self.data_engine.create_sequences(df, lookahead=3)
        
        if len(X) == 0:
            raise ValueError("Failed to create sequences")
        
        # 4. Time-based split
        X_train, X_test, y_train, y_test = self.data_engine.time_split(X, y, TRAIN_TEST_SPLIT)
        
        # 5. Train LSTM
        print("  Training LSTM...")
        lstm_history = self.lstm_engine.train(X_train, y_train, X_test, y_test, verbose=1)
        
        # 6. Train Tree (flatten sequences for tree)
        print("  Training Decision Tree...")
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        tree_info = self.tree_engine.train(X_train_flat, y_train)
        
        # 7. Evaluate
        lstm_eval = self.lstm_engine.evaluate(X_test, y_test)
        tree_eval = self.tree_engine.evaluate(X_test, y_test)
        
        self.is_trained = True
        self.last_retrain_tick = self.tick_count
        
        elapsed = time.time() - start_time
        
        results = {
            'success': True,
            'training_time': elapsed,
            'samples_trained': len(X_train),
            'samples_tested': len(X_test),
            'lstm_accuracy': lstm_eval['accuracy'],
            'tree_accuracy': tree_eval['accuracy']
        }
        
        print(f"✅ Initial training complete in {elapsed:.2f}s")
        print(f"   LSTM accuracy: {lstm_eval['accuracy']:.2%}")
        print(f"   Tree accuracy: {tree_eval['accuracy']:.2%}")
        
        return results
    
    def predict_next_tick(self) -> Optional[dict]:
        if not self.is_trained:
            raise ValueError("Engine not trained.")
        
        df = self.data_engine.buffer.get_dataframe()
        if len(df) < SEQUENCE_LENGTH:
            return None
        
        df = self.feature_engine.transform(df)
        X = self.data_engine.prepare_for_prediction(df)
        if X is None:
            return None
        
        lstm_prob = float(self.lstm_engine.predict_proba(X)[0])
        X_flat = X.reshape(1, -1)
        tree_prob = float(self.tree_engine.predict_proba(X_flat)[0])
        
        # Use meta-learner if available
        combined_prob = self.ensemble_engine.combine_with_meta(lstm_prob, tree_prob)
        
        # Get current market context
        latest        = self.data_engine.get_latest_features(df)
        vol_zscore    = float(df['vol_zscore'].iloc[-1])    if 'vol_zscore'    in df.columns else 0.0
        buy_pressure  = float(df['buy_pressure'].iloc[-1])  if 'buy_pressure'  in df.columns else 0.5
        autocorr      = float(df['autocorr_1'].iloc[-1])    if 'autocorr_1'    in df.columns else 0.0
        entropy       = float(df['entropy_20'].iloc[-1])     if 'entropy_20'    in df.columns else np.nan
        vol_regime    = float(df['vol_regime'].iloc[-1])     if 'vol_regime'    in df.columns else 1.0
        market_regime = float(df['market_regime'].iloc[-1])  if 'market_regime' in df.columns else 0.0

        decision, confidence = self.ensemble_engine.make_decision(
            combined_prob,
            volatility=latest['volatility'],
            lstm_prob=lstm_prob,
            tree_prob=tree_prob,
            entropy=entropy,
            vol_regime=vol_regime,
            market_regime=market_regime
        )
        
        # Regime/RL filter
        risk_approved    = self.risk_engine.should_trade()
        regime_approved  = self.q_agent.should_execute(
            confidence, vol_zscore, buy_pressure, autocorr
        ) if hasattr(self.q_agent, 'should_execute') else True
        
        final_decision = decision if (regime_approved and risk_approved) else "SKIP"
        
        self.prediction_count += 1
        self._last_state = {
            'confidence':    confidence,
            'vol_zscore':    vol_zscore,
            'buy_pressure':  buy_pressure,
            'autocorr':      autocorr,
            'entropy':       entropy,
            'vol_regime':    vol_regime,
            'market_regime': market_regime
        }
        
        # Position sizing
        kelly_pct = self.risk_engine.kelly_fraction()
        
        return {
            'tick_number':      self.tick_count,
            'prediction_number': self.prediction_count,
            'lstm_prob':        lstm_prob,
            'tree_prob':        tree_prob,
            'combined_prob':    float(combined_prob),
            'ensemble_decision': decision,
            'confidence':       float(confidence),
            'regime_approved':  regime_approved,
            'risk_approved':    risk_approved,
            'final_decision':   final_decision,
            'kelly_pct':        float(kelly_pct),
            'threshold':        self.ensemble_engine.threshold,
            'entropy':          float(entropy) if not np.isnan(entropy) else None,
            'vol_regime':       int(vol_regime),
            'market_regime':    int(market_regime)
        }
    
    def update_with_outcome(self, prediction, actual_direction):
        was_correct = (
            (prediction['ensemble_decision'] == 'BUY'  and actual_direction == 1) or
            (prediction['ensemble_decision'] == 'SELL' and actual_direction == 0)
        )
        
        self.ensemble_engine.log_prediction(
            prediction['combined_prob'],
            prediction['ensemble_decision'],
            actual_direction
        )
        
        if prediction['final_decision'] != 'SKIP':
            self.risk_engine.update(1 if was_correct else 0)
            
            # Feed regime filter with real outcome
            if hasattr(self.q_agent, 'record_outcome') and hasattr(self, '_last_state'):
                s = self._last_state
                self.q_agent.record_outcome(
                    s['confidence'], s['vol_zscore'],
                    s['buy_pressure'], s['autocorr'],
                    was_correct
                )
        
        # Log performance
        self.performance_log.append({
            'tick': self.tick_count,
            'correct': was_correct,
            'executed': prediction['final_decision'] != 'SKIP'
        })
        
        # Check if retraining needed
        self._check_retrain()
    
    def _check_retrain(self):
        """
        Check if retraining is needed
        
        Triggers:
        - Every N ticks
        - OR accuracy drop
        """
        ticks_since_retrain = self.tick_count - self.last_retrain_tick
        
        # Trigger 1: Tick interval
        if ticks_since_retrain >= RETRAIN_CONFIG['tick_interval']:
            self.retrain()
            return
        
        # Trigger 2: Accuracy drop
        if RETRAIN_CONFIG['force_retrain_on_accuracy_drop']:
            recent_acc = self.ensemble_engine.get_recent_accuracy(200)
            
            if recent_acc is not None and recent_acc < 0.48:
                print(f"⚠️ Accuracy dropped to {recent_acc:.2%}, triggering retrain")
                self.retrain()
    
    def retrain(self):
        """
        Retrain models on latest data
        
        Uses incremental/warm restart approach
        """
        print(f"🔄 Retraining at tick {self.tick_count}...")
        start_time = time.time()
        
        # Get latest data
        df = self.data_engine.buffer.get_dataframe()
        df = self.feature_engine.transform(df)
        
        X, y = self.data_engine.create_sequences(df)
        
        if len(X) < 100:
            print("  ⚠️ Insufficient data for retrain, skipping")
            return
        
        # Use only recent data for retraining (last 80%)
        train_size = int(len(X) * 0.8)
        X_train = X[-train_size:]
        y_train = y[-train_size:]
        
        # Retrain LSTM (few epochs only)
        self.lstm_engine.train(X_train, y_train, verbose=0)
        
        # Retrain Tree
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        self.tree_engine.train(X_train_flat, y_train)
        
        # Decay RL exploration (if applicable)
        if hasattr(self.q_agent, 'decay_epsilon'):
            self.q_agent.decay_epsilon()
        
        # Update adaptive threshold
        recent_acc = self.ensemble_engine.get_recent_accuracy(200)
        if recent_acc is not None:
            self.ensemble_engine.update_threshold(recent_acc)
        
        self.last_retrain_tick = self.tick_count
        
        elapsed = time.time() - start_time
        print(f"✅ Retrain complete in {elapsed:.2f}s")
    
    def get_status(self) -> dict:
        """
        Get comprehensive engine status
        
        Returns:
            Dict with all subsystem stats
        """
        return {
            'engine': {
                'is_trained': self.is_trained,
                'tick_count': self.tick_count,
                'prediction_count': self.prediction_count,
                'last_retrain_tick': self.last_retrain_tick
            },
            'ensemble': self.ensemble_engine.get_statistics(),
            'rl': self.q_agent.get_statistics(),
            'risk': self.risk_engine.get_statistics()
        }
    
    def save_models(self, base_path: str):
        """
        Save all models
        
        Args:
            base_path: Directory to save models
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        
        self.lstm_engine.save(f"{base_path}/lstm_model.h5")
        self.tree_engine.save(f"{base_path}/tree_model.pkl")
        self.q_agent.save(f"{base_path}/q_table.pkl")
        
        print(f"✅ All models saved to {base_path}")
    
    def load_models(self, base_path: str):
        """
        Load all models
        
        Args:
            base_path: Directory containing models
        """
        self.lstm_engine.load(f"{base_path}/lstm_model.h5")
        self.tree_engine.load(f"{base_path}/tree_model.pkl")
        self.q_agent.load(f"{base_path}/q_table.pkl")
        
        self.is_trained = True
        print(f"✅ All models loaded from {base_path}")
