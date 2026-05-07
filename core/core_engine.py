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
from .models import LSTMEngine, TreeEngine, EnsembleEngine, QAgent
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
        self.q_agent = QAgent()
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
        X, y = self.data_engine.create_sequences(df)
        
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
            'tree_accuracy': tree_eval['accuracy'],
            'tree_depth': tree_info['tree_depth']
        }
        
        print(f"✅ Initial training complete in {elapsed:.2f}s")
        print(f"   LSTM accuracy: {lstm_eval['accuracy']:.2%}")
        print(f"   Tree accuracy: {tree_eval['accuracy']:.2%}")
        
        return results
    
    def predict_next_tick(self) -> Optional[dict]:
        """
        Predict next tick direction
        
        Returns:
            Dict with prediction details or None if insufficient data
        """
        if not self.is_trained:
            raise ValueError("Engine not trained. Call initial_train() first.")
        
        # 1. Get latest data
        df = self.data_engine.buffer.get_dataframe()
        
        if len(df) < SEQUENCE_LENGTH:
            return None
        
        # 2. Generate features
        df = self.feature_engine.transform(df)
        
        # 3. Prepare sequence for prediction
        X = self.data_engine.prepare_for_prediction(df)
        
        if X is None:
            return None
        
        # 4. Get predictions from both models
        lstm_prob = self.lstm_engine.predict_proba(X)[0]
        
        X_flat = X.reshape(1, -1)
        tree_prob = self.tree_engine.predict_proba(X_flat)[0]
        
        # 5. Ensemble fusion
        combined_prob = self.ensemble_engine.combine_probabilities(lstm_prob, tree_prob)
        decision, confidence = self.ensemble_engine.make_decision(combined_prob)
        
        # 6. Get RL state
        latest_features = self.data_engine.get_latest_features(df)
        recent_acc = self.ensemble_engine.get_recent_accuracy(50)
        
        rl_state = self.q_agent.get_state(
            confidence=confidence,
            win_rate=recent_acc if recent_acc is not None else 0.5,
            volatility=latest_features['volatility'],
            streak=latest_features['streak']
        )
        
        # 7. RL execution decision
        rl_action = self.q_agent.act(rl_state, explore=True)
        
        # 8. Risk check
        risk_approved = self.risk_engine.should_trade()
        
        # 9. Final decision
        final_decision = decision if (rl_action == 1 and risk_approved) else "SKIP"
        
        self.prediction_count += 1
        
        result = {
            'tick_number': self.tick_count,
            'prediction_number': self.prediction_count,
            'lstm_prob': float(lstm_prob),
            'tree_prob': float(tree_prob),
            'combined_prob': float(combined_prob),
            'ensemble_decision': decision,
            'confidence': float(confidence),
            'rl_action': 'EXECUTE' if rl_action == 1 else 'SKIP',
            'risk_approved': risk_approved,
            'final_decision': final_decision,
            'threshold': self.ensemble_engine.threshold
        }
        
        return result
    
    def update_with_outcome(self, prediction: dict, actual_direction: int):
        """
        Update models with actual outcome
        
        Args:
            prediction: Result from predict_next_tick()
            actual_direction: Actual tick direction (1=UP, 0=DOWN)
        """
        # Determine if prediction was correct
        was_correct = (
            (prediction['ensemble_decision'] == 'BUY' and actual_direction == 1) or
            (prediction['ensemble_decision'] == 'SELL' and actual_direction == 0)
        )
        
        # Update ensemble history
        self.ensemble_engine.log_prediction(
            prediction['combined_prob'],
            prediction['ensemble_decision'],
            actual_direction
        )
        
        # Update RL (only if trade was executed)
        if prediction['final_decision'] != 'SKIP':
            reward = 1.0 if was_correct else -1.0
            
            # Get current state (simplified - would need to reconstruct)
            # For now, use dummy next state
            current_state = 0  # Would need to store from prediction
            next_state = 0
            action = 1 if prediction['rl_action'] == 'EXECUTE' else 0
            
            self.q_agent.update(current_state, action, reward, next_state)
            
            # Update risk engine
            self.risk_engine.update(1 if was_correct else 0)
        
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
        
        # Decay RL exploration
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
