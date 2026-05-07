"""
Ensemble Engine: Probability Fusion + Adaptive Thresholding
Combines LSTM and Tree predictions with confidence filtering
"""

import numpy as np
from typing import Tuple, Optional
from ..config import ENSEMBLE_WEIGHTS, INITIAL_THRESHOLD, THRESHOLD_ADJUST_STEP, MIN_THRESHOLD, MAX_THRESHOLD


class EnsembleEngine:
    """
    Combines predictions from multiple models
    
    Key Features:
    - Weighted probability fusion
    - Adaptive confidence thresholding
    - Trade filtering (SKIP low-confidence signals)
    
    Philosophy:
    Edge comes from FILTERING, not just prediction
    """
    
    def __init__(self):
        from sklearn.calibration import CalibratedClassifierCV
        self.lstm_weight = ENSEMBLE_WEIGHTS['lstm']
        self.tree_weight = ENSEMBLE_WEIGHTS['tree']
        self.threshold = INITIAL_THRESHOLD
        self.prediction_history = []
        self.accuracy_history = []
        self.meta_learner = None
        self.calibration_data = []
        self.MAX_THRESHOLD = MAX_THRESHOLD
    
    def combine_probabilities(
        self, 
        lstm_prob: float, 
        tree_prob: float
    ) -> float:
        """
        Weighted fusion of probabilities
        """
        combined = (
            self.lstm_weight * lstm_prob + 
            self.tree_weight * tree_prob
        )
        return combined

    def calibrate_and_stack(self, lstm_probs, tree_probs, y_true):
        """
        Train a meta-learner (logistic regression) on top of base model outputs.
        """
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        X_meta = np.column_stack([lstm_probs, tree_probs,
                                   lstm_probs * tree_probs,           # interaction
                                   np.abs(lstm_probs - tree_probs)])  # disagreement
        
        self.meta_learner = LogisticRegression(C=0.1, random_state=42)
        self.meta_learner.fit(X_meta, y_true)
        print("✅ Meta-learner (stacking) trained")

    def combine_with_meta(self, lstm_prob, tree_prob):
        """Use meta-learner if available, else fall back to weighted average."""
        if self.meta_learner is not None:
            import numpy as np
            X_meta = np.array([[lstm_prob, tree_prob,
                                 lstm_prob * tree_prob,
                                 abs(lstm_prob - tree_prob)]])
            return float(self.meta_learner.predict_proba(X_meta)[0, 1])
        return self.combine_probabilities(lstm_prob, tree_prob)
    
    def combine_batch(
        self,
        lstm_probs: np.ndarray,
        tree_probs: np.ndarray
    ) -> np.ndarray:
        """
        Batch probability fusion
        
        Args:
            lstm_probs: Array of LSTM probabilities
            tree_probs: Array of tree probabilities
        
        Returns:
            Array of combined probabilities
        """
        return (
            self.lstm_weight * lstm_probs + 
            self.tree_weight * tree_probs
        )
    
    def make_decision(self, combined_prob, volatility=None, threshold=None,
                      lstm_prob=None, tree_prob=None,
                      entropy=None, vol_regime=None, market_regime=None):
        """
        Enhanced decision with:
        - Model disagreement filter  (Phase 1)
        - Entropy-based SKIP filter  (Phase 1)
        - Minimum confidence gate    (Phase 1)
        - Regime-based dynamic weights (Phase 1)
        - Volatility-aware threshold widening
        """
        if threshold is None:
            threshold = self.threshold

        # ── 1. DYNAMIC WEIGHTS based on market regime ──────────────────
        if market_regime is not None:
            if market_regime == 1:   # TREND → LSTM is better
                effective_lstm = 0.70
                effective_tree = 0.30
            elif market_regime == 0: # RANGE → Tree is better
                effective_lstm = 0.40
                effective_tree = 0.60
            else:                    # HIGH VOL → balanced but conservative
                effective_lstm = 0.50
                effective_tree = 0.50
            # Recompute combined_prob with regime weights
            if lstm_prob is not None and tree_prob is not None:
                combined_prob = effective_lstm * lstm_prob + effective_tree * tree_prob

        # ── 2. VOLATILITY THRESHOLD WIDENING ───────────────────────────
        if volatility is not None and volatility > 1.5:
            threshold = min(threshold + 0.03, self.MAX_THRESHOLD)

        # ── 3. MODEL DISAGREEMENT FILTER ───────────────────────────────
        if lstm_prob is not None and tree_prob is not None:
            disagreement = abs(lstm_prob - tree_prob)
            if disagreement > 0.20:   # >20% disagreement → unreliable
                return "SKIP", 0.0

        # ── 4. ENTROPY FILTER ──────────────────────────────────────────
        if entropy is not None and not np.isnan(entropy):
            if entropy > 2.3:         # market is too random → skip
                return "SKIP", 0.0

        # ── 5. VOL REGIME FILTER ───────────────────────────────────────
        if vol_regime is not None and vol_regime >= 2:
            return "SKIP", 0.0        # high volatility regime → skip

        confidence = abs(combined_prob - 0.5)

        # ── 6. MINIMUM CONFIDENCE GATE ─────────────────────────────────
        if confidence < 0.06:
            return "SKIP", confidence

        # ── 7. FINAL DECISION ──────────────────────────────────────────
        if combined_prob > threshold:
            decision = "BUY"
        elif combined_prob < (1 - threshold):
            decision = "SELL"
        else:
            decision = "SKIP"

        return decision, confidence

    
    def make_batch_decisions(
        self,
        combined_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch decision making
        
        Returns:
            Tuple of (decisions, confidences)
            decisions: Array of 0 (SKIP), 1 (BUY), -1 (SELL)
            confidences: Array of confidence scores
        """
        decisions = np.zeros(len(combined_probs))
        confidences = np.abs(combined_probs - 0.5)
        
        # BUY signals
        decisions[combined_probs > self.threshold] = 1
        
        # SELL signals
        decisions[combined_probs < (1 - self.threshold)] = -1
        
        # Rest are SKIP (already 0)
        
        return decisions, confidences
    
    def update_threshold(self, recent_accuracy: float):
        """
        Adapt threshold based on recent performance
        
        Logic:
        - If accuracy is low → increase threshold (trade less)
        - If accuracy is high → decrease threshold (trade more)
        
        Args:
            recent_accuracy: Accuracy over recent window (0-1)
        """
        if recent_accuracy < 0.52:
            # Performance poor → be more conservative
            self.threshold = min(
                self.threshold + THRESHOLD_ADJUST_STEP,
                MAX_THRESHOLD
            )
        elif recent_accuracy > 0.60:
            # Performance good → can be more aggressive
            self.threshold = max(
                self.threshold - THRESHOLD_ADJUST_STEP,
                MIN_THRESHOLD
            )
        
        # Track history
        self.accuracy_history.append(recent_accuracy)
    
    def log_prediction(
        self,
        combined_prob: float,
        decision: str,
        actual_outcome: Optional[int] = None
    ):
        """
        Log prediction for performance tracking
        
        Args:
            combined_prob: The combined probability
            decision: "BUY", "SELL", or "SKIP"
            actual_outcome: Actual result (1=UP, 0=DOWN) if known
        """
        self.prediction_history.append({
            'probability': combined_prob,
            'decision': decision,
            'threshold': self.threshold,
            'outcome': actual_outcome
        })
    
    def get_recent_accuracy(self, window: int = 200) -> float:
        """
        Calculate accuracy over recent predictions
        
        Args:
            window: Number of recent predictions to consider
        
        Returns:
            Accuracy (0-1) or None if insufficient data
        """
        if len(self.prediction_history) < 10:
            return None
        
        recent = self.prediction_history[-window:]
        
        # Filter only executed trades (not SKIP)
        executed = [p for p in recent if p['decision'] != 'SKIP' and p['outcome'] is not None]
        
        if len(executed) == 0:
            return None
        
        # Calculate accuracy
        correct = sum(
            1 for p in executed
            if (p['decision'] == 'BUY' and p['outcome'] == 1) or
               (p['decision'] == 'SELL' and p['outcome'] == 0)
        )
        
        return correct / len(executed)
    
    def get_trade_frequency(self, window: int = 200) -> float:
        """
        Calculate what % of signals result in trades
        
        Returns:
            Trade frequency (0-1)
        """
        if len(self.prediction_history) < 10:
            return 0.0
        
        recent = self.prediction_history[-window:]
        
        executed = sum(1 for p in recent if p['decision'] != 'SKIP')
        
        return executed / len(recent)
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive ensemble statistics
        
        Returns:
            Dict with performance metrics
        """
        if len(self.prediction_history) == 0:
            return {
                'total_predictions': 0,
                'current_threshold': self.threshold
            }
        
        total = len(self.prediction_history)
        
        # Count decisions
        buy_count = sum(1 for p in self.prediction_history if p['decision'] == 'BUY')
        sell_count = sum(1 for p in self.prediction_history if p['decision'] == 'SELL')
        skip_count = sum(1 for p in self.prediction_history if p['decision'] == 'SKIP')
        
        # Accuracies
        acc_50 = self.get_recent_accuracy(50)
        acc_200 = self.get_recent_accuracy(200)
        
        return {
            'total_predictions': total,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'skip_count': skip_count,
            'trade_frequency': self.get_trade_frequency(),
            'current_threshold': self.threshold,
            'accuracy_50': acc_50,
            'accuracy_200': acc_200,
            'lstm_weight': self.lstm_weight,
            'tree_weight': self.tree_weight
        }
    
    def reset_history(self):
        """Clear prediction history (use when retraining)"""
        self.prediction_history = []
        self.accuracy_history = []
    
    def adjust_weights(self, lstm_weight: float, tree_weight: float):
        """
        Manually adjust model weights
        
        Args:
            lstm_weight: Weight for LSTM (0-1)
            tree_weight: Weight for tree (0-1)
        
        Note: Weights should sum to 1.0
        """
        total = lstm_weight + tree_weight
        
        if total == 0:
            raise ValueError("Weights cannot both be zero")
        
        # Normalize
        self.lstm_weight = lstm_weight / total
        self.tree_weight = tree_weight / total
