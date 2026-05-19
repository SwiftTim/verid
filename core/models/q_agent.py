class RegimeFilter:
    """
    Replaces Q-agent with a regime classifier.
    Detects HIGH-CONFIDENCE regimes where the model actually has edge.
    This is more sample-efficient than Q-learning.
    """
    
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.regime_model = LogisticRegression(C=0.5)
        self.history = []   # (features, was_correct)
        self.is_fitted = False
        self.min_samples = 200
        self.is_trained = False # For compatibility check
    
    def record_outcome(self, confidence, vol_zscore, buy_pressure,
                       autocorr_1, was_correct):
        self.history.append({
            'confidence': confidence,
            'vol_zscore': vol_zscore,
            'buy_pressure': buy_pressure,
            'autocorr': autocorr_1,
            'correct': int(was_correct)
        })
        
        if len(self.history) >= self.min_samples:
            self._fit()
    
    def _fit(self):
        import numpy as np
        import pandas as pd
        df = pd.DataFrame(self.history)
        X = df[['confidence', 'vol_zscore', 'buy_pressure', 'autocorr']].values
        y = df['correct'].values
        
        if y.mean() > 0.1 and y.mean() < 0.9:  # need both classes
            self.regime_model.fit(X, y)
            self.is_fitted = True
            self.is_trained = True
    
    def should_execute(self, confidence, vol_zscore, buy_pressure, autocorr):
        """Returns True if current regime is favorable."""
        if not self.is_fitted:
            return confidence > 0.08  # fallback: confidence threshold
        
        import numpy as np
        X = np.array([[confidence, vol_zscore, buy_pressure, autocorr]])
        prob_correct = self.regime_model.predict_proba(X)[0, 1]
        return prob_correct > 0.55  # only execute if >55% chance of being right

    def decay_epsilon(self):
        """No-op — RegimeFilter doesn't use epsilon. Kept for API compatibility."""
        pass

    def get_statistics(self) -> dict:
        """Return regime filter stats for monitoring."""
        return {
            'type': 'RegimeFilter',
            'is_fitted': self.is_fitted,
            'history_size': len(self.history),
            'min_samples_needed': self.min_samples,
            'favorable_rate': (
                sum(1 for h in self.history if h['correct']) / len(self.history)
                if self.history else 0.0
            )
        }

    def save(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.regime_model, 'history': self.history, 'is_fitted': self.is_fitted}, f)

            
    def load(self, filepath):
        import pickle
        import os
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.regime_model = data['model']
                self.history = data['history']
                self.is_fitted = data['is_fitted']
                self.is_trained = self.is_fitted
