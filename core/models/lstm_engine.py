"""
LSTM Engine: Shallow Temporal Pattern Detector
Optimized for fast adaptation with live retraining
"""

import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    KerasModel = keras.Model
except ImportError:
    TF_AVAILABLE = False
    KerasModel = object  # Fallback type
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")

from ..config import LSTM_CONFIG


class LSTMEngine:
    """
    Shallow LSTM for temporal pattern detection
    
    Architecture Philosophy:
    - Small network (32 units) for fast training
    - Single LSTM layer to prevent overfitting
    - Dropout for regularization
    - Binary classification (UP/DOWN)
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.is_trained = False
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM engine")
    
    def build(self, input_shape: Tuple[int, int]) -> KerasModel:
        """
        Build LSTM model
        
        Args:
            input_shape: (sequence_length, n_features)
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Single LSTM layer (shallow = fast + less overfitting)
            LSTM(
                LSTM_CONFIG['units_layer1'],
                input_shape=input_shape,
                return_sequences=False  # Only final output
            ),
            
            # Dropout for regularization
            Dropout(LSTM_CONFIG['dropout']),
            
            # Output layer (sigmoid for binary classification)
            Dense(1, activation='sigmoid')
        ])
        
        # Compile
        model.compile(
            optimizer=LSTM_CONFIG['optimizer'],
            loss=LSTM_CONFIG['loss'],
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 0
    ) -> dict:
        """
        Train LSTM model
        
        Args:
            X_train: (n_samples, sequence_length, n_features)
            y_train: (n_samples,)
            X_val: Optional validation data
            y_val: Optional validation labels
            verbose: 0=silent, 1=progress bar, 2=one line per epoch
        
        Returns:
            Training history dict
        """
        if self.model is None:
            # Auto-build if not already built
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build(input_shape)
        
        # Early stopping to prevent overfitting
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=2,
                restore_best_weights=True
            )
        ]
        
        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        elif LSTM_CONFIG['validation_split'] > 0:
            # Use built-in validation split
            pass
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=LSTM_CONFIG['epochs'],
            batch_size=LSTM_CONFIG['batch_size'],
            validation_data=validation_data,
            validation_split=LSTM_CONFIG['validation_split'] if validation_data is None else 0,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        self.is_trained = True
        
        return self.history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of UP movement
        
        Args:
            X: (n_samples, sequence_length, n_features)
        
        Returns:
            Array of probabilities (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get probabilities
        probs = self.model.predict(X, verbose=0)
        
        return probs.flatten()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class (0=DOWN, 1=UP)
        
        Args:
            X: (n_samples, sequence_length, n_features)
            threshold: Decision threshold
        
        Returns:
            Array of predictions (0 or 1)
        """
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Returns:
            Dict with metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get predictions
        probs = self.predict_proba(X_test)
        preds = (probs > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = (preds == y_test).mean()
        
        # Confusion matrix
        tp = ((preds == 1) & (y_test == 1)).sum()
        tn = ((preds == 0) & (y_test == 0)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fn = ((preds == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn),
                'fp': int(fp), 'fn': int(fn)
            }
        }
    
    def save(self, filepath: str):
        """Save model weights"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"✅ LSTM model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"✅ LSTM model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built"
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    def incremental_update(
        self, 
        X_new: np.ndarray, 
        y_new: np.ndarray,
        epochs: int = 3
    ):
        """
        Incremental training on new data
        
        Used for live retraining without full retrain
        
        Args:
            X_new: New sequences
            y_new: New labels
            epochs: Number of epochs (keep low)
        """
        if not self.is_trained:
            raise ValueError("Model must be initially trained before incremental updates")
        
        self.model.fit(
            X_new, y_new,
            epochs=epochs,
            batch_size=LSTM_CONFIG['batch_size'],
            verbose=0
        )
