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
    
    def build(self, input_shape):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, LSTM, GRU, Dense, Dropout, 
            Conv1D, BatchNormalization, Add, 
            GlobalAveragePooling1D, Concatenate
        )
        
        inputs = Input(shape=input_shape)
        
        # --- Branch 1: LSTM (temporal memory) ---
        x1 = LSTM(64, return_sequences=True)(inputs)
        x1 = Dropout(0.3)(x1)
        x1 = LSTM(32, return_sequences=False)(x1)
        x1 = Dropout(0.2)(x1)
        
        # --- Branch 2: GRU (faster, catches short patterns) ---
        x2 = GRU(32, return_sequences=False)(inputs)
        x2 = Dropout(0.2)(x2)
        
        # --- Branch 3: 1D CNN (local pattern detector) ---
        x3 = Conv1D(32, kernel_size=3, padding='causal', activation='relu')(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(16, kernel_size=3, padding='causal', activation='relu')(x3)
        x3 = GlobalAveragePooling1D()(x3)
        
        # --- Merge all branches ---
        merged = Concatenate()([x1, x2, x3])
        out = Dense(32, activation='relu')(merged)
        out = Dropout(0.2)(out)
        out = Dense(1, activation='sigmoid')(out)
        
        model = Model(inputs, out)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=0):
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        if self.model is None:
            # Auto-build if not already built
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build(input_shape)
        
        # Balance UP vs DOWN classes
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                          patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5)
        ]
        
        val_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=15,           # more epochs, early stopping handles overfit
            batch_size=128,
            validation_data=val_data,
            validation_split=0.1 if val_data is None else 0.0,
            callbacks=callbacks,
            class_weight=class_weight,
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
