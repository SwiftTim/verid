"""
Decision Tree Engine: Fast Conditional Pattern Detector
Optimized for interpretability and speed
"""

import numpy as np
import pickle
from typing import Optional
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ..config import TREE_CONFIG


class TreeEngine:
    """
    Shallow Decision Tree for conditional logic
    
    Advantages:
    - Fast training (no GPU needed)
    - Interpretable rules
    - Captures non-linear patterns
    - Complements LSTM temporal patterns
    
    Design:
    - Depth-limited (4-6) to prevent overfitting
    - Min samples constraints for stability
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_importance = None
    
    def build(self) -> DecisionTreeClassifier:
        """
        Build decision tree model
        
        Returns:
            Sklearn DecisionTreeClassifier
        """
        model = DecisionTreeClassifier(
            max_depth=TREE_CONFIG['max_depth'],
            min_samples_split=TREE_CONFIG['min_samples_split'],
            min_samples_leaf=TREE_CONFIG['min_samples_leaf'],
            random_state=42,  # For reproducibility
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Train decision tree
        
        Args:
            X_train: (n_samples, n_features) - flattened sequences or features
            y_train: (n_samples,)
        
        Returns:
            Training info dict
        """
        if self.model is None:
            self.build()
        
        # Flatten if LSTM sequences (tree needs 2D input)
        if len(X_train.shape) == 3:
            # Reshape from (n_samples, seq_len, n_features) to (n_samples, seq_len * n_features)
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        self.is_trained = True
        
        # Training accuracy
        train_acc = self.model.score(X_train, y_train)
        
        return {
            'train_accuracy': train_acc,
            'tree_depth': self.model.get_depth(),
            'n_leaves': self.model.get_n_leaves()
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of UP movement
        
        Args:
            X: (n_samples, n_features) or (n_samples, seq_len, n_features)
        
        Returns:
            Array of probabilities for class 1 (UP)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Flatten if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        # Get probabilities for class 1 (UP)
        probs = self.model.predict_proba(X)[:, 1]
        
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class (0=DOWN, 1=UP)
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Flatten if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Returns:
            Dict with metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Flatten if needed
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Get predictions
        preds = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        
        # Confusion matrix
        tp = ((preds == 1) & (y_test == 1)).sum()
        tn = ((preds == 0) & (y_test == 0)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fn = ((preds == 0) & (y_test == 1)).sum()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn),
                'fp': int(fp), 'fn': int(fn)
            }
        }
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> dict:
        """
        Get feature importance scores
        
        Args:
            feature_names: Optional list of feature names
        
        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_dict = dict(zip(feature_names, self.feature_importance))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def get_decision_path(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision path for interpretability
        
        Returns:
            Sparse matrix of decision paths
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Flatten if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.decision_path(X)
    
    def save(self, filepath: str):
        """Save model using pickle"""
        if self.model is None:
            raise ValueError("No model to save")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained
            }, f)
        
        print(f"✅ Tree model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from pickle"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_importance = data['feature_importance']
        self.is_trained = data['is_trained']
        
        print(f"✅ Tree model loaded from {filepath}")
    
    def export_rules(self, feature_names: Optional[list] = None) -> str:
        """
        Export decision tree rules as text
        
        Useful for understanding model logic
        """
        if not self.is_trained:
            return "Model not trained"
        
        from sklearn.tree import export_text
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.model.n_features_in_)]
        
        rules = export_text(self.model, feature_names=feature_names)
        return rules
