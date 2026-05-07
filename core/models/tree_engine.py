import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

class TreeEngine:
    """
    Gradient Boosting ensemble (XGBoost + LightGBM + RandomForest)
    Much stronger than a single Decision Tree
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_trained = False
        self.feature_importance = None
    
    def build(self):
        # Try XGBoost first (best for tabular data)
        try:
            from xgboost import XGBClassifier
            self.models['xgb'] = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )
            self.weights['xgb'] = 0.5
        except ImportError:
            print("⚠️ XGBoost not installed: pip install xgboost")
        
        # LightGBM (fast, handles large data well)
        try:
            from lightgbm import LGBMClassifier
            self.models['lgbm'] = LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.weights['lgbm'] = 0.35
        except ImportError:
            print("⚠️ LightGBM not installed: pip install lightgbm")
        
        # Random Forest (robust baseline, always available)
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.weights['rf'] = 0.15
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def train(self, X_train, y_train):
        if not self.models:
            self.build()
        
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            acc = model.score(X_train, y_train)
            results[name] = acc
            print(f"   {name} train accuracy: {acc:.2%}")
        
        # Feature importance from best model
        if 'xgb' in self.models:
            self.feature_importance = self.models['xgb'].feature_importances_
        elif 'rf' in self.models:
            self.feature_importance = self.models['rf'].feature_importances_
        
        self.is_trained = True
        return {'train_accuracy': np.mean(list(results.values())), 
                'model_accuracies': results}
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Not trained")
        
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        # Weighted average of all models
        combined = np.zeros(len(X))
        for name, model in self.models.items():
            prob = model.predict_proba(X)[:, 1]
            combined += self.weights[name] * prob
        
        return combined
    
    def evaluate(self, X_test, y_test):
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)
        preds = (self.predict_proba(X_test) > 0.5).astype(int)
        return {'accuracy': accuracy_score(y_test, preds)}
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'models': self.models, 'weights': self.weights,
                        'feature_importance': self.feature_importance,
                        'is_trained': self.is_trained}, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.models = data['models']
        self.weights = data['weights']
        self.feature_importance = data['feature_importance']
        self.is_trained = data['is_trained']
