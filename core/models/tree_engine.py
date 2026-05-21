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
        
        # CatBoost (Handles tabular data very well)
        try:
            from catboost import CatBoostClassifier
            self.models['catboost'] = CatBoostClassifier(
                iterations=200,
                depth=5,
                learning_rate=0.03,
                l2_leaf_reg=5,
                random_strength=1,
                verbose=False,
                random_state=42,
                thread_count=-1
            )
            self.weights['catboost'] = 0.4
        except ImportError:
            print("⚠️ CatBoost not installed: pip install catboost")

        # Random Forest (robust baseline)
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.weights['rf'] = 0.1
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def train(self, X_train, y_train):
        if not self.models:
            self.build()
        
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        # 1. Walk-forward validation to get baseline accuracy
        for name, model in self.models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]
                model.fit(X_t, y_t)
                preds = model.predict(X_v)
                scores.append(accuracy_score(y_v, preds))
            results[name] = np.mean(scores)

        # 2. SHAP Feature Selection (using best model from CV)
        try:
            import shap
            best_name = max(results, key=results.get)
            best_model = self.models[best_name]
            explainer = shap.TreeExplainer(best_model)
            shap_sample = X_train[:300] if len(X_train) > 300 else X_train
            shap_values = explainer.shap_values(shap_sample)
            if isinstance(shap_values, list): shap_values = shap_values[1]
            importance = np.abs(shap_values).mean(0)
            self.top_features_idx = np.argsort(importance)[-25:]
            print(f"   SHAP: Selected top 25 features based on {best_name}")
        except Exception as e:
            print(f"⚠️ SHAP selection skipped: {e}")
            self.top_features_idx = None

        # 3. Final fit on all data (using SHAP features if available)
        for name, model in self.models.items():
            X_final = X_train[:, self.top_features_idx] if self.top_features_idx is not None else X_train
            model.fit(X_final, y_train)
            print(f"   {name} final fit complete (CV Acc: {results[name]:.2%})")

        self.is_trained = True
        return {'train_accuracy': np.mean(list(results.values())), 
                'model_accuracies': results}
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Not trained")
        
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        # SHAP selection filter
        if getattr(self, 'top_features_idx', None) is not None:
            X = X[:, self.top_features_idx]

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
