"""
Configuration for Deriv Hybrid Predictor
Optimized for single-tick prediction with live retraining
"""

# ==================== DATA CONFIGURATION ====================
# Optimized for ~1 tick/second markets (Deriv synthetic indices)
BUFFER_SIZE = 15000  # Rolling window size (ticks) - ~4 hours of data
SEQUENCE_LENGTH = 20  # Short memory for fast adaptation
TRAIN_TEST_SPLIT = 0.8  # Time-based split ratio

# ==================== FEATURE ENGINEERING ====================
ROLLING_WINDOWS = {
    'mean': [3, 5],
    'std': [5, 10],
    'momentum': [3, 5]
}

# Keep features minimal to prevent overfitting
MAX_FEATURES = 15

# ==================== MODEL CONFIGURATION ====================

# LSTM (Shallow for fast adaptation)
LSTM_CONFIG = {
    'units_layer1': 32,
    'dropout': 0.2,
    'epochs': 5,  # Short training for live retraining
    'batch_size': 64,
    'validation_split': 0.1,
    'optimizer': 'adam',
    'loss': 'binary_crossentropy'
}

# Decision Tree (Depth-limited to prevent overfitting)
TREE_CONFIG = {
    'max_depth': 5,
    'min_samples_split': 50,
    'min_samples_leaf': 20
}

# ==================== ENSEMBLE CONFIGURATION ====================
ENSEMBLE_WEIGHTS = {
    'lstm': 0.5,
    'tree': 0.5
}

# Adaptive threshold (starts conservative)
INITIAL_THRESHOLD = 0.55
THRESHOLD_ADJUST_STEP = 0.01
MIN_THRESHOLD = 0.52
MAX_THRESHOLD = 0.65

# ==================== REINFORCEMENT LEARNING ====================
RL_CONFIG = {
    'state_size': 4,  # [confidence, win_rate, volatility, streak]
    'action_size': 2,  # [SKIP, EXECUTE]
    'learning_rate': 0.1,
    'gamma': 0.9,  # Discount factor
    'epsilon': 0.1,  # Exploration rate
    'epsilon_decay': 0.995,
    'epsilon_min': 0.05
}

# ==================== RISK MANAGEMENT ====================
RISK_CONFIG = {
    'max_drawdown': 0.10,  # 10% max drawdown
    'min_accuracy_200': 0.49,  # Shutdown if accuracy < 49%
    'min_accuracy_50': 0.47,  # Warning threshold
    'max_consecutive_losses': 10
}

# ==================== RETRAINING STRATEGY ====================
# Optimized for ~1 tick/second (retrains every ~30-40 minutes)
RETRAIN_CONFIG = {
    'tick_interval': 2000,  # Retrain every N ticks (~33 minutes at 1 tick/sec)
    'min_buffer_size': 500,  # Minimum data before first train (~8 minutes)
    'force_retrain_on_accuracy_drop': True,
    'accuracy_drop_threshold': 0.05  # Retrain if accuracy drops 5%
}

# ==================== PERFORMANCE MONITORING ====================
MONITOR_CONFIG = {
    'rolling_windows': [50, 200],  # Track accuracy over these windows
    'log_interval': 100,  # Log stats every N predictions
    'save_interval': 500  # Save models every N ticks
}

# ==================== GOOGLE COLAB SETTINGS ====================
COLAB_CONFIG = {
    'use_gpu': True,  # Use GPU for LSTM training
    'drive_mount_path': '/content/drive/MyDrive/deriv_predictor',
    'model_save_path': 'models/',
    'qtable_save_path': 'qtable/',
    'logs_path': 'logs/',
    'checkpoint_interval': 1000
}

# ==================== DERIV API SETTINGS ====================
# Optimized for Deriv synthetic indices (~1 tick/second)
DERIV_CONFIG = {
    'app_id': 'V35FbErHFzWjhj5',  # Your Deriv API key
    'api_token': None,  # Optional: for authenticated operations
    'symbol': 'R_100',  # Volatility 100 Index (~1.5 ticks/sec)
    # Alternative symbols:
    # 'R_50': ~2 ticks/sec
    # 'R_75': ~1.3 ticks/sec
    # 'R_25': ~4 ticks/sec
    'tick_stream_endpoint': 'wss://ws.binaryws.com/websockets/v3',
    'reconnect_delay': 5,  # Seconds to wait before reconnecting
    'max_reconnect_attempts': 10
}

# ==================== VALIDATION RULES ====================
VALIDATION = {
    'min_ticks_for_prediction': SEQUENCE_LENGTH + 1,
    'max_tick_gap_seconds': 10,  # Alert if tick stream stalls
    'sanity_check_enabled': True
}
