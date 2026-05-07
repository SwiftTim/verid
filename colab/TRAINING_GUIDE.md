# 🚀 Deriv Hybrid Predictor - Google Colab Training Notebook

This notebook demonstrates how to:
1. Set up the environment
2. Load historical tick data
3. Train the hybrid prediction engine
4. Evaluate performance
5. Save models to Google Drive

## 📦 Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create project directory
!mkdir -p /content/drive/MyDrive/deriv_predictor
%cd /content/drive/MyDrive/deriv_predictor

# Clone repository (or upload files)
# !git clone <your-repo-url> .

# Install dependencies
!pip install -q tensorflow scikit-learn pandas numpy matplotlib seaborn
```

## 📥 Load Core Engine

```python
import sys
sys.path.append('/content/drive/MyDrive/deriv_predictor')

from core import HybridEngine
import pandas as pd
import numpy as np
```

## 📊 Load Historical Data

```python
# Example: Load tick data from CSV
# Format: timestamp, quote, symbol
df_ticks = pd.read_csv('historical_ticks.csv')

# Or generate synthetic data for testing
def generate_synthetic_ticks(n=10000):
    """Generate synthetic tick data for testing"""
    np.random.seed(42)
    
    ticks = []
    base_price = 1000.0
    
    for i in range(n):
        # Random walk with slight drift
        change = np.random.randn() * 0.5
        base_price += change
        
        ticks.append({
            'timestamp': 1700000000 + i,
            'quote': base_price,
            'symbol': 'R_100'
        })
    
    return ticks

# Generate test data
ticks = generate_synthetic_ticks(10000)
print(f"Generated {len(ticks)} ticks")
```

## 🧠 Initialize Engine

```python
# Create engine
engine = HybridEngine(verbose=True)

# Add historical ticks
for tick in ticks:
    engine.add_tick(tick)

print(f"✅ Loaded {engine.tick_count} ticks")
```

## 🏋️ Train Models

```python
# Initial training
results = engine.initial_train()

print("\n📊 Training Results:")
print(f"  Training time: {results['training_time']:.2f}s")
print(f"  LSTM accuracy: {results['lstm_accuracy']:.2%}")
print(f"  Tree accuracy: {results['tree_accuracy']:.2%}")
```

## 🔮 Make Predictions

```python
# Predict next tick
prediction = engine.predict_next_tick()

if prediction:
    print("\n🎯 Prediction:")
    print(f"  LSTM probability: {prediction['lstm_prob']:.2%}")
    print(f"  Tree probability: {prediction['tree_prob']:.2%}")
    print(f"  Combined: {prediction['combined_prob']:.2%}")
    print(f"  Decision: {prediction['final_decision']}")
    print(f"  Confidence: {prediction['confidence']:.2%}")
```

## 📈 Evaluate Performance

```python
# Get engine status
status = engine.get_status()

print("\n📊 Engine Status:")
print(f"  Total predictions: {status['engine']['prediction_count']}")
print(f"  Ensemble accuracy (200): {status['ensemble'].get('accuracy_200', 'N/A')}")
print(f"  Trade frequency: {status['ensemble']['trade_frequency']:.2%}")
print(f"  Current threshold: {status['ensemble']['current_threshold']:.3f}")
print(f"  Risk score: {engine.risk_engine.get_risk_score():.2%}")
```

## 💾 Save Models

```python
# Save to Google Drive
save_path = '/content/drive/MyDrive/deriv_predictor/models'
engine.save_models(save_path)

print(f"✅ Models saved to {save_path}")
```

## 🔄 Load Models (for later use)

```python
# Load pre-trained models
engine_new = HybridEngine()
engine_new.load_models('/content/drive/MyDrive/deriv_predictor/models')

print("✅ Models loaded successfully")
```

## 📊 Visualize Performance

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot ensemble statistics
stats = engine.ensemble_engine.get_statistics()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Decision distribution
decisions = [stats['buy_count'], stats['sell_count'], stats['skip_count']]
labels = ['BUY', 'SELL', 'SKIP']
axes[0, 0].pie(decisions, labels=labels, autopct='%1.1f%%')
axes[0, 0].set_title('Decision Distribution')

# 2. Accuracy over time
if len(engine.ensemble_engine.accuracy_history) > 0:
    axes[0, 1].plot(engine.ensemble_engine.accuracy_history)
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Random')
    axes[0, 1].set_title('Accuracy Over Time')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

# 3. Threshold adaptation
thresholds = [p['threshold'] for p in engine.ensemble_engine.prediction_history]
if thresholds:
    axes[1, 0].plot(thresholds)
    axes[1, 0].set_title('Adaptive Threshold')
    axes[1, 0].set_ylabel('Threshold')

# 4. Risk metrics
risk_stats = engine.risk_engine.get_statistics()
axes[1, 1].bar(['Drawdown', 'Win Rate'], 
               [risk_stats['drawdown'], risk_stats.get('win_rate_all', 0)])
axes[1, 1].set_title('Risk Metrics')
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/deriv_predictor/performance.png', dpi=300)
plt.show()
```

## 🎯 Backtesting

```python
def backtest(engine, test_ticks):
    """
    Backtest engine on historical data
    
    Args:
        engine: Trained HybridEngine
        test_ticks: List of test ticks
    
    Returns:
        Backtest results dict
    """
    results = []
    
    for i in range(len(test_ticks) - 1):
        # Add tick
        engine.add_tick(test_ticks[i])
        
        # Predict
        pred = engine.predict_next_tick()
        
        if pred and pred['final_decision'] != 'SKIP':
            # Get actual outcome
            actual = 1 if test_ticks[i+1]['quote'] > test_ticks[i]['quote'] else 0
            
            # Check if correct
            correct = (
                (pred['final_decision'] == 'BUY' and actual == 1) or
                (pred['final_decision'] == 'SELL' and actual == 0)
            )
            
            results.append({
                'tick': i,
                'prediction': pred['final_decision'],
                'actual': 'UP' if actual == 1 else 'DOWN',
                'correct': correct,
                'confidence': pred['confidence']
            })
            
            # Update engine
            engine.update_with_outcome(pred, actual)
    
    # Calculate metrics
    accuracy = sum(r['correct'] for r in results) / len(results) if results else 0
    
    return {
        'total_trades': len(results),
        'accuracy': accuracy,
        'results': results
    }

# Run backtest
# backtest_results = backtest(engine, test_ticks)
# print(f"Backtest Accuracy: {backtest_results['accuracy']:.2%}")
```

## 🚨 Important Notes

1. **Data Quality**: Ensure tick data is clean and properly formatted
2. **Retraining**: Engine auto-retrains every 1000 ticks
3. **Risk Management**: Monitor drawdown and accuracy
4. **GPU Usage**: TensorFlow will auto-detect GPU for LSTM training
5. **Persistence**: Always save models to Google Drive

## 📚 Next Steps

1. Connect to live Deriv WebSocket API
2. Implement real-time prediction loop
3. Add logging and monitoring
4. Deploy backend API for execution

---

**⚠️ Disclaimer**: Trading involves risk. This is for educational purposes only.
