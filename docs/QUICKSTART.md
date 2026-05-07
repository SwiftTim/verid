# 🚀 Quick Start Guide

## Prerequisites

- Python 3.10+
- Google Colab account (free tier works)
- Deriv API credentials (optional for live trading)

---

## 📥 Installation

### Option 1: Google Colab (Recommended)

1. **Open Google Colab**: https://colab.research.google.com

2. **Mount Google Drive**:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Upload Project Files**:
   - Upload the entire `der` folder to `/content/drive/MyDrive/`

4. **Install Dependencies**:
```python
!pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Option 2: Local Installation

```bash
cd /home/tim/Downloads/2026/der
pip install -r requirements.txt
```

---

## 🎯 Basic Usage

### 1. Import Engine

```python
from core import HybridEngine

# Create engine
engine = HybridEngine(verbose=True)
```

### 2. Load Historical Data

```python
# Example: Load from CSV
import pandas as pd

df = pd.read_csv('ticks.csv')
# Expected columns: timestamp, quote, symbol

# Add ticks to engine
for _, row in df.iterrows():
    engine.add_tick({
        'timestamp': row['timestamp'],
        'quote': row['quote'],
        'symbol': row['symbol']
    })

print(f"Loaded {engine.tick_count} ticks")
```

### 3. Train Models

```python
# Initial training (requires 500+ ticks)
results = engine.initial_train()

print(f"LSTM Accuracy: {results['lstm_accuracy']:.2%}")
print(f"Tree Accuracy: {results['tree_accuracy']:.2%}")
```

### 4. Make Predictions

```python
# Predict next tick
prediction = engine.predict_next_tick()

if prediction:
    print(f"Decision: {prediction['final_decision']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Combined Probability: {prediction['combined_prob']:.2%}")
```

### 5. Update with Actual Outcome

```python
# After next tick arrives
actual_direction = 1  # 1=UP, 0=DOWN

engine.update_with_outcome(prediction, actual_direction)
```

### 6. Save Models

```python
# Save to Google Drive
engine.save_models('/content/drive/MyDrive/deriv_predictor/models')
```

---

## 📊 Example: Complete Workflow

```python
from core import HybridEngine
import numpy as np

# 1. Initialize
engine = HybridEngine(verbose=True)

# 2. Generate synthetic data (for testing)
def generate_test_data(n=5000):
    ticks = []
    price = 1000.0
    
    for i in range(n):
        price += np.random.randn() * 0.5
        ticks.append({
            'timestamp': 1700000000 + i,
            'quote': price,
            'symbol': 'R_100'
        })
    
    return ticks

ticks = generate_test_data(5000)

# 3. Load data
for tick in ticks:
    engine.add_tick(tick)

# 4. Train
results = engine.initial_train()
print(f"Training complete: {results['lstm_accuracy']:.2%} accuracy")

# 5. Predict
prediction = engine.predict_next_tick()
print(f"Prediction: {prediction['final_decision']}")

# 6. Get status
status = engine.get_status()
print(f"Trade frequency: {status['ensemble']['trade_frequency']:.2%}")
```

---

## 🔄 Live Trading Loop (Conceptual)

```python
import asyncio
import websockets
import json

async def live_trading_loop():
    """
    Conceptual live trading loop
    (Requires Deriv WebSocket integration)
    """
    engine = HybridEngine(verbose=True)
    
    # Load pre-trained models
    engine.load_models('/content/drive/MyDrive/deriv_predictor/models')
    
    # Connect to Deriv WebSocket
    uri = "wss://ws.binaryws.com/websockets/v3?app_id=YOUR_APP_ID"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to tick stream
        await websocket.send(json.dumps({
            "ticks": "R_100",
            "subscribe": 1
        }))
        
        while True:
            # Receive tick
            response = await websocket.recv()
            data = json.loads(response)
            
            if 'tick' in data:
                tick = {
                    'timestamp': data['tick']['epoch'],
                    'quote': data['tick']['quote'],
                    'symbol': data['tick']['symbol']
                }
                
                # Add to engine
                engine.add_tick(tick)
                
                # Predict
                prediction = engine.predict_next_tick()
                
                if prediction and prediction['final_decision'] != 'SKIP':
                    print(f"🎯 Signal: {prediction['final_decision']} "
                          f"(confidence: {prediction['confidence']:.2%})")
                    
                    # TODO: Execute trade via API
                    
                # Check risk status
                if not engine.risk_engine.should_trade():
                    print("🛑 Risk limit reached, stopping")
                    break

# Run (in async environment)
# await live_trading_loop()
```

---

## 📈 Monitoring

### Check Engine Status

```python
status = engine.get_status()

print("Engine Status:")
print(f"  Ticks: {status['engine']['tick_count']}")
print(f"  Predictions: {status['engine']['prediction_count']}")
print(f"  Accuracy (200): {status['ensemble'].get('accuracy_200', 'N/A')}")
print(f"  Trade frequency: {status['ensemble']['trade_frequency']:.2%}")
print(f"  Threshold: {status['ensemble']['current_threshold']:.3f}")
print(f"  Risk score: {engine.risk_engine.get_risk_score():.2%}")
```

### Check Risk Status

```python
risk_stats = engine.risk_engine.get_statistics()

print("Risk Status:")
print(f"  Active: {risk_stats['is_active']}")
print(f"  Drawdown: {risk_stats['drawdown']:.2%}")
print(f"  Win rate: {risk_stats['win_rate_all']:.2%}")
print(f"  Consecutive losses: {risk_stats['consecutive_losses']}")
```

---

## 🛠️ Troubleshooting

### Issue: "Insufficient data for training"

**Solution**: Ensure you have at least 500 ticks before calling `initial_train()`.

```python
if engine.tick_count >= 500:
    engine.initial_train()
else:
    print(f"Need {500 - engine.tick_count} more ticks")
```

### Issue: "Model not trained"

**Solution**: Call `initial_train()` before `predict_next_tick()`.

```python
if not engine.is_trained:
    engine.initial_train()
```

### Issue: TensorFlow not found

**Solution**: Install TensorFlow.

```python
!pip install tensorflow
```

### Issue: All predictions are "SKIP"

**Reason**: Adaptive threshold is too high (low confidence signals).

**Check**:
```python
print(f"Current threshold: {engine.ensemble_engine.threshold}")
```

**Solution**: This is normal if accuracy is low. The system is protecting you.

---

## 📚 Next Steps

1. **Read Architecture**: See `docs/ARCHITECTURE.md`
2. **Colab Training**: See `colab/TRAINING_GUIDE.md`
3. **Backend Integration**: Build API to connect to Deriv
4. **Frontend Dashboard**: Visualize predictions and performance

---

## ⚠️ Important Notes

1. **Synthetic indices are near-random**: Expect 51-54% accuracy max
2. **Edge comes from filtering**: Not all signals should be traded
3. **Risk management is critical**: Respect drawdown limits
4. **Retrain regularly**: Engine auto-retrains every 1000 ticks
5. **Save models frequently**: Use Google Drive for persistence

---

## 🆘 Support

For issues or questions:
1. Check `docs/ARCHITECTURE.md` for detailed explanations
2. Review `colab/TRAINING_GUIDE.md` for examples
3. Examine the code comments in `core/` modules

---

**Happy Trading! 🚀**

*(Remember: Trading involves risk. This is for educational purposes.)*
