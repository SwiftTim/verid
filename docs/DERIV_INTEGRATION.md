# 🎯 DERIV API INTEGRATION GUIDE

## ✅ Your Configuration

- **API Key**: `V35FbErHFzWjhj5`
- **Symbol**: `R_100` (Volatility 100 Index)
- **Tick Rate**: ~1.5 ticks/second
- **Buffer Size**: 15,000 ticks (~4 hours)
- **Retrain Interval**: 2,000 ticks (~33 minutes)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd /home/tim/Downloads/2026/der

# Core dependencies
pip install -r requirements.txt

# Backend dependencies
pip install -r backend/requirements.txt
```

### Step 2: Test Deriv Connection

```bash
# Test WebSocket connection (collects 100 ticks)
python run_backend.py
```

Expected output:
```
✅ Connected to R_100
📊 Tick #1: Quote: 1234.56
📊 Tick #2: Quote: 1234.58
...
✅ TEST SUCCESSFUL!
📊 Total ticks: 100
⚡ Average rate: 1.52 ticks/second
```

### Step 3: Start Full Backend

```bash
# Start FastAPI backend with auto-reload
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Access:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws

---

## 📊 Available Symbols

Your API key works with all Deriv synthetic indices:

| Symbol | Name | Tick Rate | Volatility |
|--------|------|-----------|------------|
| **R_100** | Volatility 100 | ~1.5/sec | High |
| R_75 | Volatility 75 | ~1.3/sec | Medium-High |
| R_50 | Volatility 50 | ~2.0/sec | Medium |
| R_25 | Volatility 25 | ~4.0/sec | Low |
| R_10 | Volatility 10 | ~10/sec | Very Low |

**Recommended**: Start with `R_100` (already configured)

---

## 🔌 Backend API Endpoints

### REST API

```bash
# Get system status
curl http://localhost:8000/api/status

# Get recent ticks
curl http://localhost:8000/api/ticks?limit=10

# Get prediction (TODO: integrate with Colab)
curl -X POST http://localhost:8000/api/predict

# Start/stop tick stream
curl -X POST http://localhost:8000/api/start
curl -X POST http://localhost:8000/api/stop
```

### WebSocket (Real-time)

```javascript
// Connect from frontend
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'tick') {
    console.log('New tick:', data.data.quote);
  }
  
  if (data.type === 'prediction') {
    console.log('Prediction:', data.data.decision);
  }
};
```

---

## 🧠 Integration with Core Engine

### Option 1: Direct Integration (Local)

```python
# backend/deriv_websocket.py (already has example)

from core import HybridEngine

engine = HybridEngine(verbose=True)

async def process_tick(tick):
    # Add to engine
    engine.add_tick(tick)
    
    # Train if needed
    if not engine.is_trained and engine.tick_count >= 500:
        results = engine.initial_train()
        print(f"✅ Trained: {results['lstm_accuracy']:.2%}")
    
    # Predict
    if engine.is_trained:
        prediction = engine.predict_next_tick()
        
        if prediction['final_decision'] != 'SKIP':
            print(f"🎯 {prediction['final_decision']} "
                  f"({prediction['confidence']:.2%})")

# Use in client
client = DerivTickStream(
    app_id='V35FbErHFzWjhj5',
    symbol='R_100',
    on_tick=process_tick
)
```

### Option 2: Remote Integration (Colab)

```python
# backend/main.py

import httpx

# Colab API URL (from ngrok)
COLAB_URL = "https://xxxx.ngrok.io"

async def predict_via_colab(ticks):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{COLAB_URL}/predict",
            json={"ticks": ticks}
        )
        return response.json()

# In your tick handler
async def handle_tick(tick):
    tick_buffer.append(tick)
    
    # Send batch to Colab every 10 ticks
    if len(tick_buffer) >= 10:
        prediction = await predict_via_colab(tick_buffer)
        print(f"Prediction: {prediction}")
        tick_buffer.clear()
```

---

## 📈 Performance Expectations

### With R_100 (~1.5 ticks/sec)

| Metric | Value | Time |
|--------|-------|------|
| Initial training | 500 ticks | ~5.5 minutes |
| Retrain interval | 2000 ticks | ~22 minutes |
| Buffer fills | 15000 ticks | ~2.8 hours |
| Predictions/hour | ~5400 | (1.5/sec × 3600) |

### Accuracy Targets

- **Expected**: 51-54% (on executed trades)
- **Trade frequency**: 40-60% of signals
- **Retraining**: Adapts every ~22 minutes

---

## 🔧 Configuration Tuning

### For Different Tick Rates

If you switch symbols, adjust in `core/config.py`:

```python
# For R_50 (~2 ticks/sec)
BUFFER_SIZE = 20000  # ~2.8 hours
RETRAIN_CONFIG['tick_interval'] = 3000  # ~25 minutes

# For R_25 (~4 ticks/sec)
BUFFER_SIZE = 30000  # ~2.1 hours
RETRAIN_CONFIG['tick_interval'] = 5000  # ~21 minutes
```

### For Better Performance

```python
# More aggressive filtering
INITIAL_THRESHOLD = 0.60  # Higher = fewer trades

# Faster retraining
RETRAIN_CONFIG['tick_interval'] = 1500  # ~17 minutes

# Larger buffer (more history)
BUFFER_SIZE = 20000  # ~3.7 hours
```

---

## 🛠️ Troubleshooting

### "Connection refused"

**Solution**: Check internet connection and API key

```bash
# Test connection manually
python run_backend.py
```

### "No ticks received"

**Possible causes**:
1. Symbol not available (try R_100, R_50)
2. API key invalid
3. Network firewall blocking WebSocket

### "Ticks too slow/fast"

**Solution**: Adjust symbol

```python
# In core/config.py or backend
DERIV_CONFIG['symbol'] = 'R_50'  # Faster
# or
DERIV_CONFIG['symbol'] = 'R_75'  # Slower
```

---

## 📊 Monitoring

### Check Backend Status

```bash
# API status
curl http://localhost:8000/api/status

# Response:
{
  "deriv_connected": true,
  "tick_count": 1523,
  "ticks_per_second": 1.48,
  "prediction_count": 45
}
```

### Check Deriv Connection

```python
# In Python
from backend.deriv_websocket import DerivTickStream

client = DerivTickStream(app_id='V35FbErHFzWjhj5', symbol='R_100')
await client.connect()

stats = client.get_statistics()
print(stats)
# {
#   'is_connected': True,
#   'tick_count': 1523,
#   'ticks_per_second': 1.48,
#   'uptime': 1028.5
# }
```

---

## 🚀 Production Deployment

### Environment Variables

```bash
# .env file
DERIV_API_KEY=V35FbErHFzWjhj5
DERIV_SYMBOL=R_100
COLAB_API_URL=https://xxxx.ngrok.io
```

### Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN pip install -r backend/requirements.txt

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🎯 Next Steps

1. ✅ **Test connection**: `python run_backend.py`
2. ✅ **Start backend**: `uvicorn backend.main:app --reload`
3. ✅ **Set up Colab**: Follow `docs/COLAB_INTEGRATION.md`
4. ✅ **Build frontend**: Create dashboard to display predictions
5. ✅ **Go live**: Deploy and monitor

---

## 📚 Additional Resources

- **Deriv API Docs**: https://api.deriv.com
- **WebSocket Guide**: https://api.deriv.com/websockets
- **Synthetic Indices**: https://deriv.com/markets/synthetic

---

**🎉 You're all set to start receiving live market data!**

Run `python run_backend.py` to test your connection now.
