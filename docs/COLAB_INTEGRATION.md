# 🔌 Google Colab Integration Guide

This guide shows how to offload heavy computation to Google Colab while keeping your backend/frontend local.

---

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    LOCAL (Your Server)                   │
├─────────────────────────────────────────────────────────┤
│  • Frontend Dashboard (React/Next.js)                   │
│  • Backend API (FastAPI)                                 │
│  • Tick Stream Ingestion (WebSocket)                    │
│  • Trade Execution Logic                                 │
│  • Database (PostgreSQL)                                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ HTTP/REST API
                  │
┌─────────────────▼───────────────────────────────────────┐
│              GOOGLE COLAB (Cloud GPU)                    │
├─────────────────────────────────────────────────────────┤
│  • Core Prediction Engine                                │
│  • LSTM Training (GPU accelerated)                       │
│  • Model Retraining                                      │
│  • Feature Engineering                                   │
│  • Ensemble Logic                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup Steps

### 1. Prepare Google Colab Notebook

Create a new notebook: `deriv_predictor_api.ipynb`

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install dependencies
!pip install -q fastapi uvicorn pyngrok tensorflow scikit-learn pandas numpy

# Cell 3: Upload core engine
import sys
sys.path.append('/content/drive/MyDrive/deriv_predictor')

from core import HybridEngine
import json

# Cell 4: Initialize engine
engine = HybridEngine(verbose=True)

# Load pre-trained models if available
try:
    engine.load_models('/content/drive/MyDrive/deriv_predictor/models')
    print("✅ Loaded pre-trained models")
except:
    print("⚠️ No pre-trained models found, will train on first request")

# Cell 5: Create FastAPI app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(title="Deriv Predictor API")

class Tick(BaseModel):
    timestamp: int
    quote: float
    symbol: str

class PredictionRequest(BaseModel):
    ticks: List[Tick]
    train_if_needed: bool = True

class PredictionResponse(BaseModel):
    prediction: Optional[Dict]
    status: Dict
    message: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Main prediction endpoint
    """
    try:
        # Add ticks to engine
        for tick in request.ticks:
            engine.add_tick(tick.dict())
        
        # Train if not trained and requested
        if not engine.is_trained and request.train_if_needed:
            if engine.tick_count >= 500:
                results = engine.initial_train()
                return PredictionResponse(
                    prediction=None,
                    status=engine.get_status(),
                    message=f"Training complete: {results['lstm_accuracy']:.2%} accuracy"
                )
            else:
                return PredictionResponse(
                    prediction=None,
                    status=engine.get_status(),
                    message=f"Need {500 - engine.tick_count} more ticks for training"
                )
        
        # Make prediction
        prediction = engine.predict_next_tick()
        
        return PredictionResponse(
            prediction=prediction,
            status=engine.get_status(),
            message="Prediction successful"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
async def update_outcome(prediction_id: int, actual_direction: int):
    """
    Update engine with actual outcome
    """
    # TODO: Implement outcome tracking
    return {"message": "Outcome recorded"}

@app.get("/status")
async def get_status():
    """
    Get engine status
    """
    return engine.get_status()

@app.post("/retrain")
async def force_retrain():
    """
    Force model retraining
    """
    try:
        engine.retrain()
        return {"message": "Retrain successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save")
async def save_models():
    """
    Save models to Google Drive
    """
    try:
        engine.save_models('/content/drive/MyDrive/deriv_predictor/models')
        return {"message": "Models saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Cell 6: Expose via ngrok
from pyngrok import ngrok
import uvicorn
import nest_asyncio

nest_asyncio.apply()

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print(f"🌐 Public URL: {public_url}")

# Run FastAPI
uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🔗 Connect from Your Backend

### Python (FastAPI) Example

```python
# backend/services/colab_predictor.py

import httpx
from typing import List, Dict

class ColabPredictor:
    """
    Client for Google Colab prediction API
    """
    
    def __init__(self, colab_url: str):
        self.base_url = colab_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def predict(self, ticks: List[Dict]) -> Dict:
        """
        Get prediction from Colab
        
        Args:
            ticks: List of tick dicts
        
        Returns:
            Prediction response
        """
        response = await self.client.post(
            f"{self.base_url}/predict",
            json={
                "ticks": ticks,
                "train_if_needed": True
            }
        )
        
        response.raise_for_status()
        return response.json()
    
    async def get_status(self) -> Dict:
        """Get engine status"""
        response = await self.client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    async def force_retrain(self):
        """Force model retraining"""
        response = await self.client.post(f"{self.base_url}/retrain")
        response.raise_for_status()
        return response.json()
    
    async def save_models(self):
        """Save models to Google Drive"""
        response = await self.client.post(f"{self.base_url}/save")
        response.raise_for_status()
        return response.json()

# Usage in your backend
from fastapi import FastAPI

app = FastAPI()

# Initialize Colab client
colab = ColabPredictor(colab_url="https://xxxx.ngrok.io")

@app.post("/api/predict")
async def predict_endpoint(ticks: List[Dict]):
    """
    Your backend endpoint that calls Colab
    """
    result = await colab.predict(ticks)
    
    # Store in database, execute trade, etc.
    
    return result
```

---

## 📊 Data Flow

### 1. Tick Ingestion (Local)

```python
# Your backend receives ticks from Deriv WebSocket
async def on_tick(tick_data):
    # Store in local buffer
    tick_buffer.append(tick_data)
    
    # Send batch to Colab every 10 ticks
    if len(tick_buffer) >= 10:
        prediction = await colab.predict(tick_buffer)
        tick_buffer.clear()
        
        # Use prediction
        if prediction['prediction']['final_decision'] != 'SKIP':
            await execute_trade(prediction)
```

### 2. Prediction (Colab)

```
Local Backend → HTTP POST → Colab API
                              ↓
                         Add ticks to buffer
                              ↓
                         Generate features
                              ↓
                         LSTM + Tree predict
                              ↓
                         Ensemble fusion
                              ↓
                         RL filter
                              ↓
                         Risk check
                              ↓
Local Backend ← JSON Response ← Return prediction
```

### 3. Execution (Local)

```python
# Your backend executes the trade
async def execute_trade(prediction):
    if prediction['prediction']['final_decision'] == 'BUY':
        # Place BUY order via Deriv API
        await deriv_api.buy(...)
    elif prediction['prediction']['final_decision'] == 'SELL':
        # Place SELL order
        await deriv_api.sell(...)
```

---

## ⚡ Performance Optimization

### Batching

Send ticks in batches to reduce HTTP overhead:

```python
# Instead of 1 request per tick
for tick in ticks:
    await colab.predict([tick])  # ❌ Slow

# Batch requests
await colab.predict(ticks)  # ✅ Fast
```

### Caching

Cache predictions locally:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_prediction(tick_hash):
    # Return cached prediction if available
    pass
```

### Async Processing

Use async/await for non-blocking calls:

```python
import asyncio

# Process multiple predictions concurrently
predictions = await asyncio.gather(
    colab.predict(batch1),
    colab.predict(batch2),
    colab.predict(batch3)
)
```

---

## 🔒 Security

### 1. Authentication

Add API key authentication:

```python
# In Colab notebook
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict", dependencies=[Security(verify_api_key)])
async def predict(...):
    ...
```

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(...):
    ...
```

---

## 💾 Persistence

### Auto-save Models

```python
# In Colab notebook
import schedule
import time
import threading

def save_models_job():
    engine.save_models('/content/drive/MyDrive/deriv_predictor/models')
    print("✅ Models auto-saved")

# Save every 30 minutes
schedule.every(30).minutes.do(save_models_job)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

# Run in background
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()
```

---

## 🔄 Retraining Strategy

### Option 1: Scheduled Retraining

```python
# Retrain every 6 hours
schedule.every(6).hours.do(lambda: engine.retrain())
```

### Option 2: On-Demand Retraining

```python
# Your backend triggers retrain
if accuracy < 0.50:
    await colab.force_retrain()
```

### Option 3: Automatic (Built-in)

The engine auto-retrains every 1000 ticks (already implemented).

---

## 📱 Frontend Integration

### React Example

```typescript
// services/predictor.ts

export async function getPrediction(ticks: Tick[]) {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticks })
  });
  
  return response.json();
}

// components/PredictionDisplay.tsx

function PredictionDisplay() {
  const [prediction, setPrediction] = useState(null);
  
  useEffect(() => {
    const interval = setInterval(async () => {
      const result = await getPrediction(latestTicks);
      setPrediction(result.prediction);
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div>
      <h2>Prediction: {prediction?.final_decision}</h2>
      <p>Confidence: {(prediction?.confidence * 100).toFixed(1)}%</p>
    </div>
  );
}
```

---

## ⚠️ Important Notes

1. **Ngrok Free Tier**: URLs change on restart (use paid tier for static URLs)
2. **Colab Timeout**: Free tier disconnects after 12 hours (use Colab Pro)
3. **GPU Availability**: Not always guaranteed on free tier
4. **Network Latency**: Expect 100-500ms per request
5. **Data Privacy**: Don't send sensitive data to Colab

---

## 🚀 Production Deployment

For production, consider:

1. **Google Cloud Run**: Deploy as containerized service
2. **AWS Lambda**: Serverless prediction API
3. **Dedicated GPU Server**: Rent from Vast.ai, RunPod, etc.

---

**You now have a complete separation of concerns:**
- ✅ Heavy ML computation → Google Colab (free GPU)
- ✅ Business logic → Your backend
- ✅ User interface → Your frontend

🎉 **Best of both worlds!**
