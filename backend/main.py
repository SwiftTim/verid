"""
Backend API for Deriv Predictor
Integrates: Deriv WebSocket → Core Engine → Frontend

Run with: uvicorn backend.main:app --reload
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
import os
from datetime import datetime

# Import Deriv WebSocket client
from .deriv_websocket import DerivTickStream
from .colab_client import ColabClient

app = FastAPI(
    title="Deriv Hybrid Predictor API",
    description="Real-time market prediction using hybrid ML models",
    version="1.0.0"
)

# CORS middleware (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURATION
# ============================================================

# Colab API URL (from environment variable or ngrok)
COLAB_URL = os.getenv("COLAB_URL", "https://a8ee-34-87-27-170.ngrok-free.app")

# Auto-predict settings
AUTO_PREDICT_ENABLED = True
AUTO_PREDICT_BATCH_SIZE = 50
MIN_TICKS_FOR_PREDICTION = 100

# ============================================================
# GLOBAL STATE
# ============================================================

# Deriv WebSocket client
deriv_client: Optional[DerivTickStream] = None

# Colab API client
colab_client: Optional[ColabClient] = None

# Tick buffer
tick_buffer: List[Dict] = []

# Prediction buffer
prediction_buffer: List[Dict] = []

# Total ticks processed
total_ticks: int = 0

# WebSocket connections (for real-time updates to frontend)
active_connections: List[WebSocket] = []

# ============================================================
# MODELS
# ============================================================

class TickData(BaseModel):
    timestamp: int
    quote: float
    symbol: str
    ask: Optional[float] = None
    bid: Optional[float] = None

class PredictionResponse(BaseModel):
    prediction: Optional[Dict]
    status: Dict
    timestamp: str

class StatusResponse(BaseModel):
    deriv_connected: bool
    tick_count: int
    ticks_per_second: float
    prediction_count: int
    last_prediction: Optional[Dict]

# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup
    """
    global deriv_client, colab_client
    
    print("🚀 Starting Deriv Predictor API...")
    
    # Initialize Colab client if URL is set
    if COLAB_URL:
        colab_client = ColabClient(COLAB_URL)
        
        # Check if Colab is reachable
        is_healthy = await colab_client.health_check()
        if is_healthy:
            print(f"✅ Colab API connected: {COLAB_URL}")
        else:
            print(f"⚠️ Colab API not reachable: {COLAB_URL}")
            print("   Make sure Colab notebook is running")
    else:
        print("⚠️ COLAB_URL not set - predictions will be mocked")
        print("   Set COLAB_URL to your ngrok URL from Colab")
    
    # Initialize Deriv WebSocket client
    deriv_client = DerivTickStream(
        app_id='1089',
        symbol='1HZ100V',
        on_tick=handle_tick,
        on_error=handle_error,
        on_connect=handle_connect
    )
    
    # Start tick stream in background
    asyncio.create_task(start_tick_stream())
    
    print("✅ API ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown
    """
    global deriv_client
    
    if deriv_client:
        await deriv_client.disconnect()
    
    print("👋 API shutdown complete")

# ============================================================
# DERIV WEBSOCKET HANDLERS
# ============================================================

async def handle_tick(tick: Dict):
    """
    Handle incoming tick from Deriv
    """
    global tick_buffer, total_ticks, prediction_buffer, active_connections, colab_client
    
    # Add to buffer
    tick_buffer.append(tick)
    total_ticks += 1
    
    # Keep buffer size manageable
    if len(tick_buffer) > 2000:
        tick_buffer = tick_buffer[-2000:]
    
    # Broadcast to connected frontends
    if active_connections:
        await broadcast_to_clients({
            'type': 'tick',
            'data': tick,
            'total_ticks': total_ticks
        })
    
    # Only predict when we have enough data AND on batch boundary
    enough_data = len(tick_buffer) >= MIN_TICKS_FOR_PREDICTION
    on_batch    = total_ticks % AUTO_PREDICT_BATCH_SIZE == 0
    
    if AUTO_PREDICT_ENABLED and colab_client and enough_data and on_batch:
        # RUN IN BACKGROUND TO AVOID BLOCKING TICK STREAM
        asyncio.create_task(run_prediction(list(tick_buffer[-200:]), total_ticks))

async def run_prediction(ticks: List[Dict], current_total_ticks: int):
    """
    Background task for Colab prediction to avoid blocking the main tick stream
    """
    global colab_client, prediction_buffer, active_connections
    if not colab_client:
        return
        
    try:
        result = await colab_client.predict(ticks)
        
        if result:
            prediction = result.get('prediction')
            status     = result.get('status', {})
            
            if prediction:
                prediction_buffer.append(prediction)
                if len(prediction_buffer) > 500:
                    prediction_buffer = prediction_buffer[-500:]
                
                # Broadcast to frontends
                if active_connections:
                    await broadcast_to_clients({
                        'type': 'prediction',
                        'data': prediction
                    })
                
                # Log significant predictions
                decision = prediction.get('final_decision', 'SKIP')
                conf     = prediction.get('confidence', 0)
                print(f"{'🎯' if decision != 'SKIP' else '⏩'} {decision} "
                        f"(conf: {conf:.2%}) | ticks: {current_total_ticks}")
            else:
                # No prediction yet (waiting for training)
                tick_count = status.get('tick_count', 0)
                print(f"🧠 AI Learning... {tick_count}/500 ticks | Total seen: {current_total_ticks}")
    except Exception as e:
        print(f"⚠️ Predict error: {e}")
    
    # Log progress
    if len(tick_buffer) % 100 == 0:
        print(f"📊 Buffered {len(tick_buffer)} ticks | "
              f"Predictions: {len(prediction_buffer)}")

async def handle_error(error: Exception):
    """
    Handle Deriv WebSocket errors
    """
    print(f"❌ Deriv Error: {error}")
    
    # Broadcast error to frontends
    if active_connections:
        await broadcast_to_clients({
            'type': 'error',
            'message': str(error)
        })

async def handle_connect(symbol: str):
    """
    Handle Deriv WebSocket connection
    """
    print(f"✅ Connected to Deriv: {symbol}")
    
    # Broadcast connection status
    if active_connections:
        await broadcast_to_clients({
            'type': 'status',
            'connected': True,
            'symbol': symbol
        })

async def start_tick_stream():
    """
    Start Deriv tick stream (runs in background)
    """
    global deriv_client
    
    try:
        await deriv_client.connect()
        await deriv_client.listen()
    except Exception as e:
        print(f"❌ Tick stream error: {e}")

# ============================================================
# WEBSOCKET (Real-time updates to frontend)
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates to frontend
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to Deriv Predictor API'
        })
        
        # Keep connection alive
        while True:
            # Wait for messages from client (ping/pong)
            data = await websocket.receive_text()
            
            if data == 'ping':
                await websocket.send_text('pong')
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("Client disconnected")

async def broadcast_to_clients(message: Dict):
    """
    Broadcast message to all connected WebSocket clients (SAFE ITERATION)
    """
    # Create a copy to iterate safely
    for connection in list(active_connections):
        try:
            await connection.send_json(message)
        except Exception as e:
            print(f"⚠️ WebSocket broadcast error: {e}")
            if connection in active_connections:
                active_connections.remove(connection)
                print("🔌 Removed dead WebSocket connection")

# ============================================================
# REST API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """
    API root endpoint
    """
    return {
        "name": "Deriv Hybrid Predictor API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "status": "/api/status",
            "ticks": "/api/ticks",
            "predict": "/api/predict",
            "websocket": "/ws"
        }
    }

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """
    Get system status
    """
    global deriv_client, tick_buffer, prediction_buffer
    
    stats = deriv_client.get_statistics() if deriv_client else {}
    
    return StatusResponse(
        deriv_connected=stats.get('is_connected', False),
        tick_count=stats.get('tick_count', 0),
        ticks_per_second=stats.get('ticks_per_second', 0.0),
        prediction_count=len(prediction_buffer),
        last_prediction=prediction_buffer[-1] if prediction_buffer else None
    )

@app.get("/api/ticks")
async def get_ticks(limit: int = 100):
    """
    Get recent ticks
    
    Args:
        limit: Number of recent ticks to return
    """
    global tick_buffer
    
    return {
        "count": len(tick_buffer),
        "ticks": tick_buffer[-limit:]
    }

@app.post("/api/predict")
async def predict():
    """
    Get prediction from Colab engine
    """
    global tick_buffer, prediction_buffer, colab_client
    
    if len(tick_buffer) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data. Need 20 ticks, have {len(tick_buffer)}"
        )
    
    # Check if Colab is configured
    if not colab_client:
        raise HTTPException(
            status_code=503,
            detail="Colab API not configured. Set COLAB_URL in backend/main.py"
        )
    
    # Call Colab API
    try:
        result = await colab_client.predict(tick_buffer[-100:])
        
        if not result:
            raise HTTPException(
                status_code=503,
                detail="Colab API request failed. Check if notebook is running."
            )
        
        prediction = result.get('prediction')
        status = result.get('status', {})
        
        if prediction:
            prediction_buffer.append(prediction)
            
            # Broadcast to frontends
            if active_connections:
                await broadcast_to_clients({
                    'type': 'prediction',
                    'data': prediction
                })
        
        return PredictionResponse(
            prediction=prediction,
            status=status,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/api/start")
async def start_stream():
    """
    Start Deriv tick stream
    """
    global deriv_client
    
    if deriv_client and deriv_client.is_connected:
        return {"message": "Already connected"}
    
    asyncio.create_task(start_tick_stream())
    
    return {"message": "Tick stream started"}

@app.post("/api/stop")
async def stop_stream():
    """
    Stop Deriv tick stream
    """
    global deriv_client
    
    if deriv_client:
        await deriv_client.disconnect()
    
    return {"message": "Tick stream stopped"}

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Deriv Predictor API...")
    print("📊 Deriv Symbol: R_100")
    print("🔑 API Key: V35FbErHFzWjhj5")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
