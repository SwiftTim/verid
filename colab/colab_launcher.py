import os
import sys
import subprocess
import time
import re
import threading
import nest_asyncio

# 1. Setup Environment
def setup():
    print("📦 Installing dependencies...")
    packages = [
        "fastapi", "uvicorn[standard]", "pyngrok",
        "nest-asyncio", "httpx", "python-multipart",
        "xgboost", "lightgbm", "scipy", "scikit-learn"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages)
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    return project_root

# 2. Define API
def create_app(engine, MODEL_DIR):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Dict, Optional
    from datetime import datetime
    import traceback
    import tensorflow as tf

    app = FastAPI(title="Deriv Hybrid Predictor API", version="2.5.0")

    class Tick(BaseModel):
        timestamp: int
        quote: float
        symbol: str

    @app.get("/")
    async def root():
        return {"status": "ok", "engine_ready": True, "tick_count": engine.tick_count}

    @app.post("/predict")
    async def predict(request: Dict):
        try:
            ticks = request.get('ticks', [])
            for tick in ticks:
                engine.add_tick(tick)

            if not engine.is_trained and engine.tick_count >= 500:
                print(f"🚀 Training starting ({engine.tick_count} ticks)...")
                engine.initial_train()
                engine.save_models(MODEL_DIR)

            prediction = engine.predict_next_tick() if engine.is_trained else None
            return {
                "prediction": prediction,
                "status": {
                    "is_trained": engine.is_trained,
                    "tick_count": engine.tick_count
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

    @app.get("/status")
    async def status():
        return engine.get_status()

    @app.post("/retrain")
    async def retrain():
        res = engine.initial_train() if not engine.is_trained else engine.retrain()
        engine.save_models(MODEL_DIR)
        return res

    return app

# 3. Port & Tunnel Management
def run_launcher():
    nest_asyncio.apply()
    project_root = setup()
    
    from core.core_engine import HybridEngine
    MODEL_DIR = os.path.join(project_root, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    engine = HybridEngine(verbose=True)
    try:
        engine.load_models(MODEL_DIR)
        print("✅ Models loaded")
    except:
        print("ℹ️ Starting fresh engine")

    app = create_app(engine, MODEL_DIR)
    
    # Kill existing port 8000
    os.system("fuser -k 8000/tcp > /dev/null 2>&1")
    
    # Run uvicorn in thread
    import uvicorn
    def start_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    
    threading.Thread(target=start_uvicorn, daemon=True).start()
    time.sleep(3)

    # Cloudflare Tunnel
    print("🌐 Launching Cloudflare Tunnel...")
    os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /tmp/cloudflared && chmod +x /tmp/cloudflared")
    
    tunnel_proc = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://localhost:8000"], stderr=subprocess.PIPE)
    
    public_url = None
    for _ in range(30):
        line = tunnel_proc.stderr.readline().decode()
        match = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', line)
        if match:
            public_url = match.group(0)
            break
        time.sleep(1)

    if public_url:
        print("\n" + "═" * 50)
        print(f"🚀 SERVER LIVE: {public_url}")
        print("═" * 50)
        
        # Keep alive monitor
        while True:
            s = engine.get_status()['engine']
            print(f"\r⏰ Ticks: {s['tick_count']} | Trained: {s['is_trained']}", end="")
            time.sleep(5)

if __name__ == "__main__":
    run_launcher()
