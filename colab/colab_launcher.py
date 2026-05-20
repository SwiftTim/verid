import os
import sys
import subprocess
import time
import re
import threading
import nest_asyncio
import asyncio

# ==================== 1. SETUP & PERSISTENCE ====================
def setup_colab():
    print("🚀 INITIALIZING ANTIGRAVITY ENVIRONMENT...")
    
    # Check if we're in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        PROJECT_PATH = "/content/drive/MyDrive/deriv_predictor"
    except:
        PROJECT_PATH = os.getcwd()
        print("ℹ️ Running in local mode (No Drive detected)")

    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_PATH, "models"), exist_ok=True)
    
    # Critical Dependencies
    packages = ["fastapi", "uvicorn[standard]", "nest-asyncio", "httpx", "python-multipart"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages)
    
    sys.path.insert(0, PROJECT_PATH)
    return PROJECT_PATH

# ==================== 2. THE API & ENGINE ====================
def create_app(engine, model_dir):
    from fastapi import FastAPI
    from typing import Dict, List
    from datetime import datetime

    app = FastAPI(title="Deriv Hybrid Predictor (Master Engine)")

    @app.post("/predict")
    async def predict(request: Dict):
        ticks = request.get('ticks', [])
        for tick in ticks:
            engine.add_tick(tick)

        # Automated Training Gate (Zero-config)
        if not engine.is_trained and engine.tick_count >= 1000:
            print(f"\n🔥 THRESHOLD MET ({engine.tick_count} ticks) | Initializing Neural Training...")
            results = engine.initial_train()
            engine.save_models(model_dir)
            print(f"✅ Success: LSTM Acc {results['lstm_accuracy']:.1%} | Progress saved to Drive.")

        prediction = engine.predict_next_tick() if engine.is_trained else None
        return {
            "prediction": prediction,
            "status": {
                "is_trained": engine.is_trained,
                "tick_count": engine.tick_count,
                "mode": "PRODUCTION" if engine.is_trained else "COLLECTING"
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/")
    async def health():
        return {"status": "ok", "gpu": "detected", "engine_ticks": engine.tick_count}

    @app.post("/retrain")
    async def force_retrain():
        res = engine.initial_train() if not engine.is_trained else engine.retrain()
        engine.save_models(model_dir)
        return {"message": "Optimization complete", "data": res}

    return app

# ==================== 3. LAUNCHER CORE ====================
def start_server(app):
    import uvicorn
    nest_asyncio.apply()
    
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

def main():
    PROJECT_PATH = setup_colab()
    MODEL_DIR = os.path.join(PROJECT_PATH, "models")
    
    # Load Core Engine
    from core.core_engine import HybridEngine
    engine = HybridEngine(verbose=True)
    
    try:
        engine.load_models(MODEL_DIR)
        print(f"✅ PERSISTENCE: Previous models loaded from {MODEL_DIR}")
    except:
        print("ℹ️ COLD START: No previous models found. Waiting for 1,000 ticks to begin training.")

    app = create_app(engine, MODEL_DIR)
    
    # Kill ports
    os.system("fuser -k 8000/tcp > /dev/null 2>&1")
    
    # Start API in background
    threading.Thread(target=start_server, args=(app,), daemon=True).start()
    time.sleep(5)

    # 4. TUNNEL (Cloudflare)
    print("🌐 Opening High-Speed Neural Tunnel...")
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
        print("\n" + "═"*60)
        print(f"  🚀  COLAB API IS LIVE")
        print(f"  🔗  URL: {public_url}")
        print(f"  📍  PATH: {PROJECT_PATH}")
        print("═"*60)
        
        # Monitor Loop (Keep cell alive)
        while True:
            try:
                s = engine.get_status()['engine']
                print(f"\r📡 Signal Core Active | Ticks Seeded: {s['tick_count']} | State: {'PRODUCTION' if engine.is_trained else 'LEARNING'}", end="")
                time.sleep(10)
            except KeyboardInterrupt:
                break
    else:
        print("❌ Tunnel connection timed out.")

if __name__ == "__main__":
    main()
