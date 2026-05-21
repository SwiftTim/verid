#!/usr/bin/env python3
"""
Quick Start Script for Deriv Predictor Backend
Tests the Deriv WebSocket connection with your API key

Run: python run_backend.py
"""

import asyncio
import sys
from datetime import datetime

print("=" * 70)
print("🚀 DERIV PREDICTOR - QUICK START")
print("=" * 70)
print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔑 API Key: pat_810acea9189247e547b2187dcacb7eb1c39ee880603c329fa1c642acb582b3e6")
print(f"📊 Symbol: R_100 (Volatility 100 Index)")
print(f"⚡ Expected tick rate: ~1.5 ticks/second")
print("=" * 70)

# Check dependencies
print("\n1️⃣ Checking dependencies...")
try:
    import websockets
    print("   ✅ websockets installed")
except ImportError:
    print("   ❌ websockets not found")
    print("   Install with: pip install websockets")
    sys.exit(1)

try:
    import fastapi
    print("   ✅ fastapi installed")
except ImportError:
    print("   ⚠️ fastapi not found (optional for API)")
    print("   Install with: pip install fastapi uvicorn")

print("\n2️⃣ Testing Deriv WebSocket connection...")

# Import the client
from backend.deriv_websocket import DerivTickStream

# Statistics
tick_count = 0
start_time = None

async def test_connection():
    """
    Test Deriv WebSocket connection
    """
    global tick_count, start_time
    
    # Tick handler
    async def on_tick(tick):
        global tick_count, start_time
        
        if start_time is None:
            start_time = datetime.now()
        
        tick_count += 1
        
        # Print first 5 ticks
        if tick_count <= 5:
            print(f"\n   📊 Tick #{tick_count}:")
            print(f"      Quote: {tick['quote']}")
            print(f"      Time: {datetime.fromtimestamp(tick['timestamp'])}")
            print(f"      Symbol: {tick['symbol']}")
        
        # Print summary every 50 ticks
        elif tick_count % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = tick_count / elapsed if elapsed > 0 else 0
            print(f"\n   📈 Progress: {tick_count} ticks | Rate: {rate:.2f}/sec")
        
        # Stop after 100 ticks (demo)
        if tick_count >= 100:
            print("\n" + "=" * 70)
            print("✅ TEST SUCCESSFUL!")
            print("=" * 70)
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = tick_count / elapsed if elapsed > 0 else 0
            print(f"📊 Total ticks: {tick_count}")
            print(f"⏱️ Duration: {elapsed:.1f} seconds")
            print(f"⚡ Average rate: {rate:.2f} ticks/second")
            print("\n🎯 Next steps:")
            print("   1. Start the full backend: python -m uvicorn backend.main:app --reload")
            print("   2. Or integrate with core engine (see backend/deriv_websocket.py)")
            print("=" * 70)
            
            # Disconnect
            await client.disconnect()
            return
    
    # Error handler
    async def on_error(error):
        print(f"\n   ❌ Error: {error}")
    
    # Connection handler
    async def on_connect(symbol):
        print(f"\n   ✅ Connected to {symbol}")
        print(f"   ⏳ Receiving ticks... (will collect 100 ticks)")
    
    # Create client
    global client
    client = DerivTickStream(
        app_id='33k5VK8DBmgx4PmY9BKVB',
        symbol='R_100',
        on_tick=on_tick,
        on_error=on_error,
        on_connect=on_connect
    )
    
    try:
        # Connect and listen
        await client.connect()
        await client.listen()
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")
        await client.disconnect()
    
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Verify API key is correct: V35FbErHFzWjhj5")
        print("   3. Try a different symbol (R_50, R_75)")
        sys.exit(1)

# Run the test
if __name__ == "__main__":
    try:
        asyncio.run(test_connection())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
