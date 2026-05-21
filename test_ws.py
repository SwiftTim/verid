import websockets
import asyncio

async def test():
    try:
        url = "wss://api.derivws.com/trading/v1/options/ws/public"
        async with websockets.connect(url) as ws:
            print(f"Version: {websockets.__version__}")
            print(f"Attributes: {[a for a in dir(ws) if not a.startswith('_')]}")
            # Try some common ones
            try: print(f"ws.open: {ws.open}")
            except Exception as e: print(f"ws.open failed: {e}")
            try: print(f"ws.closed: {ws.closed}")
            except Exception as e: print(f"ws.closed failed: {e}")
            try: print(f"ws.state: {ws.state}")
            except Exception as e: print(f"ws.state failed: {e}")
            try: print(f"ws.connection_state: {ws.connection_state}")
            except Exception as e: print(f"ws.connection_state failed: {e}")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test())
