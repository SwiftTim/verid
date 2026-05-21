"""
Deriv WebSocket Client
Real-time tick stream ingestion from Deriv API

Optimized for:
- ~1 tick/second markets (R_100, R_75, R_50)
- Automatic reconnection
- Error handling
- Async/await pattern
"""

import asyncio
import websockets
import json
from typing import Callable, Optional, Dict, List
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DerivTickStream:
    """
    WebSocket client for Deriv tick stream
    
    Features:
    - Auto-reconnection
    - Error handling
    - Callback-based tick delivery
    - Connection status monitoring
    """
    
    def __init__(
        self,
        app_id: str,
        symbol: str = 'R_100',
        on_tick=None,
        on_error=None,
        on_connect=None
    ):
        """
        Initialize the stream client
        """
        self.app_id = app_id
        self.symbol = symbol
        # Use the verified public endpoint for price data
        self.endpoint = "wss://api.derivws.com/trading/v1/options/ws/public"
        
        # Callbacks
        self.on_tick = on_tick or self._default_on_tick
        self.on_error = on_error or self._default_on_error
        self.on_connect = on_connect or self._default_on_connect
        
        # State
        self.websocket = None
        self.is_connected = False
        self.tick_count = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        
        # Statistics
        self.start_time = None
        self.last_tick_time = None
        self.ticks_per_second = 0.0
    
    async def connect(self):
        """
        Connect to Deriv WebSocket API
        """
        try:
            logger.info(f"Connecting to Deriv API: {self.symbol}")
            
            # Match settings that worked in TradeExecutor
            headers = { "Origin": "https://developers.deriv.com" }
            ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            
            self.websocket = await websockets.connect(
                self.endpoint,
                additional_headers=headers,
                user_agent_header=ua,
                ping_interval=20,
                ping_timeout=20
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            self.start_time = datetime.now()
            
            logger.info(f"✅ Connected to Deriv API")
            await self.on_connect(self.symbol)
            
            # Subscribe to tick stream
            await self._subscribe_ticks()
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self.on_error(e)
            raise
    
    async def _subscribe_ticks(self):
        """
        Subscribe to tick stream for specified symbol
        """
        subscribe_message = {
            "ticks": self.symbol,
            "subscribe": 1
        }
        
        await self.websocket.send(json.dumps(subscribe_message))
        logger.info(f"📊 Subscribed to {self.symbol} tick stream")
    
    async def listen(self):
        """
        Listen for incoming tick data
        
        This is the main loop that receives and processes ticks
        """
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle tick data
                    if 'tick' in data:
                        await self._process_tick(data['tick'])
                    
                    # Handle errors from API
                    elif 'error' in data:
                        error_msg = data['error'].get('message', 'Unknown error')
                        logger.error(f"API Error: {error_msg}")
                        await self.on_error(Exception(error_msg))
                    
                    # Handle subscription confirmation
                    elif 'subscription' in data:
                        logger.info(f"✅ Subscription confirmed: {data['subscription']}")
                
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                    continue
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed")
            self.is_connected = False
            await self._handle_disconnect()
        
        except Exception as e:
            logger.error(f"Listen error: {e}")
            await self.on_error(e)
            self.is_connected = False
    
    async def _process_tick(self, tick_data: Dict):
        """
        Process incoming tick data
        
        Args:
            tick_data: Raw tick data from Deriv API
        """
        # Extract tick information
        tick = {
            'timestamp': tick_data.get('epoch'),
            'quote': tick_data.get('quote'),
            'symbol': tick_data.get('symbol'),
            'ask': tick_data.get('ask'),
            'bid': tick_data.get('bid'),
            'pip_size': tick_data.get('pip_size')
        }
        
        # Update statistics
        self.tick_count += 1
        self.last_tick_time = datetime.now()
        
        if self.start_time:
            elapsed = (self.last_tick_time - self.start_time).total_seconds()
            if elapsed > 0:
                self.ticks_per_second = self.tick_count / elapsed
        
        # Log every 100 ticks
        if self.tick_count % 100 == 0:
            logger.info(
                f"📊 Ticks: {self.tick_count} | "
                f"Rate: {self.ticks_per_second:.2f}/sec | "
                f"Latest: {tick['quote']}"
            )
        
        # Call user callback
        await self.on_tick(tick)
    
    async def _handle_disconnect(self):
        """
        Handle disconnection and attempt reconnection
        """
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.warning(
                f"Reconnecting... (attempt {self.reconnect_attempts}/"
                f"{self.max_reconnect_attempts})"
            )
            
            await asyncio.sleep(self.reconnect_delay)
            
            try:
                await self.connect()
                await self.listen()
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                await self._handle_disconnect()
        else:
            logger.error("Max reconnection attempts reached")
            await self.on_error(Exception("Connection lost"))
    
    async def disconnect(self):
        """
        Gracefully disconnect from WebSocket
        """
        if self.websocket and self.is_connected:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from Deriv API")
    
    def get_statistics(self) -> Dict:
        """
        Get connection statistics
        
        Returns:
            Dict with statistics
        """
        return {
            'is_connected': self.is_connected,
            'tick_count': self.tick_count,
            'ticks_per_second': self.ticks_per_second,
            'symbol': self.symbol,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'last_tick': self.last_tick_time.isoformat() if self.last_tick_time else None
        }
    
    # Default callbacks
    async def _default_on_tick(self, tick: Dict):
        """Default tick handler (just logs)"""
        pass
    
    async def _default_on_error(self, error: Exception):
        """Default error handler"""
        logger.error(f"Error: {error}")
    
    async def _default_on_connect(self, symbol: str):
        """Default connection handler"""
        logger.info(f"Connected to {symbol}")


# ============================================================
# EXAMPLE USAGE
# ============================================================

async def example_usage():
    """
    Example: How to use the DerivTickStream client
    """
    
    # Storage for ticks
    tick_buffer = []
    
    # Define callbacks
    async def handle_tick(tick):
        """Process each tick"""
        tick_buffer.append(tick)
        print(f"Tick: {tick['quote']} at {tick['timestamp']}")
        
        # Send to prediction engine every 10 ticks
        if len(tick_buffer) >= 10:
            # TODO: Send to Colab API or local engine
            print(f"📦 Batch ready: {len(tick_buffer)} ticks")
            tick_buffer.clear()
    
    async def handle_error(error):
        """Handle errors"""
        print(f"❌ Error: {error}")
    
    async def handle_connect(symbol):
        """Handle connection"""
        print(f"✅ Connected to {symbol}")
    
    # Create client
    client = DerivTickStream(
        app_id='V35FbErHFzWjhj5',
        symbol='R_100',
        on_tick=handle_tick,
        on_error=handle_error,
        on_connect=handle_connect
    )
    
    # Connect and listen
    try:
        await client.connect()
        await client.listen()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
        await client.disconnect()
    except Exception as e:
        print(f"❌ Fatal error: {e}")


# ============================================================
# INTEGRATION WITH CORE ENGINE
# ============================================================

async def stream_to_engine():
    """
    Example: Stream ticks directly to HybridEngine
    """
    from core import HybridEngine
    
    # Initialize engine
    engine = HybridEngine(verbose=True)
    
    # Tick handler
    async def process_tick(tick):
        # Add to engine
        engine.add_tick(tick)
        
        # Train if needed
        if not engine.is_trained and engine.tick_count >= 500:
            print("🚀 Starting initial training...")
            results = engine.initial_train()
            print(f"✅ Training complete: {results['lstm_accuracy']:.2%}")
        
        # Predict if trained
        if engine.is_trained:
            prediction = engine.predict_next_tick()
            
            if prediction and prediction['final_decision'] != 'SKIP':
                print(f"🎯 Signal: {prediction['final_decision']} "
                      f"(confidence: {prediction['confidence']:.2%})")
    
    # Create client
    client = DerivTickStream(
        app_id='V35FbErHFzWjhj5',
        symbol='R_100',
        on_tick=process_tick
    )
    
    # Run
    await client.connect()
    await client.listen()


# Run example
if __name__ == "__main__":
    print("🚀 Deriv Tick Stream Client")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Run the example
    asyncio.run(example_usage())
    
    # Or run with engine integration:
    # asyncio.run(stream_to_engine())
