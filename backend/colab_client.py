"""
Colab API Client
Connects local backend to Google Colab prediction engine

Usage:
    from backend.colab_client import ColabClient
    
    client = ColabClient("https://xxxx.ngrok.io")
    prediction = await client.predict(ticks)
"""

import httpx
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ColabClient:
    """
    Client for Google Colab prediction API
    
    Handles:
    - Prediction requests
    - Status checks
    - Error handling
    - Retries
    """
    
    def __init__(
        self,
        colab_url: str,
        timeout: float = 300.0,  # 5 minutes — allows for initial LSTM training
        max_retries: int = 3
    ):
        """
        Initialize Colab API client
        
        Args:
            colab_url: ngrok URL from Colab (e.g., "https://xxxx.ngrok.io")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
        """
        self.colab_url = colab_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(f"Colab client initialized: {self.colab_url}")
    
    async def predict(self, ticks: List[Dict]) -> Optional[Dict]:
        """
        Get prediction from Colab engine
        
        Args:
            ticks: List of tick dictionaries with keys:
                   - timestamp (int)
                   - quote (float)
                   - symbol (str)
        
        Returns:
            Prediction dictionary or None on error
        """
        url = f"{self.colab_url}/predict"
        
        # Format ticks
        formatted_ticks = [
            {
                "timestamp": tick.get('timestamp'),
                "quote": tick.get('quote'),
                "symbol": tick.get('symbol', 'R_100')
            }
            for tick in ticks
        ]
        
        payload = {"ticks": formatted_ticks}
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    pred = data.get('prediction') or {}
                    logger.info(
                        f"Prediction received: {pred.get('final_decision', 'WAITING')}"
                    )
                    
                    return data
            
            except httpx.TimeoutException:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    logger.error("Max retries reached")
                    return None
            
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code}")
                return None
            
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return None
        
        return None
    
    async def get_status(self) -> Optional[Dict]:
        """Get Colab engine status from root URL."""
        url = f"{self.colab_url}/"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                # Map 'engine_ticks' to 'tick_count' for dashboard compatibility
                if 'engine_ticks' in data:
                    data['tick_count'] = data['engine_ticks']
                return data
        
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return None
    
    async def force_retrain(self) -> Optional[Dict]:
        """
        Force model retraining
        
        Returns:
            Retrain results or None on error
        """
        url = f"{self.colab_url}/retrain"
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min for big retrains
                response = await client.post(url)
                response.raise_for_status()
                
                logger.info("Retrain triggered successfully")
                return response.json()
        
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            return None
    
    async def health_check(self) -> bool:
        """
        Check if Colab API is reachable
        
        Returns:
            True if healthy, False otherwise
        """
        url = f"{self.colab_url}/"
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        
        except:
            return False


# ============================================================
# EXAMPLE USAGE
# ============================================================

async def example_usage():
    """
    Example: How to use ColabClient
    """
    
    # Initialize client with ngrok URL from Colab
    client = ColabClient("https://xxxx.ngrok.io")
    
    # Check if Colab is reachable
    is_healthy = await client.health_check()
    print(f"Colab healthy: {is_healthy}")
    
    # Get status
    status = await client.get_status()
    print(f"Status: {status}")
    
    # Send ticks for prediction
    ticks = [
        {"timestamp": 1700000000 + i, "quote": 1234.56 + i * 0.01, "symbol": "R_100"}
        for i in range(100)
    ]
    
    prediction = await client.predict(ticks)
    
    if prediction:
        pred_data = prediction.get('prediction', {})
        print(f"Decision: {pred_data.get('final_decision')}")
        print(f"Confidence: {pred_data.get('confidence'):.2%}")
    else:
        print("Prediction failed")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
