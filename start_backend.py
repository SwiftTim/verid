#!/usr/bin/env python3
"""
Launcher script for Deriv Predictor Backend
"""
import uvicorn
import sys
import os

if __name__ == "__main__":
    print("🚀 Starting Deriv Predictor Backend...")
    from backend.main import COLAB_URL
    print(f"🔗 Colab URL: {COLAB_URL}")
    print("=" * 60)
    
    # Run uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
