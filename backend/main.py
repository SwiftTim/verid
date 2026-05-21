"""
main.py — Production FastAPI backend with full automated trading loop.

Architecture:
  Deriv tick stream → buffer → ML prediction (Colab) → position sizer
  → trade executor → DB → WebSocket broadcast → dashboard

Run:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Dict, Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from .deriv_websocket import DerivTickStream
from .colab_client    import ColabClient
from .trade_executor  import TradeExecutor, TradeDirection
from .position_sizer  import PositionSizer, RiskConfig

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("deriv.main")

# ─────────────────────────────────────────────
# Config — override via environment variables
# ─────────────────────────────────────────────
COLAB_URL              = os.getenv("COLAB_URL", "https://specialists-comm-knife-because.trycloudflare.com")
DERIV_API_TOKEN        = os.getenv("DERIV_API_TOKEN", "pat_c493adf4cbbab38b6bb677308c26df1715d40d54ead2fdfda4b33c6167445a6d")
DERIV_APP_ID           = os.getenv("DERIV_APP_ID", "33k5VK8DBmgx4PmY9BKVB")
DERIV_ACCOUNT_ID       = os.getenv("DERIV_ACCOUNT_ID", "DOT92180896")   # Demo account
DERIV_SYMBOL           = os.getenv("DERIV_SYMBOL", "1HZ100V")
AUTO_TRADE_ENABLED     = os.getenv("AUTO_TRADE", "false").lower() == "true"
TRADE_DURATION         = int(os.getenv("TRADE_DURATION", "5"))     # ticks
STARTING_BALANCE       = float(os.getenv("STARTING_BALANCE", "100.0"))
API_SECRET             = os.getenv("API_SECRET", "changeme")       # dashboard auth

# Prediction batching
PREDICT_BATCH_SIZE     = 50    # send to Colab every N ticks
MIN_TICKS_FOR_PREDICT  = 100

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
tick_buffer:      List[Dict] = []
prediction_cache: List[Dict] = []
open_positions:   int        = 0
total_ticks:      int        = 0

deriv_client:  Optional[DerivTickStream] = None
colab_client:  Optional[ColabClient]     = None
trade_executor: Optional[TradeExecutor]  = None
position_sizer: Optional[PositionSizer]  = None

ws_clients: List[WebSocket] = []


# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Depends(api_key_header)):
    if key != API_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# ─────────────────────────────────────────────
# App lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield
    await _shutdown()


app = FastAPI(
    title="Deriv Auto-Trader",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _startup():
    global deriv_client, colab_client, trade_executor, position_sizer

    logger.info("=== Deriv Auto-Trader starting ===")

    # ── Colab client ───────────────────────────────────────────────────────
    colab_client = ColabClient(COLAB_URL)
    ok = await colab_client.health_check()
    logger.info(f"Colab API: {'✅ connected' if ok else '⚠️  not reachable'}")

    # ── Trade executor ─────────────────────────────────────────────────────
    if DERIV_API_TOKEN:
        trade_executor = TradeExecutor(
            api_token=DERIV_API_TOKEN,
            app_id=DERIV_APP_ID,
            account_id=DERIV_ACCOUNT_ID,
            on_trade_update=_on_trade_update,
        )
        await trade_executor.connect()
        bal = await trade_executor.get_balance()
        logger.info(f"Deriv account connected | balance: ${bal:.2f}")
    else:
        logger.warning("DERIV_API_TOKEN not set — paper-trade mode only")

    # ── Position sizer ─────────────────────────────────────────────────────
    position_sizer = PositionSizer(
        account_balance=STARTING_BALANCE,
        config=RiskConfig(),
    )

    # ── Deriv tick stream ──────────────────────────────────────────────────
    deriv_client = DerivTickStream(
        app_id=DERIV_APP_ID,
        symbol=DERIV_SYMBOL,
        on_tick=_on_tick,
        on_error=_on_deriv_error,
        on_connect=_on_deriv_connect,
    )
    asyncio.create_task(_run_tick_stream())

    logger.info(f"Auto-trade: {'ENABLED' if AUTO_TRADE_ENABLED else 'DISABLED'}")
    logger.info("=== Startup complete ===")


async def _shutdown():
    if deriv_client:
        await deriv_client.disconnect()
    if trade_executor:
        await trade_executor.disconnect()


# ─────────────────────────────────────────────
# Tick handling
# ─────────────────────────────────────────────
async def _on_tick(tick: Dict):
    global tick_buffer, total_ticks

    tick_buffer.append(tick)
    total_ticks += 1

    # Rolling buffer — keep last 2000 ticks
    if len(tick_buffer) > 2000:
        tick_buffer = tick_buffer[-2000:]

    # Broadcast tick to dashboard
    await _broadcast({"type": "tick", "data": tick, "total_ticks": total_ticks})

    # Trigger prediction on every batch boundary
    enough = len(tick_buffer) >= MIN_TICKS_FOR_PREDICT
    batch  = total_ticks % PREDICT_BATCH_SIZE == 0
    if enough and batch:
        asyncio.create_task(_predict_and_trade(list(tick_buffer[-200:])))


async def _predict_and_trade(ticks: List[Dict]):
    global prediction_cache, open_positions, position_sizer

    if not colab_client:
        return

    try:
        result = await colab_client.predict(ticks)
    except Exception as exc:
        logger.error(f"Colab predict error: {exc}")
        return

    if not result:
        return

    prediction = result.get("prediction")
    status     = result.get("status", {})

    if not prediction:
        tick_count = status.get("tick_count", 0)
        logger.info(f"Waiting for training… {tick_count}/500 ticks seen by Colab")
        await _broadcast({"type": "training_progress", "tick_count": tick_count})
        return

    # Cache and broadcast prediction
    prediction_cache.append(prediction)
    if len(prediction_cache) > 500:
        prediction_cache = prediction_cache[-500:]
    
    await _broadcast({
        "type": "prediction", 
        "data": prediction,
        "status": status  # Keep memory bar updated
    })

    decision = prediction.get("final_decision", "SKIP")
    logger.info(
        f"Prediction: {decision:4s} | conf={prediction.get('confidence', 0):.3f} "
        f"| lstm={prediction.get('lstm_prob', 0):.3f} "
        f"| tree={prediction.get('tree_prob', 0):.3f}"
    )

    # ── Auto-trade gate ───────────────────────────────────────────────────
    if not AUTO_TRADE_ENABLED or not trade_executor or decision == "SKIP":
        return

    sizing = position_sizer.evaluate(prediction, open_positions=open_positions)
    if not sizing.approved:
        logger.info(f"Trade blocked by sizer: {sizing.reason}")
        await _broadcast({"type": "trade_blocked", "reason": sizing.reason})
        return

    # Place trade
    try:
        trade = await trade_executor.open_trade(
            direction  = decision,
            symbol     = DERIV_SYMBOL,
            stake      = sizing.stake_usd,
            duration   = TRADE_DURATION,
            duration_unit = "t",
        )
        open_positions += 1
        await _broadcast({
            "type":       "trade_opened",
            "trade_id":   trade.trade_id,
            "direction":  decision,
            "stake":      sizing.stake_usd,
            "contract_id": trade.contract_id,
        })
        logger.info(f"Trade opened: {trade.trade_id} | {decision} | ${sizing.stake_usd}")
    except Exception as exc:
        logger.error(f"Trade execution error: {exc}")


# ─────────────────────────────────────────────
# Trade outcome callback
# ─────────────────────────────────────────────
def _on_trade_update(trade):
    global open_positions
    if trade.status in ("WON", "LOST", "ERROR", "CANCELED"):
        open_positions = max(0, open_positions - 1)
        if position_sizer and trade.profit is not None:
            position_sizer.record_outcome(trade.profit)

    async def _update_task():
        bal = await trade_executor.get_balance() if trade_executor else 0.0
        await _broadcast({
            "type":      "trade_update",
            "trade_id":  trade.trade_id,
            "status":    trade.status,
            "profit":    trade.profit,
            "direction": trade.direction,
            "balance":   bal
        })
    asyncio.create_task(_update_task())

    logger.info(
        f"Trade settled: {trade.trade_id} | {trade.status} "
        f"| P&L ${trade.profit or 0:.2f}"
    )


# ─────────────────────────────────────────────
# Deriv WS callbacks
# ─────────────────────────────────────────────
async def _on_deriv_error(error: Exception):
    logger.error(f"Deriv WS error: {error}")
    await _broadcast({"type": "error", "message": str(error)})


async def _on_deriv_connect(symbol: str):
    logger.info(f"Deriv WS connected: {symbol}")
    await _broadcast({"type": "connected", "symbol": symbol})


async def _run_tick_stream():
    try:
        await deriv_client.connect()
        await deriv_client.listen()
    except Exception as exc:
        logger.error(f"Tick stream fatal: {exc}")


# ─────────────────────────────────────────────
# WebSocket broadcast
# ─────────────────────────────────────────────
async def _broadcast(msg: Dict):
    dead = []
    for ws in list(ws_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ─────────────────────────────────────────────
# WebSocket endpoint (dashboard)
# ─────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        bal = await trade_executor.get_balance() if trade_executor else 0.0
        await websocket.send_json({
            "type": "connected", 
            "msg": "Deriv Auto-Trader",
            "balance": bal
        })
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)


# ─────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Deriv Auto-Trader",
        "auto_trade": AUTO_TRADE_ENABLED,
        "symbol": DERIV_SYMBOL,
        "tick_count": total_ticks,
    }


@app.get("/api/status")
async def get_status():
    stats = deriv_client.get_statistics() if deriv_client else {}
    sizer_pnl = position_sizer._session_pnl if position_sizer else 0.0
    colab_status = await colab_client.get_status() if colab_client else {}
    return {
        "deriv":       stats,
        "auto_trade":  AUTO_TRADE_ENABLED,
        "session_pnl": sizer_pnl,
        "open_positions": open_positions,
        "prediction_count": len(prediction_cache),
        "colab":       colab_status,
    }


@app.get("/api/ticks")
async def get_ticks(limit: int = 100):
    return {"count": len(tick_buffer), "ticks": tick_buffer[-limit:]}


@app.get("/api/predictions")
async def get_predictions(limit: int = 20):
    return {"predictions": prediction_cache[-limit:]}


@app.get("/api/trades")
async def get_trades(limit: int = 50, _: str = Depends(require_api_key)):
    if not trade_executor:
        return {"trades": []}
    return {"trades": await trade_executor.recent_trades(limit)}


@app.get("/api/performance")
async def get_performance(_: str = Depends(require_api_key)):
    if not trade_executor:
        return {}
    return await trade_executor.performance_stats()


class TradeRequest(BaseModel):
    direction: str   # BUY or SELL
    stake: float
    duration: int = 5


@app.post("/api/trade/manual", dependencies=[Depends(require_api_key)])
async def manual_trade(req: TradeRequest):
    """Manually open a trade from the dashboard."""
    if not trade_executor:
        raise HTTPException(503, "Trade executor not configured")
    trade = await trade_executor.open_trade(
        direction=req.direction,
        symbol=DERIV_SYMBOL,
        stake=req.stake,
        duration=req.duration,
    )
    return {"trade_id": trade.trade_id, "status": trade.status}


@app.post("/api/auto_trade/toggle", dependencies=[Depends(require_api_key)])
async def toggle_auto_trade(enabled: bool):
    global AUTO_TRADE_ENABLED
    AUTO_TRADE_ENABLED = enabled
    logger.info(f"Auto-trade toggled: {enabled}")
    return {"auto_trade": AUTO_TRADE_ENABLED}


@app.post("/api/retrain", dependencies=[Depends(require_api_key)])
async def force_retrain():
    if not colab_client:
        raise HTTPException(503, "Colab not connected")
    result = await colab_client.force_retrain()
    return result or {"message": "retrain triggered"}


@app.get("/health")
async def health():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}
