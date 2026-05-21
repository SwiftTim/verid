"""
trade_executor.py — Industry-ready Deriv trade execution engine.

Handles the full lifecycle of a binary-options / Rise-Fall contract:
    open_trade()  → buy a contract via the Deriv WS API
    close_trade() → sell / early-exit if supported
    track_open()  → poll for outcome once the contract settles
    record()      → write results to SQLite for later analysis

All network calls are async; the executor holds one persistent
WebSocket connection and reconnects automatically on failure.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, Callable

import httpx
import websockets
import aiosqlite

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
OTP_ENDPOINT    = "https://api.derivws.com/trading/v1/options/accounts/{account_id}/otp"
DEFAULT_APP_ID  = "33k5VK8DBmgx4PmY9BKVB"
DEFAULT_ACCOUNT = "DOT92180896"            # Demo account ID
RECONNECT_DELAY = 3              # seconds between reconnect attempts
MAX_RECONNECT   = 10
HEARTBEAT_SECS  = 20             # keep-alive ping interval


class TradeDirection(str, Enum):
    BUY  = "BUY"   # predict price rises  → CALL / RISE
    SELL = "SELL"  # predict price falls  → PUT  / FALL


class TradeStatus(str, Enum):
    PENDING  = "PENDING"
    OPEN     = "OPEN"
    WON      = "WON"
    LOST     = "LOST"
    CANCELED = "CANCELED"
    ERROR    = "ERROR"


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────
@dataclass
class TradeRecord:
    trade_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))
    direction:      str   = TradeDirection.BUY
    symbol:         str   = "1HZ100V"
    stake:          float = 1.0          # USD
    duration:       int   = 1            # ticks
    duration_unit:  str   = "t"          # t=ticks, s=seconds, m=minutes
    contract_id:    Optional[str]  = None
    entry_price:    Optional[float] = None
    exit_price:     Optional[float] = None
    payout:         Optional[float] = None
    profit:         Optional[float] = None
    status:         str   = TradeStatus.PENDING
    opened_at:      Optional[str]  = None
    closed_at:      Optional[str]  = None
    prediction_ref: Optional[str]  = None   # links back to ML prediction id
    error_msg:      Optional[str]  = None


# ─────────────────────────────────────────────
# Database layer
# ─────────────────────────────────────────────
class TradeDB:
    """Lightweight async SQLite store for trade records."""

    def __init__(self, db_path: str = "/home/tim/Downloads/2026/trades.db"):
        self.db_path = db_path

    async def init(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id       TEXT PRIMARY KEY,
                    direction      TEXT,
                    symbol         TEXT,
                    stake          REAL,
                    duration       INTEGER,
                    duration_unit  TEXT,
                    contract_id    TEXT,
                    entry_price    REAL,
                    exit_price     REAL,
                    payout         REAL,
                    profit         REAL,
                    status         TEXT,
                    opened_at      TEXT,
                    closed_at      TEXT,
                    prediction_ref TEXT,
                    error_msg      TEXT
                )
            """)
            await db.commit()

    async def upsert(self, trade: TradeRecord):
        d = asdict(trade)
        cols   = ", ".join(d.keys())
        vals   = ", ".join(["?" for _ in d])
        update = ", ".join([f"{k}=excluded.{k}" for k in d if k != "trade_id"])
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"INSERT INTO trades ({cols}) VALUES ({vals}) "
                f"ON CONFLICT(trade_id) DO UPDATE SET {update}",
                list(d.values())
            )
            await db.commit()

    async def get_recent(self, limit: int = 50) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,)
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def stats(self) -> Dict:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("""
                SELECT
                    COUNT(*)                                   AS total,
                    SUM(CASE WHEN status='WON'  THEN 1 END)   AS wins,
                    SUM(CASE WHEN status='LOST' THEN 1 END)   AS losses,
                    SUM(profit)                                AS net_profit,
                    AVG(CASE WHEN status IN ('WON','LOST')
                             THEN profit END)                  AS avg_profit
                FROM trades
            """)
            row = await cur.fetchone()
            total = row[0] or 0
            wins  = row[1] or 0
            return {
                "total":      total,
                "wins":       wins,
                "losses":     row[2] or 0,
                "win_rate":   wins / total if total else 0.0,
                "net_profit": row[3] or 0.0,
                "avg_profit": row[4] or 0.0,
            }


# ─────────────────────────────────────────────
# Trade Executor
# ─────────────────────────────────────────────
class TradeExecutor:
    """
    Manages a single persistent WebSocket connection to Deriv and exposes
    async methods to open / monitor trades.

    Usage:
        executor = TradeExecutor(api_token="YOUR_TOKEN", app_id="YOUR_APP_ID")
        await executor.connect()

        trade = await executor.open_trade(
            direction="BUY",
            symbol="1HZ100V",
            stake=1.0,
            duration=5,
        )
        # trade.status will be updated as the contract settles
    """

    def __init__(
        self,
        api_token: str,
        app_id: str = DEFAULT_APP_ID,
        account_id: str = DEFAULT_ACCOUNT,
        db_path: str = "/home/tim/Downloads/2026/trades.db",
        on_trade_update: Optional[Callable[[TradeRecord], None]] = None,
    ):
        self.api_token  = api_token
        self.app_id     = app_id
        self.account_id = account_id
        self.db         = TradeDB(db_path)
        self.on_update  = on_trade_update

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected   = False
        self._reconnects  = 0
        self._pending_req: Dict[int, asyncio.Future] = {}   # req_id → Future
        self._open_trades: Dict[str, TradeRecord]    = {}   # contract_id → record

        self._listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self):
        await self.db.init()
        await self._connect_ws()

    async def _get_otp_url(self) -> str:
        """Step 1: Call the REST OTP endpoint to get a pre-authenticated WebSocket URL."""
        otp_url = OTP_ENDPOINT.format(account_id=self.account_id)
        rest_headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Deriv-App-ID": self.app_id,
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(otp_url, headers=rest_headers, json={})
            resp.raise_for_status()
            data = resp.json()
            ws_url = data["data"]["url"]
            logger.info(f"TradeExecutor: Got OTP url → {ws_url}")
            return ws_url

    async def _connect_ws(self):
        while self._reconnects < MAX_RECONNECT:
            try:
                # Step 1: Get pre-authenticated OTP WebSocket URL
                url = await self._get_otp_url()

                # Step 2: Connect — no authorize message needed!
                headers = {"Origin": "https://developers.deriv.com"}
                ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

                self._ws = await websockets.connect(
                    url,
                    additional_headers=headers,
                    user_agent_header=ua,
                    ping_interval=20,
                    ping_timeout=30
                )
                self._connected = True
                self._reconnects = 0
                logger.info("TradeExecutor: WebSocket connected & authenticated via OTP")

                # Start background listener & heartbeat
                self._listener_task  = asyncio.create_task(self._listen())
                self._heartbeat_task = asyncio.create_task(self._heartbeat())
                return

            except Exception as exc:
                self._reconnects += 1
                logger.warning(f"TradeExecutor: connect failed ({exc}), "
                               f"retry {self._reconnects}/{MAX_RECONNECT}")
                await asyncio.sleep(RECONNECT_DELAY)

        raise RuntimeError("TradeExecutor: max reconnect attempts reached")

    async def disconnect(self):
        self._connected = False
        for task in [self._listener_task, self._heartbeat_task]:
            if task:
                task.cancel()
        if self._ws:
            await self._ws.close()

    # ── Messaging ──────────────────────────────────────────────────────────

    async def _send(self, payload: Dict) -> int:
        req_id = int(time.time() * 1000)
        payload["req_id"] = req_id
        raw_msg = json.dumps(payload)
        logger.info(f"→ SENDING {payload.get('msg_type', 'request')} (ID: {req_id})")
        await self._ws.send(raw_msg)
        return req_id

    async def _request(self, payload: Dict, timeout: float = 60.0) -> Dict:
        """Send a request and wait for its response by req_id."""
        if not self._connected or self._ws is None or self._ws.state != 1:
            logger.info("TradeExecutor: Connection lost, re-authenticating...")
            await self._connect_ws()

        loop   = asyncio.get_event_loop()
        future = loop.create_future()
        req_id = await self._send(payload)
        self._pending_req[req_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_req.pop(req_id, None)
            raise TimeoutError(f"Request timed out: {payload}")

    async def _listen(self):
        """Background task: receive all WS messages and route them."""
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                logger.debug(f"← RECEIVED: {msg}")
                req_id = msg.get("req_id")

                # Resolve pending request futures
                if req_id in self._pending_req:
                    fut = self._pending_req.pop(req_id)
                    if not fut.done():
                        if "error" in msg:
                            fut.set_exception(
                                RuntimeError(msg["error"].get("message", "API error"))
                            )
                        else:
                            fut.set_result(msg)

                # Contract updates (subscription)
                if msg.get("msg_type") == "proposal_open_contract":
                    await self._handle_contract_update(msg)

        except websockets.ConnectionClosed:
            logger.warning("TradeExecutor: connection closed")
            self._connected = False
        except Exception as e:
            logger.error(f"TradeExecutor listener error: {e}")
            self._connected = False

    async def _heartbeat(self):
        """Send keep-alive pings."""
        while self._connected:
            await asyncio.sleep(HEARTBEAT_SECS)
            try:
                await self._send({"ping": 1})
            except Exception:
                pass

    # ── Trade Operations ───────────────────────────────────────────────────

    async def open_trade(
        self,
        direction: str,
        symbol:    str   = "1HZ100V",
        stake:     float = 1.0,
        duration:  int   = 5,
        duration_unit: str = "t",
        prediction_ref: Optional[str] = None,
    ) -> TradeRecord:
        """
        Open a Rise/Fall binary-options contract.

        direction: "BUY"  → CALL (price rises)
                   "SELL" → PUT  (price falls)
        stake:     USD amount to risk
        duration:  contract length
        duration_unit: "t"=ticks, "s"=seconds, "m"=minutes, "h"=hours, "d"=days
        """
        contract_type = "CALL" if direction == TradeDirection.BUY else "PUT"

        trade = TradeRecord(
            direction=direction,
            symbol=symbol,
            stake=stake,
            duration=duration,
            duration_unit=duration_unit,
            prediction_ref=prediction_ref,
            status=TradeStatus.PENDING,
            opened_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.db.upsert(trade)

        try:
            # Step 1: get a price proposal
            proposal_resp = await self._request({
                "proposal": 1,
                "amount":   stake,
                "basis":    "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": duration,
                "duration_unit": duration_unit,
                "underlying_symbol": symbol,
            })
            proposal_id   = proposal_resp["proposal"]["id"]
            expected_pay  = proposal_resp["proposal"]["payout"]

            # Step 2: buy the proposal
            buy_resp = await self._request({
                "buy":   proposal_id,
                "price": stake,
            })

            contract_id        = str(buy_resp["buy"]["contract_id"])
            trade.contract_id  = contract_id
            trade.entry_price  = buy_resp["buy"].get("buy_price", stake)
            trade.payout       = expected_pay
            trade.status       = TradeStatus.OPEN
            self._open_trades[contract_id] = trade
            await self.db.upsert(trade)

            logger.info(f"Trade opened: {contract_id} | {direction} | ${stake}")

            # Step 3: subscribe to contract updates
            await self._send({
                "proposal_open_contract": 1,
                "contract_id": int(contract_id),
                "subscribe": 1,
            })

        except Exception as exc:
            trade.status    = TradeStatus.ERROR
            trade.error_msg = str(exc)
            await self.db.upsert(trade)
            logger.error(f"Trade open failed: {exc}")

        if self.on_update:
            self.on_update(trade)

        return trade

    async def _handle_contract_update(self, msg: Dict):
        """Called whenever Deriv pushes a contract status update."""
        poc = msg.get("proposal_open_contract", {})
        cid = str(poc.get("contract_id", ""))
        if cid not in self._open_trades:
            return

        trade = self._open_trades[cid]

        if poc.get("is_expired") or poc.get("is_sold"):
            profit_raw = poc.get("profit", 0.0)
            profit     = float(profit_raw)
            trade.exit_price = float(poc.get("exit_tick", 0.0))
            trade.profit     = profit
            trade.status     = TradeStatus.WON if profit > 0 else TradeStatus.LOST
            trade.closed_at  = datetime.now(timezone.utc).isoformat()
            self._open_trades.pop(cid, None)
            await self.db.upsert(trade)
            logger.info(f"Trade settled: {cid} | {trade.status} | P&L ${profit:.2f}")

            if self.on_update:
                self.on_update(trade)

    # ── Helpers ────────────────────────────────────────────────────────────

    async def get_balance(self) -> float:
        resp = await self._request({"balance": 1})
        return resp["balance"]["balance"]

    async def recent_trades(self, limit: int = 20) -> List[Dict]:
        return await self.db.get_recent(limit)

    async def performance_stats(self) -> Dict:
        return await self.db.stats()
