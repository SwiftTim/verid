"""
monitor.py — Structured performance monitor with Prometheus-style metrics.

Tracks:
  • Prediction accuracy (rolling windows: 50, 200, 1000)
  • Trade PnL, win-rate, drawdown
  • Tick latency (time between ticks)
  • Colab round-trip latency
  • Auto-alerts when thresholds are breached

All data exposed via /api/metrics for Grafana / Prometheus scraping.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class PredictionRecord:
    ts:         float    # unix timestamp
    decision:   str
    confidence: float
    correct:    Optional[bool] = None  # None until outcome known


@dataclass
class TradeRecord:
    ts:      float
    profit:  float
    stake:   float
    correct: bool


@dataclass
class Alert:
    level:   str    # "INFO" | "WARNING" | "CRITICAL"
    message: str
    ts:      float  = field(default_factory=time.time)


# ─────────────────────────────────────────────
# Monitor
# ─────────────────────────────────────────────

class PerformanceMonitor:
    """
    Call these methods from the main backend loop:
        monitor.record_prediction(pred_dict)
        monitor.record_outcome(pred_id, actual_direction)
        monitor.record_trade(profit, stake)
        monitor.record_tick_latency(seconds)
        monitor.record_colab_latency(seconds)
    Then serve monitor.snapshot() from /api/metrics.
    """

    def __init__(self, alert_cb=None):
        self._predictions:     Deque[PredictionRecord] = deque(maxlen=2000)
        self._trades:          Deque[TradeRecord]       = deque(maxlen=2000)
        self._tick_latencies:  Deque[float]             = deque(maxlen=500)
        self._colab_latencies: Deque[float]             = deque(maxlen=200)
        self._alerts:          Deque[Alert]             = deque(maxlen=100)

        self._session_start    = time.time()
        self._total_ticks      = 0
        self._alert_cb         = alert_cb   # optional async fn(alert)

        # Thresholds that trigger alerts
        self._thresholds = {
            "min_accuracy_50":    0.47,
            "min_accuracy_200":   0.49,
            "max_drawdown":       0.08,
            "max_colab_latency":  5.0,   # seconds
            "max_tick_gap":       10.0,  # seconds (missed ticks)
        }

    # ── Recording ──────────────────────────────────────────────────────────

    def record_prediction(self, pred: Dict) -> str:
        """Returns a unique key to match against later outcome."""
        rec = PredictionRecord(
            ts         = time.time(),
            decision   = pred.get("final_decision", "SKIP"),
            confidence = pred.get("confidence", 0.0),
        )
        self._predictions.append(rec)
        return str(len(self._predictions))

    def record_outcome(self, actual_direction: int, last_n: int = 1):
        """
        Mark the last `last_n` un-resolved predictions as correct/wrong.
        actual_direction: 1=UP, 0=DOWN
        """
        resolved = 0
        for rec in reversed(self._predictions):
            if rec.correct is not None:
                break
            if rec.decision == "SKIP":
                rec.correct = None   # no outcome for skips
            else:
                expected = 1 if rec.decision == "BUY" else 0
                rec.correct = (expected == actual_direction)
            resolved += 1
            if resolved >= last_n:
                break

        # Check accuracy thresholds
        acc50 = self._accuracy(50)
        if acc50 is not None and acc50 < self._thresholds["min_accuracy_50"]:
            self._alert("WARNING", f"50-trade accuracy dropped to {acc50:.1%}")

        acc200 = self._accuracy(200)
        if acc200 is not None and acc200 < self._thresholds["min_accuracy_200"]:
            self._alert("CRITICAL", f"200-trade accuracy dropped to {acc200:.1%}")

    def record_trade(self, profit: float, stake: float):
        self._trades.append(TradeRecord(
            ts=time.time(), profit=profit, stake=stake, correct=(profit > 0)
        ))
        dd = self._max_drawdown()
        if dd > self._thresholds["max_drawdown"]:
            self._alert("CRITICAL", f"Max drawdown {dd:.1%} exceeded threshold")

    def record_tick(self):
        now = time.time()
        if self._tick_latencies:
            gap = now - (self._session_start + sum(self._tick_latencies))
        else:
            gap = 0.0
        self._tick_latencies.append(now)
        self._total_ticks += 1
        if gap > self._thresholds["max_tick_gap"]:
            self._alert("WARNING", f"Tick gap of {gap:.1f}s detected")

    def record_colab_latency(self, seconds: float):
        self._colab_latencies.append(seconds)
        if seconds > self._thresholds["max_colab_latency"]:
            self._alert("WARNING", f"Colab latency {seconds:.1f}s exceeded threshold")

    # ── Computed metrics ───────────────────────────────────────────────────

    def _accuracy(self, window: int) -> Optional[float]:
        resolved = [p for p in self._predictions
                    if p.correct is not None and p.decision != "SKIP"]
        if len(resolved) < window:
            return None
        last = resolved[-window:]
        return sum(1 for p in last if p.correct) / len(last)

    def _max_drawdown(self) -> float:
        if not self._trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self._trades:
            cumulative += t.profit
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / (peak + 1e-8)
            max_dd = max(max_dd, dd)
        return max_dd

    def _net_pnl(self) -> float:
        return sum(t.profit for t in self._trades)

    def _win_rate(self) -> Optional[float]:
        settled = [t for t in self._trades]
        if not settled:
            return None
        return sum(1 for t in settled if t.correct) / len(settled)

    # ── Snapshot (serve from API) ──────────────────────────────────────────

    def snapshot(self) -> Dict:
        acc50  = self._accuracy(50)
        acc200 = self._accuracy(200)
        acc1k  = self._accuracy(1000)

        cl = list(self._colab_latencies)
        cl_avg = mean(cl) if cl else 0.0
        cl_p95 = sorted(cl)[int(len(cl) * 0.95)] if len(cl) > 5 else cl_avg

        return {
            "session_uptime_s": time.time() - self._session_start,
            "total_ticks":      self._total_ticks,
            "predictions": {
                "total":    len(self._predictions),
                "executed": sum(1 for p in self._predictions if p.decision != "SKIP"),
                "accuracy_50":   acc50,
                "accuracy_200":  acc200,
                "accuracy_1000": acc1k,
            },
            "trades": {
                "total":     len(self._trades),
                "win_rate":  self._win_rate(),
                "net_pnl":   self._net_pnl(),
                "max_dd":    self._max_drawdown(),
            },
            "latency": {
                "colab_avg_s": cl_avg,
                "colab_p95_s": cl_p95,
            },
            "alerts": [asdict(a) for a in list(self._alerts)[-10:]],
        }

    # ── Alerting ───────────────────────────────────────────────────────────

    def _alert(self, level: str, msg: str):
        a = Alert(level=level, message=msg)
        self._alerts.append(a)
        log_fn = logger.critical if level == "CRITICAL" else logger.warning
        log_fn(f"[ALERT][{level}] {msg}")
        if self._alert_cb:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._alert_cb(a))
            except RuntimeError:
                pass

    def recent_alerts(self, n: int = 20) -> List[Dict]:
        return [asdict(a) for a in list(self._alerts)[-n:]]
