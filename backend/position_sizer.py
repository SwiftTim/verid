"""
position_sizer.py — Production position-sizing and pre-trade risk gate.

Every signal from the ML engine passes through here before any money moves.
Three layers:
  1. Session-level circuit breakers  (daily loss limit, max open positions)
  2. Per-trade sizing                 (Kelly fraction with hard caps)
  3. Signal quality gate              (entropy, confidence, regime checks)

Usage:
    sizer = PositionSizer(account_balance=100.0, config=RISK_CONFIG)

    decision = sizer.evaluate(
        signal       = prediction,       # dict from HybridEngine.predict_next_tick()
        tick_buffer  = last_200_ticks,   # for volatility context
        open_positions = 1,              # how many contracts are already live
    )

    if decision.approved:
        stake = decision.stake_usd
        # → place trade
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config (override in core/config.py or .env)
# ─────────────────────────────────────────────
@dataclass
class RiskConfig:
    # Session limits
    daily_loss_limit_pct:  float = 0.05   # stop trading if session P&L < -5%
    max_open_positions:    int   = 1       # only 1 contract at a time (binary)
    max_trades_per_hour:   int   = 30
    
    # Capital Preservation
    max_consecutive_losses: int  = 5       # stop after 5 losses in a row
    drawdown_stake_reduction: bool = True  # shrink stake on losing streak

    # Per-trade sizing
    base_stake_pct:        float = 0.01   # 1% of balance per trade
    kelly_fraction:        float = 0.25   # quarter-Kelly
    min_stake_usd:         float = 0.35   # Deriv minimum
    max_stake_usd:         float = 50.0   # hard cap

    # Signal quality gates
    min_confidence:        float = 0.06
    max_entropy:           float = 2.3
    min_model_agreement:   float = 0.80   # fraction — both models same direction
    forbidden_vol_regime:  int   = 2      # skip high-vol regime

    # Win-rate window for Kelly
    kelly_window:          int   = 100     # last N trades


DEFAULT_CONFIG = RiskConfig()


# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────
@dataclass
class SizingDecision:
    approved:    bool
    stake_usd:   float
    reason:      str          # why approved or rejected
    details:     Dict = None  # diagnostic info for logging


# ─────────────────────────────────────────────
# Position Sizer
# ─────────────────────────────────────────────
class PositionSizer:

    def __init__(
        self,
        account_balance: float,
        config: RiskConfig = DEFAULT_CONFIG,
    ):
        self.balance        = account_balance
        self.cfg            = config
        self._session_pnl   = 0.0
        self._hour_trades   = []    # timestamps of trades in rolling 1-hour window
        self._recent_results: List[int] = []  # 1=win, 0=loss; last N

    # ── Main entry point ───────────────────────────────────────────────────

    def evaluate(
        self,
        signal: Dict,
        open_positions: int = 0,
    ) -> SizingDecision:
        """
        Full pre-trade gate.  Returns a SizingDecision with .approved and .stake_usd.
        """

        # ── 1. Session circuit breakers ────────────────────────────────────
        loss_limit = self.balance * self.cfg.daily_loss_limit_pct
        if self._session_pnl <= -loss_limit:
            return self._reject(f"Daily loss limit hit (P&L ${self._session_pnl:.2f})")

        if open_positions >= self.cfg.max_open_positions:
            return self._reject(f"Max open positions ({self.cfg.max_open_positions}) reached")

        # Rate limiting — max N trades per hour
        now = _ts()
        self._hour_trades = [t for t in self._hour_trades if now - t < 3600]
        if len(self._hour_trades) >= self.cfg.max_trades_per_hour:
            return self._reject("Trade rate limit (per-hour) exceeded")

        # Consecutive Loss circuit breaker
        consec_losses = self._calculate_consecutive_losses()
        if consec_losses >= self.cfg.max_consecutive_losses:
            return self._reject(f"Consecutive loss limit hit ({consec_losses} >= {self.cfg.max_consecutive_losses})")

        # ── 2. Signal quality gate ─────────────────────────────────────────
        confidence   = signal.get("confidence", 0.0)
        entropy      = signal.get("entropy")
        vol_regime   = signal.get("vol_regime", 0)
        lstm_prob    = signal.get("lstm_prob", 0.5)
        tree_prob    = signal.get("tree_prob", 0.5)
        decision_str = signal.get("final_decision", "SKIP")

        if decision_str == "SKIP":
            return self._reject("Signal is SKIP")

        if confidence < self.cfg.min_confidence:
            return self._reject(f"Confidence too low ({confidence:.3f} < {self.cfg.min_confidence})")

        if entropy is not None and not math.isnan(entropy) and entropy > self.cfg.max_entropy:
            return self._reject(f"Entropy too high ({entropy:.2f})")

        if vol_regime >= self.cfg.forbidden_vol_regime:
            return self._reject(f"Forbidden volatility regime ({vol_regime})")

        # Model agreement: both models point same direction?
        bullish_lstm = lstm_prob > 0.5
        bullish_tree = tree_prob > 0.5
        if bullish_lstm != bullish_tree:
            return self._reject("LSTM and Tree models disagree on direction")

        # ── 3. Kelly sizing ────────────────────────────────────────────────
        win_rate = self._estimate_win_rate()
        kelly    = self._kelly(win_rate, vol_regime=vol_regime)
        raw_stake = self.balance * kelly

        # Apply base_stake_pct as floor (use whichever is larger, up to max)
        base_stake = self.balance * self.cfg.base_stake_pct
        stake = max(raw_stake, base_stake)

        # Apply capital preservation multipliers (Drawdown engineering)
        if self.cfg.drawdown_stake_reduction:
            consec_losses = self._calculate_consecutive_losses()
            if consec_losses >= 2:
                reduction = 0.5 ** (consec_losses - 1)
                stake *= reduction
                logger.info(f"📉 Sizing: Reducing stake due to {consec_losses} consecutive losses (Multiplier: {reduction:.2f})")

        stake = max(stake, self.cfg.min_stake_usd)
        stake = min(stake, self.cfg.max_stake_usd)
        stake = round(stake, 2)

        self._hour_trades.append(now)
        return SizingDecision(
            approved  = True,
            stake_usd = stake,
            reason    = "OK",
            details   = {
                "confidence": confidence,
                "win_rate":   win_rate,
                "kelly":      kelly,
                "stake":      stake,
            }
        )

    # ── Feedback loop ──────────────────────────────────────────────────────

    def record_outcome(self, profit: float):
        """Call after every settled trade."""
        self._session_pnl += profit
        self.balance      += profit
        self._recent_results.append(1 if profit > 0 else 0)
        if len(self._recent_results) > 500:
            self._recent_results = self._recent_results[-500:]

    def reset_session(self, new_balance: float):
        """Call at start of each trading session."""
        self.balance      = new_balance
        self._session_pnl = 0.0
        self._hour_trades.clear()

    def _calculate_consecutive_losses(self) -> int:
        """Count current consecutive losses in recent history."""
        if not self._recent_results:
            return 0
        
        count = 0
        for i in range(len(self._recent_results) - 1, -1, -1):
            if self._recent_results[i] == 0:
                count += 1
            else:
                break
        return count

    # ── Internal helpers ───────────────────────────────────────────────────

    def _estimate_win_rate(self) -> float:
        window = self._recent_results[-self.cfg.kelly_window:]
        if len(window) < 10:
            return 0.50    # conservative prior
        return sum(window) / len(window)

    def _kelly(self, win_rate, win_loss_ratio=0.95, vol_regime=1):
        """
        Kelly adjusted for current volatility regime.
        In high-vol regimes, shrink stake automatically.
        """
        q = 1 - win_rate
        b = win_loss_ratio
        k = (win_rate * b - q) / b
        k = max(0.0, k)

        # Volatility penalty: shrink in high-vol regimes
        # vol_regime: 0=low, 1=med, 2=high
        vol_penalties = {0: 1.0, 1: 0.75, 2: 0.25}
        vol_mult = vol_penalties.get(vol_regime, 0.75)

        return k * self.cfg.kelly_fraction * vol_mult

    @staticmethod
    def _reject(reason: str) -> SizingDecision:
        logger.debug(f"Trade rejected: {reason}")
        return SizingDecision(approved=False, stake_usd=0.0, reason=reason)


def _ts() -> float:
    import time
    return time.time()
