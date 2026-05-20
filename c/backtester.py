"""
backtester.py — Walk-forward backtesting engine.

Simulates the full prediction → position-sizing → PnL loop on
historical tick data WITHOUT any lookahead bias.

Usage:
    bt = Backtester(engine, config=BacktestConfig())
    results = await bt.run(ticks)          # list of tick dicts
    report  = bt.report(results)
    print(report)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_balance:    float = 100.0
    stake_pct:          float = 0.01      # 1% per trade
    min_confidence:     float = 0.06
    trade_cost_pct:     float = 0.0       # Deriv synthetic = no spread cost
    train_on_first_n:   int   = 500       # warm-up ticks before trading
    retrain_every_n:    int   = 2000
    max_stake_usd:      float = 50.0


@dataclass
class BacktestTrade:
    tick_index:  int
    direction:   str
    stake:       float
    entry_price: float
    exit_price:  float
    profit:      float
    confidence:  float
    correct:     bool


@dataclass
class BacktestResults:
    config:      BacktestConfig
    trades:      List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float]        = field(default_factory=list)
    start_balance: float = 0.0
    end_balance:   float = 0.0

    # Computed metrics
    total_trades:  int   = 0
    wins:          int   = 0
    losses:        int   = 0
    win_rate:      float = 0.0
    net_profit:    float = 0.0
    max_drawdown:  float = 0.0
    sharpe:        float = 0.0
    profit_factor: float = 0.0


class Backtester:
    def __init__(self, engine, config: BacktestConfig = BacktestConfig()):
        self.engine = engine
        self.cfg    = config

    def run(self, ticks: List[Dict]) -> BacktestResults:
        """
        Walk-forward backtest.  No async needed — runs synchronously.
        The engine's internal state is reset-safe: we just add ticks
        sequentially, train when enough data exists, and predict ahead.
        """
        from .position_sizer import PositionSizer, RiskConfig

        cfg     = self.cfg
        engine  = self.engine
        sizer   = PositionSizer(cfg.initial_balance, RiskConfig(
            base_stake_pct=cfg.stake_pct,
            max_stake_usd=cfg.max_stake_usd,
        ))

        balance = cfg.initial_balance
        equity  = [balance]
        trades: List[BacktestTrade] = []

        trained      = False
        last_retrain = 0

        for i, tick in enumerate(ticks[:-1]):   # leave last tick for exit price
            engine.add_tick(tick)

            # ── Training ──────────────────────────────────────────────────
            if not trained and engine.tick_count >= cfg.train_on_first_n:
                try:
                    engine.initial_train()
                    trained      = True
                    last_retrain = i
                    logger.info(f"[Backtest] Initial train at tick {i}")
                except Exception as exc:
                    logger.warning(f"[Backtest] Train failed: {exc}")
                    continue

            if trained and (i - last_retrain) >= cfg.retrain_every_n:
                try:
                    engine.retrain()
                    last_retrain = i
                except Exception:
                    pass

            if not trained:
                equity.append(balance)
                continue

            # ── Prediction ────────────────────────────────────────────────
            try:
                pred = engine.predict_next_tick()
            except Exception:
                equity.append(balance)
                continue

            if not pred or pred.get("final_decision", "SKIP") == "SKIP":
                equity.append(balance)
                continue

            # ── Sizing ────────────────────────────────────────────────────
            sizing = sizer.evaluate(pred, open_positions=0)
            if not sizing.approved:
                equity.append(balance)
                continue

            # ── Simulate trade outcome ────────────────────────────────────
            entry_price = ticks[i]["quote"]
            exit_price  = ticks[i + 1]["quote"]
            direction   = pred["final_decision"]  # BUY or SELL
            correct     = (
                (direction == "BUY"  and exit_price > entry_price) or
                (direction == "SELL" and exit_price < entry_price)
            )

            # Binary options: win ≈ 95% of stake, lose = full stake
            payout   = sizing.stake_usd * 0.95
            profit   = payout if correct else -sizing.stake_usd
            balance += profit
            sizer.record_outcome(profit)

            trade = BacktestTrade(
                tick_index  = i,
                direction   = direction,
                stake       = sizing.stake_usd,
                entry_price = entry_price,
                exit_price  = exit_price,
                profit      = profit,
                confidence  = pred.get("confidence", 0.0),
                correct     = correct,
            )
            trades.append(trade)

            # Feedback to engine
            actual = 1 if exit_price > entry_price else 0
            try:
                engine.update_with_outcome(pred, actual)
            except Exception:
                pass

            equity.append(balance)

        # ── Compute metrics ───────────────────────────────────────────────
        results = BacktestResults(
            config        = cfg,
            trades        = trades,
            equity_curve  = equity,
            start_balance = cfg.initial_balance,
            end_balance   = balance,
        )
        return self._compute_metrics(results)

    @staticmethod
    def _compute_metrics(r: BacktestResults) -> BacktestResults:
        if not r.trades:
            return r

        wins   = [t for t in r.trades if t.correct]
        losses = [t for t in r.trades if not t.correct]

        gross_win  = sum(t.profit for t in wins)
        gross_loss = abs(sum(t.profit for t in losses))

        r.total_trades = len(r.trades)
        r.wins         = len(wins)
        r.losses       = len(losses)
        r.win_rate     = r.wins / r.total_trades
        r.net_profit   = r.end_balance - r.start_balance
        r.profit_factor = gross_win / gross_loss if gross_loss else float("inf")

        # Max drawdown from equity curve
        peak = r.equity_curve[0]
        dd   = 0.0
        for v in r.equity_curve:
            peak = max(peak, v)
            dd   = max(dd, (peak - v) / peak)
        r.max_drawdown = dd

        # Sharpe (simplified, annualised assuming 1-tick ≈ 1 sec)
        import statistics, math
        returns = [
            r.equity_curve[i] / r.equity_curve[i-1] - 1
            for i in range(1, len(r.equity_curve))
            if r.equity_curve[i-1] != 0
        ]
        if len(returns) > 2:
            mu  = statistics.mean(returns)
            std = statistics.stdev(returns)
            r.sharpe = (mu / std * math.sqrt(86400)) if std else 0.0

        return r

    @staticmethod
    def report(r: BacktestResults) -> str:
        if not r.trades:
            return "No trades executed."
        lines = [
            "=" * 55,
            "  BACKTEST REPORT",
            "=" * 55,
            f"  Trades:        {r.total_trades}",
            f"  Win rate:      {r.win_rate:.1%}",
            f"  Net profit:    ${r.net_profit:+.2f}",
            f"  Start balance: ${r.start_balance:.2f}",
            f"  End balance:   ${r.end_balance:.2f}",
            f"  Max drawdown:  {r.max_drawdown:.1%}",
            f"  Profit factor: {r.profit_factor:.2f}",
            f"  Sharpe ratio:  {r.sharpe:.2f}",
            "=" * 55,
        ]
        return "\n".join(lines)
