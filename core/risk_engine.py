"""
Risk Engine: Drawdown Protection + Performance Monitoring
Prevents catastrophic losses and enforces discipline
"""

import numpy as np
from typing import Optional, List
from .config import RISK_CONFIG


class RiskEngine:
    """
    Risk management and performance monitoring
    
    Key Functions:
    - Track drawdown
    - Monitor rolling accuracy
    - Auto-shutdown on performance decay
    - Prevent death spiral
    
    Philosophy:
    Survival > Optimization
    """
    
    def __init__(
        self,
        max_drawdown: float = RISK_CONFIG['max_drawdown'],
        min_accuracy_200: float = RISK_CONFIG['min_accuracy_200']
    ):
        # Risk parameters
        self.max_drawdown = max_drawdown
        self.min_accuracy_200 = min_accuracy_200
        self.min_accuracy_50 = RISK_CONFIG['min_accuracy_50']
        self.max_consecutive_losses = RISK_CONFIG['max_consecutive_losses']
        
        # State tracking
        self.peak_value = 1.0  # Normalized starting capital
        self.current_value = 1.0
        self.trade_results = []  # List of 1 (win) or 0 (loss)
        self.consecutive_losses = 0
        
        # Flags
        self.is_active = True
        self.shutdown_reason = None
    
    def update(self, trade_result: int) -> bool:
        """
        Update risk state after trade
        
        Args:
            trade_result: 1 (win) or 0 (loss)
        
        Returns:
            True if trading should continue, False if shutdown
        """
        # Record result
        self.trade_results.append(trade_result)
        
        # Update consecutive losses
        if trade_result == 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update portfolio value (simplified: +1% win, -1% loss)
        if trade_result == 1:
            self.current_value *= 1.01
        else:
            self.current_value *= 0.99
        
        # Update peak
        self.peak_value = max(self.peak_value, self.current_value)
        
        # Check risk conditions
        return self._check_risk_conditions()
    
    def _check_risk_conditions(self) -> bool:
        """
        Check if any risk limits are breached
        
        Returns:
            True if safe to continue, False if shutdown required
        """
        # 1. Check drawdown
        drawdown = self._calculate_drawdown()
        if drawdown > self.max_drawdown:
            self.is_active = False
            self.shutdown_reason = f"Max drawdown exceeded: {drawdown:.2%}"
            return False
        
        # 2. Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_active = False
            self.shutdown_reason = f"Max consecutive losses: {self.consecutive_losses}"
            return False
        
        # 3. Check rolling accuracy (200 trades)
        if len(self.trade_results) >= 200:
            acc_200 = self.get_rolling_accuracy(200)
            if acc_200 < self.min_accuracy_200:
                self.is_active = False
                self.shutdown_reason = f"Accuracy too low: {acc_200:.2%}"
                return False
        
        # 4. Check rolling accuracy (50 trades) - warning only
        if len(self.trade_results) >= 50:
            acc_50 = self.get_rolling_accuracy(50)
            if acc_50 < self.min_accuracy_50:
                print(f"⚠️ Warning: 50-trade accuracy low: {acc_50:.2%}")
        
        return True
    
    def _calculate_drawdown(self) -> float:
        """
        Calculate current drawdown from peak
        
        Returns:
            Drawdown as decimal (0.1 = 10%)
        """
        if self.peak_value == 0:
            return 0.0
        
        drawdown = (self.peak_value - self.current_value) / self.peak_value
        return drawdown
    
    def get_rolling_accuracy(self, window: int) -> Optional[float]:
        """
        Calculate accuracy over recent trades
        
        Args:
            window: Number of recent trades
        
        Returns:
            Accuracy (0-1) or None if insufficient data
        """
        if len(self.trade_results) < window:
            return None
        
        recent = self.trade_results[-window:]
        return sum(recent) / len(recent)
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive risk statistics
        
        Returns:
            Dict with risk metrics
        """
        drawdown = self._calculate_drawdown()
        
        stats = {
            'is_active': self.is_active,
            'shutdown_reason': self.shutdown_reason,
            'current_value': self.current_value,
            'peak_value': self.peak_value,
            'drawdown': drawdown,
            'total_trades': len(self.trade_results),
            'consecutive_losses': self.consecutive_losses,
            'win_rate_all': sum(self.trade_results) / len(self.trade_results) if self.trade_results else 0
        }
        
        # Add rolling accuracies
        acc_50 = self.get_rolling_accuracy(50)
        acc_200 = self.get_rolling_accuracy(200)
        
        if acc_50 is not None:
            stats['accuracy_50'] = acc_50
        if acc_200 is not None:
            stats['accuracy_200'] = acc_200
        
        return stats
    
    def should_trade(self) -> bool:
        """
        Check if trading is allowed
        
        Returns:
            True if safe to trade
        """
        return self.is_active
    
    def force_shutdown(self, reason: str):
        """
        Manually shutdown trading
        
        Args:
            reason: Reason for shutdown
        """
        self.is_active = False
        self.shutdown_reason = reason
        print(f"🛑 Trading shutdown: {reason}")
    
    def reset(self):
        """
        Reset risk engine (use with caution)
        
        Only use after addressing root cause of shutdown
        """
        self.peak_value = self.current_value
        self.consecutive_losses = 0
        self.is_active = True
        self.shutdown_reason = None
        print("✅ Risk engine reset")
    
    def get_risk_score(self) -> float:
        """
        Calculate overall risk score (0-1)
        
        0 = Safe
        1 = Maximum risk
        
        Returns:
            Risk score
        """
        if not self.is_active:
            return 1.0
        
        # Components
        drawdown_score = self._calculate_drawdown() / self.max_drawdown
        loss_score = self.consecutive_losses / self.max_consecutive_losses
        
        # Accuracy score
        acc_200 = self.get_rolling_accuracy(200)
        if acc_200 is not None:
            acc_score = max(0, (self.min_accuracy_200 - acc_200) / self.min_accuracy_200)
        else:
            acc_score = 0
        
        # Combined risk score (weighted average)
        risk_score = (
            0.4 * drawdown_score +
            0.3 * loss_score +
            0.3 * acc_score
        )
        
        return min(risk_score, 1.0)
    
    def get_trade_recommendation(self) -> str:
        """
        Get trading recommendation based on risk
        
        Returns:
            "FULL" | "REDUCED" | "STOP"
        """
        if not self.is_active:
            return "STOP"
        
        risk_score = self.get_risk_score()
        
        if risk_score < 0.3:
            return "FULL"
        elif risk_score < 0.7:
            return "REDUCED"
        else:
            return "STOP"
