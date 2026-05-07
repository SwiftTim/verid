"""
Q-Learning Agent: Execution Optimizer
Learns WHEN to trust the ensemble predictions
"""

import numpy as np
import pickle
from typing import Tuple, Optional
from ..config import RL_CONFIG


class QAgent:
    """
    Q-Learning agent for execution optimization
    
    Philosophy:
    - Model predicts WHAT (direction)
    - RL learns WHEN (to execute)
    
    State Space:
    - Ensemble confidence (binned)
    - Recent win rate (binned)
    - Volatility regime (low/medium/high)
    - Streak length (binned)
    
    Action Space:
    - 0: SKIP (don't trade)
    - 1: EXECUTE (trust the prediction)
    
    Reward:
    - +1 for correct prediction
    - -1 for wrong prediction
    - 0 for SKIP (neutral)
    """
    
    def __init__(
        self,
        state_size: int = RL_CONFIG['state_size'],
        action_size: int = RL_CONFIG['action_size']
    ):
        # Q-table: state -> action values
        self.q_table = np.zeros((state_size ** 4, action_size))  # Simplified state space
        
        # Hyperparameters
        self.lr = RL_CONFIG['learning_rate']
        self.gamma = RL_CONFIG['gamma']
        self.epsilon = RL_CONFIG['epsilon']
        self.epsilon_decay = RL_CONFIG['epsilon_decay']
        self.epsilon_min = RL_CONFIG['epsilon_min']
        
        # State/action sizes
        self.state_size = state_size
        self.action_size = action_size
        
        # History
        self.action_history = []
        self.reward_history = []
        
    def _discretize_state(
        self,
        confidence: float,
        win_rate: float,
        volatility: float,
        streak: int
    ) -> int:
        """
        Convert continuous state to discrete state index
        
        Args:
            confidence: Ensemble confidence (0-1)
            win_rate: Recent win rate (0-1)
            volatility: Current volatility (normalized)
            streak: Current streak (-10 to +10)
        
        Returns:
            State index for Q-table
        """
        # Bin each dimension
        conf_bin = min(int(confidence * self.state_size), self.state_size - 1)
        win_bin = min(int(win_rate * self.state_size), self.state_size - 1)
        
        # Volatility bins: low, medium, high
        if volatility < 0.3:
            vol_bin = 0
        elif volatility < 0.7:
            vol_bin = 1
        else:
            vol_bin = 2
        vol_bin = min(vol_bin, self.state_size - 1)
        
        # Streak bins
        streak_normalized = (streak + 10) / 20  # Normalize to 0-1
        streak_bin = min(int(streak_normalized * self.state_size), self.state_size - 1)
        
        # Combine into single state index
        state_idx = (
            conf_bin * (self.state_size ** 3) +
            win_bin * (self.state_size ** 2) +
            vol_bin * self.state_size +
            streak_bin
        )
        
        return min(state_idx, len(self.q_table) - 1)
    
    def get_state(
        self,
        confidence: float,
        win_rate: float,
        volatility: float,
        streak: int
    ) -> int:
        """Public interface for state discretization"""
        return self._discretize_state(confidence, win_rate, volatility, streak)
    
    def act(self, state: int, explore: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state index
            explore: Whether to use epsilon-greedy (False = pure exploitation)
        
        Returns:
            Action: 0 (SKIP) or 1 (EXECUTE)
        """
        # Epsilon-greedy exploration
        if explore and np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            # Exploit: choose best action
            action = np.argmax(self.q_table[state])
        
        self.action_history.append(action)
        return action
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int
    ):
        """
        Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Get best next action value
        best_next_value = np.max(self.q_table[next_state])
        
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # TD target
        target = reward + self.gamma * best_next_value
        
        # Update Q-value
        self.q_table[state][action] += self.lr * (target - current_q)
        
        # Track reward
        self.reward_history.append(reward)
    
    def decay_epsilon(self):
        """
        Decay exploration rate over time
        
        Call this after each episode/batch
        """
        self.epsilon = max(
            self.epsilon * self.epsilon_decay,
            self.epsilon_min
        )
    
    def get_action_distribution(self, state: int) -> np.ndarray:
        """
        Get probability distribution over actions
        
        Useful for analysis
        """
        q_values = self.q_table[state]
        
        # Softmax
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()
        
        return probs
    
    def get_statistics(self) -> dict:
        """
        Get RL agent statistics
        
        Returns:
            Dict with metrics
        """
        if len(self.reward_history) == 0:
            return {
                'total_actions': 0,
                'epsilon': self.epsilon
            }
        
        recent_rewards = self.reward_history[-200:]
        recent_actions = self.action_history[-200:]
        
        return {
            'total_actions': len(self.action_history),
            'epsilon': self.epsilon,
            'avg_reward_200': np.mean(recent_rewards),
            'execute_rate': sum(1 for a in recent_actions if a == 1) / len(recent_actions),
            'skip_rate': sum(1 for a in recent_actions if a == 0) / len(recent_actions),
            'cumulative_reward': sum(self.reward_history)
        }
    
    def save(self, filepath: str):
        """Save Q-table and parameters"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'action_history': self.action_history,
            'reward_history': self.reward_history,
            'lr': self.lr,
            'gamma': self.gamma
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Q-Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table and parameters"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.action_history = data['action_history']
        self.reward_history = data['reward_history']
        self.lr = data['lr']
        self.gamma = data['gamma']
        
        print(f"✅ Q-Agent loaded from {filepath}")
    
    def reset_history(self):
        """Clear history (keep Q-table)"""
        self.action_history = []
        self.reward_history = []
    
    def get_q_value(self, state: int, action: int) -> float:
        """Get Q-value for specific state-action pair"""
        return self.q_table[state][action]
    
    def get_best_action(self, state: int) -> Tuple[int, float]:
        """
        Get best action and its Q-value
        
        Returns:
            Tuple of (action, q_value)
        """
        action = np.argmax(self.q_table[state])
        q_value = self.q_table[state][action]
        
        return action, q_value
