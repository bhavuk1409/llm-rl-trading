"""
Simplified Trading Environment
Gym-compatible environment for stock trading
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """Simple but realistic trading environment."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Features (all numeric columns except ticker and date)
        self.feature_cols = [
            col for col in df.columns 
            if col not in ['ticker', 'date', 'Date'] and df[col].dtype in [np.float64, np.int64]
        ]
        
        # Action space: continuous [-1, 1] where -1=sell all, 0=hold, 1=buy all
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: features + portfolio state
        obs_dim = len(self.feature_cols) + 3  # features + [cash, shares, value]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.total_value = self.initial_capital
        
        self.history = []
        
        return self._get_obs(), {}
    
    def step(self, action):
        action = float(action[0])
        current_price = self.df.loc[self.current_step, 'close']
        
        # Calculate target position
        max_shares = int(self.cash / (current_price * (1 + self.commission + self.slippage)))
        target_shares = int(action * max_shares) if action > 0 else int(action * self.shares)
        shares_to_trade = target_shares - self.shares
        
        # Execute trade with costs
        if shares_to_trade != 0:
            trade_value = abs(shares_to_trade) * current_price
            cost = trade_value * (self.commission + self.slippage)
            
            self.shares += shares_to_trade
            self.cash -= (shares_to_trade * current_price + cost)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        if self.current_step < len(self.df):
            new_price = self.df.loc[self.current_step, 'close']
        else:
            new_price = current_price
        
        new_value = self.cash + self.shares * new_price
        
        # Reward is change in portfolio value
        reward = (new_value - self.total_value) / self.initial_capital
        
        self.total_value = new_value
        
        # Record history
        self.history.append({
            'step': self.current_step,
            'price': new_price,
            'action': action,
            'shares': self.shares,
            'cash': self.cash,
            'value': self.total_value,
            'return': (self.total_value / self.initial_capital - 1) * 100
        })
        
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.total_value <= 0
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        # Get market features
        features = self.df.loc[self.current_step, self.feature_cols].values.astype(np.float32)
        
        # Add portfolio state (normalized)
        portfolio_state = np.array([
            self.cash / self.initial_capital,
            self.shares / 1000,  # Normalize shares
            self.total_value / self.initial_capital
        ], dtype=np.float32)
        
        return np.concatenate([features, portfolio_state])
    
    def _get_info(self):
        return {
            'step': self.current_step,
            'total_value': self.total_value,
            'return': (self.total_value / self.initial_capital - 1) * 100
        }
    
    def get_history(self):
        return pd.DataFrame(self.history)


if __name__ == "__main__":
    from data_handler import DataHandler
    
    logging.basicConfig(level=logging.INFO)
    
    # Test environment
    handler = DataHandler(["AAPL"], "2023-01-01", "2024-01-01")
    df = handler.fetch_and_process()
    
    env = TradingEnv(df)
    
    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space)
    
    # Random episode
    obs, info = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    
    print(f"\nFinal return: {info['return']:.2f}%")