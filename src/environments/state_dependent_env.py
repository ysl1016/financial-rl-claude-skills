"""
State-Dependent Reward Environment for Financial Trading

This environment implements state-dependent rewards based on market conditions (RSI, MACD)
to prevent extreme trading behaviors (Buy 100% or Sell 100%).

Key Features:
- RSI-based Buy/Sell rewards (oversold → Buy bonus, overbought → Sell bonus)
- MACD-based trend rewards (golden cross → Buy, death cross → Sell)
- Position-aware rewards (cash ratio, position ratio)
- Profit realization bonus

Author: Based on architecture_modification_plan.md
Date: 2025-11-23
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd


class StateDependentRewardEnv(gym.Env):
    """
    Trading environment with state-dependent rewards.
    
    Rewards are calculated based on:
    1. Market conditions (RSI, MACD, SMA crossovers)
    2. Portfolio state (cash ratio, position ratio)
    3. Action appropriateness (Buy in oversold, Sell in overbought)
    4. Profit realization
    """
    
    def __init__(self, data, initial_capital=100000, trading_cost=0.0001,
                 buy_ratio=0.2, sell_ratio=0.5):
        """
        Initialize state-dependent reward environment.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            initial_capital: Starting capital
            trading_cost: Transaction cost ratio
            buy_ratio: Percentage of cash to use per Buy action (0.2 = 20%)
            sell_ratio: Percentage of shares to sell per Sell action (0.5 = 50%)
        """
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        
        # Feature columns (must exist in data)
        self.feature_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 
            'SMA_20', 'SMA_50', 'Returns', 'Volatility'
        ]
        
        # Validate required columns
        for col in self.feature_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # State space: 8 features + 2 portfolio info = 10
        self.n_features = len(self.feature_columns) + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_features,), dtype=np.float32
        )
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Buy & Hold benchmark
        self.buy_hold_returns = (
            self.data['Close'].values / self.data['Close'].values[0] - 1
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.portfolio_values = [self.initial_capital]
        self.trades = []
        self.last_buy_price = 0  # Track last buy price for profit calculation
        
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get current state observation.
        
        Returns:
            np.array: [8 technical features + cash_ratio + position_ratio]
        """
        # Basic technical features
        obs = self.data.loc[self.current_step, self.feature_columns].values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Portfolio state
        current_price = self.data.loc[self.current_step, 'Close']
        shares_value = self.shares * current_price
        
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        position_ratio = shares_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Extended observation: [8 features + 2 portfolio info]
        extended_obs = np.append(obs, [cash_ratio, position_ratio])
        
        return extended_obs.astype(np.float32)
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            
        Returns:
            observation, reward, done, info
        """
        current_price = self.data.loc[self.current_step, 'Close']
        
        # Get market conditions for reward calculation
        rsi = self.data.loc[self.current_step, 'RSI']
        macd = self.data.loc[self.current_step, 'MACD']
        sma_20 = self.data.loc[self.current_step, 'SMA_20']
        sma_50 = self.data.loc[self.current_step, 'SMA_50']
        
        # Execute action
        if action == 1:  # Buy
            amount_to_invest = self.cash * self.buy_ratio
            if amount_to_invest > current_price * (1 + self.trading_cost):
                shares_to_buy = amount_to_invest / (current_price * (1 + self.trading_cost))
                cost = shares_to_buy * current_price * (1 + self.trading_cost)
                self.shares += shares_to_buy
                self.cash -= cost
                self.last_buy_price = current_price  # Record buy price
                self.trades.append(('buy', self.current_step, current_price, shares_to_buy))
        
        elif action == 2:  # Sell
            shares_to_sell = self.shares * self.sell_ratio
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.trading_cost)
                self.cash += proceeds
                self.trades.append(('sell', self.current_step, current_price, shares_to_sell))
                self.shares -= shares_to_sell
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Update portfolio value
        if not done:
            next_price = self.data.loc[self.current_step, 'Close']
            self.portfolio_value = self.cash + self.shares * next_price
        else:
            self.portfolio_value = self.cash + self.shares * current_price
        
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate STATE-DEPENDENT reward
        reward = self._calculate_state_dependent_reward(
            action, current_price, rsi, macd, sma_20, sma_50
        )
        
        # Get next observation
        obs = self._get_observation() if not done else np.zeros(self.n_features, dtype=np.float32)
        
        # Info
        current_price_now = self.data.loc[min(self.current_step, len(self.data)-1), 'Close']
        shares_value = self.shares * current_price_now
        position_ratio = shares_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        strategy_return = (self.portfolio_value / self.initial_capital - 1)
        bh_return = self.buy_hold_returns[min(self.current_step, len(self.buy_hold_returns)-1)]
        alpha = strategy_return - bh_return
        
        info = {
            'portfolio_value': self.portfolio_value,
            'trades': len(self.trades),
            'alpha': alpha,
            'position_ratio': position_ratio,
            'cash_ratio': self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        }
        
        return obs, reward, done, info
    
    def _calculate_state_dependent_reward(self, action, current_price, rsi, macd, sma_20, sma_50):
        """
        Calculate reward based on market state and action appropriateness.
        
        Key principle: Same action gets different rewards in different market conditions.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            current_price: Current asset price
            rsi: Relative Strength Index (0-100)
            macd: MACD indicator
            sma_20: 20-day Simple Moving Average
            sma_50: 50-day Simple Moving Average
            
        Returns:
            float: State-dependent reward
        """
        reward = 0.0
        
        # Portfolio state
        shares_value = self.shares * current_price
        position_ratio = shares_value / self.portfolio_value if self.portfolio_value > 0 else 0
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # 1. Portfolio value change (baseline)
        if len(self.portfolio_values) > 1:
            pv_change = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
            reward += pv_change * 10
        
        # 2. STATE-DEPENDENT action rewards
        if action == 0:  # Hold
            # Neutral market → Hold is good
            if 40 < rsi < 60:
                reward += 1.0
            # Extreme conditions → Hold is opportunity loss
            elif rsi < 30 or rsi > 70:
                reward -= 0.5
        
        elif action == 1:  # Buy
            # EXCELLENT: Oversold + Cash available
            if rsi < 30 and cash_ratio > 0.2:
                reward += 5.0  # Strong buy signal!
            
            # GOOD: Golden cross + Oversold
            elif sma_20 > sma_50 and rsi < 40:
                reward += 3.0
            
            # GOOD: Positive MACD + Oversold
            elif macd > 0 and rsi < 35:
                reward += 3.0
            
            # BAD: Overbought
            elif rsi > 70:
                reward -= 3.0  # Bad timing!
            
            # BAD: No cash
            elif cash_ratio < 0.1:
                reward -= 2.0
            
            # NEUTRAL
            else:
                reward += 1.0
        
        elif action == 2:  # Sell
            # EXCELLENT: Overbought + Shares available
            if rsi > 70 and position_ratio > 0.3:
                reward += 5.0  # Strong sell signal!
            
            # GOOD: Death cross + Overbought
            elif sma_20 < sma_50 and rsi > 60:
                reward += 3.0
            
            # GOOD: Negative MACD + Overbought
            elif macd < 0 and rsi > 65:
                reward += 3.0
            
            # BAD: Oversold
            elif rsi < 30:
                reward -= 3.0  # Bad timing!
            
            # BAD: No shares to sell
            elif position_ratio < 0.1:
                reward -= 2.0
            
            # NEUTRAL
            else:
                reward += 1.0
        
        # 3. Profit realization bonus
        if action == 2 and self.last_buy_price > 0:
            profit_ratio = (current_price - self.last_buy_price) / self.last_buy_price
            if profit_ratio > 0:
                reward += profit_ratio * 20  # Reward profit taking!
        
        # 4. Alpha (small weight to avoid overfitting to benchmark)
        strategy_return = (self.portfolio_value / self.initial_capital - 1)
        bh_return = self.buy_hold_returns[min(self.current_step, len(self.buy_hold_returns)-1)]
        alpha = strategy_return - bh_return
        reward += alpha * 5
        
        return reward
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Shares: {self.shares:.2f}")
            print(f"Trades: {len(self.trades)}")
