"""
Tri-Axial Reward Environment for Financial Trading

Based on architecture document 2 (보고서 2, 4절):
- Axis 1: Financial Performance (Log returns, Alpha, Liquidity)
- Axis 2: Risk Management (Sortino, MDD penalty)
- Axis 3: Process/Reasoning (Consistency check)

Author: Based on DeepSeek-R1 GRPO architecture documents
Date: 2025-11-23
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from collections import deque


class TriAxialRewardEnv(gym.Env):
    """
    Trading environment with Tri-Axial reward system.
    
    Implements 보고서 2, 4절:
    - Financial axis (4.1절)
    - Risk axis (4.2절)
    - Process axis (4.3절)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        trading_cost: float = 0.0001,
        buy_ratio: float = 0.2,
        sell_ratio: float = 0.5,
        # Reward weights (보고서 2, Table line 69-75)
        w_log_return: float = 1.0,
        w_alpha: float = 0.5,
        w_liquidity: float = 0.3,
        w_sortino: float = 0.8,
        w_mdd: float = 2.0,
        w_consistency: float = 0.5,
        # Risk parameters
        eta_liquidity: float = 0.001,  # Liquidity penalty coefficient
        beta_mdd: float = 2.0,  # MDD penalty base
        k_mdd: float = 10.0,  # MDD exponential factor
        mdd_threshold: float = 0.10  # 10% MDD threshold
    ):
        """
        Initialize Tri-Axial Reward Environment.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            initial_capital: Starting capital
            trading_cost: Transaction cost ratio
            buy_ratio: Percentage of cash to use per Buy (0.2 = 20%)
            sell_ratio: Percentage of shares to sell per Sell (0.5 = 50%)
            w_*: Reward weights for each component
            eta_liquidity: Liquidity penalty coefficient (Almgren-Chriss)
            beta_mdd: MDD penalty base
            k_mdd: MDD exponential factor
            mdd_threshold: MDD threshold for exponential penalty
        """
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        
        # Reward weights
        self.w_log_return = w_log_return
        self.w_alpha = w_alpha
        self.w_liquidity = w_liquidity
        self.w_sortino = w_sortino
        self.w_mdd = w_mdd
        self.w_consistency = w_consistency
        
        # Risk parameters
        self.eta_liquidity = eta_liquidity
        self.beta_mdd = beta_mdd
        self.k_mdd = k_mdd
        self.mdd_threshold = mdd_threshold
        
        # Feature columns
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
        
        # Returns history for Sortino calculation
        self.returns_history = deque(maxlen=20)
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.portfolio_values = [self.initial_capital]
        self.trades = []
        self.last_buy_price = 0
        
        # Track peak for MDD calculation
        self.peak_portfolio_value = self.initial_capital
        self.current_drawdown = 0.0
        
        # Clear returns history
        self.returns_history.clear()
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        obs = self.data.loc[self.current_step, self.feature_columns].values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        current_price = self.data.loc[self.current_step, 'Close']
        shares_value = self.shares * current_price
        
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        position_ratio = shares_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        extended_obs = np.append(obs, [cash_ratio, position_ratio])
        
        return extended_obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            
        Returns:
            observation, reward, done, info
        """
        current_price = self.data.loc[self.current_step, 'Close']
        
        # Get market conditions
        rsi = self.data.loc[self.current_step, 'RSI']
        macd = self.data.loc[self.current_step, 'MACD']
        sma_20 = self.data.loc[self.current_step, 'SMA_20']
        sma_50 = self.data.loc[self.current_step, 'SMA_50']
        
        # Track trade execution and volume
        trade_executed = False
        trade_volume = 0.0
        
        # Execute action
        if action == 1:  # Buy
            amount_to_invest = self.cash * self.buy_ratio
            if amount_to_invest > current_price * (1 + self.trading_cost):
                shares_to_buy = amount_to_invest / (current_price * (1 + self.trading_cost))
                cost = shares_to_buy * current_price * (1 + self.trading_cost)
                self.shares += shares_to_buy
                self.cash -= cost
                self.last_buy_price = current_price
                self.trades.append(('buy', self.current_step, current_price, shares_to_buy))
                trade_executed = True
                trade_volume = shares_to_buy * current_price
        
        elif action == 2:  # Sell
            shares_to_sell = self.shares * self.sell_ratio
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.trading_cost)
                self.cash += proceeds
                self.trades.append(('sell', self.current_step, current_price, shares_to_sell))
                self.shares -= shares_to_sell
                trade_executed = True
                trade_volume = shares_to_sell * current_price
        
        # Move to next step
        prev_portfolio_value = self.portfolio_value
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Update portfolio value
        if not done:
            next_price = self.data.loc[self.current_step, 'Close']
            self.portfolio_value = self.cash + self.shares * next_price
        else:
            self.portfolio_value = self.cash + self.shares * current_price
        
        self.portfolio_values.append(self.portfolio_value)
        
        # Update peak and drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        
        # Calculate return for this step
        step_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns_history.append(step_return)
        
        # Calculate TRI-AXIAL reward
        reward = self._calculate_tri_axial_reward(
            action=action,
            current_price=current_price,
            trade_executed=trade_executed,
            trade_volume=trade_volume,
            rsi=rsi,
            macd=macd,
            sma_20=sma_20,
            sma_50=sma_50
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
            'cash_ratio': self.cash / self.portfolio_value if self.portfolio_value > 0 else 0,
            'current_drawdown': self.current_drawdown,
            'peak_value': self.peak_portfolio_value
        }
        
        return obs, reward, done, info
    
    def _calculate_tri_axial_reward(
        self,
        action: int,
        current_price: float,
        trade_executed: bool,
        trade_volume: float,
        rsi: float,
        macd: float,
        sma_20: float,
        sma_50: float
    ) -> float:
        """
        Calculate Tri-Axial reward.
        
        Implementation of 보고서 2, 4절:
        R_total = w1*r_financial + w2*r_risk + w3*r_process
        
        Returns:
            total_reward: Combined reward from all three axes
        """
        # Axis 1: Financial Performance (보고서 2, 4.1절)
        r_financial = self._financial_reward(trade_volume)
        
        # Axis 2: Risk Management (보고서 2, 4.2절)
        r_risk = self._risk_reward()
        
        # Axis 3: Process/Reasoning (보고서 2, 4.3절)
        r_process = self._process_reward(action, trade_executed, rsi, macd, sma_20, sma_50)
        
        # Combine (보고서 2, line 140)
        total_reward = r_financial + r_risk + r_process
        
        return total_reward
    
    def _financial_reward(self, trade_volume: float) -> float:
        """
        Financial axis reward (보고서 2, 4.1절).
        
        Components:
        1. Log returns (대칭적)
        2. Alpha (벤치마크 초과 수익)
        3. Liquidity penalty (Almgren-Chriss)
        
        Returns:
            financial_reward
        """
        reward = 0.0
        
        # 1. Log returns (보고서 2, line 79)
        if len(self.portfolio_values) > 1:
            log_return = np.log(self.portfolio_values[-1] / self.portfolio_values[-2])
            reward += self.w_log_return * log_return
        
        # 2. Alpha (보고서 2, Table line 70)
        if self.current_step > 0:
            strategy_return = (self.portfolio_value / self.initial_capital - 1)
            bh_return = self.buy_hold_returns[min(self.current_step, len(self.buy_hold_returns)-1)]
            alpha = strategy_return - bh_return
            reward += self.w_alpha * alpha
        
        # 3. Liquidity penalty (보고서 2, line 82)
        # Market impact: -η * Volume^1.5
        if trade_volume > 0:
            liquidity_penalty = -self.eta_liquidity * (trade_volume ** 1.5)
            reward += self.w_liquidity * liquidity_penalty
        
        return reward
    
    def _risk_reward(self) -> float:
        """
        Risk axis reward (보고서 2, 4.2절).
        
        Components:
        1. Sortino Ratio (하방 변동성만 페널티)
        2. Exponential MDD penalty (파산 방지)
        
        Returns:
            risk_reward
        """
        reward = 0.0
        
        # 1. Sortino Ratio (보고서 2, line 88-89)
        if len(self.returns_history) >= 5:
            returns_array = np.array(self.returns_history)
            downside_returns = returns_array[returns_array < 0]
            
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns) + 1e-8
                sortino = np.mean(returns_array) / downside_std
                reward += self.w_sortino * sortino
        
        # 2. Exponential MDD penalty (보고서 2, line 90-92)
        # P_MDD = -β * (e^(k * MDD) - 1)
        if self.current_drawdown > 0:
            mdd_penalty = -self.beta_mdd * (np.exp(self.k_mdd * self.current_drawdown) - 1)
            reward += self.w_mdd * mdd_penalty
        
        return reward
    
    def _process_reward(
        self,
        action: int,
        trade_executed: bool,
        rsi: float,
        macd: float,
        sma_20: float,
        sma_50: float
    ) -> float:
        """
        Process axis reward (보고서 2, 4.3절).
        
        Components:
        1. Logical consistency check
        2. Trade execution penalty
        
        Returns:
            process_reward
        """
        reward = 0.0
        
        # 1. Logical consistency (보고서 2, line 98)
        # RSI > 70 (과매수) → Buy는 논리적 모순
        # RSI < 30 (과매도) → Sell은 논리적 모순
        if action == 1:  # Buy
            if rsi > 70:
                reward -= 5.0  # 논리적 모순 페널티
            elif rsi < 30:
                reward += 2.0  # 논리적 일관성 보상
        
        elif action == 2:  # Sell
            if rsi < 30:
                reward -= 5.0  # 논리적 모순 페널티
            elif rsi > 70:
                reward += 2.0  # 논리적 일관성 보상
        
        # 2. Trade execution penalty (impossible trades)
        if (action == 1 or action == 2) and not trade_executed:
            reward -= 10.0  # 거래 불가능 시 큰 페널티
        
        # 3. Consistency bonus for executed trades
        if trade_executed:
            reward += self.w_consistency * 0.5
        
        return reward


if __name__ == "__main__":
    # Test initialization
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("Testing TriAxialRewardEnv...")
    
    # Download test data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    ticker = yf.Ticker('SPY')
    data = ticker.history(start=start_date, end=end_date)
    
    # Calculate indicators
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    data = data.dropna()
    
    # Initialize environment
    env = TriAxialRewardEnv(data)
    
    print("✓ TriAxialRewardEnv initialized")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.n}")
    print(f"  - Data length: {len(data)} days")
    
    # Test episode
    state = env.reset()
    done = False
    total_reward = 0
    
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"\n✓ Test episode completed")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"  - Alpha: {info['alpha']*100:+.2f}%")
    print(f"  - Current drawdown: {info['current_drawdown']*100:.2f}%")
