import json

# Load the notebook
with open('train_100_episodes.ipynb', 'r') as f:
    nb = json.load(f)

# Replace Cell 4 with complete environment code
env_code = '''# StateDependentRewardEnv - Complete Implementation
import gym
from gym import spaces

class StateDependentRewardEnv(gym.Env):
    """Trading environment with state-dependent rewards."""
    
    def __init__(self, data, initial_capital=100000, trading_cost=0.0001,
                 buy_ratio=0.2, sell_ratio=0.5):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        
        self.feature_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 
            'SMA_20', 'SMA_50', 'Returns', 'Volatility'
        ]
        
        self.n_features = len(self.feature_columns) + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_features,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)
        
        self.buy_hold_returns = (
            self.data['Close'].values / self.data['Close'].values[0] - 1
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.portfolio_values = [self.initial_capital]
        self.trades = []
        self.last_buy_price = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        obs = self.data.loc[self.current_step, self.feature_columns].values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        current_price = self.data.loc[self.current_step, 'Close']
        shares_value = self.shares * current_price
        
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        position_ratio = shares_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        extended_obs = np.append(obs, [cash_ratio, position_ratio])
        
        return extended_obs.astype(np.float32)
    
    def step(self, action):
        current_price = self.data.loc[self.current_step, 'Close']
        
        rsi = self.data.loc[self.current_step, 'RSI']
        macd = self.data.loc[self.current_step, 'MACD']
        sma_20 = self.data.loc[self.current_step, 'SMA_20']
        sma_50 = self.data.loc[self.current_step, 'SMA_50']
        
        if action == 1:  # Buy
            amount_to_invest = self.cash * self.buy_ratio
            if amount_to_invest > current_price * (1 + self.trading_cost):
                shares_to_buy = amount_to_invest / (current_price * (1 + self.trading_cost))
                cost = shares_to_buy * current_price * (1 + self.trading_cost)
                self.shares += shares_to_buy
                self.cash -= cost
                self.last_buy_price = current_price
                self.trades.append(('buy', self.current_step, current_price, shares_to_buy))
        
        elif action == 2:  # Sell
            shares_to_sell = self.shares * self.sell_ratio
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.trading_cost)
                self.cash += proceeds
                self.trades.append(('sell', self.current_step, current_price, shares_to_sell))
                self.shares -= shares_to_sell
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        if not done:
            next_price = self.data.loc[self.current_step, 'Close']
            self.portfolio_value = self.cash + self.shares * next_price
        else:
            self.portfolio_value = self.cash + self.shares * current_price
        
        self.portfolio_values.append(self.portfolio_value)
        
        reward = self._calculate_state_dependent_reward(
            action, current_price, rsi, macd, sma_20, sma_50
        )
        
        obs = self._get_observation() if not done else np.zeros(self.n_features, dtype=np.float32)
        
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
        reward = 0.0
        
        shares_value = self.shares * current_price
        position_ratio = shares_value / self.portfolio_value if self.portfolio_value > 0 else 0
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # 1. Portfolio value change
        if len(self.portfolio_values) > 1:
            pv_change = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
            reward += pv_change * 10
        
        # 2. STATE-DEPENDENT action rewards
        if action == 0:  # Hold
            if 40 < rsi < 60:
                reward += 1.0
            elif rsi < 30 or rsi > 70:
                reward -= 0.5
        
        elif action == 1:  # Buy
            if rsi < 30 and cash_ratio > 0.2:
                reward += 5.0
            elif sma_20 > sma_50 and rsi < 40:
                reward += 3.0
            elif macd > 0 and rsi < 35:
                reward += 3.0
            elif rsi > 70:
                reward -= 3.0
            elif cash_ratio < 0.1:
                reward -= 2.0
            else:
                reward += 1.0
        
        elif action == 2:  # Sell
            if rsi > 70 and position_ratio > 0.3:
                reward += 5.0
            elif sma_20 < sma_50 and rsi > 60:
                reward += 3.0
            elif macd < 0 and rsi > 65:
                reward += 3.0
            elif rsi < 30:
                reward -= 3.0
            elif position_ratio < 0.1:
                reward -= 2.0
            else:
                reward += 1.0
        
        # 3. Profit realization bonus
        if action == 2 and self.last_buy_price > 0:
            profit_ratio = (current_price - self.last_buy_price) / self.last_buy_price
            if profit_ratio > 0:
                reward += profit_ratio * 20
        
        # 4. Alpha
        strategy_return = (self.portfolio_value / self.initial_capital - 1)
        bh_return = self.buy_hold_returns[min(self.current_step, len(self.buy_hold_returns)-1)]
        alpha = strategy_return - bh_return
        reward += alpha * 5
        
        return reward

print("✓ StateDependentRewardEnv defined")'''

nb['cells'][3]['source'] = env_code

# Save
with open('train_100_episodes.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("✓ Notebook fixed with complete environment code")
