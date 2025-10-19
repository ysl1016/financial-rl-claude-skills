# DeepSeek-R1 Financial Trading Reinforcement Learning Model API Documentation

This document explains the API usage of the DeepSeek-R1 based financial trading reinforcement learning model. This model uses the GRPO (Generalized Reward-Penalty Optimization) algorithm to optimize financial trading decisions.

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [Main Modules](#2-main-modules)
3. [Data Processing API](#3-data-processing-api)
4. [Model API](#4-model-api)
5. [Environment API](#5-environment-api)
6. [Evaluation and Backtesting API](#6-evaluation-and-backtesting-api)
7. [Utility API](#7-utility-api)
8. [API Usage Examples](#8-api-usage-examples)

## 1. Model Overview

The DeepSeek-R1 financial trading reinforcement learning model provides the following core features:

- Single and multi-asset trading environments
- Custom reward functions
- Online learning and market regime detection
- Transformer-based reinforcement learning algorithms
- Comprehensive backtesting and performance evaluation

## 2. Main Modules

The model consists of the following main modules:

- `data`: Data processing and management
- `models`: Reinforcement learning model implementation
- `utils`: Utility functions
- `environments`: Trading environments
- `rewards`: Reward function components
- `evaluation`: Performance evaluation tools

## 3. Data Processing API

### 3.1 Data Download

```python
from src.data.data_processor import download_stock_data, process_data

try:
    # Download single asset data
    data = download_stock_data(symbol="SPY", start_date="2020-01-01", end_date="2022-12-31")
    # Process data and calculate technical indicators
    processed_data = process_data(symbol="SPY", start_date="2020-01-01", end_date="2022-12-31")
except ValueError as e:
    print(f"Data preparation failed: {e}")
```

#### Key Parameters

- `symbol` (str): Asset symbol (e.g., "SPY", "AAPL")
- `start_date` (str, optional): Start date (YYYY-MM-DD format)
- `end_date` (str, optional): End date (YYYY-MM-DD format)

#### Return Values

- `download_stock_data`: pandas DataFrame with OHLCV data. Raises `ValueError` if the data cannot be retrieved or is invalid.
- `process_data`: processed pandas DataFrame with added technical indicators. Propagates `ValueError` from `download_stock_data`.

### 3.2 Multi-Asset Data Processing

```python
from src.data.multi_asset_processor import download_multi_assets, process_multi_assets

# Download multiple asset data
symbols = ["SPY", "QQQ", "GLD", "TLT"]
multi_data = download_multi_assets(symbols=symbols, start_date="2020-01-01", end_date="2022-12-31")

# Process multi-asset data
processed_multi_data = process_multi_assets(symbols=symbols, start_date="2020-01-01", end_date="2022-12-31")
```

#### Key Parameters

- `symbols` (list of str): List of asset symbols
- `start_date` (str, optional): Start date (YYYY-MM-DD format)
- `end_date` (str, optional): End date (YYYY-MM-DD format)
- `include_correlation` (bool, optional): Whether to calculate correlation between assets

#### Return Values

- `download_multi_assets`: Dictionary containing OHLCV data for each asset
- `process_multi_assets`: Dictionary containing processed data and correlation information for each asset

## 4. Model API

### 4.1 GRPO Agent

```python
from src.models.grpo_agent import GRPOAgent

# Create GRPO agent
agent = GRPOAgent(
    state_dim=14,
    action_dim=3,
    hidden_dim=128,
    lr=3e-4,
    gamma=0.99,
    reward_scale=1.0,
    penalty_scale=0.5
)

# Select action
action = agent.select_action(state)

# Store transition
agent.store_transition(state, action, reward, next_state, done)

# Update model
update_info = agent.update()

# Save model
agent.save("models/grpo_model.pt")

# Load model
agent.load("models/grpo_model.pt")
```

#### Key Parameters

- `state_dim` (int): State space dimension
- `action_dim` (int): Action space dimension
- `hidden_dim` (int, optional): Hidden layer dimension (default: 128)
- `lr` (float, optional): Learning rate (default: 3e-4)
- `gamma` (float, optional): Discount factor (default: 0.99)
- `reward_scale` (float, optional): Reward scale (default: 1.0)
- `penalty_scale` (float, optional): Penalty scale (default: 0.5)
- `device` (str, optional): Training device ('cuda' or 'cpu')

### 4.2 DeepSeek GRPO Agent

```python
from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent

# Create DeepSeek GRPO agent
agent = DeepSeekGRPOAgent(
    state_dim=14,
    action_dim=3,
    seq_length=30,
    hidden_dim=256,
    lr=3e-4,
    gamma=0.99
)

# Select action (with history)
action = agent.select_action(state, history)

# Store transition
agent.store_transition(state, action, reward, next_state, done, history)

# Update model
update_info = agent.update()
```

#### Key Parameters

- `state_dim` (int): State space dimension
- `action_dim` (int): Action space dimension
- `seq_length` (int): Sequence length (history data)
- `hidden_dim` (int, optional): Hidden layer dimension (default: 256)
- `lr` (float, optional): Learning rate (default: 3e-4)
- `gamma` (float, optional): Discount factor (default: 0.99)
- `device` (str, optional): Training device ('cuda' or 'cpu')

## 5. Environment API

### 5.1 Single Asset Trading Environment

```python
from src.models.trading_env import TradingEnv

# Create trading environment
env = TradingEnv(
    data=processed_data,
    initial_capital=100000,
    trading_cost=0.0005,
    slippage=0.0001,
    risk_free_rate=0.02,
    max_position_size=1.0,
    stop_loss_pct=0.02
)

# Reset environment
state = env.reset()

# Take action
next_state, reward, done, info = env.step(action)
```

#### Key Parameters

- `data` (pandas.DataFrame): Processed OHLCV data
- `initial_capital` (float, optional): Initial capital (default: 100000)
- `trading_cost` (float, optional): Trading cost (default: 0.0005)
- `slippage` (float, optional): Slippage (default: 0.0001)
- `risk_free_rate` (float, optional): Risk-free rate (default: 0.02)
- `max_position_size` (float, optional): Maximum position size (default: 1.0)
- `stop_loss_pct` (float, optional): Stop-loss percentage (default: 0.02)

### 5.2 Multi-Asset Trading Environment

```python
from src.environments.multi_asset_env import MultiAssetTradingEnv

# Create multi-asset trading environment
env = MultiAssetTradingEnv(
    asset_data=processed_multi_data,
    initial_capital=100000,
    trading_cost=0.0005,
    slippage=0.0001,
    risk_free_rate=0.02,
    max_position_size=1.5,
    stop_loss_pct=0.02,
    max_asset_weight=0.5,
    correlation_lookback=30
)

# Reset environment
state = env.reset()

# Take action (position ratio for each asset)
action = np.array([0.5, -0.3, 0.0, 0.2])  # Position for each asset (-1.0 ~ 1.0)
next_state, reward, done, info = env.step(action)
```

#### Key Parameters

- `asset_data` (dict): Processed data for each asset
- `initial_capital` (float, optional): Initial capital (default: 100000)
- `trading_cost` (float, optional): Trading cost (default: 0.0005)
- `slippage` (float, optional): Slippage (default: 0.0001)
- `risk_free_rate` (float, optional): Risk-free rate (default: 0.02)
- `max_position_size` (float, optional): Maximum total leverage (default: 1.5)
- `stop_loss_pct` (float, optional): Stop-loss percentage (default: 0.02)
- `max_asset_weight` (float, optional): Maximum weight for a single asset (default: 0.5)
- `correlation_lookback` (int, optional): Correlation calculation lookback period (default: 30)

## 6. Evaluation and Backtesting API

### 6.1 Performance Evaluation

```python
from src.utils.evaluation import calculate_metrics, plot_performance

# Calculate performance metrics
metrics = calculate_metrics(
    portfolio_values=env.portfolio_values,
    benchmark_values=benchmark_values,
    risk_free_rate=0.02
)

# Visualize performance
plot_performance(
    portfolio_values=env.portfolio_values,
    benchmark_values=benchmark_values,
    trades=env.trades
)
```

#### Key Metrics

- Total Return
- Annual Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Volatility
- Win Rate
- Profit-Loss Ratio
- Alpha
- Beta

### 6.2 Backtesting

```python
from src.utils.backtest_utils import run_backtest

# Run backtest
results = run_backtest(
    env_factory=lambda: TradingEnv(data=test_data, **env_params),
    agent=agent,
    episodes=10,
    render=True
)
```

#### Key Parameters

- `env_factory` (callable): Environment creation function
- `agent` (object): Trained agent
- `episodes` (int, optional): Number of backtest episodes (default: 10)
- `render` (bool, optional): Whether to visualize the backtest (default: False)

#### Return Value

Dictionary containing backtest results:

- `portfolio_values`: Portfolio value time series
- `returns`: Daily returns
- `trades`: Trade history
- `metrics`: Performance metrics

## 7. Utility API

### 7.1 Technical Indicators

```python
from src.utils.indicators import calculate_technical_indicators

# Calculate technical indicators
indicators = calculate_technical_indicators(data)
```

### 7.2 Market Regime Detection

```python
from src.utils.regime_detection import MarketRegimeDetector

# Create market regime detector
regime_detector = MarketRegimeDetector(
    n_regimes=4,
    lookback_period=60,
    min_samples_per_regime=20,
    stability_threshold=0.6
)

# Detect market regimes
regimes = regime_detector.detect_regimes(market_data)
```

## 8. API Usage Examples

### 8.1 Basic Trading Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_processor import process_data
from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent

# Process data and create train/test split
data = process_data('SPY', start_date='2020-01-01', end_date='2022-01-01')
split_idx = int(len(data) * 0.8)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

# Create environment
env = TradingEnv(
    data=train_data,
    initial_capital=100000,
    trading_cost=0.0005,
    slippage=0.0001
)

# Create agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = GRPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=128,
    lr=3e-4,
    gamma=0.99
)

# Training
n_episodes = 100
for episode in range(n_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
    
    # Update model
    update_info = agent.update()
    
    print(f"Episode {episode+1}/{n_episodes}, Reward: {episode_reward:.2f}")

# Save model
agent.save("models/grpo_model.pt")

# Backtest
test_env = TradingEnv(
    data=test_data,
    initial_capital=100000,
    trading_cost=0.0005,
    slippage=0.0001
)

state = test_env.reset()
done = False

while not done:
    action = agent.select_action(state)
    next_state, reward, done, info = test_env.step(action)
    state = next_state

# Visualize results
plt.figure(figsize=(15, 7))
plt.plot(test_env.portfolio_values)
plt.title('Portfolio Value During Backtest')
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.show()
```

### 8.2 Multi-Asset Trading Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.multi_asset_processor import process_multi_assets
from src.environments.multi_asset_env import MultiAssetTradingEnv
from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent

# Process multi-asset data
symbols = ['SPY', 'QQQ', 'GLD', 'TLT']
asset_data = process_multi_assets(symbols=symbols, start_date='2020-01-01', end_date='2022-01-01')

# Split train/test data
train_length = int(len(asset_data[symbols[0]]) * 0.8)
train_data = {symbol: data.iloc[:train_length] for symbol, data in asset_data.items()}
test_data = {symbol: data.iloc[train_length:] for symbol, data in asset_data.items()}

# Create environment
env = MultiAssetTradingEnv(
    asset_data=train_data,
    initial_capital=100000,
    trading_cost=0.0005,
    slippage=0.0001,
    max_position_size=1.5,
    max_asset_weight=0.5
)

# Create agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # Continuous action for each asset
seq_length = 30  # History length

agent = DeepSeekGRPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    seq_length=seq_length,
    hidden_dim=256,
    lr=3e-4,
    gamma=0.99
)

# Training code omitted...

# Backtest
test_env = MultiAssetTradingEnv(
    asset_data=test_data,
    initial_capital=100000,
    trading_cost=0.0005,
    slippage=0.0001,
    max_position_size=1.5,
    max_asset_weight=0.5
)

# Backtesting code omitted...

# Compare performance with individual assets
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(test_env.portfolio_values, label='Portfolio')
for symbol in symbols:
    # Calculate and visualize single asset Buy & Hold performance
    pass
plt.legend()
plt.title('Portfolio Value vs Individual Assets (Buy & Hold)')
plt.grid(True)

plt.subplot(2, 1, 2)
# Visualize asset allocation
pass
plt.title('Asset Allocation Over Time')
plt.tight_layout()
plt.show()
```

## Additional Resources

For more details, please refer to the following documents:

- [Model Architecture Documentation](https://github.com/ysl1016/financial-rl-trading/blob/master/docs/architecture.md)
- [Use Case Examples](https://github.com/ysl1016/financial-rl-trading/blob/master/examples/README.md)
- [Tutorial Notebooks](https://github.com/ysl1016/financial-rl-trading/blob/master/notebooks/tutorial.ipynb)