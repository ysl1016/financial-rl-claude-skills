# Financial RL Trading Model with Claude Skills

This repository contains a Reinforcement Learning based Financial Trading Model with integrated Claude AI capabilities. The project combines traditional RL trading with advanced AI analysis and professional investment reporting.

## Key Features

### Trading & RL Capabilities
* Custom OpenAI Gym environment for trading
* Multiple technical indicators implementation
* Risk management with stop-loss
* Transaction costs and slippage simulation
* Sharpe ratio based reward function
* GRPO (Generalized Reward-Penalty Optimization) agent implementation
* Comprehensive evaluation metrics
* Advanced testing & optimization tools
* Benchmarking framework for strategy comparison
* DeepSeek-R1 transformer-based model integration

### Claude AI Integration 🆕
* **Claude Skills**: Natural language investment report generation
* **AI-Powered Analysis**: Market sentiment and trend analysis using Claude 3.7 Sonnet
* **Professional Reporting**: Institutional-grade investment reports
* **Advanced Charting**: 4 types of high-resolution charts (300 DPI)
* **Dual Interface**: Both conversational (Skills) and programmatic (Python API) access

## Installation

```bash
pip install -r requirements.txt
```

## Package Structure

```
financial-rl-claude-skills/
├── .claude/skills/trading-analysis/  # 🆕 Claude Skills integration
│   ├── SKILL.md                      # Skill definition
│   ├── reference.md                  # Technical reference
│   ├── examples.md                   # Usage examples
│   └── scripts/                      # Execution wrapper
│
├── docs/                             # 📚 Documentation (organized)
│   ├── guides/                       # User guides
│   │   ├── QUICKSTART.md
│   │   ├── CLAUDE_INTEGRATION_GUIDE.md
│   │   ├── CLAUDE_SKILLS_INTEGRATION.md
│   │   └── TESTING_GUIDE.md
│   ├── reports/                      # Technical reports
│   ├── architecture/                 # Architecture docs
│   ├── api/                          # API documentation
│   └── security/                     # Security guidelines
│
├── scripts/                          # 🔧 Executable scripts
│   ├── reports/                      # Report generation
│   │   └── generate_investment_report.py
│   └── utils/                        # Utility scripts
│
├── src/                              # 💻 Source code
│   ├── api/                          # REST API
│   ├── claude_integration/           # 🆕 Claude AI integration
│   ├── data/                         # Data processing
│   ├── deployment/                   # Model deployment
│   ├── models/                       # RL agents & environments
│   ├── monitoring/                   # Performance monitoring
│   ├── reporting/                    # 🆕 Report generation
│   └── utils/                        # Utilities
│
├── tests/                            # ✅ Unit & integration tests
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── run_tests.py
│
├── examples/                         # 📝 Example scripts
├── reports/                          # 📊 Generated investment reports
├── monitoring/                       # 📈 Monitoring configs
├── .env.example
├── requirements.txt
└── README.md
```

## Documentation

### Quick Start Guides
* **🆕 [Quick Start](docs/guides/QUICKSTART.md)** - Get started in 5 minutes
* **🆕 [Claude Skills Integration](docs/guides/CLAUDE_SKILLS_INTEGRATION.md)** - Natural language report generation
* [Claude Integration Guide](docs/guides/CLAUDE_INTEGRATION_GUIDE.md) - Hybrid RL-Claude trading
* [Testing Guide](docs/guides/TESTING_GUIDE.md) - Guide for testing and optimization

### Technical Documentation
* [API Documentation](docs/api/api_documentation.md) - Detailed API reference
* [Architecture Overview](docs/architecture/DeepSeek-R1_Financial_Trading_Model_Architecture.md) - System architecture
* [Security Guidelines](docs/security/SECURITY.md) - Security best practices

### Reports & Summaries
* **🆕 [Investment Report Summary](docs/reports/INVESTMENT_REPORT_SUMMARY.md)** - Report system overview
* [Project Reorganization Plan](PROJECT_REORGANIZATION_PLAN.md) - Structure improvements

## Quick Start

### 🆕 Using Claude Skills (Natural Language)

The easiest way to generate investment reports is through natural language in Claude Code:

```
"Generate an investment report for SPY"
"Analyze AAPL and create a professional report"
"Compare QQQ and SPY - which should I invest in?"
```

Claude automatically discovers and uses the `trading-analysis` skill to generate comprehensive reports with:
- Real-time market data from Yahoo Finance
- 10+ technical indicators (RSI, MACD, Moving Averages, etc.)
- Claude AI market analysis and sentiment
- 4 high-resolution charts (300 DPI)
- Institutional-grade markdown reports
- Investment recommendations with entry/exit criteria

**See [Claude Skills Integration Guide](docs/guides/CLAUDE_SKILLS_INTEGRATION.md) for detailed examples.**

### 🆕 Using Python API (Programmatic)

Generate reports programmatically for automation:

```bash
# Generate investment report for SPY
python3 scripts/reports/generate_investment_report.py --symbol SPY --client "Acme Capital"

# Custom title
python3 scripts/reports/generate_investment_report.py --symbol AAPL --title "Q4 2025 AAPL Analysis"
```

Or use in your Python code:

```python
import sys
sys.path.insert(0, 'scripts/reports')
from generate_investment_report import generate_complete_report

success = generate_complete_report(
    symbol='SPY',
    client_name='Institutional Investors',
    report_title='SPY Market Analysis'
)
# Generates reports in reports/ directory
```

## Traditional RL Trading Usage

### Data Processing

```python
from src.data.data_processor import process_data

try:
    # Download and process stock data
    data = process_data('SPY', start_date='2020-01-01')
except ValueError as e:
    print(f"Data processing failed: {e}")
```

### Setting Up Trading Environment

```python
from src.models.trading_env import TradingEnv

# Create trading environment
env = TradingEnv(
    data=data,
    initial_capital=100000,
    trading_cost=0.0005,
    slippage=0.0001
)
```

### Creating and Training GRPO Agent

```python
from src.models.grpo_agent import GRPOAgent

# Create agent
agent = GRPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=128,
    lr=3e-4
)

# Basic training loop
for episode in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        
        # Update every 10 steps
        if episode % 10 == 0:
            agent.update()
        
        state = next_state
```

### Using DeepSeek-R1 GRPO Agent

```python
from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent

# Create DeepSeek-R1 based GRPO agent
agent = DeepSeekGRPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    seq_length=30,  # For temporal context
    hidden_dim=256,
    lr=3e-4
)

# See API documentation for full usage details
```

### Using Pre-built Examples

```bash
# Basic example of environment usage
python examples/trading_example.py

# Training with GRPO
python examples/train_grpo.py

# Hyperparameter optimization and benchmarking
python examples/optimize_and_benchmark.py --symbol SPY --optimize --train --benchmark
```

## GRPO Model Details

The GRPO (Generalized Reward-Penalty Optimization) agent implements a policy optimization algorithm with the following features:

* Actor-Critic architecture with shared network layers
* Generalized Advantage Estimation (GAE)
* Proximal Policy Optimization (PPO) clipping
* Entropy regularization for exploration
* Value function loss with coefficient
* Gradient clipping for stability

The model uses the following hyperparameters by default:
* Learning rate: 3e-4
* Discount factor (gamma): 0.99
* GAE lambda: 0.95
* PPO clip ratio: 0.2
* Entropy coefficient: 0.01
* Value function coefficient: 0.5
* Max gradient norm: 0.5

## Advanced Features

### Testing and Optimization

The project includes comprehensive testing and optimization tools:

```bash
# Run all tests
python tests/run_tests.py --type all

# Optimize hyperparameters
python examples/optimize_and_benchmark.py --symbol SPY --optimize --n_iter 30

# Benchmark against traditional strategies
python examples/optimize_and_benchmark.py --symbol SPY --benchmark
```

For detailed information on testing and optimization, see [Testing Guide](docs/guides/TESTING_GUIDE.md).

### Hyperparameter Optimization

The project includes Bayesian Optimization for finding the best hyperparameters:

```python
from src.utils.hyperparameter_optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(train_data, val_data)
result = optimizer.optimize(n_iter=30)
```

### Strategy Benchmarking

Compare GRPO agent against traditional trading strategies:

```python
from src.utils.benchmarking import StrategyBenchmark

benchmark = StrategyBenchmark(test_data)
benchmark.create_standard_strategies()  # Add Buy & Hold, MA Crossover, RSI, etc.
benchmark.add_grpo_agent(agent, name="GRPO")
results = benchmark.run_all_benchmarks()
benchmark.plot_results()
```

## Training Process

The training process includes:

1. Data collection using the trading environment
2. Advantage estimation using GAE
3. Policy and value function updates
4. Regular evaluation of agent performance
5. Model checkpointing
6. Performance visualization

## Performance Metrics

The toolkit includes various performance metrics:

* **Profitability Metrics**: Total return, annualized return, win rate
* **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio
* **Risk Metrics**: Maximum drawdown, volatility
* **Statistical Significance Testing**: t-test with effect size calculation

## License

MIT