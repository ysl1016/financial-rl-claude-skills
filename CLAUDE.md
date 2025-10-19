# Financial RL Trading System - Claude Code Guide

**Project**: Reinforcement Learning-based Trading System with Claude AI Integration
**Version**: 2.0
**Last Updated**: 2025-10-19
**Repository**: github.com/ysl1016/financial-rl-claude-skills

---

## Project Overview

This project implements an **AI-powered trading system** combining:
- **Reinforcement Learning (RL)**: GRPO and DeepSeek-R1 agents for decision-making
- **Claude AI Integration**: Market analysis and hybrid decision-making
- **Backtesting Framework**: Historical data simulation using OpenAI Gym
- **Real-time Trading System**: Multi-broker live trading (Phase 2 - In Development)

### Architecture at a Glance

```
financial-rl-claude-skills/
│
├── src/                          # Backtesting System (Production-Ready)
│   ├── models/                   # RL agents (GRPO, DeepSeek)
│   ├── claude_integration/       # Claude AI + Hybrid Agent
│   ├── data/                     # Data processors (Yahoo Finance)
│   └── utils/                    # Technical indicators, visualization
│
├── realtime_trading/             # Real-time Trading System (In Development)
│   ├── CLAUDE.md                 # Detailed architecture guide (READ THIS FOR REAL-TIME WORK)
│   ├── WORK_PLAN_REPORT.md       # Implementation roadmap
│   └── src/                      # Real-time components (brokers, execution, risk)
│
├── docs/                         # Documentation
├── examples/                     # Usage examples
├── tests/                        # Unit and integration tests
└── reports/                      # Generated analysis reports
```

---

## System Components

### 1. Backtesting System (src/)

**Status**: ✅ Production-Ready

#### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **GRPO Agent** | `src/models/grpo_agent.py` | Generalized Reward-Penalty Optimization RL agent |
| **DeepSeek Agent** | `src/models/deepseek_grpo_agent.py` | Transformer-based RL agent with historical context |
| **Trading Environment** | `src/models/trading_env.py` | OpenAI Gym environment for episodic training |
| **Hybrid Agent** | `src/claude_integration/hybrid_agent.py` | RL + Claude AI decision-making |
| **Claude Analyzer** | `src/claude_integration/claude_analyzer.py` | Market state analysis using Claude AI |
| **Data Processor** | `src/data/data_processor.py` | Yahoo Finance data fetching and preprocessing |
| **Technical Indicators** | `src/utils/indicators.py` | RSI, MACD, Moving Averages, etc. |

#### Key Features

- **Historical Backtesting**: Simulate trading on past data
- **RL Training**: Train agents using episodic Gym environment
- **Claude Analysis**: AI-powered market sentiment and risk assessment
- **Reporting**: Auto-generate markdown reports with charts
- **Visualization**: Price charts, technical indicators, performance metrics

#### Usage Example

```bash
# Generate investment analysis report
claude /trading-analysis NVDA
claude /trading-analysis PLTR
```

---

### 2. Real-time Trading System (realtime_trading/)

**Status**: 🚧 In Development (Architecture Design Phase)

#### Architecture Overview

**4-Layer Design**:
1. **Decision Layer**: Hybrid RL-Claude agent for trading signals
2. **Data Layer**: Real-time market data streaming (WebSocket)
3. **Execution Layer**: Order management and position tracking
4. **Broker Abstraction Layer**: Multi-broker support (KIS + IB)

#### Key Components (Planned)

| Component | File | Purpose |
|-----------|------|---------|
| **BaseBroker** | `realtime_trading/src/brokers/base_broker.py` | Unified broker interface |
| **KIS Broker** | `realtime_trading/src/brokers/kis_broker.py` | Korea Investment Securities API |
| **IB Broker** | `realtime_trading/src/brokers/ib_broker.py` | Interactive Brokers API |
| **Market Data Stream** | `realtime_trading/src/core/market_data_stream.py` | Real-time data streaming |
| **Order Manager** | `realtime_trading/src/core/order_manager.py` | Order lifecycle management |
| **Risk Manager** | `realtime_trading/src/risk/risk_manager.py` | Real-time risk validation |
| **Realtime Env Adapter** | `realtime_trading/src/adapters/realtime_env_adapter.py` | Bridge existing RL agents to real-time |

#### Implementation Roadmap

| Phase | Timeline | Status |
|-------|----------|--------|
| **Phase 1**: Foundation (Broker connectivity) | Week 1-2 | 📋 Planned |
| **Phase 2**: Execution Engine | Week 3-4 | 📋 Planned |
| **Phase 3**: Agent Integration | Week 5 | 📋 Planned |
| **Phase 4**: Paper Trading Validation | Month 2-7 | 📋 Planned |
| **Phase 5**: Live Trading (Optional) | Month 8+ | 📋 Planned |

**📖 For detailed implementation guide, see: `realtime_trading/CLAUDE.md`**

---

## Working with This Project

### For Backtesting Tasks

**When to use**: Analysis, backtesting, model training, report generation

**Key Files**:
- RL agents: `src/models/grpo_agent.py`, `src/models/deepseek_grpo_agent.py`
- Claude integration: `src/claude_integration/hybrid_agent.py`
- Environment: `src/models/trading_env.py`
- Data: `src/data/data_processor.py`

**Example Tasks**:
- Train RL agent on historical data
- Generate investment analysis reports
- Backtest trading strategies
- Visualize performance metrics

**Important Notes**:
- ✅ This system is production-ready
- ✅ Reuse components for real-time system (zero-modification)
- ✅ All backtesting code remains unchanged

---

### For Real-time Trading Tasks

**When to use**: Live trading, broker integration, real-time data streaming

**Key Files**:
- Architecture guide: `realtime_trading/CLAUDE.md` ⭐ **READ THIS FIRST**
- Work plan: `realtime_trading/WORK_PLAN_REPORT.md`
- Implementation: `realtime_trading/src/` (in development)

**Example Tasks**:
- Implement broker connectors (KIS, IB)
- Build real-time data streaming
- Create order management system
- Integrate RL agents for live trading

**Important Notes**:
- 🚧 This system is under development
- 📖 **ALWAYS refer to `realtime_trading/CLAUDE.md` for detailed architecture**
- ⚠️ Do NOT modify existing backtesting code in `src/`
- ✅ Reuse existing components via adapters

---

## Code Integration Strategy

### Reusable Components (Zero-Modification)

The following components from the backtesting system can be reused **directly** in the real-time system:

| Component | Location | Usage in Real-time |
|-----------|----------|-------------------|
| **GRPO Agent** | `src/models/grpo_agent.py` | Call `select_action(state)` directly |
| **DeepSeek Agent** | `src/models/deepseek_grpo_agent.py` | Call `select_action(state, history)` |
| **Hybrid Agent** | `src/claude_integration/hybrid_agent.py` | Use for decision-making |
| **Claude Analyzer** | `src/claude_integration/claude_analyzer.py` | Use for market analysis |
| **Technical Indicators** | `src/utils/indicators.py` | Reuse `calculate_technical_indicators()` |

### Adapter Pattern

**Problem**: Backtesting uses episodic environment (Gym), real-time is continuous.

**Solution**: Use `RealtimeEnvAdapter` to bridge the gap.

```python
# realtime_trading/src/adapters/realtime_env_adapter.py
from src.models.grpo_agent import GRPOAgent  # Reuse existing agent

class RealtimeEnvAdapter:
    def __init__(self, agent: GRPOAgent, data_buffer, position_manager):
        self.agent = agent  # No modification needed
        self.data_buffer = data_buffer
        self.position_manager = position_manager

    def select_action(self) -> int:
        state = self._construct_state()
        return self.agent.select_action(state)  # Direct call
```

---

## Configuration

### Backtesting Configuration

**Data Source**: Yahoo Finance (historical data)
**Symbols**: NVDA, PLTR (currently analyzed)
**Indicators**: RSI, MACD, SMA, Bollinger Bands
**Claude API**: Anthropic Claude (API key required)

### Real-time Configuration

**Brokers**:
- Korea Investment Securities (KIS) - Korean stocks + US stocks
- Interactive Brokers (IB) - Global markets

**Trading Mode**: Paper trading (default) → Live trading (after validation)

**Risk Limits**:
- Daily loss limit: 5%
- Position size limit: 30%
- Stop-loss: 2% per position
- Emergency shutdown: 10% total loss

**Configuration Files** (in `realtime_trading/config/`):
- `brokers.yaml`: Broker credentials and settings
- `trading.yaml`: Trading parameters and agent configuration
- `risk.yaml`: Risk management limits
- `.env`: API keys and secrets (DO NOT COMMIT)

---

## Development Guidelines

### When Working on Backtesting Code

1. **Read existing code first**: Understand current implementation
2. **Test thoroughly**: Use `tests/` for unit and integration tests
3. **Document changes**: Update docstrings and comments
4. **Preserve compatibility**: Real-time system depends on these components

### When Working on Real-time Code

1. **Start with architecture**: Read `realtime_trading/CLAUDE.md` first
2. **Follow roadmap**: Use `realtime_trading/WORK_PLAN_REPORT.md` for task planning
3. **Implement by phase**: Complete Phase 1 before Phase 2
4. **Reuse, don't modify**: Use adapters to integrate existing components
5. **Test with paper trading**: Validate for 3-6 months before live trading

### Code Style

- **Language**: Python 3.8+
- **Async**: Use `asyncio` for real-time components
- **Type hints**: Use type annotations for clarity
- **Documentation**: Write clear docstrings for all classes and methods
- **Error handling**: Comprehensive try-catch with logging
- **Logging**: Use Python logging module, structured logs

---

## Testing Strategy

### Backtesting Tests

**Location**: `tests/`

**Test Types**:
- Unit tests: Individual components (agents, indicators)
- Integration tests: End-to-end backtesting flows
- Performance tests: Training speed, inference latency

### Real-time Tests

**Location**: `realtime_trading/tests/`

**Test Types**:
- Broker connection tests
- Data streaming tests
- Order execution tests (paper trading)
- Risk limit validation tests
- End-to-end integration tests

**Validation Period**: 3-6 months paper trading before live

---

## Important Reminders

### ⚠️ Critical Safety Rules

1. **NEVER commit API keys**: Use `.env` file (gitignored)
2. **NEVER skip paper trading**: Minimum 3 months validation
3. **NEVER disable risk limits**: Always enforce safety checks
4. **ALWAYS log all trades**: For audit and analysis
5. **ALWAYS test in paper mode first**: Before live trading

### 📖 Documentation Hierarchy

**For overall project understanding**:
- Read this file (`CLAUDE.md`)

**For backtesting work**:
- `src/models/`, `src/claude_integration/`, `src/data/`
- `docs/guides/CLAUDE_INTEGRATION_GUIDE.md`

**For real-time trading work**:
- **Primary**: `realtime_trading/CLAUDE.md` ⭐
- **Roadmap**: `realtime_trading/WORK_PLAN_REPORT.md`
- **Implementation**: `realtime_trading/src/`

### 🔄 Integration Points

**Backtesting → Real-time**:
- RL agents: Direct reuse via adapter
- Claude integration: Direct reuse
- Technical indicators: Direct reuse
- Data processing: Replace Yahoo Finance with real-time streams

**Real-time → Backtesting**:
- Trade history: Can be used for retraining
- Performance metrics: Compare with backtest predictions
- Market insights: Improve future backtests

---

## Quick Start

### Backtesting Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate analysis report
claude /trading-analysis NVDA

# 3. View report
cat reports/NVDA_analysis_report_*.md
```

### Real-time Development Workflow

```bash
# 1. Navigate to real-time directory
cd realtime_trading

# 2. Read architecture guide
cat CLAUDE.md

# 3. Read work plan
cat WORK_PLAN_REPORT.md

# 4. Create directory structure
mkdir -p src/{brokers,core,risk,state,adapters} config scripts tests logs

# 5. Install real-time dependencies
pip install python-kis ib_insync pyyaml python-dotenv

# 6. Start implementing Phase 1
# Follow realtime_trading/WORK_PLAN_REPORT.md
```

---

## External Resources

### API Documentation

- **Korea Investment Securities**: https://apiportal.koreainvestment.com
- **Interactive Brokers**: https://www.interactivebrokers.com/en/trading/ib-api.php
- **Claude API**: https://docs.anthropic.com

### Libraries

- **ib_insync**: https://ib-insync.readthedocs.io
- **python-kis**: https://github.com/Soju06/python-kis
- **OpenAI Gym**: https://gymnasium.farama.org

### Project Documentation

- **README**: Project overview and setup
- **Architecture**: `docs/architecture/PROJECT_STRUCTURE.md`
- **Testing Guide**: `docs/guides/TESTING_GUIDE.md`
- **Claude Integration**: `docs/guides/CLAUDE_INTEGRATION_GUIDE.md`

---

## Project Status Summary

| Component | Status | Next Action |
|-----------|--------|-------------|
| **Backtesting System** | ✅ Production | Continuous improvement |
| **RL Agents (GRPO, DeepSeek)** | ✅ Trained | Ready for real-time |
| **Claude Integration** | ✅ Working | Ready for real-time |
| **Real-time Architecture** | ✅ Designed | Start Phase 1 implementation |
| **Broker Integration** | 📋 Planned | Implement KIS/IB connectors |
| **Paper Trading** | 📋 Planned | 3-6 months validation |
| **Live Trading** | 📋 Future | After paper trading success |

---

## Contact & Support

**Repository**: https://github.com/ysl1016/financial-rl-claude-skills
**Issues**: https://github.com/ysl1016/financial-rl-claude-skills/issues

---

**Last Updated**: 2025-10-19
**Maintained By**: Claude Code
**Version**: 2.0

---

## Appendix: Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-10-19 | Multi-broker approach (KIS + IB) | Flexibility for Korean and global markets |
| 2025-10-19 | 4-layer architecture for real-time | Clear separation of concerns |
| 2025-10-19 | Adapter pattern for RL integration | Preserve existing code, zero modification |
| 2025-10-19 | Minimum 3 months paper trading | Risk management and validation |
| 2025-10-19 | Initial symbols: NVDA, PLTR | Already analyzed, proven volatility |

---

**End of Claude Code Guide**

This document serves as the master guide for all development work on this project. Always refer to this file for overall context, and dive into specific subdirectories (`src/`, `realtime_trading/`) for detailed implementation work.
