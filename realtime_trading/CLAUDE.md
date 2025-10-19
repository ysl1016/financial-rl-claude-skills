# Real-Time Trading System Architecture Guide

**Project**: Financial RL Trading with Multi-Broker Integration
**Version**: 1.0
**Last Updated**: 2025-10-19
**Status**: Architecture Design Phase

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [System Layers](#system-layers)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Directory Structure](#directory-structure)
7. [Integration with Existing Code](#integration-with-existing-code)
8. [Multi-Broker Strategy](#multi-broker-strategy)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Risk Management](#risk-management)
11. [Configuration](#configuration)
12. [Next Steps](#next-steps)

---

## Project Overview

### Objective

Transform the existing **backtesting-based RL trading system** into a **real-time production trading system** that supports:

- **Multi-broker execution** (Korea Investment & Securities + Interactive Brokers)
- **Real-time market data streaming** (WebSocket)
- **Live order execution** with actual broker APIs
- **Hybrid decision-making** (RL agents + Claude AI)
- **Risk management** and position monitoring
- **24/7 global market coverage**

### Current State

**Existing System** (Backtesting):
- Data source: Yahoo Finance (historical data)
- Environment: OpenAI Gym-based `TradingEnv` (episodic)
- Agents: GRPO, DeepSeek-R1 (trained on historical data)
- Claude Integration: Hybrid decision-making with Claude AI
- Order execution: Simulated only

**Gap Analysis**:
- ❌ No real-time data streaming
- ❌ No broker API integration
- ❌ No live order execution
- ❌ No connection monitoring/recovery
- ❌ No real-time risk management

### Target State

**Real-Time Trading System**:
- ✅ Real-time market data (WebSocket streams)
- ✅ Multi-broker support (KIS + IB)
- ✅ Live order execution and tracking
- ✅ Continuous trading loop (not episodic)
- ✅ Automatic reconnection and error handling
- ✅ Real-time risk monitoring and alerts

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  Real-Time Trading System                        │
│         (Korea Investment Securities + Interactive Brokers)      │
└─────────────────────────────────────────────────────────────────┘

                        ┌──────────────────┐
                        │   Main Process   │
                        │  - Trading Loop  │
                        │  - State Manager │
                        └────────┬─────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ↓                        ↓                        ↓
┌───────────────┐      ┌────────────────┐      ┌─────────────────┐
│ Decision Layer│      │   Data Layer   │      │ Execution Layer │
│  (Layer 1)    │      │   (Layer 2)    │      │   (Layer 3)     │
└───────────────┘      └────────────────┘      └─────────────────┘
        │                        │                        │
        │                        │                        │
┌───────┴──────────┐   ┌────────┴─────────┐   ┌─────────┴────────┐
│ Hybrid Agent     │←──│ Market Data      │   │ Order Manager    │
│ - GRPO/DeepSeek  │   │ Stream           │   │ - Place Orders   │
│ - Claude AI      │   │ - Real-time      │   │ - Track Status   │
│   Analyzer       │   │ - Indicators     │   │ - Cancel Orders  │
└──────────────────┘   └──────────────────┘   └──────────────────┘
        │                        │                        │
        ↓                        ↓                        ↓
┌──────────────────────────────────────────────────────────────────┐
│              Broker Abstraction Layer (Layer 4)                   │
│  ┌──────────────────┐            ┌──────────────────┐            │
│  │  Broker Router   │            │  Unified         │            │
│  │  - Select Broker │            │  Interface       │            │
│  │  - Load Balance  │            │                  │            │
│  └──────────────────┘            └──────────────────┘            │
└────────┬──────────────────────────────────┬──────────────────────┘
         │                                  │
    ┌────┴─────┐                     ┌─────┴────┐
    │          │                     │          │
    ↓          ↓                     ↓          ↓
┌────────┐ ┌──────┐           ┌──────────┐ ┌──────┐
│KIS API │ │ IB   │           │ Data     │ │ Risk │
│        │ │ API  │           │ Store    │ │ Mgmt │
└────────┘ └──────┘           └──────────┘ └──────┘

┌──────────────────────────────────────────────────────────────────┐
│                   Supporting Services                             │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐      │
│  │ Logging │  │Monitoring│  │ Alerting │  │ Dashboard   │      │
│  │         │  │(Grafana) │  │(Slack/   │  │(Streamlit)  │      │
│  │         │  │          │  │ Email)   │  │             │      │
│  └─────────┘  └──────────┘  └──────────┘  └─────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Layered Architecture**: Clear separation of concerns
2. **Asynchronous Processing**: Non-blocking I/O for real-time data
3. **Scalability**: Support multiple brokers and symbols
4. **Reliability**: Auto-reconnect, error handling, graceful shutdown
5. **Code Reuse**: Leverage existing RL agents and Claude integration

---

## System Layers

### Layer 1: Decision Layer

**Responsibilities**:
- Market state analysis
- Trading signal generation
- RL agent inference
- Claude AI consultation

**Key Components**:
```
HybridRLClaudeAgent (existing)
├─ select_action()
├─ consult_claude()
└─ decision_info

DecisionManager (new)
├─ aggregate_signals()
├─ apply_filters()
└─ generate_orders()
```

**Integration Points**:
- ✅ Reuse existing `HybridRLClaudeAgent`
- ✅ Reuse `GRPO Agent`, `DeepSeek Agent`
- 🆕 Add `DecisionManager` for multi-agent coordination

---

### Layer 2: Data Layer

**Responsibilities**:
- Real-time market data streaming (WebSocket)
- Data normalization and preprocessing
- Technical indicator calculation
- Data buffering and caching

**Key Components**:
```
MarketDataStream (new)
├─ subscribe(symbols, callback)
├─ unsubscribe(symbols)
└─ get_latest_bar()

DataBuffer (new)
├─ update(new_data)
├─ get_features()
└─ calculate_indicators()

DataNormalizer (existing - reuse)
└─ normalize()
```

**Integration Points**:
- 🆕 `MarketDataStream` - WebSocket real-time data
- ✅ Reuse `calculate_technical_indicators()` from `utils/indicators.py`
- 🆕 `DataBuffer` - Real-time data buffering

---

### Layer 3: Execution Layer

**Responsibilities**:
- Order creation and submission
- Order status tracking
- Position management
- Execution reporting

**Key Components**:
```
OrderManager (new)
├─ place_order(symbol, side, qty, order_type)
├─ cancel_order(order_id)
├─ get_order_status(order_id)
└─ track_orders()

PositionManager (new)
├─ get_positions()
├─ calculate_pnl()
└─ check_risk_limits()

ExecutionRouter (new)
├─ route_to_broker(order)
└─ split_orders(large_order)
```

---

### Layer 4: Broker Abstraction Layer

**Responsibilities**:
- Unified broker interface
- Abstract broker-specific differences
- Multi-broker routing
- Connection management

**Key Components**:
```
BaseBroker (new - abstract interface)
├─ connect()
├─ disconnect()
├─ place_order()
├─ cancel_order()
├─ get_positions()
├─ subscribe_data()
└─ get_account_info()

KISBroker (new - implementation)
└─ implements BaseBroker

IBBroker (new - implementation)
└─ implements BaseBroker

BrokerRouter (new)
├─ select_broker(symbol, strategy)
├─ get_available_brokers()
└─ health_check()
```

---

## Core Components

### Component 1: BaseBroker Interface

**Purpose**: Define a unified interface that all broker implementations must follow.

**Required Methods**:

| Method | Description | Return Type |
|--------|-------------|-------------|
| `connect()` | Connect to broker API | `bool` |
| `disconnect()` | Disconnect from broker | `None` |
| `place_order()` | Execute an order | `Order` |
| `cancel_order()` | Cancel an existing order | `bool` |
| `get_order_status()` | Query order status | `Order` |
| `get_positions()` | Get current positions | `List[Position]` |
| `subscribe_realtime_data()` | Subscribe to market data | `None` |
| `get_current_price()` | Get latest price | `float` |
| `get_account_info()` | Get account information | `AccountInfo` |

**Data Models**:

```python
@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide  # BUY/SELL
    order_type: OrderType  # MARKET/LIMIT/STOP
    quantity: float
    price: Optional[float]
    status: OrderStatus  # PENDING/FILLED/CANCELLED
    filled_quantity: float
    average_fill_price: Optional[float]
    broker: str
    timestamp: Optional[str]

@dataclass
class Position:
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    broker: str

@dataclass
class AccountInfo:
    broker: str
    account_id: str
    cash: float
    equity: float
    buying_power: float
    portfolio_value: float
    positions: List[Position]
```

---

### Component 2: MarketDataStream

**Purpose**: Stream real-time market data from brokers via WebSocket.

**Key Features**:
- Multi-symbol subscription
- Asynchronous event handling
- Automatic reconnection
- Data validation

**Interface**:

```python
class MarketDataStream:
    def __init__(self, broker: BaseBroker):
        self.broker = broker
        self.subscriptions = {}
        self.is_streaming = False

    async def subscribe(self, symbols: List[str], callback: Callable):
        """Subscribe to real-time data for symbols"""
        pass

    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        pass

    def get_latest_bar(self, symbol: str) -> Dict:
        """Get most recent data bar"""
        pass

    async def start(self):
        """Start streaming"""
        pass

    async def stop(self):
        """Stop streaming"""
        pass
```

---

### Component 3: DataBuffer

**Purpose**: Buffer real-time data and calculate technical indicators.

**Key Features**:
- Rolling window buffer (e.g., last 100 bars)
- Efficient indicator calculation
- State extraction for RL agent

**Interface**:

```python
class DataBuffer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)

    def update(self, new_bar: Dict):
        """Add new data bar"""
        self.data.append(new_bar)

    def get_features(self) -> np.ndarray:
        """Extract features for RL agent"""
        # Calculate indicators using existing utils
        indicators = calculate_technical_indicators(self.to_dataframe())
        return self._normalize_features(indicators)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame"""
        return pd.DataFrame(list(self.data))
```

---

### Component 4: OrderManager

**Purpose**: Manage order lifecycle from creation to execution.

**Key Features**:
- Order validation
- Broker routing
- Status tracking
- Error handling

**Interface**:

```python
class OrderManager:
    def __init__(self, broker_router: BrokerRouter):
        self.broker_router = broker_router
        self.active_orders = {}
        self.order_history = []

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None
    ) -> Order:
        """Place an order via appropriate broker"""
        broker = self.broker_router.select_broker(symbol)
        order = await broker.place_order(symbol, side, quantity, order_type, limit_price)
        self.active_orders[order.order_id] = order
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass

    def get_order_status(self, order_id: str) -> Order:
        """Get current order status"""
        pass

    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self.active_orders.values())
```

---

### Component 5: BrokerRouter

**Purpose**: Select the optimal broker for each order.

**Routing Logic**:

| Symbol Type | Broker | Reason |
|------------|--------|--------|
| Korean stocks (KOSPI/KOSDAQ) | KIS | Only KIS supports Korean market |
| US stocks (Low latency priority) | KIS | Lower latency from Korea |
| US stocks (Low fees priority) | IB | $0.005/share |
| Options/Futures | IB | Better derivative support |
| International (EU/Asia) | IB | Global coverage |

**Interface**:

```python
class BrokerRouter:
    def __init__(self, brokers: Dict[str, BaseBroker]):
        self.brokers = brokers  # {"KIS": kis_broker, "IB": ib_broker}

    def select_broker(self, symbol: str, strategy: str = "auto") -> BaseBroker:
        """Select optimal broker for symbol"""
        if self._is_korean_stock(symbol):
            return self.brokers["KIS"]
        elif strategy == "low_latency":
            return self.brokers["KIS"]
        elif strategy == "low_fees":
            return self.brokers["IB"]
        else:
            return self.brokers["IB"]  # Default

    def get_available_brokers(self) -> List[str]:
        """Get list of connected brokers"""
        return [name for name, broker in self.brokers.items()
                if broker.is_connected]

    def health_check(self) -> Dict[str, bool]:
        """Check connection status of all brokers"""
        return {name: broker.is_connected
                for name, broker in self.brokers.items()}
```

---

## Data Flow

### Real-Time Trading Flow

```
[1] Market Data Stream (WebSocket)
      │
      ↓ (Real-time tick data)
[2] Data Buffer
      │
      ├─→ Calculate Technical Indicators
      └─→ Extract Features
              │
              ↓
[3] State Construction
      │ (features + position info)
      ↓
[4] Hybrid Agent Decision
      ├─→ RL Agent (GRPO/DeepSeek) → Action
      └─→ Claude Analyzer (periodic) → Risk Assessment
              │
              ↓
[5] Action Selection
      │ (action: 0=hold, 1=buy, 2=sell)
      ↓
[6] Risk Manager Check
      │
      ↓ (approved)
[7] Order Manager
      │
      ├─→ Broker Router (select broker)
      │
      ↓
[8] Broker API (KIS or IB)
      │
      ↓
[9] Order Execution
      │
      ↓
[10] Order Status Update
       │
       ↓
[11] Position Manager Update
       │
       └─→ Loop back to [1] (next tick)
```

### Agent Learning Flow (Optional)

```
[1] Trading Loop Running
      │
      ↓
[2] Collect Experiences
      │ (state, action, reward, next_state)
      ↓
[3] Store in Replay Buffer
      │
      ↓
[4] Every N steps (e.g., 10)
      │
      ↓
[5] Agent.update()
      │
      └─→ Update GRPO/DeepSeek weights
              │
              ↓
[6] Performance Monitoring
      │
      ↓
[7] Save Model Checkpoint
```

---

## Directory Structure

### Planned Directory Layout

```
financial-rl-claude-skills/
├── realtime_trading/           # 🆕 Real-time trading system root
│   ├── CLAUDE.md              # This file
│   ├── README.md              # User-facing documentation
│   │
│   ├── src/
│   │   ├── brokers/           # 🆕 Broker integration layer
│   │   │   ├── __init__.py
│   │   │   ├── base_broker.py        # Abstract interface
│   │   │   ├── kis_broker.py         # Korea Investment Securities
│   │   │   ├── ib_broker.py          # Interactive Brokers
│   │   │   └── broker_router.py      # Broker selection logic
│   │   │
│   │   ├── core/              # 🆕 Core trading components
│   │   │   ├── __init__.py
│   │   │   ├── market_data_stream.py # Real-time data streaming
│   │   │   ├── data_buffer.py        # Data buffering
│   │   │   ├── order_manager.py      # Order lifecycle
│   │   │   ├── position_manager.py   # Position tracking
│   │   │   └── trading_loop.py       # Main execution loop
│   │   │
│   │   ├── risk/              # 🆕 Risk management
│   │   │   ├── __init__.py
│   │   │   ├── risk_manager.py       # Risk validation
│   │   │   ├── position_sizer.py     # Position sizing
│   │   │   └── stop_loss_manager.py  # Stop-loss automation
│   │   │
│   │   ├── state/             # 🆕 State management
│   │   │   ├── __init__.py
│   │   │   ├── state_manager.py      # Global state
│   │   │   └── connection_monitor.py # Connection health
│   │   │
│   │   └── adapters/          # 🆕 Integration adapters
│   │       ├── __init__.py
│   │       ├── realtime_env_adapter.py  # RL agent adapter
│   │       └── hybrid_agent_adapter.py  # Claude integration adapter
│   │
│   ├── config/                # 🆕 Configuration files
│   │   ├── brokers.yaml           # Broker credentials/settings
│   │   ├── trading.yaml           # Trading parameters
│   │   ├── risk.yaml              # Risk limits
│   │   └── symbols.yaml           # Symbol mappings
│   │
│   ├── scripts/               # 🆕 Execution scripts
│   │   ├── run_live_trading.py    # Live trading runner
│   │   ├── run_paper_trading.py   # Paper trading runner
│   │   └── backtest_realtime.py   # Backtest with real-time setup
│   │
│   ├── tests/                 # 🆕 Tests
│   │   ├── test_brokers.py
│   │   ├── test_order_manager.py
│   │   └── test_integration.py
│   │
│   └── logs/                  # Runtime logs
│       ├── trading_YYYYMMDD.log
│       └── orders_YYYYMMDD.log
│
├── src/                       # ✅ Existing source code (unchanged)
│   ├── models/                # RL agents (GRPO, DeepSeek)
│   ├── claude_integration/    # Hybrid agent, Claude analyzer
│   ├── data/                  # Backtesting data processors
│   ├── utils/                 # Technical indicators
│   └── ...
│
└── ... (other existing directories)
```

---

## Integration with Existing Code

### Zero-Modification Components (Reuse As-Is)

| Component | File | Usage |
|-----------|------|-------|
| **GRPO Agent** | `src/models/grpo_agent.py` | ✅ Call `select_action(state)` directly |
| **DeepSeek Agent** | `src/models/deepseek_grpo_agent.py` | ✅ Call `select_action(state, history)` |
| **Claude Analyzer** | `src/claude_integration/claude_analyzer.py` | ✅ Call `analyze_market_state()` |
| **Hybrid Agent** | `src/claude_integration/hybrid_agent.py` | ✅ Call `select_action()` |
| **Indicators** | `src/utils/indicators.py` | ✅ Reuse `calculate_technical_indicators()` |

### Adapter Layer (Bridge to Real-Time)

**Problem**: Existing agents expect episodic environment (Gym), but real-time is continuous.

**Solution**: Create `RealtimeEnvAdapter`

```python
# realtime_trading/src/adapters/realtime_env_adapter.py

class RealtimeEnvAdapter:
    """
    Adapter to use existing RL agents in real-time environment.
    Converts real-time data stream into Gym-like state representation.
    """
    def __init__(self, agent, data_buffer, position_manager):
        self.agent = agent  # GRPO or DeepSeek agent
        self.data_buffer = data_buffer
        self.position_manager = position_manager

    def get_state(self) -> np.ndarray:
        """Construct state from real-time data"""
        # Get latest features from buffer
        features = self.data_buffer.get_features()

        # Get position information
        position_info = self.position_manager.get_position_info()

        # Combine into state vector (same format as backtesting)
        state = np.concatenate([features, position_info])
        return state

    def select_action(self) -> int:
        """Get action from agent"""
        state = self.get_state()

        # Call agent's select_action (no modification needed)
        if isinstance(self.agent, DeepSeekGRPOAgent):
            history = self.data_buffer.get_history_sequence()
            action = self.agent.select_action(state, history)
        else:
            action = self.agent.select_action(state)

        return action

    def execute_action(self, action: int, order_manager: OrderManager):
        """Convert action to actual order"""
        if action == 1:  # Buy
            order_manager.place_order("NVDA", OrderSide.BUY, quantity=100)
        elif action == 2:  # Sell
            order_manager.place_order("NVDA", OrderSide.SELL, quantity=100)
        # action == 0: Hold (do nothing)
```

### Configuration Bridge

**Existing**: Hard-coded parameters
**New**: YAML configuration files

**Example** (`config/trading.yaml`):

```yaml
trading:
  symbols: ["NVDA", "PLTR"]
  update_frequency: 5  # seconds between decisions
  claude_consultation_frequency: 10  # steps

agents:
  rl_agent:
    type: "GRPO"  # or "DeepSeek"
    model_path: "../models/grpo_trained.pth"
    state_dim: 50
    action_dim: 3

  hybrid_agent:
    decision_mode: "sequential"
    rl_weight: 0.7
    claude_weight: 0.3
    enable_claude_override: true
    risk_threshold: 0.7

risk:
  max_daily_loss: 0.05  # 5% of capital
  max_position_size: 0.3  # 30% of capital
  stop_loss_pct: 0.02  # 2% stop-loss
  emergency_shutdown_loss: 0.10  # 10% total loss
```

---

## Multi-Broker Strategy

### Broker Selection Logic

| Scenario | Broker | Reason |
|----------|--------|--------|
| Korean stocks (KOSPI/KOSDAQ) | **KIS** | Only KIS supports Korean market |
| US stocks (low latency) | **KIS** | Lower latency from Korea (~30-150ms) |
| US stocks (low fees) | **IB** | $0.005/share vs KIS standard fees |
| Options/Futures | **IB** | Better derivative instrument support |
| European/Asian stocks | **IB** | Global market access (150+ countries) |
| Cryptocurrencies | None | Neither supports crypto (use separate API) |

### Portfolio Unification

**Challenge**: Manage positions across two brokers.

**Solution**: `UnifiedPortfolioManager`

```python
class UnifiedPortfolioManager:
    """Aggregates positions from multiple brokers"""

    def __init__(self, brokers: Dict[str, BaseBroker]):
        self.brokers = brokers

    def get_total_positions(self) -> List[Position]:
        """Combine positions from all brokers"""
        all_positions = []
        for broker_name, broker in self.brokers.items():
            positions = broker.get_positions()
            for pos in positions:
                pos.broker = broker_name
                all_positions.append(pos)
        return all_positions

    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value across brokers"""
        total = 0.0
        for broker in self.brokers.values():
            account_info = broker.get_account_info()
            total += account_info.portfolio_value
        return total

    def get_symbol_exposure(self, symbol: str) -> float:
        """Get total exposure to a symbol across all brokers"""
        exposure = 0.0
        for pos in self.get_total_positions():
            if pos.symbol == symbol:
                exposure += pos.market_value
        return exposure
```

---

## Implementation Roadmap

### Phase 1: Foundation (1-2 weeks)

**Goal**: Establish broker connectivity and data streaming.

| # | Task | Output | Test |
|---|------|--------|------|
| 1 | Design `BaseBroker` interface | `base_broker.py` | Unit test |
| 2 | Implement `KISBroker` | `kis_broker.py` | Connection test |
| 3 | Implement `IBBroker` | `ib_broker.py` | Connection test |
| 4 | Implement `MarketDataStream` | `market_data_stream.py` | Data streaming test |
| 5 | Implement `DataBuffer` | `data_buffer.py` | Indicator calculation test |

**Completion Criteria**:
- ✅ Both brokers connect successfully
- ✅ Real-time market data streaming works
- ✅ Technical indicators calculated correctly

---

### Phase 2: Execution Engine (1-2 weeks)

**Goal**: Order execution and position management.

| # | Task | Output | Test |
|---|------|--------|------|
| 6 | Implement `OrderManager` | `order_manager.py` | Paper trading orders |
| 7 | Implement `PositionManager` | `position_manager.py` | Position tracking |
| 8 | Implement `RiskManager` | `risk_manager.py` | Risk limit validation |
| 9 | Implement `BrokerRouter` | `broker_router.py` | Broker selection test |

**Completion Criteria**:
- ✅ Paper trading orders execute successfully
- ✅ Positions tracked accurately
- ✅ Risk limits enforced

---

### Phase 3: Agent Integration (1 week)

**Goal**: Integrate existing RL agents with real-time system.

| # | Task | Output | Test |
|---|------|--------|------|
| 10 | Implement `RealtimeEnvAdapter` | `realtime_env_adapter.py` | State construction test |
| 11 | Integrate `HybridAgent` | `trading_loop.py` | End-to-end test |
| 12 | Create configuration files | `config/*.yaml` | Config loading test |

**Completion Criteria**:
- ✅ GRPO agent runs in real-time
- ✅ Claude consultation works
- ✅ Decision → Order flow complete

---

### Phase 4: Paper Trading Validation (3-6 months)

**Goal**: Validate system stability and performance.

| # | Task | Test Period |
|---|------|-------------|
| 13 | Run continuous paper trading | 1 month |
| 14 | Test various market conditions | Rising/Falling/Sideways |
| 15 | Add monitoring and alerts | Grafana dashboard |
| 16 | Performance tuning | Latency optimization |

**Completion Criteria**:
- ✅ 3+ months stable operation
- ✅ Sharpe Ratio > 1.5
- ✅ Max Drawdown < 15%

---

### Phase 5: Live Trading (Optional)

**Goal**: Transition to real money trading.

| # | Task | Notes |
|---|------|-------|
| 17 | Re-verify all risk limits | Critical safety check |
| 18 | Start with small capital | < $1,000 |
| 19 | Daily monitoring | Review performance daily |
| 20 | Gradual scale-up | Increase if successful |

---

## Risk Management

### Pre-Launch Checklist

Before starting real-time trading, ensure:

- [ ] **Connection Monitoring**: Heartbeat implemented
- [ ] **Auto-Reconnect**: Automatic reconnection on disconnect
- [ ] **Daily Loss Limit**: E.g., 5% of capital
- [ ] **Position Limit**: E.g., 30% per symbol
- [ ] **Stop-Loss**: Automatically applied to all positions
- [ ] **Emergency Shutdown**: Close all positions if total loss > 10%
- [ ] **Logging**: All orders and trades logged
- [ ] **Alerts**: Critical events sent via Slack/Email
- [ ] **Paper Trading**: Minimum 3 months validation
- [ ] **Broker Credentials**: Secure storage (environment variables)

### Risk Limits

**Daily Limits**:
- Max Daily Loss: 5% of capital
- Max Daily Orders: 100 orders
- Max Claude API Calls: 200/day

**Position Limits**:
- Max Position Size: 30% of portfolio
- Max Leverage: 1x (no leverage)
- Stop-Loss: 2% per position

**System Limits**:
- Connection Timeout: 10 seconds
- Order Timeout: 30 seconds
- Max Retry Attempts: 3

### Error Handling

**Connection Lost**:
1. Log error
2. Attempt reconnection (3 retries)
3. If failed: Close all positions and shutdown
4. Send alert

**Order Rejection**:
1. Log rejection reason
2. Notify via alert
3. Do not retry automatically
4. Wait for manual intervention

**Unexpected Error**:
1. Log full traceback
2. Close all positions
3. Shutdown system gracefully
4. Send emergency alert

---

## Configuration

### Environment Variables

```bash
# .env file (DO NOT commit to git)

# Korea Investment Securities
KIS_APP_KEY=your_app_key_here
KIS_APP_SECRET=your_app_secret_here
KIS_ACCOUNT_NO=your_account_number

# Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497  # 7497=paper, 7496=live
IB_CLIENT_ID=1

# Claude API
ANTHROPIC_API_KEY=your_claude_api_key

# Alerts
SLACK_WEBHOOK_URL=your_slack_webhook
ALERT_EMAIL=your_email@example.com

# Mode
TRADING_MODE=paper  # paper or live
```

### Configuration Files

**`config/brokers.yaml`**:
```yaml
brokers:
  kis:
    enabled: true
    markets: ["KR", "US", "CN", "JP"]
    api_type: "REST + WebSocket"

  ib:
    enabled: true
    markets: ["US", "EU", "ASIA"]
    host: "${IB_HOST}"
    port: "${IB_PORT}"
    client_id: "${IB_CLIENT_ID}"
```

**`config/risk.yaml`**:
```yaml
risk_limits:
  daily_loss_limit: 0.05  # 5%
  position_size_limit: 0.30  # 30%
  stop_loss_pct: 0.02  # 2%
  emergency_shutdown_loss: 0.10  # 10%

  max_daily_orders: 100
  max_position_count: 10

alerts:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
  email:
    enabled: true
    to: "${ALERT_EMAIL}"
```

---

## Next Steps

### Immediate Actions

1. **Create Directory Structure**
   ```bash
   cd realtime_trading
   mkdir -p src/{brokers,core,risk,state,adapters} config scripts tests logs
   ```

2. **Install Dependencies**
   ```bash
   pip install python-kis ib_insync pyyaml python-dotenv asyncio
   ```

3. **Create Configuration Files**
   - `config/brokers.yaml`
   - `config/trading.yaml`
   - `config/risk.yaml`
   - `.env` (with credentials)

4. **Implement Core Interfaces**
   - Start with `src/brokers/base_broker.py`

### Decision Points

Please provide input on:

1. **Which broker to start with?** (KIS or IB)
   - Recommendation: Start with **KIS** (easier for Korean residents)

2. **Paper trading duration?**
   - Recommendation: **3-6 months**

3. **Target trading capital?**
   - For paper trading: Any amount
   - For live trading: Start with **< $1,000**

4. **Priority symbols?**
   - Recommendation: **NVDA, PLTR** (as analyzed earlier)

5. **Trading frequency?**
   - High-frequency (< 1 min): Requires ultra-low latency
   - Medium-frequency (5-15 min): Recommended
   - Low-frequency (hourly/daily): Easier to manage

---

## References

### External Documentation

- **Korea Investment Securities API**: https://apiportal.koreainvestment.com
- **Interactive Brokers API**: https://www.interactivebrokers.com/en/trading/ib-api.php
- **ib_insync Documentation**: https://ib-insync.readthedocs.io
- **python-kis GitHub**: https://github.com/Soju06/python-kis

### Internal Documentation

- **Project README**: `../README.md`
- **Architecture**: `../docs/architecture/PROJECT_STRUCTURE.md`
- **Claude Integration**: `../docs/guides/CLAUDE_INTEGRATION_GUIDE.md`
- **Testing Guide**: `../docs/guides/TESTING_GUIDE.md`

---

## Appendix

### Key Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Primary language | 3.8+ |
| **asyncio** | Asynchronous I/O | Built-in |
| **ib_insync** | IB API wrapper | Latest |
| **python-kis** | KIS API wrapper | Latest |
| **PyTorch** | RL agent inference | 2.0+ |
| **Anthropic Claude** | AI analysis | API v1 |
| **WebSocket** | Real-time data | Standard |
| **YAML** | Configuration | PyYAML |

### Glossary

- **GRPO**: Generalized Reward-Penalty Optimization (RL algorithm)
- **DeepSeek**: Transformer-based RL agent
- **KIS**: Korea Investment & Securities (한국투자증권)
- **IB**: Interactive Brokers
- **Paper Trading**: Simulated trading with real market data
- **Latency**: Time delay from data reception to order execution
- **Slippage**: Price difference between expected and executed price

---

**End of Architecture Guide**

For questions or clarifications, refer to this document when working on the real-time trading system implementation.
