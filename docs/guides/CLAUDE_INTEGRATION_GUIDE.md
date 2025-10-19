# Claude Skills Integration Guide

## 개요

이 가이드는 강화학습 기반 트레이딩 모델에 Anthropic Claude의 자연어 처리 및 추론 능력을 통합하는 방법을 설명합니다.

## 목차

1. [아키텍처](#아키텍처)
2. [설치 및 설정](#설치-및-설정)
3. [핵심 컴포넌트](#핵심-컴포넌트)
4. [사용 예시](#사용-예시)
5. [의사결정 모드](#의사결정-모드)
6. [고급 기능](#고급-기능)
7. [성능 최적화](#성능-최적화)
8. [FAQ](#faq)

---

## 아키텍처

### 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│           Hybrid RL-Claude Trading System               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐         ┌──────────────┐            │
│  │  RL Agent    │◄───────►│ Claude API   │            │
│  │  (GRPO/      │         │ (GPT-4 class)│            │
│  │   DeepSeek)  │         └──────────────┘            │
│  └──────┬───────┘                 │                    │
│         │                         │                    │
│         │        ┌────────────────▼────────┐           │
│         │        │  Hybrid Decision Layer  │           │
│         └───────►│  - Weighted             │           │
│                  │  - Sequential           │           │
│                  │  - Ensemble             │           │
│                  └────────────┬────────────┘           │
│                               │                        │
│                  ┌────────────▼────────────┐           │
│                  │   Final Trading Action  │           │
│                  └─────────────────────────┘           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 주요 특징

1. **하이브리드 의사결정**: RL의 빠른 판단 + Claude의 깊은 분석
2. **자연어 인사이트**: 복잡한 시장 패턴을 자연어로 해석
3. **리스크 평가**: 다차원적 리스크 분석 및 시나리오 평가
4. **레짐 해석**: 시장 레짐 변화를 감지하고 전략 조정
5. **설명 가능성**: 모든 의사결정에 대한 자연어 설명 제공

---

## 설치 및 설정

### 1. 패키지 설치

```bash
# 기본 요구사항
pip install -r requirements.txt

# Claude API 패키지
pip install anthropic
```

### 2. API 키 설정

```bash
# 환경변수 설정
export ANTHROPIC_API_KEY="your-api-key-here"

# 또는 .env 파일 생성
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

### 3. 설정 확인

```python
import os
print(os.environ.get("ANTHROPIC_API_KEY"))  # API 키 확인
```

---

## 핵심 컴포넌트

### 1. ClaudeMarketAnalyzer

시장 데이터를 분석하고 자연어 인사이트를 제공합니다.

```python
from src.claude_integration import ClaudeMarketAnalyzer

analyzer = ClaudeMarketAnalyzer()

# 시장 상황 분석
analysis = analyzer.analyze_market_state(
    market_data=market_df,
    technical_indicators={'RSI_norm': 0.65, 'MACD_norm': 0.23},
    current_position=1,  # Long
    portfolio_value=105000
)

print(analysis['market_sentiment'])      # 'bullish'
print(analysis['confidence_level'])       # 0.75
print(analysis['trading_recommendation']) # {'suggested_action': 'hold', ...}
```

**주요 메서드:**
- `analyze_market_state()`: 종합적 시장 분석
- `explain_indicator_divergence()`: 가격-지표 다이버전스 설명
- `interpret_pattern()`: 차트 패턴 해석
- `generate_trading_narrative()`: 트레이딩 내러티브 생성

### 2. HybridRLClaudeAgent

RL과 Claude를 결합한 하이브리드 에이전트입니다.

```python
from src.claude_integration import HybridRLClaudeAgent
from src.models.grpo_agent import GRPOAgent

# RL 에이전트 생성
rl_agent = GRPOAgent(state_dim=16, action_dim=3, hidden_dim=128)

# 하이브리드 에이전트 생성
hybrid_agent = HybridRLClaudeAgent(
    rl_agent=rl_agent,
    decision_mode='weighted',          # weighted/sequential/ensemble
    rl_weight=0.7,                     # RL 가중치
    claude_weight=0.3,                 # Claude 가중치
    claude_consultation_frequency=10,  # Claude 상담 빈도
    enable_claude_override=True        # Claude 거부권
)

# 행동 선택
action, decision_info = hybrid_agent.select_action(
    state=state,
    market_data=market_df,
    technical_indicators=indicators,
    current_position=position,
    portfolio_value=value
)

# 의사결정 설명
print(hybrid_agent.explain_last_decision())
```

### 3. ClaudeRiskAssessor

포트폴리오 리스크를 평가하고 시나리오를 분석합니다.

```python
from src.claude_integration import ClaudeRiskAssessor

risk_assessor = ClaudeRiskAssessor(risk_tolerance='moderate')

# 포지션 리스크 평가
risk_assessment = risk_assessor.assess_position_risk(
    current_position=1.0,
    portfolio_value=105000,
    max_drawdown=0.08,
    volatility=0.25,
    market_data=market_df,
    technical_indicators=indicators
)

print(risk_assessment['overall_risk_score'])   # 0.0-1.0
print(risk_assessment['risk_level'])            # 'low/medium/high/critical'
print(risk_assessment['immediate_actions'])     # 권장 조치 리스트

# 거래 리스크 평가
trade_risk = risk_assessor.evaluate_trade_risk(
    proposed_action=1,  # Buy
    current_position=0,
    entry_price=450.0,
    current_price=455.0,
    position_duration=5,
    recent_returns=[0.01, -0.005, 0.02, 0.015, -0.01]
)

print(trade_risk['approval_status'])  # 'approved/cautioned/rejected'
```

### 4. ClaudeRegimeInterpreter

시장 레짐을 해석하고 전략을 조정합니다.

```python
from src.claude_integration import ClaudeRegimeInterpreter

regime_interpreter = ClaudeRegimeInterpreter()

# 레짐 해석
interpretation = regime_interpreter.interpret_regime(
    regime_id=2,
    regime_features={'volatility': 0.35, 'trend': -0.1, 'volume': 1.2},
    regime_stability=0.85,
    market_data=market_df,
    historical_regimes=[0, 0, 1, 1, 2, 2, 2]
)

print(interpretation['regime_name'])           # 'High Volatility Bear Market'
print(interpretation['regime_type'])           # 'trending_down'
print(interpretation['trading_strategy'])      # 전략 딕셔너리

# 레짐 전환 감지
transition = regime_interpreter.detect_regime_transition(
    current_regime=2,
    previous_regime=1,
    transition_probability=0.75,
    market_data=market_df
)

print(transition['transition_type'])           # 'gradual/abrupt/false_signal'
print(transition['immediate_actions'])         # 권장 조치 리스트
```

---

## 사용 예시

### 기본 사용법

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

from src.data.data_processor import process_data
from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent
from src.claude_integration import HybridRLClaudeAgent, ClaudeMarketAnalyzer

# 1. 데이터 준비
data = process_data('SPY', start_date='2020-01-01')

# 2. 환경 생성
env = TradingEnv(data=data, initial_capital=100000)

# 3. RL 에이전트 생성
rl_agent = GRPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=128
)

# 4. 하이브리드 에이전트 생성
hybrid_agent = HybridRLClaudeAgent(
    rl_agent=rl_agent,
    decision_mode='weighted',
    claude_consultation_frequency=20
)

# 5. 트레이딩 루프
state = env.reset()
done = False

while not done:
    # 기술적 지표 추출
    current_row = data.iloc[env.index]
    indicators = {
        col: float(current_row[col])
        for col in data.columns
        if col.endswith('_norm')
    }

    # 하이브리드 행동 선택
    action, decision_info = hybrid_agent.select_action(
        state=state,
        market_data=data.iloc[max(0, env.index-50):env.index+1],
        technical_indicators=indicators,
        current_position=env.position,
        portfolio_value=env.portfolio_values[-1]
    )

    # 의사결정 정보 출력
    if decision_info['claude_consulted']:
        print(f"Claude consulted: {decision_info['reasoning']}")

    # 환경 스텝
    next_state, reward, done, info = env.step(action)
    state = next_state

# 통계 출력
stats = hybrid_agent.get_decision_statistics()
print(f"Claude influence rate: {stats['claude_influence_rate']:.1%}")
print(f"Agreement rate: {stats['agreement_rate']:.1%}")
```

### 실전 예시 실행

```bash
# 기본 실행
python examples/hybrid_claude_trading.py

# 커스텀 설정
python examples/hybrid_claude_trading.py \
    --symbol AAPL \
    --start_date 2021-01-01 \
    --end_date 2023-12-31 \
    --episodes 20 \
    --mode sequential \
    --frequency 15
```

---

## 의사결정 모드

### 1. Weighted Mode (가중 평균)

RL과 Claude의 의견을 신뢰도 기반으로 가중 평균합니다.

```python
hybrid_agent = HybridRLClaudeAgent(
    rl_agent=rl_agent,
    decision_mode='weighted',
    rl_weight=0.7,        # RL 가중치
    claude_weight=0.3      # Claude 가중치
)
```

**특징:**
- Claude 신뢰도에 따라 가중치 동적 조정
- 양쪽 의견을 모두 반영
- 안정적이지만 때로 중립적인 결정

**적합한 상황:**
- 일반적인 시장 상황
- RL 모델이 충분히 학습된 경우
- 균형잡힌 의사결정 필요

### 2. Sequential Mode (순차 승인)

RL이 제안하고 Claude가 승인/거부합니다.

```python
hybrid_agent = HybridRLClaudeAgent(
    rl_agent=rl_agent,
    decision_mode='sequential',
    enable_claude_override=True  # Claude 거부권 활성화
)
```

**특징:**
- RL의 빠른 반응 + Claude의 검증
- 고위험 상황에서 Claude가 개입
- 신뢰도 높은 Claude 의견 우선

**적합한 상황:**
- 변동성이 큰 시장
- RL 모델이 미숙한 경우
- 리스크 관리 중시

### 3. Ensemble Mode (앙상블)

양쪽이 동의할 때만 행동합니다.

```python
hybrid_agent = HybridRLClaudeAgent(
    rl_agent=rl_agent,
    decision_mode='ensemble'
)
```

**특징:**
- 가장 보수적인 접근
- 의견 불일치 시 hold
- 높은 정확도, 낮은 거래 빈도

**적합한 상황:**
- 불확실한 시장
- 보수적 트레이딩
- 신호 확실성 중시

---

## 고급 기능

### 1. 다이버전스 분석

```python
analyzer = ClaudeMarketAnalyzer()

explanation = analyzer.explain_indicator_divergence(
    price_data=data['Close'],
    indicator_data=data['RSI'],
    indicator_name='RSI'
)

print(explanation)
# "A bearish divergence is forming as price makes higher highs
#  while RSI shows lower highs, suggesting weakening momentum..."
```

### 2. 패턴 인식

```python
pattern_analysis = analyzer.interpret_pattern(
    recent_candles=data.tail(20),
    pattern_type='head_and_shoulders'
)

print(pattern_analysis['pattern_identified'])  # 'Head and Shoulders'
print(pattern_analysis['reliability'])         # 'high'
print(pattern_analysis['key_levels'])          # {'support': 440, 'resistance': 460}
```

### 3. 포트폴리오 상관관계 분석

```python
risk_assessor = ClaudeRiskAssessor()

correlation_risk = risk_assessor.analyze_portfolio_correlation_risk(
    assets_data={'SPY': spy_data, 'QQQ': qqq_data, 'GLD': gld_data},
    positions={'SPY': 0.5, 'QQQ': 0.3, 'GLD': 0.2}
)

print(correlation_risk['concentration_risk'])   # 'medium'
print(correlation_risk['hedging_opportunities']) # 전략 리스트
```

### 4. 레짐 기반 전략 조정

```python
regime_interpreter = ClaudeRegimeInterpreter()

strategy_adaptation = regime_interpreter.recommend_strategy_adaptation(
    current_regime=2,
    regime_interpretation=interpretation,
    current_strategy_performance={'sharpe': 0.8, 'max_dd': 0.12},
    agent_params={'learning_rate': 0.0003, 'gamma': 0.99}
)

print(strategy_adaptation['parameter_adjustments'])  # 파라미터 조정 제안
print(strategy_adaptation['risk_management_changes']) # 리스크 관리 변경
```

---

## 성능 최적화

### 1. API 호출 최적화

```python
# Claude 상담 빈도 조정 (비용 절감)
hybrid_agent = HybridRLClaudeAgent(
    rl_agent=rl_agent,
    claude_consultation_frequency=50,  # 50 스텝마다만 상담
    # ...
)

# 중요한 시점에만 강제 상담
action, info = hybrid_agent.select_action(
    # ...
    force_claude_consultation=(high_volatility or regime_change)
)
```

### 2. 캐싱 활용

```python
# 최근 Claude 분석 재사용
latest_analysis = hybrid_agent.get_latest_claude_analysis()

if latest_analysis and is_recent(latest_analysis['timestamp']):
    # 캐시된 분석 사용
    use_cached_analysis(latest_analysis)
else:
    # 새로운 분석 요청
    new_analysis = analyzer.analyze_market_state(...)
```

### 3. 배치 분석

```python
# 여러 시나리오를 한 번에 분석 (미래 기능)
scenarios = [
    {'action': 0, 'position': 0},
    {'action': 1, 'position': 0},
    {'action': 2, 'position': 1}
]

# TODO: 배치 API 지원 시
batch_analysis = analyzer.analyze_batch(scenarios)
```

---

## FAQ

### Q1: Claude API 비용이 걱정됩니다.

**A:** 다음 방법으로 비용을 관리할 수 있습니다:
- `claude_consultation_frequency` 증가 (예: 50-100)
- 중요한 의사결정에만 Claude 사용
- 프롬프트 캐싱 활용 (자동으로 지원)
- 더 저렴한 모델 사용 (`claude-3-haiku`)

```python
analyzer = ClaudeMarketAnalyzer(
    model="claude-3-haiku-20240307",  # 저렴한 모델
    cache_enabled=True                 # 캐싱 활성화
)
```

### Q2: RL과 Claude가 자주 충돌합니다.

**A:** 의사결정 모드를 조정하거나 Claude 신뢰도를 조정하세요:

```python
# Sequential 모드: 높은 신뢰도에서만 Claude 우선
hybrid_agent = HybridRLClaudeAgent(
    decision_mode='sequential',
    # Claude 신뢰도 > 0.8일 때만 override
)

# Weighted 모드: RL 가중치 증가
hybrid_agent = HybridRLClaudeAgent(
    decision_mode='weighted',
    rl_weight=0.8,      # RL 우선
    claude_weight=0.2
)
```

### Q3: 실시간 트레이딩에 사용할 수 있나요?

**A:** 가능하지만 주의사항이 있습니다:
- API 지연시간 고려 (0.5-2초)
- 빠른 의사결정이 필요한 경우 RL만 사용
- Claude는 전략적 의사결정에만 활용
- 네트워크 오류 대비 fallback 구현

```python
try:
    action, info = hybrid_agent.select_action(...)
except Exception as e:
    # Fallback: RL만 사용
    action = rl_agent.select_action(state)
    info = {'error': str(e), 'fallback': True}
```

### Q4: 어떤 의사결정 모드가 가장 좋나요?

**A:** 상황에 따라 다릅니다:
- **초보자**: Sequential (안전)
- **중급자**: Weighted (균형)
- **고급자**: 상황별 모드 전환
- **백테스트**: 모든 모드 비교 테스트

### Q5: Claude 분석을 어떻게 검증하나요?

**A:** 다음 방법으로 검증하세요:

```python
# 의사결정 히스토리 분석
history = hybrid_agent.get_decision_history(last_n=100)

# Claude 영향 분석
claude_influenced = [d for d in history if d['claude_consulted']]
claude_correct = sum(1 for d in claude_influenced if was_correct(d))
accuracy = claude_correct / len(claude_influenced)

print(f"Claude accuracy: {accuracy:.1%}")

# 통계 비교
stats = hybrid_agent.get_decision_statistics()
print(f"Agreement rate: {stats['agreement_rate']:.1%}")
print(f"Override success rate: {stats['claude_override_rate']:.1%}")
```

---

## 다음 단계

1. **예시 실행**: `examples/hybrid_claude_trading.py` 실행
2. **파라미터 튜닝**: 의사결정 모드 및 가중치 실험
3. **백테스트**: 다양한 시장 조건에서 성능 비교
4. **커스터마이징**: 자신만의 분석 프롬프트 작성
5. **프로덕션**: 실전 트레이딩 시스템 구축

---

## 참고 자료

- [Anthropic Claude API 문서](https://docs.anthropic.com/)
- [GRPO 알고리즘 설명](../docs/DeepSeek-R1_Financial_Trading_Model_Architecture.md)
- [API 문서](../docs/api_documentation.md)
- [테스팅 가이드](../TESTING_GUIDE.md)

---

**버전**: 1.0.0
**최종 업데이트**: 2025-10-18
**작성자**: Financial RL Trading Team
