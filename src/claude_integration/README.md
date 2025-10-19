# Claude Integration Module

Anthropic Claude를 활용한 RL 트레이딩 시스템 강화 모듈입니다.

## 빠른 시작

```python
# 1. API 키 설정
export ANTHROPIC_API_KEY="your-key-here"

# 2. 패키지 설치
pip install anthropic

# 3. 기본 사용
from src.claude_integration import HybridRLClaudeAgent, ClaudeMarketAnalyzer

analyzer = ClaudeMarketAnalyzer()
analysis = analyzer.analyze_market_state(...)
```

## 모듈 구조

```
claude_integration/
├── __init__.py                # 패키지 초기화
├── claude_analyzer.py         # 시장 분석 (ClaudeMarketAnalyzer)
├── hybrid_agent.py            # 하이브리드 에이전트 (HybridRLClaudeAgent)
├── risk_assessor.py           # 리스크 평가 (ClaudeRiskAssessor)
├── regime_interpreter.py      # 레짐 해석 (ClaudeRegimeInterpreter)
└── README.md                  # 이 파일
```

## 주요 클래스

### ClaudeMarketAnalyzer
시장 데이터를 자연어로 분석하고 인사이트를 제공합니다.

**주요 메서드:**
- `analyze_market_state()` - 종합 시장 분석
- `explain_indicator_divergence()` - 다이버전스 설명
- `interpret_pattern()` - 차트 패턴 해석
- `generate_trading_narrative()` - 트레이딩 내러티브 생성

### HybridRLClaudeAgent
RL 에이전트와 Claude를 결합한 하이브리드 의사결정 시스템입니다.

**의사결정 모드:**
- `weighted` - RL과 Claude 의견 가중 평균
- `sequential` - RL 제안 → Claude 검증
- `ensemble` - 양쪽 동의 시에만 행동

### ClaudeRiskAssessor
포트폴리오 리스크를 다차원적으로 평가합니다.

**주요 메서드:**
- `assess_position_risk()` - 포지션 리스크 평가
- `evaluate_trade_risk()` - 개별 거래 리스크 평가
- `analyze_portfolio_correlation_risk()` - 상관관계 리스크 분석
- `generate_risk_report()` - 종합 리스크 리포트 생성

### ClaudeRegimeInterpreter
시장 레짐을 해석하고 전략을 조정합니다.

**주요 메서드:**
- `interpret_regime()` - 레짐 해석
- `detect_regime_transition()` - 레짐 전환 감지
- `recommend_strategy_adaptation()` - 전략 조정 제안
- `explain_regime_sequence()` - 레짐 시퀀스 설명

## 사용 예시

### 1. 기본 시장 분석

```python
from src.claude_integration import ClaudeMarketAnalyzer

analyzer = ClaudeMarketAnalyzer()
analysis = analyzer.analyze_market_state(
    market_data=df,
    technical_indicators={'RSI_norm': 0.7, 'MACD_norm': 0.3},
    current_position=1,
    portfolio_value=105000
)

print(analysis['market_sentiment'])       # bullish/bearish/neutral
print(analysis['confidence_level'])        # 0.0-1.0
print(analysis['trading_recommendation']) # 거래 추천
```

### 2. 하이브리드 트레이딩

```python
from src.claude_integration import HybridRLClaudeAgent
from src.models.grpo_agent import GRPOAgent

# RL 에이전트
rl_agent = GRPOAgent(state_dim=16, action_dim=3)

# 하이브리드 에이전트
hybrid_agent = HybridRLClaudeAgent(
    rl_agent=rl_agent,
    decision_mode='weighted',
    claude_consultation_frequency=20
)

# 트레이딩
action, info = hybrid_agent.select_action(
    state=state,
    market_data=df,
    technical_indicators=indicators,
    current_position=position,
    portfolio_value=value
)

# 의사결정 설명
print(hybrid_agent.explain_last_decision())

# 통계
stats = hybrid_agent.get_decision_statistics()
print(f"Claude 영향률: {stats['claude_influence_rate']:.1%}")
```

### 3. 리스크 평가

```python
from src.claude_integration import ClaudeRiskAssessor

assessor = ClaudeRiskAssessor(risk_tolerance='moderate')

# 포지션 리스크
risk = assessor.assess_position_risk(
    current_position=1.0,
    portfolio_value=105000,
    max_drawdown=0.08,
    volatility=0.25,
    market_data=df,
    technical_indicators=indicators
)

print(risk['overall_risk_score'])  # 0.0-1.0
print(risk['risk_level'])           # low/medium/high/critical
print(risk['immediate_actions'])    # 권장 조치

# 종합 리포트
report = assessor.generate_risk_report(
    portfolio_metrics={'sharpe': 1.2, 'max_dd': 0.08},
    trade_history=trades,
    current_positions={'SPY': 1.0}
)
print(report)
```

### 4. 레짐 해석

```python
from src.claude_integration import ClaudeRegimeInterpreter

interpreter = ClaudeRegimeInterpreter()

# 레짐 해석
interpretation = interpreter.interpret_regime(
    regime_id=2,
    regime_features={'volatility': 0.35, 'trend': -0.1},
    regime_stability=0.85,
    market_data=df
)

print(interpretation['regime_name'])      # 'High Volatility Bear Market'
print(interpretation['trading_strategy']) # 전략 딕셔너리

# 전환 감지
transition = interpreter.detect_regime_transition(
    current_regime=2,
    previous_regime=1,
    transition_probability=0.75,
    market_data=df
)

print(transition['immediate_actions'])  # 권장 조치
```

## 설정 옵션

### API 설정

```python
# 환경변수로 설정 (권장)
export ANTHROPIC_API_KEY="sk-ant-..."

# 또는 코드에서 설정
analyzer = ClaudeMarketAnalyzer(api_key="sk-ant-...")
```

### 모델 선택

```python
# Sonnet (기본) - 균형잡힌 성능
analyzer = ClaudeMarketAnalyzer(model="claude-3-5-sonnet-20241022")

# Haiku - 빠르고 저렴
analyzer = ClaudeMarketAnalyzer(model="claude-3-haiku-20240307")

# Opus - 최고 성능 (비용 높음)
analyzer = ClaudeMarketAnalyzer(model="claude-3-opus-20240229")
```

### 캐싱 설정

```python
# 프롬프트 캐싱 활성화 (비용 절감)
analyzer = ClaudeMarketAnalyzer(cache_enabled=True)
```

## 성능 고려사항

### API 호출 빈도

```python
# 비용 절감: 덜 자주 상담
hybrid_agent = HybridRLClaudeAgent(
    rl_agent=rl_agent,
    claude_consultation_frequency=50  # 50 스텝마다
)

# 높은 정확도: 자주 상담
hybrid_agent = HybridRLClaudeAgent(
    rl_agent=rl_agent,
    claude_consultation_frequency=10  # 10 스텝마다
)
```

### 응답 시간

- Claude API: 평균 0.5-2초
- 실시간 트레이딩: RL만 사용하거나 비동기 처리
- 백테스팅: Claude 완전 활용 가능

### 비용 관리

- Haiku 모델: ~$0.25/백만 토큰
- Sonnet 모델: ~$3/백만 토큰
- 프롬프트 캐싱: 90% 비용 절감
- 상담 빈도 조절로 비용 통제

## 디버깅

### 로깅 활성화

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('claude_integration')
```

### 의사결정 히스토리

```python
# 최근 의사결정 확인
history = hybrid_agent.get_decision_history(last_n=10)

for decision in history:
    print(f"RL: {decision['rl_action']}, "
          f"Claude: {decision.get('claude_action', 'N/A')}, "
          f"Final: {decision['final_action']}")
```

### 에러 처리

```python
try:
    analysis = analyzer.analyze_market_state(...)
except Exception as e:
    print(f"Analysis failed: {e}")
    # Fallback logic
    analysis = default_analysis()
```

## 제한사항

1. **API 의존성**: 인터넷 연결 필요
2. **지연시간**: 0.5-2초 응답 시간
3. **비용**: API 사용량에 따른 과금
4. **레이트 리밋**: API 호출 제한 존재

## 다음 단계

1. [통합 가이드](../../docs/CLAUDE_INTEGRATION_GUIDE.md) 읽기
2. [예시 스크립트](../../examples/hybrid_claude_trading.py) 실행
3. 파라미터 튜닝 및 백테스트
4. 실전 트레이딩 시스템 구축

## 지원

- 문서: [Claude Integration Guide](../../docs/CLAUDE_INTEGRATION_GUIDE.md)
- 이슈: [GitHub Issues](https://github.com/ysl1016/financial-rl-trading/issues)
- API 문서: [Anthropic Docs](https://docs.anthropic.com/)
