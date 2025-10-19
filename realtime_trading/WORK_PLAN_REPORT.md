# 실시간 트레이딩 시스템 구축 작업계획 보고서

**프로젝트명**: Financial RL Trading - Real-Time Multi-Broker Integration
**보고서 작성일**: 2025년 10월 19일
**프로젝트 상태**: 아키텍처 설계 완료 → 구현 착수 대기
**예상 총 소요기간**: 5-12개월 (페이퍼 트레이딩 검증 포함)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [구현 단계별 작업계획](#2-구현-단계별-작업계획)
3. [상세 작업 일정](#3-상세-작업-일정)
4. [기술 스택 및 리소스](#4-기술-스택-및-리소스)
5. [위험 요인 및 대응 방안](#5-위험-요인-및-대응-방안)
6. [성공 기준](#6-성공-기준)
7. [즉시 착수 가능한 작업](#7-즉시-착수-가능한-작업)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목표

**현재 시스템**: 백테스팅 기반 강화학습 트레이딩 시스템
- Yahoo Finance 히스토리컬 데이터 활용
- OpenAI Gym 환경에서 에피소드 단위 학습
- GRPO, DeepSeek-R1 에이전트 학습 완료
- Claude AI 하이브리드 의사결정 구현

**목표 시스템**: 실시간 멀티 브로커 트레이딩 시스템
- 실시간 시장 데이터 스트리밍 (WebSocket)
- 한국투자증권(KIS) + Interactive Brokers(IB) 동시 연동
- 실제 주문 체결 및 포지션 관리
- 24/7 글로벌 마켓 커버리지
- 실시간 리스크 관리 및 자동화된 손절

### 1.2 핵심 가치 제안

| 구분 | 기존 시스템 | 목표 시스템 | 개선 효과 |
|------|------------|-----------|----------|
| **데이터** | 과거 데이터 (지연) | 실시간 스트리밍 | 즉시성 확보 |
| **실행** | 시뮬레이션 | 실제 주문 체결 | 수익 실현 가능 |
| **브로커** | 없음 | KIS + IB 멀티 브로커 | 유연한 시장 접근 |
| **리스크** | 백테스트 한정 | 실시간 모니터링 | 자본 보호 |
| **운영** | 수동 재실행 | 24/7 자동 운영 | 인력 절감 |

### 1.3 아키텍처 설계

**4계층 구조**:
```
[Layer 1] Decision Layer - 의사결정 (RL 에이전트 + Claude AI)
[Layer 2] Data Layer - 데이터 스트리밍 및 지표 계산
[Layer 3] Execution Layer - 주문 실행 및 포지션 관리
[Layer 4] Broker Abstraction Layer - 브로커 추상화 및 라우팅
```

**핵심 컴포넌트**:
- `BaseBroker`: 통합 브로커 인터페이스
- `KISBroker`, `IBBroker`: 브로커별 구현체
- `MarketDataStream`: 실시간 데이터 스트리밍
- `OrderManager`: 주문 생명주기 관리
- `RiskManager`: 실시간 리스크 검증
- `RealtimeEnvAdapter`: 기존 RL 에이전트 연동 어댑터

---

## 2. 구현 단계별 작업계획

### Phase 1: 기반 구축 (Foundation) - 1-2주

**목표**: 브로커 연결 및 데이터 스트리밍 구현

#### 주요 작업

| 순번 | 작업명 | 산출물 | 검증 방법 |
|------|--------|--------|----------|
| 1.1 | 디렉토리 구조 생성 | `realtime_trading/src/` 하위 폴더 | 파일 시스템 확인 |
| 1.2 | 의존성 패키지 설치 | `requirements.txt` 업데이트 | `pip list` 확인 |
| 1.3 | `BaseBroker` 인터페이스 설계 | `base_broker.py` | 단위 테스트 |
| 1.4 | `KISBroker` 구현 | `kis_broker.py` | 연결 테스트 |
| 1.5 | `IBBroker` 구현 | `ib_broker.py` | 연결 테스트 |
| 1.6 | `MarketDataStream` 구현 | `market_data_stream.py` | 실시간 데이터 수신 확인 |
| 1.7 | `DataBuffer` 구현 | `data_buffer.py` | 지표 계산 정확성 검증 |

#### 완료 기준
- ✅ KIS API 연결 성공 (REST + WebSocket)
- ✅ IB API 연결 성공 (TWS 또는 Gateway)
- ✅ 실시간 시장 데이터 수신 (NVDA, PLTR)
- ✅ RSI, MACD 등 기술 지표 계산 정확성 검증

#### 예상 이슈 및 대응
- **이슈**: KIS API 인증 오류
  - **대응**: API 키 유효성 확인, 개발자 포털 문서 참조
- **이슈**: IB TWS 연결 타임아웃
  - **대응**: 방화벽 설정 확인, 포트(7497/7496) 개방

---

### Phase 2: 실행 엔진 구축 (Execution Engine) - 1-2주

**목표**: 주문 실행 및 포지션 관리 시스템 구현

#### 주요 작업

| 순번 | 작업명 | 산출물 | 검증 방법 |
|------|--------|--------|----------|
| 2.1 | `OrderManager` 구현 | `order_manager.py` | 페이퍼 트레이딩 주문 체결 |
| 2.2 | `PositionManager` 구현 | `position_manager.py` | 포지션 추적 정확성 |
| 2.3 | `RiskManager` 구현 | `risk_manager.py` | 리스크 한도 검증 |
| 2.4 | `BrokerRouter` 구현 | `broker_router.py` | 브로커 선택 로직 테스트 |
| 2.5 | 통합 주문 플로우 테스트 | 엔드투엔드 테스트 | 주문→체결→포지션 반영 |

#### 완료 기준
- ✅ 페이퍼 트레이딩 주문 성공 (KIS, IB)
- ✅ 실시간 포지션 추적 정확성 검증
- ✅ 일일 손실 한도 5% 자동 차단
- ✅ 심볼당 포지션 크기 30% 제한 적용

#### 예상 이슈 및 대응
- **이슈**: 주문 체결 지연 (슬리피지)
  - **대응**: 시장가 주문 우선 사용, 지연 로깅 및 분석
- **이슈**: 포지션 불일치 (브로커 API vs 내부 관리)
  - **대응**: 주기적 재조정(reconciliation) 로직 추가

---

### Phase 3: 에이전트 통합 (Agent Integration) - 1주

**목표**: 기존 RL 에이전트를 실시간 시스템에 통합

#### 주요 작업

| 순번 | 작업명 | 산출물 | 검증 방법 |
|------|--------|--------|----------|
| 3.1 | `RealtimeEnvAdapter` 구현 | `realtime_env_adapter.py` | 상태 벡터 생성 검증 |
| 3.2 | `HybridAgent` 연동 | `trading_loop.py` | 엔드투엔드 실시간 의사결정 |
| 3.3 | 설정 파일 작성 | `config/*.yaml` | 설정 로딩 테스트 |
| 3.4 | 실시간 트레이딩 루프 구현 | `run_paper_trading.py` | 1시간 연속 실행 |

#### 완료 기준
- ✅ GRPO 에이전트 실시간 추론 성공
- ✅ Claude AI 주기적 컨설팅 (10스텝마다)
- ✅ 의사결정 → 주문 생성 → 체결 전체 플로우 작동
- ✅ 1시간 동안 에러 없이 안정적 실행

#### 통합 플로우
```
[1] MarketDataStream → 실시간 데이터 수신
[2] DataBuffer → 지표 계산 (RSI, MACD)
[3] RealtimeEnvAdapter → 상태 벡터 생성
[4] HybridAgent.select_action() → 행동 결정 (0:hold, 1:buy, 2:sell)
[5] RiskManager.validate() → 리스크 검증
[6] OrderManager.place_order() → 주문 실행
[7] PositionManager.update() → 포지션 업데이트
[8] Loop back to [1]
```

---

### Phase 4: 페이퍼 트레이딩 검증 (Paper Trading Validation) - 3-6개월

**목표**: 시스템 안정성 및 수익성 검증

#### 주요 작업

| 순번 | 작업명 | 테스트 기간 | 검증 지표 |
|------|--------|------------|----------|
| 4.1 | 연속 페이퍼 트레이딩 실행 | 1개월 | 99% 업타임 |
| 4.2 | 다양한 시장 환경 테스트 | 3개월 | 상승/하락/횡보장 대응 |
| 4.3 | 모니터링 시스템 구축 | 1주 | Grafana 대시보드 |
| 4.4 | 알림 시스템 구축 | 1주 | Slack/Email 알림 |
| 4.5 | 성능 최적화 | 2주 | 레이턴시 < 100ms |
| 4.6 | 수익성 분석 | 3개월 | Sharpe Ratio, MDD |

#### 완료 기준
- ✅ **3개월 이상** 안정적 운영 (99% 업타임)
- ✅ **Sharpe Ratio > 1.5** (위험 대비 수익)
- ✅ **Max Drawdown < 15%** (최대 손실폭)
- ✅ **일평균 거래 빈도**: 10-20회
- ✅ **Claude API 비용**: < $50/월

#### 검증 시나리오

| 시나리오 | 조건 | 예상 결과 |
|----------|------|----------|
| **급격한 상승장** | 일일 상승률 > 3% | 적절한 매수 타이밍 포착 |
| **급격한 하락장** | 일일 하락률 > 3% | 손절 자동 실행 |
| **횡보장** | 일일 변동률 < 1% | 불필요한 거래 억제 (hold) |
| **연결 끊김** | API 연결 실패 | 자동 재연결 또는 안전 종료 |
| **API 오류** | 주문 거부 | 로깅 및 알림, 재시도 안함 |

---

### Phase 5: 실전 트레이딩 (Live Trading) - 선택 사항

**⚠️ 경고**: 실제 자본 투입 전 충분한 검증 필수

#### 주요 작업

| 순번 | 작업명 | 초기 자본 | 위험 관리 |
|------|--------|----------|----------|
| 5.1 | 리스크 한도 재검증 | - | 모든 한도 재확인 |
| 5.2 | 소액 실전 테스트 | $500-1,000 | 손실 허용 범위 설정 |
| 5.3 | 일일 성과 모니터링 | - | 매일 리뷰 |
| 5.4 | 점진적 자본 확대 | +$500/월 | 수익성 검증 후 진행 |

#### 실전 전환 조건
- ✅ 페이퍼 트레이딩 **6개월 이상** 성공적 운영
- ✅ Sharpe Ratio **> 2.0**
- ✅ Max Drawdown **< 10%**
- ✅ 모든 리스크 관리 시스템 검증 완료
- ✅ 법적/세무 자문 완료

---

## 3. 상세 작업 일정

### 3.1 단계별 타임라인

```
Week 1-2:  Phase 1 - Foundation (기반 구축)
  ├─ W1: BaseBroker, KISBroker, IBBroker
  └─ W2: MarketDataStream, DataBuffer

Week 3-4:  Phase 2 - Execution Engine (실행 엔진)
  ├─ W3: OrderManager, PositionManager
  └─ W4: RiskManager, BrokerRouter

Week 5:    Phase 3 - Agent Integration (에이전트 통합)
  ├─ RealtimeEnvAdapter
  └─ Trading Loop 구현

Week 6:    통합 테스트 및 버그 수정

Month 2-7: Phase 4 - Paper Trading (페이퍼 트레이딩)
  ├─ Month 2: 초기 안정화
  ├─ Month 3-4: 다양한 시장 환경 테스트
  ├─ Month 5-6: 성능 최적화
  └─ Month 7: 최종 검증 및 리포트

Month 8+:  Phase 5 - Live Trading (선택 사항)
```

### 3.2 마일스톤

| 마일스톤 | 목표 날짜 | 완료 기준 |
|----------|----------|----------|
| **M1**: 브로커 연결 완료 | Week 2 | KIS, IB 연결 및 데이터 수신 |
| **M2**: 페이퍼 주문 체결 | Week 4 | 주문 → 체결 → 포지션 반영 |
| **M3**: 실시간 에이전트 실행 | Week 5 | 1시간 연속 자동 트레이딩 |
| **M4**: 1개월 안정 운영 | Month 2 | 99% 업타임 달성 |
| **M5**: 3개월 수익성 검증 | Month 4 | Sharpe > 1.5, MDD < 15% |
| **M6**: 최종 승인 | Month 7 | 실전 전환 여부 결정 |

---

## 4. 기술 스택 및 리소스

### 4.1 필수 기술 스택

#### 핵심 라이브러리

| 라이브러리 | 버전 | 용도 | 설치 명령 |
|-----------|------|------|----------|
| **python-kis** | latest | 한국투자증권 API | `pip install python-kis` |
| **ib_insync** | latest | Interactive Brokers API | `pip install ib_insync` |
| **asyncio** | built-in | 비동기 I/O | Python 3.7+ 기본 포함 |
| **websockets** | latest | WebSocket 통신 | `pip install websockets` |
| **pyyaml** | latest | 설정 파일 파싱 | `pip install pyyaml` |
| **python-dotenv** | latest | 환경 변수 관리 | `pip install python-dotenv` |

#### 기존 의존성 (재사용)

| 라이브러리 | 용도 |
|-----------|------|
| **PyTorch** | RL 에이전트 추론 |
| **pandas** | 데이터 처리 |
| **numpy** | 수치 연산 |
| **anthropic** | Claude AI API |

### 4.2 개발 환경

- **Python**: 3.8 이상 (3.10 권장)
- **OS**: Linux/macOS (Windows WSL2 가능)
- **RAM**: 최소 8GB (16GB 권장)
- **네트워크**: 안정적인 초고속 인터넷 (레이턴시 중요)

### 4.3 브로커 계정 요구사항

#### 한국투자증권 (KIS)
- 해외주식 거래 계좌 개설
- API 서비스 신청 (개발자 포털)
- App Key 및 App Secret 발급
- 초기 입금: 페이퍼 트레이딩 무료, 실전 최소 $1,000 권장

#### Interactive Brokers (IB)
- IB 계좌 개설 (한국 거주자 가능)
- TWS (Trader Workstation) 또는 IB Gateway 설치
- API 접근 활성화
- 초기 입금: 최소 $10,000 (정책 변동 가능)

### 4.4 외부 서비스

| 서비스 | 용도 | 비용 |
|--------|------|------|
| **Claude API** | AI 시장 분석 | ~$50/월 (예상) |
| **Slack** | 알림 | 무료 |
| **Grafana Cloud** | 모니터링 (선택) | 무료 티어 |

---

## 5. 위험 요인 및 대응 방안

### 5.1 기술적 위험

| 위험 | 발생 가능성 | 영향도 | 대응 방안 |
|------|-----------|--------|----------|
| **API 연결 불안정** | 중 | 높음 | 자동 재연결, 3회 시도 후 안전 종료 |
| **데이터 지연** | 중 | 중간 | 타임스탬프 검증, 오래된 데이터 거부 |
| **주문 거부** | 중 | 중간 | 재시도 없음, 로깅 및 알림 |
| **시스템 다운** | 낮음 | 높음 | 모든 포지션 청산 후 종료 |
| **버그/예외 처리** | 중 | 높음 | 포괄적 try-catch, 상세 로깅 |

### 5.2 재무적 위험

| 위험 | 발생 가능성 | 영향도 | 대응 방안 |
|------|-----------|--------|----------|
| **일일 손실 한도 초과** | 중 | 높음 | 자동 거래 중단 (5% 한도) |
| **단일 포지션 과다** | 낮음 | 중간 | 포지션 크기 제한 (30%) |
| **급격한 시장 변동** | 중 | 높음 | 손절 자동 실행 (2%) |
| **총 자본 손실** | 낮음 | 매우 높음 | 긴급 종료 (10% 총 손실) |

### 5.3 운영적 위험

| 위험 | 발생 가능성 | 영향도 | 대응 방안 |
|------|-----------|--------|----------|
| **인터넷 연결 끊김** | 낮음 | 높음 | 연결 모니터링, 안전 종료 |
| **브로커 API 점검** | 중 | 중간 | 점검 일정 확인, 사전 대비 |
| **API 키 만료** | 낮음 | 중간 | 만료 알림, 자동 갱신 고려 |
| **Claude API 할당량 초과** | 낮음 | 낮음 | 호출 빈도 제한, 캐싱 |

### 5.4 법적/규제 위험

| 위험 | 대응 방안 |
|------|----------|
| **알고리즘 트레이딩 규제** | 증권사 정책 확인, 필요시 신고 |
| **세무 신고** | 거래 내역 상세 기록, 전문가 상담 |
| **자본시장법 준수** | 과도한 시장 교란 행위 금지 |

---

## 6. 성공 기준

### 6.1 기술적 성공 지표

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| **시스템 업타임** | > 99% | 실행 로그 분석 |
| **데이터 레이턴시** | < 100ms | 타임스탬프 차이 |
| **주문 체결 시간** | < 5초 | 주문 생성 → 체결 시간 |
| **API 오류율** | < 1% | 오류 로그 / 전체 호출 |
| **재연결 성공률** | > 95% | 연결 끊김 후 복구 비율 |

### 6.2 재무적 성공 지표 (페이퍼 트레이딩)

| 지표 | 목표 | 산출 방법 |
|------|------|----------|
| **Sharpe Ratio** | > 1.5 | (수익률 - 무위험수익률) / 변동성 |
| **Max Drawdown** | < 15% | 최대 손실폭 / 최고 자산 |
| **Win Rate** | > 50% | 수익 거래 / 전체 거래 |
| **Profit Factor** | > 1.3 | 총 수익 / 총 손실 |
| **일평균 거래 수** | 10-20회 | 과도한 거래 방지 |

### 6.3 운영적 성공 지표

| 지표 | 목표 |
|------|------|
| **알림 응답 시간** | < 1분 (긴급 알림) |
| **일일 모니터링 시간** | < 30분 (자동화 후) |
| **Claude API 비용** | < $50/월 |
| **버그 발견 후 수정** | < 24시간 |

---

## 7. 즉시 착수 가능한 작업

### 7.1 우선순위 1: 환경 설정 (오늘 착수 가능)

#### 작업 1-1: 디렉토리 구조 생성

```bash
cd /mnt/d/drbreakfast/financial-rl-claude-skills/realtime_trading

# 핵심 디렉토리 생성
mkdir -p src/brokers
mkdir -p src/core
mkdir -p src/risk
mkdir -p src/state
mkdir -p src/adapters
mkdir -p config
mkdir -p scripts
mkdir -p tests
mkdir -p logs
```

#### 작업 1-2: 의존성 설치

```bash
# requirements.txt에 추가할 패키지
pip install python-kis
pip install ib_insync
pip install pyyaml
pip install python-dotenv
pip install websockets
pip install asyncio
```

#### 작업 1-3: 환경 변수 파일 생성

`.env` 파일 생성 (절대 Git에 커밋하지 말것):
```bash
# Korea Investment Securities
KIS_APP_KEY=your_app_key_here
KIS_APP_SECRET=your_app_secret_here
KIS_ACCOUNT_NO=your_account_number

# Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497  # 7497=paper, 7496=live
IB_CLIENT_ID=1

# Claude API
ANTHROPIC_API_KEY=your_existing_key

# Trading Mode
TRADING_MODE=paper  # paper or live

# Alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### 7.2 우선순위 2: 핵심 인터페이스 구현 (Week 1)

#### 작업 2-1: BaseBroker 추상 클래스

**파일**: `src/brokers/base_broker.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0
    average_fill_price: Optional[float] = None
    broker: str = ""
    timestamp: Optional[str] = None

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

class BaseBroker(ABC):
    """모든 브로커가 구현해야 하는 통합 인터페이스"""

    @abstractmethod
    async def connect(self) -> bool:
        """브로커 API 연결"""
        pass

    @abstractmethod
    async def disconnect(self):
        """브로커 API 연결 해제"""
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None
    ) -> Order:
        """주문 실행"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """주문 상태 조회"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """현재 포지션 조회"""
        pass

    @abstractmethod
    async def subscribe_realtime_data(self, symbols: List[str], callback):
        """실시간 시장 데이터 구독"""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """최신 가격 조회"""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """계좌 정보 조회"""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """연결 상태"""
        pass
```

#### 작업 2-2: 설정 파일 템플릿 작성

**파일**: `config/brokers.yaml`

```yaml
brokers:
  kis:
    enabled: true
    name: "Korea Investment & Securities"
    markets: ["KR", "US", "CN", "JP"]
    api_type: "REST + WebSocket"
    app_key: "${KIS_APP_KEY}"
    app_secret: "${KIS_APP_SECRET}"
    account_no: "${KIS_ACCOUNT_NO}"

  ib:
    enabled: true
    name: "Interactive Brokers"
    markets: ["US", "EU", "ASIA"]
    host: "${IB_HOST}"
    port: "${IB_PORT}"
    client_id: "${IB_CLIENT_ID}"
```

**파일**: `config/trading.yaml`

```yaml
trading:
  symbols: ["NVDA", "PLTR"]
  update_frequency: 5  # 초 단위 (5초마다 의사결정)
  claude_consultation_frequency: 10  # 스텝 (10회마다 Claude 컨설팅)

agents:
  rl_agent:
    type: "GRPO"  # 또는 "DeepSeek"
    model_path: "../models/grpo_trained.pth"
    state_dim: 50
    action_dim: 3

  hybrid_agent:
    decision_mode: "sequential"
    rl_weight: 0.7
    claude_weight: 0.3
    enable_claude_override: true
    risk_threshold: 0.7
```

**파일**: `config/risk.yaml`

```yaml
risk_limits:
  # 손실 한도
  daily_loss_limit: 0.05  # 5% of capital
  position_size_limit: 0.30  # 30% of capital per symbol
  stop_loss_pct: 0.02  # 2% stop-loss per position
  emergency_shutdown_loss: 0.10  # 10% total loss → shutdown

  # 거래 한도
  max_daily_orders: 100
  max_position_count: 10
  min_order_value: 100  # USD

  # 시스템 한도
  connection_timeout: 10  # seconds
  order_timeout: 30  # seconds
  max_retry_attempts: 3

alerts:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
    alert_levels: ["ERROR", "CRITICAL"]

  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    from_email: "${ALERT_EMAIL_FROM}"
    to_email: "${ALERT_EMAIL_TO}"
```

### 7.3 우선순위 3: 의사결정 포인트

다음 작업을 진행하기 전에 결정이 필요한 사항들:

#### 질문 1: 어떤 브로커를 먼저 구현할까?

**옵션 A**: 한국투자증권(KIS) 우선
- **장점**: 한국 거주자에게 접근성 좋음, 문서 한글 제공
- **단점**: API 문서 상대적으로 부족

**옵션 B**: Interactive Brokers(IB) 우선
- **장점**: 글로벌 스탠다드, 문서 풍부, 커뮤니티 활발
- **단점**: 계좌 개설 요구사항 높음 (최소 자본 $10K)

**권장**: **한국투자증권(KIS) 우선** → 이후 IB 추가

#### 질문 2: 페이퍼 트레이딩 목표 기간?

**옵션 A**: 3개월
- 빠른 검증, 조기 피드백

**옵션 B**: 6개월
- 다양한 시장 환경 경험, 안정성 확보

**권장**: **최소 3개월, 가능하면 6개월**

#### 질문 3: 초기 거래 심볼?

**현재 분석 완료**: NVDA, PLTR

**추가 고려 대상**:
- **대형주**: AAPL, MSFT, GOOGL (안정성)
- **중형주**: NVDA, AMD (성장성)
- **소형주**: PLTR (변동성)

**권장**: **NVDA, PLTR로 시작** → 안정적 수익 확인 후 확대

#### 질문 4: 거래 빈도?

**옵션 A**: 고빈도 (< 1분)
- 초저지연 인프라 필요, 복잡도 높음

**옵션 B**: 중빈도 (5-15분)
- 균형잡힌 접근, 관리 가능

**옵션 C**: 저빈도 (1시간 이상)
- 관리 쉬움, 수익 기회 제한

**권장**: **중빈도 (5분)** → 시작 후 조정

---

## 8. 다음 단계 (Next Actions)

### 즉시 실행 (오늘)

1. **디렉토리 구조 생성** (5분)
   ```bash
   cd realtime_trading && mkdir -p src/{brokers,core,risk,state,adapters} config scripts tests logs
   ```

2. **의존성 설치** (10분)
   ```bash
   pip install python-kis ib_insync pyyaml python-dotenv websockets
   ```

3. **환경 변수 파일 생성** (10분)
   - `.env` 파일 작성 (API 키는 추후 입력)

4. **설정 파일 작성** (20분)
   - `config/brokers.yaml`
   - `config/trading.yaml`
   - `config/risk.yaml`

### 이번 주 (Week 1)

5. **BaseBroker 인터페이스 구현** (2시간)
   - `src/brokers/base_broker.py`

6. **KISBroker 구현 시작** (4시간)
   - `src/brokers/kis_broker.py`
   - 연결 및 인증 로직

7. **MarketDataStream 스켈레톤 작성** (2시간)
   - `src/core/market_data_stream.py`

### 다음 주 (Week 2)

8. **KISBroker 완성 및 테스트**
9. **IBBroker 구현 시작**
10. **DataBuffer 구현**

---

## 9. 참고 문서

### 내부 문서
- **아키텍처 가이드**: `realtime_trading/CLAUDE.md`
- **프로젝트 구조**: `docs/architecture/PROJECT_STRUCTURE.md`
- **Claude 통합 가이드**: `docs/guides/CLAUDE_INTEGRATION_GUIDE.md`

### 외부 문서
- **한국투자증권 API**: https://apiportal.koreainvestment.com
- **Interactive Brokers API**: https://www.interactivebrokers.com/en/trading/ib-api.php
- **ib_insync 문서**: https://ib-insync.readthedocs.io
- **python-kis GitHub**: https://github.com/Soju06/python-kis

---

## 10. 결론

### 프로젝트 요약

- **목표**: 백테스팅 시스템을 실시간 멀티브로커 트레이딩 시스템으로 전환
- **기간**: 5-12개월 (페이퍼 트레이딩 포함)
- **핵심 기술**: Python, asyncio, KIS API, IB API, RL agents, Claude AI
- **예상 투자**: 시간 투자 (개발 200시간), 금전 투자 (API 비용 ~$50/월)

### 성공 가능성 평가

| 요소 | 평가 | 근거 |
|------|------|------|
| **기술적 실현 가능성** | ✅ 높음 | 필요한 API 및 라이브러리 모두 존재 |
| **아키텍처 견고성** | ✅ 높음 | 4계층 설계, 명확한 책임 분리 |
| **기존 코드 재사용** | ✅ 높음 | RL 에이전트 및 Claude 통합 그대로 활용 |
| **리스크 관리** | ✅ 적절 | 다층 안전장치 (손절, 한도, 알림) |
| **수익성** | ⚠️ 검증 필요 | 페이퍼 트레이딩으로 입증 필요 |

### 최종 권고사항

1. **단계적 접근**: Phase 1-3 완료 후 반드시 장기 페이퍼 트레이딩 (3-6개월)
2. **리스크 최우선**: 모든 안전장치 구현 및 검증 후 진행
3. **지속적 모니터링**: 초기 실행 시 매일 성과 리뷰
4. **점진적 확대**: 소액으로 시작 → 성공 확인 후 확대
5. **전문가 자문**: 필요시 금융/법률 전문가 상담

---

**보고서 작성**: Claude Code
**최종 검토일**: 2025년 10월 19일
**버전**: 1.0

---

## 부록 A: 체크리스트

### Phase 1 완료 체크리스트

- [ ] 디렉토리 구조 생성 완료
- [ ] 의존성 패키지 설치 완료
- [ ] `.env` 파일 작성 완료 (API 키 입력)
- [ ] `BaseBroker` 인터페이스 구현 완료
- [ ] `KISBroker` 연결 테스트 성공
- [ ] `IBBroker` 연결 테스트 성공
- [ ] 실시간 데이터 수신 확인 (NVDA)
- [ ] 기술 지표 계산 검증 (RSI, MACD)

### Phase 2 완료 체크리스트

- [ ] `OrderManager` 구현 완료
- [ ] 페이퍼 트레이딩 주문 체결 성공
- [ ] `PositionManager` 포지션 추적 정확성 검증
- [ ] `RiskManager` 한도 검증 테스트 통과
- [ ] `BrokerRouter` 브로커 선택 로직 검증
- [ ] 엔드투엔드 주문 플로우 테스트 통과

### Phase 3 완료 체크리스트

- [ ] `RealtimeEnvAdapter` 상태 벡터 생성 검증
- [ ] `HybridAgent` 실시간 통합 완료
- [ ] 설정 파일 로딩 테스트 통과
- [ ] 1시간 연속 실행 에러 없음
- [ ] Claude 컨설팅 주기적 실행 확인

### Phase 4 완료 체크리스트 (3개월 후)

- [ ] 3개월 이상 안정적 운영 (99% 업타임)
- [ ] Sharpe Ratio > 1.5 달성
- [ ] Max Drawdown < 15% 유지
- [ ] 모니터링 대시보드 구축 완료
- [ ] 알림 시스템 동작 확인
- [ ] 성능 최적화 완료 (레이턴시 < 100ms)

---

**이 보고서를 실행 가이드로 활용하여 단계별로 진행하시기 바랍니다.**
