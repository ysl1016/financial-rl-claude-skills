# Quick Start Guide - Yahoo Finance + Claude Integration

## 🎯 목표

실제 Yahoo Finance 데이터로 Claude 통합 RL 트레이딩 시스템을 5분 안에 실행하기

---

## 📋 사전 요구사항

- Python 3.8 이상
- 인터넷 연결 (Yahoo Finance 데이터 다운로드용)
- Anthropic API 키 (선택사항, Claude 기능 사용 시)

---

## ⚡ 빠른 시작 (3단계)

### 1단계: 패키지 설치

```bash
cd financial-rl-trading
pip install -r requirements.txt
```

또는 개별 설치:
```bash
pip install yfinance pandas numpy torch anthropic matplotlib
```

### 2단계: API 키 설정 (선택사항)

Claude 기능을 사용하려면:
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

> 💡 API 키 없이도 기본 RL 기능은 동작합니다!

### 3단계: 데모 실행

```bash
# 간단한 데모 (Yahoo Finance 데이터 테스트)
python examples/quick_demo_yahoo_finance.py

# 또는 전체 하이브리드 학습
python examples/hybrid_claude_trading.py --symbol SPY --episodes 5
```

---

## 📊 실제 사용 예시

### 예시 1: SPY (S&P 500 ETF) 트레이딩

```bash
python examples/hybrid_claude_trading.py \
    --symbol SPY \
    --start_date 2022-01-01 \
    --end_date 2023-12-31 \
    --episodes 10 \
    --mode weighted \
    --frequency 20
```

**결과:**
- Yahoo Finance에서 SPY 데이터 자동 다운로드
- 40+ 기술적 지표 자동 계산
- RL 에이전트 학습
- Claude가 매 20스텝마다 시장 분석
- 하이브리드 의사결정으로 트레이딩

### 예시 2: AAPL (애플) 단기 트레이딩

```bash
python examples/hybrid_claude_trading.py \
    --symbol AAPL \
    --start_date 2023-01-01 \
    --episodes 15 \
    --mode sequential \
    --frequency 10
```

### 예시 3: 여러 주식 비교

```bash
# Tesla
python examples/hybrid_claude_trading.py --symbol TSLA --episodes 5

# Microsoft
python examples/hybrid_claude_trading.py --symbol MSFT --episodes 5

# NVIDIA
python examples/hybrid_claude_trading.py --symbol NVDA --episodes 5
```

---

## 🔍 데이터 확인

### Yahoo Finance에서 사용 가능한 심볼

```python
import yfinance as yf

# 주식 정보 확인
ticker = yf.Ticker("AAPL")
print(ticker.info['longName'])  # Apple Inc.
print(ticker.info['sector'])    # Technology

# 데이터 다운로드
data = ticker.history(period="1y")
print(f"Downloaded {len(data)} days")
```

### 인기 있는 심볼들

| 심볼 | 이름 | 유형 |
|------|------|------|
| SPY | S&P 500 ETF | 인덱스 |
| QQQ | NASDAQ-100 ETF | 인덱스 |
| AAPL | Apple | 기술주 |
| MSFT | Microsoft | 기술주 |
| TSLA | Tesla | 자동차 |
| NVDA | NVIDIA | 반도체 |
| GOOGL | Google | 기술주 |
| AMZN | Amazon | 전자상거래 |
| JPM | JP Morgan | 금융 |
| GLD | Gold ETF | 원자재 |

---

## 🤖 Claude 통합 모드

### 모드 1: Weighted (균형)

```bash
python examples/hybrid_claude_trading.py \
    --mode weighted \
    --frequency 20
```

- RL과 Claude 의견을 가중 평균
- RL 70%, Claude 30% (기본값)
- **추천**: 일반적인 시장 상황

### 모드 2: Sequential (검증)

```bash
python examples/hybrid_claude_trading.py \
    --mode sequential \
    --frequency 15
```

- RL이 제안, Claude가 검증
- 고위험 상황에서 Claude가 거부권
- **추천**: 변동성 큰 시장

### 모드 3: Ensemble (보수적)

```bash
python examples/hybrid_claude_trading.py \
    --mode ensemble \
    --frequency 30
```

- RL과 Claude 모두 동의할 때만 행동
- 가장 보수적인 접근
- **추천**: 불확실한 시장

---

## 📈 결과 확인

### 자동 생성되는 파일들

1. **학습 그래프** - `hybrid_training_SPY_YYYYMMDD_HHMMSS.png`
   - 포트폴리오 가치 변화
   - 에피소드별 보상
   - Claude 상담 빈도
   - 의사결정 통계

2. **의사결정 로그** - `hybrid_decisions_SPY_YYYYMMDD_HHMMSS.json`
   - 모든 의사결정 기록
   - RL vs Claude 의견 비교
   - 최종 행동 및 이유

3. **학습된 모델** - `hybrid_model_SPY_YYYYMMDD_HHMMSS.pt`
   - 재사용 가능한 RL 모델
   - 나중에 로드하여 계속 학습 가능

### 결과 예시

```
Episode 5/10
  Total Reward: 45.23
  Final Portfolio: $108,450.00
  Total Return: 8.45%
  Claude Consultations: 12
  Total Trades: 34

Decision Statistics:
  Total Decisions: 245
  Claude Influenced: 58 (23.7%)
  Claude Overrides: 12 (4.9%)
  Agreement Rate: 78.4%
```

---

## 🛠️ 트러블슈팅

### 문제 1: yfinance 설치 오류

```bash
# pip 업그레이드
pip install --upgrade pip

# yfinance 재설치
pip install --upgrade yfinance
```

### 문제 2: 데이터 다운로드 실패

```bash
# 인터넷 연결 확인
ping yahoo.com

# 다른 심볼로 시도
python examples/quick_demo_yahoo_finance.py
```

### 문제 3: Claude API 오류

```bash
# API 키 확인
echo $ANTHROPIC_API_KEY

# API 키 다시 설정
export ANTHROPIC_API_KEY="sk-ant-..."

# API 키 없이 실행 (RL만 사용)
unset ANTHROPIC_API_KEY
python examples/hybrid_claude_trading.py --symbol SPY --episodes 3
```

### 문제 4: GPU 메모리 부족

```python
# CPU로 강제 실행
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

또는 배치 크기 줄이기:
```bash
# 에피소드 수 줄이기
python examples/hybrid_claude_trading.py --episodes 3
```

---

## 💰 비용 관리 (Claude API)

### 예상 비용

| 설정 | API 호출 수 | 예상 비용 (Sonnet) |
|------|------------|-------------------|
| frequency=50, episodes=5 | ~50 calls | ~$0.15 |
| frequency=20, episodes=10 | ~250 calls | ~$0.75 |
| frequency=10, episodes=20 | ~1000 calls | ~$3.00 |

### 비용 절감 팁

1. **상담 빈도 줄이기**
   ```bash
   --frequency 50  # 50 스텝마다만 Claude 상담
   ```

2. **저렴한 모델 사용**
   ```python
   analyzer = ClaudeMarketAnalyzer(
       model="claude-3-haiku-20240307"  # Sonnet보다 ~90% 저렴
   )
   ```

3. **캐싱 활용** (자동으로 활성화됨)
   - 반복적인 프롬프트 90% 할인

4. **API 키 없이 테스트**
   ```bash
   # RL만 사용하여 무료 테스트
   unset ANTHROPIC_API_KEY
   python examples/hybrid_claude_trading.py --episodes 20
   ```

---

## 📚 더 알아보기

### 문서

- [Claude Integration Guide](docs/CLAUDE_INTEGRATION_GUIDE.md) - 전체 가이드
- [API Documentation](docs/api_documentation.md) - API 레퍼런스
- [Testing Guide](TESTING_GUIDE.md) - 테스팅 방법

### 예시 코드

- `examples/quick_demo_yahoo_finance.py` - 빠른 데모
- `examples/hybrid_claude_trading.py` - 전체 하이브리드 학습
- `examples/train_grpo.py` - 기본 RL 학습
- `src/claude_integration/` - Claude 통합 모듈

### 커스터마이징

```python
# 자신만의 분석 프롬프트 작성
analyzer = ClaudeMarketAnalyzer()
custom_analysis = analyzer.analyze_market_state(
    market_data=your_data,
    technical_indicators=your_indicators,
    current_position=0,
    portfolio_value=100000,
    context="This is a high volatility period"  # 커스텀 컨텍스트
)
```

---

## 🎓 학습 경로

### 초급
1. ✅ `quick_demo_yahoo_finance.py` 실행
2. ✅ SPY로 3 에피소드 학습
3. ✅ 결과 그래프 분석

### 중급
1. 다양한 심볼 시도 (AAPL, TSLA, etc.)
2. 의사결정 모드 비교 (weighted vs sequential)
3. 파라미터 튜닝 (frequency, episodes)

### 고급
1. 커스텀 분석 프롬프트 작성
2. 새로운 기술적 지표 추가
3. 다중 자산 포트폴리오 구축
4. 실시간 데이터 스트리밍 연결

---

## 🚀 프로덕션 배포

### Docker로 실행

```bash
# 이미지 빌드
docker build -t financial-rl-trading .

# 컨테이너 실행
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           financial-rl-trading \
           python examples/hybrid_claude_trading.py --symbol SPY
```

### API 서버 시작

```bash
# API 서버 실행
python -m src.api.app run --model-path models/hybrid_model.pt

# 테스트
curl http://localhost:8000/health
```

---

## ❓ FAQ

**Q: Yahoo Finance 데이터는 무료인가요?**
A: 네! Yahoo Finance API는 개인 사용에 무료입니다.

**Q: Claude API 없이도 사용 가능한가요?**
A: 네! RL 기능은 API 없이도 완전히 동작합니다.

**Q: 실시간 트레이딩에 사용할 수 있나요?**
A: 백테스팅용으로 설계되었지만, API 연결하면 실시간 사용 가능합니다.

**Q: 어떤 주식이 가장 잘 동작하나요?**
A: 거래량이 많은 대형주 (SPY, AAPL, MSFT 등)가 안정적입니다.

**Q: 학습에 얼마나 걸리나요?**
A: 5 에피소드 기준 5-10분 정도 (Claude 사용 시 약간 더 소요)

---

**시작하세요!**

```bash
python examples/quick_demo_yahoo_finance.py
```

즐거운 트레이딩 되세요! 🎉
