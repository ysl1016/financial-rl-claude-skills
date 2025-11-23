# 100 Episodes Training Guide

**목적**: 상태 의존적 보상 환경으로 100 episodes 학습  
**예상 시간**: 20-30분

---

## 📋 실행 방법

### 1. Colab 업로드
1. https://colab.research.google.com/ 접속
2. File → Upload notebook
3. `train_100_episodes.ipynb` 선택

### 2. Runtime 설정
- Runtime → Change runtime type
- Hardware accelerator: **GPU (T4)**

### 3. 실행
- Runtime → Run all
- 또는 Shift+Enter로 셀별 실행

---

## 📊 노트북 구조

| Cell | 내용 | 소요 시간 |
|------|------|-----------|
| 1 | Title | - |
| 2 | Setup & Install | 1분 |
| 3 | Download Data (SPY) | 30초 |
| 4 | StateDependentRewardEnv | 5초 |
| 5 | CleanGRPOAgent | 5초 |
| 6 | Training (100 episodes) | 15-20분 |
| 7 | Test Evaluation | 10초 |
| 8 | Save Model | 5초 |

---

## 🎯 성공 기준

| 지표 | 목표 | 이전 (Phase 3) |
|------|------|----------------|
| Alpha | > 0% | -12.37% |
| Trades | 10-20 | 0 |
| Buy % | 30-70% | 0% |
| Sell % | > 0% | 100% |

---

## 📈 예상 결과

**학습 진행**:
```
Episode 20/100:
  Portfolio: $102,500
  Alpha: +1.2%
  Trades: 12.3
  Epsilon: 0.243

Episode 100/100:
  Portfolio: $105,000
  Alpha: +2.5%
  Trades: 15.7
  Epsilon: 0.010
```

**테스트 결과**:
```
RL Return: 8.50%
Buy & Hold: 6.20%
Alpha: +2.30%
Sharpe: 0.85
Max DD: 12.50%
Trades: 14

Action Distribution:
  Hold: 45 (45.0%)
  Buy: 32 (32.0%)
  Sell: 23 (23.0%)

✅ Alpha > 0%
✅ Trades > 10
✅ Buy 30-70%
✅ Sell > 0%

Result: 4/4 criteria passed
🎉 SUCCESS!
```

---

## 💾 저장 위치

**모델**:
```
/content/drive/MyDrive/financial-rl-trading/models/state_dependent_100ep.pt
```

**포함 내용**:
- Network weights
- Optimizer state
- Epsilon value
- Test results (alpha, sharpe, trades, etc.)

---

## ⚠️ 문제 해결

### GPU 메모리 부족
```python
# Runtime → Factory reset runtime
# 다시 실행
```

### 데이터 다운로드 실패
```python
# Cell 3 재실행
# 또는 다른 심볼 시도 (예: 'QQQ', 'IWM')
```

### 학습 중단
```python
# 마지막 저장된 모델 로드
checkpoint = torch.load(save_path)
agent.network.load_state_dict(checkpoint['network_state_dict'])
```

---

## 🔄 다음 단계

### 성공 시
1. 결과 스크린샷 저장
2. Phase 2 (GRPO Agent) 준비
3. 목표: Alpha > +5%, Sharpe > 1.0

### 실패 시 (< 3/4 기준)
1. 보상 가중치 조정
2. Epsilon decay 속도 조정
3. 학습 episodes 증가 (200)

---

**작성일**: 2025-11-23 19:25  
**문의**: 학습 완료 후 결과 공유
