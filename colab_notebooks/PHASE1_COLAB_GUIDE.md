# Phase 1 Colab 테스트 가이드

**생성일**: 2025-11-23  
**목적**: Google Colab에서 Phase 1 상태 의존적 보상 환경 테스트

---

## 📋 준비물

1. Google 계정 (Colab 접속용)
2. 생성된 노트북: `phase1_state_dependent_test.ipynb`

---

## 🚀 실행 방법

### Step 1: 노트북 생성

로컬에서 실행:
```bash
cd /Users/ihyunseo/Projects/financial-rl-claude-skills/colab_notebooks
python3 phase1_state_dependent_test.py
```

출력:
```
✓ Notebook created: phase1_state_dependent_test.ipynb
```

### Step 2: Colab 업로드

1. **Colab 접속**
   - https://colab.research.google.com/

2. **노트북 업로드**
   - File → Upload notebook
   - `phase1_state_dependent_test.ipynb` 선택

3. **GPU 설정** (선택사항)
   - Runtime → Change runtime type
   - Hardware accelerator: GPU (T4)

### Step 3: 실행

**전체 실행**:
- Runtime → Run all

**셀별 실행**:
1. ✅ Mount Drive (선택사항)
2. ✅ Install Dependencies
3. ✅ Define Environment
4. ✅ Create Test Data
5. ✅ Test 1: Reward Differentiation
6. ✅ Test 2: Action Balance
7. ✅ Visualization
8. ✅ Summary

---

## 📊 예상 결과

### Test 1: Reward Differentiation

```
【Buy Action in Different Conditions】
  Buy in Oversold (RSI=25): Reward = 5.2186
  Buy in Overbought (RSI=75): Reward = -2.7814
  Buy in Neutral (RSI=50): Reward = 1.2186

  Validation:
  ✅ Buy rewards correctly differentiated!
     Oversold (5.22) > Neutral (1.22) > Overbought (-2.78)

【Sell Action in Different Conditions】
  Sell in Overbought (RSI=75): Reward = 5.0471
  Sell in Oversold (RSI=25): Reward = -2.9529
  Sell in Neutral (RSI=50): Reward = 1.0471

  Validation:
  ✅ Sell rewards correctly differentiated!
     Overbought (5.05) > Neutral (1.05) > Oversold (-2.95)
```

### Test 2: Action Balance

```
【Oversold Region (RSI=25)】
  Hold: -0.4529
  Buy:  5.2186
  ✅ Buy > Hold in oversold region

【Overbought Region (RSI=75)】
  Hold: -0.4529
  Sell: 5.0471
  ✅ Sell > Hold in overbought region
```

### Visualization

생성되는 그래프:
1. **Buy Reward vs RSI**: RSI 감소 시 Buy 보상 증가
2. **Sell Reward vs RSI**: RSI 증가 시 Sell 보상 증가
3. **Buy vs Sell Comparison**: 교차점 확인
4. **Summary Table**: 보상 구조 요약

---

## 🎯 성공 기준

모든 테스트 통과:
- ✅ Buy: Oversold > Neutral > Overbought
- ✅ Sell: Overbought > Neutral > Oversold
- ✅ Buy > Hold (in oversold)
- ✅ Sell > Hold (in overbought)

---

## 💾 결과 저장

**Drive 저장 경로**:
```
/content/drive/MyDrive/financial-rl-trading/phase1_results/
├── reward_structure.png  # 보상 구조 시각화
```

---

## ⚠️ 문제 해결

### 문제 1: Drive 마운트 실패
```python
# Cell 2 재실행
from google.colab import drive
drive.mount('/content/drive')
```

### 문제 2: 패키지 설치 오류
```python
# Cell 3 재실행
!pip install --upgrade gym numpy pandas matplotlib seaborn
```

### 문제 3: 메모리 부족
- Runtime → Factory reset runtime
- 다시 실행

---

## 📝 노트북 구조

| Cell | 내용 | 소요 시간 |
|------|------|-----------|
| 1 | Title & Overview | - |
| 2 | Mount Drive | 10초 |
| 3 | Install Dependencies | 30초 |
| 4 | Define Environment | 5초 |
| 5 | Create Test Data | 5초 |
| 6 | Test 1: Differentiation | 10초 |
| 7 | Test 2: Balance | 10초 |
| 8 | Visualization | 20초 |
| 9 | Summary | - |

**총 소요 시간**: 약 2분

---

## 🔄 다음 단계

테스트 성공 후:

1. **100 Episodes 학습**
   - 기존 에이전트 사용
   - StateDependentRewardEnv 적용
   - 목표: Alpha > 0%

2. **Phase 2 진행**
   - GRPO Agent 구현
   - Critic 제거
   - 그룹 샘플링

---

**작성일**: 2025-11-23 19:05  
**문의**: Phase 1 테스트 결과 공유 시 스크린샷 첨부
