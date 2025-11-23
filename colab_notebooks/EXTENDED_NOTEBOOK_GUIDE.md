# Extended Notebook Usage Guide

**파일**: `phase1_state_dependent_test.ipynb`  
**업데이트**: 2025-11-23 19:31

---

## 📋 노트북 구조 (확장됨)

### Part 1: Validation Tests (기존)
1. Title & Overview
2. Mount Drive
3. Install Dependencies
4. StateDependentRewardEnv
5. Create Test Data
6. Test 1: Reward Differentiation
7. Test 2: Action Balance
8. Visualization
9. Summary

### Part 2: 100 Episodes Training (신규)
10. Part 2 Title
11. Download Real Data (SPY)
12. CleanGRPOAgent
13. Training (100 episodes)
14. Test Evaluation
15. Training Visualization
16. Save Model

**총 16개 셀** (기존 9개 + 신규 7개)

---

## 🚀 실행 방법

### 현재 런타임에서 계속 실행

**이미 Part 1 완료한 경우**:
1. Cell 10부터 실행 (Part 2 Title)
2. Shift+Enter로 순차 실행
3. 또는 Runtime → Run after (Cell 10 선택)

**처음부터 실행**:
1. Runtime → Run all
2. 전체 소요 시간: 약 25-30분
   - Part 1: 2분
   - Part 2: 20-25분

---

## 📊 예상 결과

### Part 1 (Validation)
```
✅ Buy rewards differentiated
✅ Sell rewards differentiated
✅ Action balance verified
```

### Part 2 (Training)
```
Episode 20/100:
  Portfolio: $102,500
  Alpha: +1.2%
  Trades: 12.3

Episode 100/100:
  Portfolio: $105,000
  Alpha: +2.5%
  Trades: 15.7

Test Results:
  Alpha: +2.30%
  Sharpe: 0.85
  Trades: 14
  
Success: 4/4 criteria passed
🎉 SUCCESS!
```

---

## 💾 저장 파일

**Drive 위치**:
```
/content/drive/MyDrive/financial-rl-trading/
├── phase1_results/
│   ├── reward_structure.png (Part 1)
│   └── training_results.png (Part 2)
└── models/
    └── state_dependent_100ep.pt (Part 2)
```

---

## ⚡ 빠른 실행 가이드

**현재 런타임에서 Part 2만 실행**:

```python
# Cell 10: Part 2 Title (Shift+Enter)
# Cell 11: Download Data (Shift+Enter) - 30초
# Cell 12: Agent (Shift+Enter) - 5초
# Cell 13: Training (Shift+Enter) - 15-20분
# Cell 14: Test (Shift+Enter) - 10초
# Cell 15: Viz (Shift+Enter) - 10초
# Cell 16: Save (Shift+Enter) - 5초
```

**총 소요**: 약 20분

---

## 🎯 성공 기준

| 지표 | 목표 | 이전 |
|------|------|------|
| Alpha | > 0% | -12.37% |
| Trades | > 10 | 0 |
| Buy % | 30-70% | 0% |
| Sell % | > 0% | 100% |

---

**작성일**: 2025-11-23 19:32  
**다음**: Part 2 실행 후 결과 공유
