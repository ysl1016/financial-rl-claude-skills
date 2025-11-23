# Google Colab 노트북 사용 가이드

이 디렉토리에는 Google Colab에서 DeepSeek GRPO 모델을 학습하기 위한 Jupyter 노트북이 포함되어 있습니다.

## 📓 노트북 목록

### 1. `colab_training.ipynb` - 메인 학습 노트북
Yahoo Finance 데이터를 사용하여 모델을 학습하는 완전한 워크플로우

**포함 내용:**
- Google Drive 마운트 및 환경 설정
- GPU 확인 및 설정
- Yahoo Finance 데이터 다운로드
- 기술적 지표 계산 (RSI, MACD, Bollinger Bands, ATR 등)
- 데이터 분할 (Train/Val/Test)
- 모델 학습 가이드
- 결과 시각화

## 🚀 빠른 시작

### 방법 1: Google Drive에서 직접 열기

1. 이 폴더의 `.ipynb` 파일을 Google Drive에 업로드
2. 파일을 더블클릭하여 Colab에서 열기
3. 런타임 > 런타임 유형 변경 > **T4 GPU** 선택
4. 모든 셀 실행 (런타임 > 모두 실행)

### 방법 2: GitHub에서 열기 (프로젝트를 GitHub에 업로드한 경우)

1. Colab 접속: https://colab.research.google.com/
2. File > Open notebook > GitHub 탭
3. 저장소 URL 입력
4. 노트북 선택

### 방법 3: 로컬에서 업로드

1. Colab 접속: https://colab.research.google.com/
2. File > Upload notebook
3. 이 폴더의 `.ipynb` 파일 선택

## ⚙️ 설정

### GPU 런타임 설정 (필수)

1. Colab 메뉴: **런타임** > **런타임 유형 변경**
2. **하드웨어 가속기**: T4 GPU 선택
3. 저장

**GPU 사용 시 예상 학습 시간:**
- 100 episodes: ~20-30분
- 500 episodes: ~1.5-2시간

**CPU 사용 시 (권장하지 않음):**
- 100 episodes: ~2-3시간
- 500 episodes: ~10-15시간

### 프로젝트 파일 준비

노트북에서 프로젝트의 전체 기능을 사용하려면 다음 중 하나를 수행하세요:

#### 옵션 A: GitHub 클론 (권장)
```python
!git clone https://github.com/[your-username]/financial-rl-claude-skills.git
%cd financial-rl-claude-skills
```

#### 옵션 B: ZIP 파일 업로드
1. 프로젝트 폴더를 압축 (financial-rl-claude-skills.zip)
2. Google Drive에 업로드
3. 노트북에서:
```python
!unzip /content/drive/MyDrive/financial-rl-claude-skills.zip
%cd financial-rl-claude-skills
```

## 📊 학습 프로세스

### 1. 데이터 수집
- Yahoo Finance API를 통해 실시간 주식 데이터 다운로드
- 기본 설정: SPY (S&P 500 ETF), 2020-01-01 ~ 현재
- 다중 종목 지원: `SYMBOLS = ['SPY', 'QQQ', 'AAPL']`

### 2. 기술적 지표 계산
자동으로 계산되는 지표:
- 이동평균 (SMA 20, 50, 200)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- 변동성 (Volatility)

### 3. 모델 학습
- **에이전트**: DeepSeek-R1 기반 GRPO (Actor-Critic)
- **환경**: 커스텀 트레이딩 환경 (Gym 기반)
- **보상**: Sharpe Ratio 기반
- **체크포인트**: 매 10 에피소드마다 Google Drive에 자동 저장

### 4. 결과 저장 위치

모든 결과는 Google Drive에 자동 저장됩니다:

```
/content/drive/MyDrive/financial-rl-trading/
├── models/
│   ├── checkpoints/
│   │   ├── grpo_episode_10.pt
│   │   ├── grpo_episode_20.pt
│   │   └── ...
│   └── best_model.pt
├── results/
│   ├── training_progress.png
│   ├── backtest_result.png
│   └── ...
└── data/
    └── cache/
```

## 🔧 커스터마이제이션

### 다른 종목으로 학습

```python
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']  # 원하는 종목으로 변경
```

### 학습 기간 조정

```python
NUM_EPISODES = 500  # 기본값: 100
SAVE_INTERVAL = 20  # 체크포인트 저장 간격
```

### 하이퍼파라미터 조정

```python
agent = SimpleGRPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=512,  # 기본값: 256
    lr=1e-4,         # 기본값: 3e-4
    device=device
)
```

## 📈 성과 평가

학습 후 자동으로 생성되는 시각화:
1. **Episode Rewards** - 에피소드별 보상 추이
2. **Episode Losses** - 학습 손실 추이
3. **Portfolio Value** - 최종 포트폴리오 가치
4. **Moving Average Reward** - 이동평균 보상

백테스트 결과:
- **Total Return** - 총 수익률
- **Sharpe Ratio** - 위험 대비 수익률
- **Max Drawdown** - 최대 손실
- **Strategy vs Buy & Hold** 비교 그래프

## ⚠️ 주의사항

1. **세션 타임아웃**
   - Colab 무료 버전: 최대 12시간 (90분 비활성 시 종료)
   - 정기적으로 체크포인트가 저장되므로 중단되어도 재개 가능

2. **메모리 제한**
   - Colab 무료: ~12GB RAM
   - 배치 크기를 조정하여 OOM 방지

3. **GPU 할당량**
   - Colab 무료: 제한적 GPU 시간
   - 장시간 학습 시 Colab Pro 권장

## 🆘 문제 해결

### GPU를 사용할 수 없습니다
→ 런타임 > 런타임 유형 변경 > T4 GPU 선택

### 패키지 설치 오류
→ 노트북 재시작 후 다시 실행

### 데이터 다운로드 실패
→ 인터넷 연결 확인, 다른 종목으로 시도

### 메모리 부족 (OOM)
→ `batch_size` 감소, `hidden_dim` 감소

## 📚 추가 리소스

- [Implementation Plan](../implementation_plan.md) - 전체 구현 계획
- [Quick Start Guide](../docs/guides/QUICKSTART.md) - 프로젝트 빠른 시작
- [Testing Guide](../docs/guides/TESTING_GUIDE.md) - 테스트 및 최적화

## 💡 다음 단계

1. **초기 실험**: SPY 100 episodes로 빠른 테스트
2. **성능 확인**: 백테스트 결과 검토
3. **스케일업**: 500-1000 episodes, 다중 종목
4. **최적화**: 하이퍼파라미터 튜닝
5. **프로덕션**: 최종 모델 선정 및 배포

---

**생성일**: 2025-11-22  
**버전**: 1.0  
**문의**: 프로젝트 README.md 참조
