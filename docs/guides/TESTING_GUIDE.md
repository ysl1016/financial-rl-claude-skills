# 테스팅 및 최적화 가이드

이 문서는 Financial RL Trading 프로젝트의 테스팅, 하이퍼파라미터 최적화, 및 성능 벤치마킹 도구 사용법을 설명합니다.

## 목차

1. [테스트 실행](#1-테스트-실행)
2. [하이퍼파라미터 최적화](#2-하이퍼파라미터-최적화)
3. [성능 벤치마킹](#3-성능-벤치마킹)
4. [통합 스크립트](#4-통합-스크립트)

## 1. 테스트 실행

### 1.1 필요한 패키지 설치

테스팅에 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

### 1.2 단위 테스트 실행

트레이딩 환경과 GRPO 에이전트에 대한 단위 테스트 실행:

```bash
python src/tests/run_tests.py --type unit
```

### 1.3 통합 테스트 실행

시스템 전체의 통합 테스트 실행:

```bash
python src/tests/run_tests.py --type integration
```

### 1.4 회귀 테스트 실행

성능 회귀 여부를 확인하는 테스트 실행:

```bash
python src/tests/run_tests.py --type regression
```

### 1.5 모든 테스트 실행

모든 유형의 테스트 일괄 실행:

```bash
python src/tests/run_tests.py --type all
```

조용한 모드로 실행하기:

```bash
python src/tests/run_tests.py --type all --quiet
```

## 2. 하이퍼파라미터 최적화

### 2.1 베이지안 최적화 실행

GRPO 에이전트의 하이퍼파라미터 최적화:

```bash
python -c "from src.utils.hyperparameter_optimization import run_hyperparameter_optimization; run_hyperparameter_optimization('path/to/data.csv', n_iter=30, validation_method='expanding')"
```

### 2.2 ValidationFramework 사용

데이터를 훈련/검증/테스트 세트로 분할:

```python
from src.utils.hyperparameter_optimization import ValidationFramework
import pandas as pd

# 데이터 로드
data = pd.read_csv('path/to/data.csv')

# 검증 프레임워크 생성
validation = ValidationFramework(
    data=data,
    validation_method='expanding',  # 'expanding', 'sliding', 'k_fold' 중 선택
    test_ratio=0.2,
    val_ratio=0.2,
    n_splits=5
)

# 분할 가져오기
splits = validation.get_splits()
train_data, val_data = splits[0]  # 첫 번째 분할
test_data = validation.get_test_data()
```

### 2.3 민감도 분석

하이퍼파라미터 민감도 분석:

```python
from src.utils.hyperparameter_optimization import SensitivityAnalysis

# 기준 파라미터 정의
base_params = {
    'learning_rate': 3e-4,
    'hidden_dim': 128,
    'gamma': 0.99,
    'reward_scale': 1.0,
    'penalty_scale': 0.5
}

# 민감도 분석 도구 생성
sensitivity = SensitivityAnalysis(
    train_data=train_data,
    val_data=val_data,
    base_params=base_params,
    log_dir='logs/sensitivity'
)

# 학습률 민감도 분석
lr_results = sensitivity.analyze_parameter(
    param_name='learning_rate',
    values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
)

# 전체 민감도 분석 실행
full_results = sensitivity.run_full_analysis()
```

## 3. 성능 벤치마킹

### 3.1 다양한 전략 벤치마킹

GRPO 에이전트와 기존 트레이딩 전략 비교:

```python
from src.utils.benchmarking import run_strategy_benchmark

# 벤치마크 실행
results = run_strategy_benchmark(
    data_path='path/to/data.csv',
    model_path='path/to/model.pt',
    state_dim=14,
    action_dim=3,
    log_dir='logs/benchmark',
    num_episodes=5,
    plot_results=True
)
```

### 3.2 커스텀 벤치마킹

더 많은 제어가 필요한 커스텀 벤치마킹:

```python
from src.utils.benchmarking import StrategyBenchmark
from src.models.grpo_agent import GRPOAgent
import pandas as pd

# 데이터 로드
data = pd.read_csv('path/to/data.csv')

# 벤치마크 도구 생성
benchmark = StrategyBenchmark(
    data=data,
    env_params={
        'initial_capital': 100000,
        'trading_cost': 0.0005,
        'slippage': 0.0001
    },
    log_dir='logs/benchmark'
)

# 표준 전략 추가
benchmark.create_standard_strategies()

# GRPO 에이전트 추가
agent = GRPOAgent(...)  # 에이전트 생성 또는 로드
benchmark.add_grpo_agent(agent, name='My GRPO Model')

# 벤치마크 실행
results = benchmark.run_all_benchmarks(num_episodes=5)

# 통계적 유의성 검정
significance = benchmark.statistical_significance_test(
    results=results,
    reference_strategy='Buy and Hold'
)

# 결과 시각화
benchmark.plot_results(
    results=results,
    figsize=(16, 16),
    plot_path='benchmark_results.png'
)
```

## 4. 통합 스크립트

하이퍼파라미터 최적화, 모델 학습, 및 벤치마킹을 한 번에 실행:

### 4.1 기본 사용법

```bash
python examples/optimize_and_benchmark.py --symbol SPY --optimize --train --benchmark
```

### 4.2 하이퍼파라미터 최적화만 실행

```bash
python examples/optimize_and_benchmark.py --symbol SPY --optimize
```

### 4.3 모델 학습만 실행

```bash
python examples/optimize_and_benchmark.py --symbol SPY --train --train_episodes 200
```

### 4.4 벤치마킹만 실행

```bash
python examples/optimize_and_benchmark.py --symbol SPY --benchmark --benchmark_episodes 10
```

### 4.5 전체 작업 플로우 예시

1. 데이터 다운로드 및 처리
2. 하이퍼파라미터 최적화 (30회 반복)
3. 최적 파라미터로 모델 학습 (100 에피소드)
4. 학습된 모델과 기준 전략 벤치마킹 (5회 반복)

```bash
python examples/optimize_and_benchmark.py \
  --symbol SPY \
  --start_date 2018-01-01 \
  --end_date 2022-12-31 \
  --optimize \
  --n_iter 30 \
  --train \
  --train_episodes 100 \
  --benchmark \
  --benchmark_episodes 5 \
  --seed 42 \
  --log_dir logs/SPY_full_test
```

### 4.6 매개변수 설명

자세한 매개변수 옵션 확인:

```bash
python examples/optimize_and_benchmark.py --help
```

주요 매개변수:
- `--symbol`: 주식 심볼 (예: SPY, AAPL)
- `--start_date`: 시작 날짜 (YYYY-MM-DD)
- `--end_date`: 종료 날짜 (YYYY-MM-DD)
- `--optimize`: 하이퍼파라미터 최적화 활성화
- `--n_iter`: 최적화 반복 횟수
- `--train`: 모델 학습 활성화
- `--train_episodes`: 학습 에피소드 수
- `--benchmark`: 벤치마킹 활성화
- `--benchmark_episodes`: 벤치마크 반복 횟수
- `--seed`: 랜덤 시드
- `--device`: 사용할 디바이스 ('cuda' 또는 'cpu')
- `--log_dir`: 로그 디렉토리 경로