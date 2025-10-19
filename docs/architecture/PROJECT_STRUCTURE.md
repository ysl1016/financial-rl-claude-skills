# 프로젝트 구조 가이드

**버전**: 2.0.0
**작성일**: 2025년 10월 19일
**상태**: ✅ 구조 정리 완료

---

## 개요

이 문서는 `financial-rl-claude-skills` 프로젝트의 디렉토리 구조와 각 파일/모듈의 역할을 설명합니다.

## 전체 구조

```
financial-rl-claude-skills/
├── .claude/                    # Claude Code 통합
├── docs/                       # 📚 문서 (체계적 분류)
├── scripts/                    # 🔧 실행 스크립트
├── src/                        # 💻 소스 코드
├── tests/                      # ✅ 테스트
├── examples/                   # 📝 예제
├── reports/                    # 📊 생성된 보고서
├── monitoring/                 # 📈 모니터링 설정
└── venv/                       # 가상 환경
```

---

## 디렉토리 상세 설명

### 1. `.claude/` - Claude Skills 통합

```
.claude/
└── skills/
    └── trading-analysis/
        ├── SKILL.md              # Skill 정의 (Claude Code 자동 발견)
        ├── reference.md          # 기술 참조 문서
        ├── examples.md           # 사용 예시
        ├── scripts/
        │   └── generate_report.py  # Skills 래퍼 스크립트
        └── templates/            # 향후 템플릿
```

**역할**:
- Claude Code의 Skills 기능 지원
- 자연어로 투자 보고서 생성 가능
- 기존 Python 모듈을 래핑

**주요 파일**:
- `SKILL.md`: Claude가 자동으로 발견하는 Skill 정의
- `scripts/generate_report.py`: `scripts/reports/generate_investment_report.py` 호출

---

### 2. `docs/` - 문서 (체계적 분류)

```
docs/
├── guides/                     # 사용자 가이드
│   ├── QUICKSTART.md
│   ├── CLAUDE_INTEGRATION_GUIDE.md
│   ├── CLAUDE_SKILLS_INTEGRATION.md
│   └── TESTING_GUIDE.md
├── reports/                    # 기술 보고서
│   ├── CLAUDE_API_TEST_REPORT.md
│   ├── INVESTMENT_REPORT_SUMMARY.md
│   ├── MODEL_MIGRATION_REPORT.md
│   ├── SETUP_VERIFICATION.md
│   ├── SKILLS_UPDATE_SUMMARY.md
│   ├── SPY_TEST_REPORT.md
│   └── 작업보고서_Claude_Skills_통합.md
├── architecture/               # 아키텍처 문서
│   ├── DeepSeek-R1_Financial_Trading_Model_Architecture.md
│   └── PROJECT_STRUCTURE.md (현재 문서)
├── api/                        # API 문서
│   └── api_documentation.md
└── security/                   # 보안 가이드라인
    └── SECURITY.md
```

**역할**:
- 모든 문서를 목적별로 분류
- 찾기 쉬운 구조
- 유지보수 용이

**카테고리**:
1. **guides/**: 사용자 가이드 (시작, 통합, 테스트)
2. **reports/**: 기술 보고서 및 작업 보고서
3. **architecture/**: 시스템 아키텍처 문서
4. **api/**: API 참조 문서
5. **security/**: 보안 관련 문서

---

### 3. `scripts/` - 실행 스크립트

```
scripts/
├── reports/                    # 보고서 생성 스크립트
│   └── generate_investment_report.py
├── tests/                      # 테스트 스크립트
│   ├── test_claude_simple.py
│   ├── test_claude_integration_full.py
│   ├── test_model_versions.py
│   ├── test_spy_data.py
│   └── test_spy_data_no_api.py
└── utils/                      # 유틸리티 스크립트 (향후)
```

**역할**:
- 명령줄에서 직접 실행할 수 있는 스크립트
- 소스 코드(`src/`)와 분리
- 사용 목적별로 분류

**실행 예시**:
```bash
# 보고서 생성
python3 scripts/reports/generate_investment_report.py --symbol SPY

# Claude API 테스트
python3 scripts/tests/test_claude_simple.py
```

---

### 4. `src/` - 소스 코드

```
src/
├── api/                        # REST API 서버
│   ├── __init__.py
│   └── app.py
├── claude_integration/         # Claude AI 통합
│   ├── __init__.py
│   ├── claude_analyzer.py      # 시장 분석기
│   ├── hybrid_agent.py         # RL + Claude 하이브리드
│   ├── risk_assessor.py        # 리스크 평가기
│   ├── regime_interpreter.py   # 레짐 해석기
│   └── README.md
├── data/                       # 데이터 처리
│   ├── __init__.py
│   ├── data_processor.py       # 기본 데이터 처리
│   ├── advanced_normalizer.py  # 정규화
│   └── macro_sentiment.py      # 매크로 감성 분석
├── deployment/                 # 모델 배포
│   ├── __init__.py
│   ├── model_optimization.py
│   └── model_packaging.py
├── models/                     # RL 모델 및 환경
│   ├── __init__.py
│   ├── trading_env.py          # 기본 트레이딩 환경 (Gym)
│   ├── enhanced_trading_env.py # 향상된 환경
│   ├── multi_asset_env.py      # 다중 자산 환경
│   ├── grpo_agent.py           # GRPO 에이전트
│   ├── deepseek_grpo_agent.py  # DeepSeek-R1 GRPO
│   ├── deepseek_transformer.py # DeepSeek 트랜스포머
│   ├── deepseek_trading_model.py
│   └── hybrid_temporal_model.py
├── monitoring/                 # 성능 모니터링
│   ├── __init__.py
│   ├── performance_tracker.py
│   ├── anomaly_detection.py
│   └── alerting.py
├── reporting/                  # 보고서 생성
│   ├── __init__.py
│   ├── report_generator.py     # 마크다운 보고서 생성
│   └── chart_generator.py      # 차트 생성
└── utils/                      # 유틸리티
    ├── __init__.py
    ├── config.py               # 설정 관리
    ├── indicators.py           # 기술적 지표
    ├── advanced_indicators.py
    ├── evaluation.py           # 성능 평가
    ├── backtest_utils.py       # 백테스팅
    ├── benchmarking.py         # 벤치마킹
    ├── hyperparameter_optimization.py
    ├── feature_selection.py
    ├── reward_functions.py
    ├── lr_scheduler.py
    ├── online_learning.py
    ├── distributed_utils.py
    └── visualization.py
```

**역할**:
- 순수 소스 코드만 포함
- 재사용 가능한 모듈 및 클래스
- 명확한 책임 분리

**주요 모듈**:
- `api/`: Flask 기반 REST API
- `claude_integration/`: Claude AI 통합 (시장 분석, 하이브리드 에이전트)
- `data/`: 데이터 다운로드 및 전처리
- `models/`: RL 에이전트 및 트레이딩 환경
- `reporting/`: 투자 보고서 및 차트 생성
- `utils/`: 공통 유틸리티 함수

---

### 5. `tests/` - 테스트

```
tests/
├── unit/                       # 단위 테스트
│   ├── test_data_processor.py
│   ├── test_trading_env.py
│   ├── test_trading_env_src.py
│   ├── test_enhanced_trading_env.py
│   ├── test_deepseek_grpo_agent.py
│   └── test_grpo_agent.py
├── integration/                # 통합 테스트
│   ├── test_api_prediction.py
│   ├── test_enhanced_processor.py
│   ├── test_integration.py
│   └── test_regression.py
├── __init__.py
└── run_tests.py                # 테스트 실행 스크립트
```

**역할**:
- 모든 테스트 코드 통합 관리
- 단위 테스트와 통합 테스트 분리
- 자동화된 테스트 실행

**실행 예시**:
```bash
# 모든 테스트 실행
python tests/run_tests.py --type all

# 단위 테스트만
python tests/run_tests.py --type unit

# 특정 테스트
python -m pytest tests/unit/test_grpo_agent.py
```

---

### 6. `examples/` - 예제 스크립트

```
examples/
├── trading_example.py          # 기본 트레이딩 예제
├── train_grpo.py               # GRPO 학습
├── train_deepseek_grpo.py      # DeepSeek-R1 학습
├── backtest_deepseek_grpo.py   # 백테스팅
├── hybrid_claude_trading.py    # 하이브리드 모델
├── optimize_and_benchmark.py   # 최적화 및 벤치마크
├── integration_example.py      # 통합 예제
├── multi_asset_example.py      # 다중 자산 예제
├── quick_demo_yahoo_finance.py # 빠른 데모
├── run_api_server.py           # API 서버 실행
└── streaming_server.py         # 스트리밍 서버
```

**역할**:
- 실제 사용 예시 제공
- 학습용 코드
- 빠른 데모 및 프로토타이핑

---

### 7. `reports/` - 생성된 보고서

```
reports/
├── SPY_analysis_report_*.md    # 마크다운 보고서
├── SPY_analysis_report_*_data.json  # JSON 데이터
└── charts/                     # 차트 이미지
    ├── SPY_price_chart.png
    ├── SPY_indicators_chart.png
    ├── SPY_volatility_chart.png
    └── SPY_summary_dashboard.png
```

**역할**:
- `scripts/reports/generate_investment_report.py`가 생성한 파일 저장
- 마크다운 보고서 + JSON 데이터 + PNG 차트

---

### 8. `monitoring/` - 모니터링 설정

```
monitoring/
├── grafana/                    # Grafana 대시보드 설정
└── prometheus/                 # Prometheus 설정
```

**역할**:
- 프로덕션 모니터링 설정
- 성능 추적 및 알림

---

## 주요 파일 설명

### 루트 디렉토리 파일

| 파일 | 설명 |
|------|------|
| `README.md` | 프로젝트 메인 문서 |
| `requirements.txt` | Python 의존성 |
| `.env.example` | 환경 변수 예제 |
| `.gitignore` | Git 무시 파일 목록 |
| `PROJECT_REORGANIZATION_PLAN.md` | 프로젝트 정리 계획 |

---

## 주요 사용 시나리오

### 시나리오 1: 투자 보고서 생성

**자연어 (Claude Skills)**:
```
"SPY 투자 보고서를 생성해줘"
```

**Python 스크립트**:
```bash
python3 scripts/reports/generate_investment_report.py --symbol SPY
```

**Python 코드**:
```python
import sys
sys.path.insert(0, 'scripts/reports')
from generate_investment_report import generate_complete_report

generate_complete_report(symbol='SPY')
```

### 시나리오 2: RL 모델 학습

```bash
# GRPO 에이전트 학습
python examples/train_grpo.py

# DeepSeek-R1 에이전트 학습
python examples/train_deepseek_grpo.py
```

### 시나리오 3: 하이브리드 모델 사용

```bash
# RL + Claude 하이브리드 트레이딩
python examples/hybrid_claude_trading.py --symbol SPY --mode sequential
```

### 시나리오 4: 테스트 실행

```bash
# 모든 테스트
python tests/run_tests.py --type all

# 단위 테스트만
python tests/run_tests.py --type unit
```

---

## 파일 이동 내역

### 정리 전 → 정리 후

#### 문서 파일
- `/QUICKSTART.md` → `docs/guides/QUICKSTART.md`
- `/CLAUDE_SKILLS_INTEGRATION.md` → `docs/guides/CLAUDE_SKILLS_INTEGRATION.md`
- `/TESTING_GUIDE.md` → `docs/guides/TESTING_GUIDE.md`
- `/CLAUDE_API_TEST_REPORT.md` → `docs/reports/CLAUDE_API_TEST_REPORT.md`
- `/INVESTMENT_REPORT_SUMMARY.md` → `docs/reports/INVESTMENT_REPORT_SUMMARY.md`
- 기타 보고서 → `docs/reports/`

#### 스크립트 파일
- `/generate_investment_report.py` → `scripts/reports/generate_investment_report.py`
- `/test_*.py` (5개) → `scripts/tests/test_*.py`

#### 테스트 파일
- `tests/test_*.py` → `tests/unit/` 또는 `tests/integration/`
- `src/tests/test_*.py` → `tests/unit/` 또는 `tests/integration/`

---

## 구조 개선 효과

### Before (정리 전)
```
❌ 루트에 파일 20+ 개 흩어짐
❌ 문서 분산 (루트 + docs/)
❌ 테스트 중복 (tests/ + src/tests/)
❌ 스크립트와 소스 코드 혼재
```

### After (정리 후)
```
✅ 루트 깔끔 (5개 파일만)
✅ 문서 체계화 (docs/ 하위 분류)
✅ 테스트 통합 (tests/ 단일화)
✅ 명확한 책임 분리
```

### 장점
1. **찾기 쉬움**: 파일 위치 예측 가능
2. **유지보수 용이**: 명확한 구조
3. **전문성**: 표준 Python 프로젝트 구조
4. **확장 가능**: 새 파일 추가 시 위치 명확

---

## 개발자 가이드

### 새 모듈 추가 시

**규칙**:
- **소스 코드** → `src/` 하위 적절한 디렉토리
- **실행 스크립트** → `scripts/` 하위 분류
- **테스트** → `tests/unit/` 또는 `tests/integration/`
- **문서** → `docs/` 하위 적절한 분류
- **예제** → `examples/`

**예시**:
```
새 기능: PDF 보고서 생성

src/reporting/pdf_generator.py       # 소스 코드
scripts/reports/generate_pdf_report.py  # 실행 스크립트
tests/unit/test_pdf_generator.py     # 단위 테스트
docs/guides/PDF_EXPORT_GUIDE.md      # 가이드
examples/pdf_export_example.py       # 예제
```

---

## 버전 히스토리

### v2.0.0 (2025-10-19) - 구조 대폭 정리
- ✅ docs/ 디렉토리 체계화 (5개 하위 분류)
- ✅ scripts/ 디렉토리 신설 (스크립트 분리)
- ✅ tests/ 통합 (src/tests 제거)
- ✅ 루트 디렉토리 정리 (파일 20+ → 5개)
- ✅ README 업데이트 (새 구조 반영)

### v1.0.0 (2025-10-18) - Claude Skills 통합
- ✅ .claude/skills/trading-analysis/ 추가
- ✅ src/reporting/ 모듈 추가
- ✅ Claude AI 통합

---

**작성자**: Claude AI Assistant
**최종 업데이트**: 2025년 10월 19일
**문서 버전**: 2.0.0
