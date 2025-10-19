# 프로젝트 구조 정리 계획

**작성일**: 2025년 10월 19일
**목적**: 프로젝트 파일 및 모듈의 체계적 재구성

---

## 현재 문제점

### 1. 중복된 테스트 디렉토리
```
tests/              # 루트 레벨 테스트 (중복)
src/tests/          # src 내부 테스트 (중복)
```

### 2. 루트 디렉토리에 흩어진 파일들
```
/generate_investment_report.py     # 스크립트
/test_*.py (5개 파일)               # 테스트 스크립트들
/*.md (10개 파일)                   # 문서들
```

### 3. 문서 파일 분산
```
/CLAUDE_API_TEST_REPORT.md
/CLAUDE_SKILLS_INTEGRATION.md
/INVESTMENT_REPORT_SUMMARY.md
/MODEL_MIGRATION_REPORT.md
/QUICKSTART.md
/SETUP_VERIFICATION.md
/SKILLS_UPDATE_SUMMARY.md
/SPY_TEST_REPORT.md
/TESTING_GUIDE.md
/작업보고서_Claude_Skills_통합.md
docs/CLAUDE_INTEGRATION_GUIDE.md
docs/api_documentation.md
```

### 4. 모니터링 디렉토리 중복
```
/monitoring/                # 루트 레벨 (설정 파일)
src/monitoring/             # 소스 코드
```

---

## 정리된 구조 (제안)

```
financial-rl-claude-skills/
├── .claude/
│   └── skills/
│       └── trading-analysis/
│
├── docs/                           # 📚 모든 문서 통합
│   ├── guides/                     # 가이드 문서
│   │   ├── QUICKSTART.md
│   │   ├── CLAUDE_INTEGRATION_GUIDE.md
│   │   ├── CLAUDE_SKILLS_INTEGRATION.md
│   │   └── TESTING_GUIDE.md
│   ├── reports/                    # 테스트/작업 보고서
│   │   ├── CLAUDE_API_TEST_REPORT.md
│   │   ├── INVESTMENT_REPORT_SUMMARY.md
│   │   ├── MODEL_MIGRATION_REPORT.md
│   │   ├── SETUP_VERIFICATION.md
│   │   ├── SKILLS_UPDATE_SUMMARY.md
│   │   ├── SPY_TEST_REPORT.md
│   │   └── 작업보고서_Claude_Skills_통합.md
│   ├── architecture/               # 아키텍처 문서
│   │   ├── DeepSeek-R1_Financial_Trading_Model_Architecture.md
│   │   └── PROJECT_STRUCTURE.md (새로 작성)
│   ├── api/                        # API 문서
│   │   └── api_documentation.md
│   └── security/                   # 보안 문서
│       └── SECURITY.md
│
├── scripts/                        # 🔧 실행 스크립트 통합
│   ├── reports/                    # 보고서 생성
│   │   └── generate_investment_report.py
│   ├── tests/                      # 테스트 스크립트
│   │   ├── test_claude_simple.py
│   │   ├── test_claude_integration_full.py
│   │   ├── test_model_versions.py
│   │   ├── test_spy_data.py
│   │   └── test_spy_data_no_api.py
│   └── utils/                      # 유틸리티 스크립트
│
├── src/                            # 💻 소스 코드
│   ├── api/
│   ├── claude_integration/
│   ├── data/
│   ├── deployment/
│   ├── models/
│   ├── monitoring/
│   ├── reporting/
│   └── utils/
│
├── tests/                          # ✅ 단위 테스트 (통합)
│   ├── unit/                       # 단위 테스트
│   │   ├── test_data_processor.py
│   │   ├── test_trading_env.py
│   │   ├── test_enhanced_trading_env.py
│   │   └── test_deepseek_grpo_agent.py
│   ├── integration/                # 통합 테스트
│   │   ├── test_api_prediction.py
│   │   └── test_enhanced_processor.py
│   └── run_tests.py
│
├── examples/                       # 📝 예제 코드 (변경 없음)
│
├── reports/                        # 📊 생성된 보고서 (변경 없음)
│
├── monitoring/                     # 📈 모니터링 설정
│   ├── grafana/
│   └── prometheus/
│
├── .env.example                    # 환경 설정 예제
├── .gitignore
├── requirements.txt
└── README.md                       # 메인 문서
```

---

## 이동할 파일 목록

### 1. 문서 파일 이동

#### docs/guides/ 로 이동
- `/QUICKSTART.md` → `docs/guides/QUICKSTART.md`
- `/CLAUDE_SKILLS_INTEGRATION.md` → `docs/guides/CLAUDE_SKILLS_INTEGRATION.md`
- `/TESTING_GUIDE.md` → `docs/guides/TESTING_GUIDE.md`

#### docs/reports/ 로 이동
- `/CLAUDE_API_TEST_REPORT.md` → `docs/reports/CLAUDE_API_TEST_REPORT.md`
- `/INVESTMENT_REPORT_SUMMARY.md` → `docs/reports/INVESTMENT_REPORT_SUMMARY.md`
- `/MODEL_MIGRATION_REPORT.md` → `docs/reports/MODEL_MIGRATION_REPORT.md`
- `/SETUP_VERIFICATION.md` → `docs/reports/SETUP_VERIFICATION.md`
- `/SKILLS_UPDATE_SUMMARY.md` → `docs/reports/SKILLS_UPDATE_SUMMARY.md`
- `/SPY_TEST_REPORT.md` → `docs/reports/SPY_TEST_REPORT.md`
- `/작업보고서_Claude_Skills_통합.md` → `docs/reports/작업보고서_Claude_Skills_통합.md`

#### docs/architecture/ 로 이동
- `docs/DeepSeek-R1_Financial_Trading_Model_Architecture.md` → `docs/architecture/DeepSeek-R1_Financial_Trading_Model_Architecture.md`

#### docs/api/ 로 이동
- `docs/api_documentation.md` → `docs/api/api_documentation.md`

#### docs/security/ 로 이동
- `docs/SECURITY.md` → `docs/security/SECURITY.md`

#### docs/guides/ 로 이동 (기존)
- `docs/CLAUDE_INTEGRATION_GUIDE.md` → `docs/guides/CLAUDE_INTEGRATION_GUIDE.md`

### 2. 스크립트 파일 이동

#### scripts/reports/ 로 이동
- `/generate_investment_report.py` → `scripts/reports/generate_investment_report.py`

#### scripts/tests/ 로 이동
- `/test_claude_simple.py` → `scripts/tests/test_claude_simple.py`
- `/test_claude_integration_full.py` → `scripts/tests/test_claude_integration_full.py`
- `/test_model_versions.py` → `scripts/tests/test_model_versions.py`
- `/test_spy_data.py` → `scripts/tests/test_spy_data.py`
- `/test_spy_data_no_api.py` → `scripts/tests/test_spy_data_no_api.py`

### 3. 테스트 파일 통합

#### tests/unit/ 로 이동
- `tests/test_data_processor.py` → `tests/unit/test_data_processor.py`
- `tests/test_trading_env.py` → `tests/unit/test_trading_env.py`
- `tests/test_enhanced_trading_env.py` → `tests/unit/test_enhanced_trading_env.py`
- `tests/test_deepseek_grpo_agent.py` → `tests/unit/test_deepseek_grpo_agent.py`

#### tests/integration/ 로 이동
- `tests/test_api_prediction.py` → `tests/integration/test_api_prediction.py`
- `tests/test_enhanced_processor.py` → `tests/integration/test_enhanced_processor.py`

#### src/tests/ 제거
- `src/tests/` 내용을 `tests/`로 통합 후 디렉토리 제거

---

## 업데이트 필요한 파일

### 1. Import 경로 변경
- `.claude/skills/trading-analysis/scripts/generate_report.py`
  - `generate_investment_report` 경로 업데이트

### 2. 문서 링크 업데이트
- `README.md` - 모든 문서 링크 업데이트
- 각 가이드 문서 내 상호 참조 링크 업데이트

### 3. 설정 파일 업데이트
- 테스트 실행 스크립트 경로 업데이트

---

## 삭제할 파일/디렉토리

1. `src/tests/` - tests/로 통합 후 삭제
2. 중복 `tests/__init__.py` 정리

---

## 장점

### 1. 명확한 구조
- **docs/**: 모든 문서 한 곳에
- **scripts/**: 실행 스크립트 분리
- **tests/**: 테스트 코드 통합
- **src/**: 순수 소스 코드만

### 2. 찾기 쉬움
- 가이드 찾기: `docs/guides/`
- 보고서 찾기: `docs/reports/`
- 테스트 스크립트: `scripts/tests/`
- 단위 테스트: `tests/unit/`

### 3. 유지보수 용이
- 문서 관리 일원화
- 테스트 구조 명확화
- 루트 디렉토리 정리

### 4. 전문성
- 일반적인 Python 프로젝트 구조 준수
- 명확한 책임 분리
- 확장 가능한 구조

---

## 구현 순서

1. ✅ 새 디렉토리 생성
2. ✅ 파일 이동
3. ✅ Import 경로 업데이트
4. ✅ 문서 링크 업데이트
5. ✅ 테스트 실행하여 검증
6. ✅ 불필요한 디렉토리 삭제
7. ✅ 최종 문서 작성

---

**작성자**: Claude AI Assistant
**승인 대기 중**
