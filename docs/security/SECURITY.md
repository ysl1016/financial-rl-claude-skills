# Security Guide - API Key Management

## 🔒 보안 개요

이 프로젝트는 Anthropic Claude API를 사용하며, API 키는 **절대로 코드에 하드코딩하거나 Git에 커밋해서는 안 됩니다.**

---

## ⚠️ 중요 보안 규칙

### ❌ 절대 하지 말아야 할 것

```python
# ❌ 나쁜 예: 코드에 API 키 직접 입력
api_key = "sk-ant-api03-vmRxnY3JzR3IW1nPMwXI..."  # 절대 금지!

# ❌ 나쁜 예: 주석에 API 키
# My API key: sk-ant-api03-vmRxnY3JzR3IW1nPMwXI...  # 절대 금지!
```

### ✅ 올바른 방법

```python
# ✅ 좋은 예: 환경 변수 사용
import os
api_key = os.environ.get("ANTHROPIC_API_KEY")

# ✅ 좋은 예: config 모듈 사용
from src.utils.config import get_anthropic_api_key
api_key = get_anthropic_api_key()
```

---

## 📁 파일 구조

### 보안 파일 계층

```
financial-rl-trading/
├── .env                    # ❌ Git에 커밋 금지 (실제 API 키 포함)
├── .env.example            # ✅ Git에 커밋 가능 (템플릿)
├── .gitignore              # ✅ .env를 무시하도록 설정됨
└── src/
    └── utils/
        └── config.py       # ✅ 안전하게 .env 로드
```

### 파일 설명

| 파일 | 목적 | Git 커밋 | 내용 |
|------|------|----------|------|
| `.env` | **실제 API 키 저장** | ❌ 절대 금지 | 실제 API 키와 설정 |
| `.env.example` | 템플릿 제공 | ✅ 가능 | 예시 값만 포함 |
| `.gitignore` | 보안 파일 제외 | ✅ 필수 | `.env` 제외 설정 |

---

## 🚀 초기 설정

### 1단계: .env 파일 생성

```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env
```

### 2단계: API 키 입력

`.env` 파일을 열고 실제 API 키 입력:

```bash
# .env 파일 편집
nano .env
# 또는
code .env
```

**변경 전:**
```env
ANTHROPIC_API_KEY=your-api-key-here
```

**변경 후:**
```env
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-ACTUAL-API-KEY-HERE
```

### 3단계: 파일 권한 설정 (Linux/Mac)

```bash
# .env 파일 권한을 소유자만 읽기/쓰기로 제한
chmod 600 .env

# 확인
ls -l .env
# 출력: -rw------- 1 user user 1234 Oct 18 23:59 .env
```

### 4단계: Git 상태 확인

```bash
# .env가 추적되지 않는지 확인
git status

# 출력에 .env가 없어야 함 (있으면 .gitignore 확인)
```

---

## 🔍 API 키 검증

### 테스트 스크립트

```bash
# 설정 검증
python -c "from src.utils.config import validate_config; validate_config()"
```

**성공 시 출력:**
```
✓ Configuration validated successfully
  API Key: sk-ant-api...kQBg
  Claude Model: claude-3-5-sonnet-20241022
  Device: cuda
```

**실패 시 출력:**
```
❌ ANTHROPIC_API_KEY is not set or using default value
   Please set your API key in .env file
```

---

## 🛡️ .gitignore 설정

프로젝트의 `.gitignore` 파일에 다음이 포함되어 있는지 확인:

```gitignore
# Security - Never commit these
.env
.env.local
.env.*.local
*.key
*.pem
credentials.json
api_keys.txt
**/ANTHROPIC_API_KEY*
```

### .gitignore 테스트

```bash
# .env 파일이 무시되는지 확인
git check-ignore .env

# 출력: .env (무시됨을 의미)
```

---

## 🚨 보안 사고 대응

### API 키가 실수로 커밋된 경우

**즉시 수행:**

1. **API 키 즉시 폐기**
   ```
   https://console.anthropic.com/settings/keys
   → 해당 키 삭제
   → 새 키 생성
   ```

2. **Git 히스토리에서 제거**
   ```bash
   # BFG Repo-Cleaner 사용 (권장)
   brew install bfg  # macOS
   # 또는
   apt-get install bfg  # Linux

   bfg --replace-text passwords.txt
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   ```

3. **.env 파일 업데이트**
   ```bash
   # .env에 새 API 키 입력
   nano .env
   ```

4. **강제 푸시 (주의!)**
   ```bash
   git push --force
   ```

### 키 노출 확인

```bash
# Git 히스토리에서 API 키 검색
git log -S "sk-ant-api" --all

# 파일 내용에서 API 키 검색
grep -r "sk-ant-api" . --exclude-dir=.git
```

---

## 🔧 환경별 설정

### 개발 환경

```bash
# 로컬 개발
.env  # 로컬 API 키
```

### 프로덕션 환경

```bash
# 서버 환경 변수 설정 (Docker)
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY ...

# 또는 Kubernetes Secret
kubectl create secret generic api-keys \
  --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY
```

### CI/CD (GitHub Actions)

```yaml
# .github/workflows/test.yml
- name: Run tests
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: pytest
```

**GitHub Secrets 설정:**
```
Settings → Secrets → Actions → New repository secret
Name: ANTHROPIC_API_KEY
Value: sk-ant-api03-...
```

---

## 📊 비용 모니터링

### API 사용량 확인

```python
# 사용량 추적 (커스텀 로깅)
import logging

logger = logging.getLogger(__name__)

def track_api_call(tokens_used):
    logger.info(f"Claude API called: {tokens_used} tokens")
    # 비용 계산 (Sonnet 기준: $3/million tokens)
    cost = (tokens_used / 1_000_000) * 3
    logger.info(f"Estimated cost: ${cost:.4f}")
```

### 사용량 제한 설정

```python
# config.py에 추가
MAX_DAILY_API_CALLS = 1000
MAX_DAILY_COST = 10.0  # USD

def check_api_budget():
    # 일일 사용량 확인 로직
    pass
```

---

## 🔐 추가 보안 조치

### 1. 로그 파일 보안

```python
# 로그에 API 키 노출 방지
import logging

class SensitiveDataFilter(logging.Filter):
    def filter(self, record):
        # API 키 마스킹
        if hasattr(record, 'msg'):
            record.msg = record.msg.replace(
                os.environ.get('ANTHROPIC_API_KEY', ''),
                'sk-ant-***'
            )
        return True

logger.addFilter(SensitiveDataFilter())
```

### 2. 환경 변수 마스킹

```python
# config.py의 print_config_summary()에서 자동 마스킹
api_key = config.get('ANTHROPIC_API_KEY', '(not set)')
print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
# 출력: API Key: sk-ant-api...kQBg
```

### 3. 팀 협업 시

```bash
# 각 팀원은 자신의 .env 파일 생성
cp .env.example .env
# 각자의 API 키 입력

# .env는 절대 공유하지 않음
# 대신 .env.example을 업데이트하여 공유
```

---

## ✅ 보안 체크리스트

배포 전 확인사항:

- [ ] `.env` 파일이 `.gitignore`에 포함되어 있는가?
- [ ] Git 히스토리에 API 키가 없는가?
- [ ] 프로덕션 환경 변수가 안전하게 설정되어 있는가?
- [ ] 로그 파일에 API 키가 노출되지 않는가?
- [ ] API 키 권한이 최소 권한으로 설정되어 있는가?
- [ ] 사용량 모니터링이 설정되어 있는가?
- [ ] 팀원들이 보안 가이드를 숙지했는가?

---

## 📞 문제 발생 시

### 도움이 필요한 경우

1. **API 키 분실**
   - https://console.anthropic.com/settings/keys
   - 새 키 생성

2. **.env 파일이 Git에 커밋됨**
   - 즉시 API 키 폐기
   - Git 히스토리 정리
   - 새 키로 재설정

3. **권한 오류**
   ```bash
   chmod 600 .env
   ```

4. **설정 로드 실패**
   ```bash
   # config 모듈 테스트
   python src/utils/config.py
   ```

---

## 📚 추가 자료

- [Anthropic API 보안 가이드](https://docs.anthropic.com/claude/reference/security)
- [환경 변수 모범 사례](https://12factor.net/config)
- [Git 비밀 관리](https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage)

---

**⚠️ 기억하세요:**
- API 키는 **비밀번호와 같습니다**
- 절대 코드에 하드코딩하지 마세요
- 절대 Git에 커밋하지 마세요
- 의심스러우면 즉시 키를 재발급하세요

**🔒 안전한 개발을 위해 이 가이드를 항상 따라주세요!**
