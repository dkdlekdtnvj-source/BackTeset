# PostgreSQL 스토리지 전환 가이드

이 문서는 SQLite 잠금 오류를 줄이고 Optuna 병렬 최적화를 안정적으로 실행하기 위해 PostgreSQL 기반 RDB 스토리지로 전환하는 절차를 정리한 것입니다. 모든 명령은 리포지토리 루트(`/workspace/BackTeset`)에서 수행한다고 가정합니다.

## 1. 왜 PostgreSQL인가?

- SQLite는 파일 잠금 기반이어서 `database is locked` 오류가 자주 발생합니다. 특히 `search.n_jobs` 또는 `search.dataset_jobs`를 2 이상으로 두고 병렬 실행할 때 충돌이 두드러집니다.
- PostgreSQL은 다중 프로세스 환경에서 검증된 트랜잭션 격리와 커넥션 풀을 제공하므로 Optuna가 동시에 수백 개의 트라이얼을 분배해도 안전하게 동작합니다.
- 이번 업데이트에서 `config/params.yaml`에 PostgreSQL 풀 및 타임아웃 설정 항목을 추가했고, `optimize/run.py`는 URL이 `postgresql://` 또는 `postgresql+psycopg://`로 시작하면 자동으로 RDB 모드를 사용합니다.

## 2. 환경 준비

### 2.1 패키지 설치

```bash
pip install -r requirements.txt  # psycopg2-binary가 함께 설치됩니다.
```

### 2.2 PostgreSQL 실행 (Docker 예시)

```bash
docker run -d \
  --name optuna-postgres \
  -e POSTGRES_USER=optuna \
  -e POSTGRES_PASSWORD=optuna \
  -e POSTGRES_DB=optuna \
  -p 5432:5432 \
  postgres:15
```

컨테이너가 올라가면 psql로 접속해 권한을 확인합니다.

```bash
psql postgresql://optuna:optuna@localhost:5432/optuna -c "SELECT current_database(), current_user;"
```

## 3. 프로젝트 설정 변경

1. `.env` 또는 쉘 프로필에 접속 URL을 등록합니다.
   ```bash
   export OPTUNA_STORAGE_URL="postgresql+psycopg://optuna:optuna@localhost:5432/optuna"
   ```
2. `config/params.yaml`의 주요 항목:
   - `search.storage_url_env`: 기본값이 `OPTUNA_STORAGE_URL`이므로 추가 수정 없이 위 환경 변수를 읽어옵니다.
   - `search.storage_pool_size`, `storage_max_overflow`, `storage_pool_timeout`, `storage_pool_recycle`: 커넥션 풀 크기와 재생성 주기를 제어합니다. 기본값(8/16/30/1800초)을 유지해도 무방합니다.
   - `search.storage_connect_timeout`: DB 연결 시도 제한(초). 네트워크 지연이 긴 환경이면 값을 늘리세요.
   - `search.storage_statement_timeout_ms`: PostgreSQL의 `statement_timeout`을 밀리초 단위로 지정합니다. 기본 300000(5분)이며, 장기 실행이 필요하면 0으로 비워 비활성화할 수 있습니다.
   - `search.storage_isolation_level`: Optuna 트랜잭션 격리 수준. 기본은 `READ COMMITTED`로 설정되어 있습니다.

환경 변수를 비워 두면 코드가 자동으로 SQLite(`studies/<이름>.db`)로 되돌아갑니다.

## 4. 실행 명령 예시

```bash
python -m optimize.run \
  --config config/params.yaml \
  --symbol BINANCE:BTCUSDT \
  --timeframe-grid "1m@15m,1m@1h" \
  --n-trials 500 \
  --storage-url-env OPTUNA_STORAGE_URL
```

또는 URL을 직접 지정할 수도 있습니다.

```bash
python -m optimize.run \
  --config config/params.yaml \
  --storage-url "postgresql+psycopg://optuna:optuna@localhost:5432/optuna"
```

## 5. 기존 SQLite 스터디 마이그레이션 (선택)

Optuna 3.x는 `optuna storage upgrade`/`optuna storage create-study` 명령을 제공합니다. 간단한 복제는 다음처럼 수행할 수 있습니다.

```bash
# (선택) 대상 스터디 이름 확인
optuna studies --storage sqlite:///studies/기존스터디.db

# SQLite → PostgreSQL 복제
optuna copy \
  --from sqlite:///studies/기존스터디.db \
  --to postgresql+psycopg://optuna:optuna@localhost:5432/optuna \
  --study 기존스터디이름
```

마이그레이션이 끝나면 `studies` 폴더의 SQLite 파일은 백업으로만 보관하고, 이후 실행부터는 PostgreSQL URL을 사용하세요.

## 6. 문제 해결 체크리스트

- **연결 실패**: 방화벽 또는 Docker 포트 매핑을 확인합니다. `psql`로 수동 접속이 되는지 먼저 점검하세요.
- **`password authentication failed`**: URL의 사용자/비밀번호를 다시 확인하세요.
- **`timeout expired`**: `storage_connect_timeout` 또는 `storage_statement_timeout_ms` 값을 늘리거나, PostgreSQL 설정에서 `statement_timeout`을 수정합니다.
- **병렬 최적화가 느릴 때**: `storage_pool_size`와 `storage_max_overflow`를 늘리고, PostgreSQL 서버의 `max_connections`도 함께 확장하세요.

이 가이드를 따르면 SQLite 잠금 문제 없이 안정적으로 Optuna 병렬 최적화를 수행할 수 있습니다.
