# Optuna 병렬/대규모 실행 가이드

이 문서는 Pine 최적화 파이프라인을 Optuna로 대규모 실행할 때 고려해야 할 병렬화, 스토리지, 타임프레임 조합 분할 전략을 정리합니다. 모든 예시는 리포지토리 루트(`/workspace/strategy`)에서 실행한다고 가정합니다.

## 1. RDBStorage 설정으로 프로세스 병렬화 활성화

`n_jobs>1`은 파이썬 스레드 기반이어서 GIL 영향을 받습니다. CPU 바운드 최적화에서는 **프로세스/노드 병렬**이 필수이며, 이를 위해 Optuna 스터디를 외부 RDB에 저장해야 합니다. 2025년 9월 이후 버전부터는 개별 데이터셋 백테스트도 `search.dataset_jobs`와 `search.dataset_executor`(thread/process) 설정으로 병렬화할 수 있으니, 단일 트라이얼이 여러 기간·심볼을 동시에 평가할 때 적극 활용하세요.

1. 데이터베이스 준비 (예: PostgreSQL)
   ```bash
   createdb pine_optuna
   psql pine_optuna -c "CREATE USER pine_optuna WITH PASSWORD 'pine_optuna';"
   psql pine_optuna -c "GRANT ALL PRIVILEGES ON DATABASE pine_optuna TO pine_optuna;"
   ```
2. 접속 URL을 환경 변수로 등록합니다.
   ```bash
   export OPTUNA_STORAGE_URL="postgresql+psycopg://pine_optuna:pine_optuna@localhost:5432/pine_optuna"
   ```
3. `config/params.yaml`의 `search.storage_url_env` 키가 기본값으로 `OPTUNA_STORAGE_URL`을 바라보도록 구성되어 있으므로 추가 수정 없이 외부 DB가 사용됩니다. 환경 변수가 비어 있으면 자동으로 로컬 SQLite(`studies/<symbol>_<ltf>_<htf>.db`)로 돌아갑니다. 필요하면 CLI에서 `--storage-url-env MY_ENV`, `--storage-url postgresql+psycopg://...` 플래그를 사용해 일시적으로 환경 변수 이름이나 URL을 덮어쓸 수 있습니다.
4. 동일한 스터디 이름(`search.study_name` 또는 CLI `--study-name`)을 공유하는 여러 프로세스를 실행하면 Optuna가 트라이얼 분배를 조율합니다.

## 2. 타임프레임 조합 × 1,000회 실행 전략

두 가지 방식이 있습니다.

### 2.1. 단일 스터디에서 타임프레임을 카테고리로 최적화

- `config/params.yaml`의 `space.timeframe`/`space.htf` 항목이 이미 카테고리 선택으로 등록되어 있습니다.
- 하나의 스터디에서 1,000회 이상 돌리고 싶다면 CLI나 YAML에서 `search.n_trials`을 키워주면 됩니다.
- 장점: Optuna가 서로 다른 타임프레임 조합을 직접 비교해가며 탐색합니다.
- 단점: 조합별 성능 로그를 분리하기 어려워집니다. 다중 프로세스에서 동일 스터디를 공유하면 통계가 섞이는 점에 유의하세요.

### 2.2. 조합별로 스터디 분리 (권장)

1. CLI `--timeframe-grid` 옵션을 사용하면 쉼표/세미콜론으로 구분된 `LTF@HTF` 목록을 한 번에 실행할 수 있습니다.
   ```bash
   python -m optimize.run \
     --params config/params.yaml \
     --backtest config/backtest.yaml \
     --symbol BINANCE:ENAUSDT \
     --timeframe-grid "1m@15m,1m@1h,3m@15m,3m@1h,5m@15m,5m@1h" \
     --n-trials 1000 \
     --enable useHTF \
     --disable useEventFilter \
     --resume-from reports/latest/bank.json \
     --pruner hyperband \
     --top-k 20
   ```
   - 각 조합은 `reports/<timestamp>_<symbol>_<ltf>_<htf>` 형태로 출력 디렉터리가 생성되고, Optuna 스터디 이름도 `심볼_LTF_HTF_해시` 형식으로 자동 분리됩니다.
   - 필요 시 `--study-template "{symbol_slug}_{ltf_slug}_{htf_slug}"`, `--run-tag-template "{ltf_slug}_{htf_slug}_{index:02d}"` 처럼 플레이스홀더 템플릿으로 이름 규칙을 제어할 수 있습니다. 사용 가능한 플레이스홀더: `{symbol}`, `{symbol_slug}`, `{ltf}`, `{ltf_slug}`, `{htf}`, `{htf_slug}`, `{index}`, `{total}`, `{suffix}`.
2. 더 큰 규모에서는 위 명령을 `GNU parallel`, `tmux`, Kubernetes 잡 등으로 감싸 동시에 여러 프로세스를 띄우면 됩니다. 동일한 RDB 스토리지를 바라보고 있으면 Optuna가 충돌 없이 트라이얼을 분산합니다.

## 3. Dask 또는 Ray 통합

보다 대규모 클러스터 환경에서는 Optuna 통합을 사용할 수 있습니다.

- **Dask**: `optuna-integration` 패키지의 `DaskStorage`를 사용하면 워커 수만큼 병렬화됩니다.
  ```python
  from optuna_integration import DaskStorage
  storage = DaskStorage(address="tcp://scheduler:8786")
  study = optuna.create_study(direction="maximize", storage=storage)
  ```
  `optimize/run.py`를 직접 수정하기 어렵다면, 동일한 로직을 모듈화하여 맞춤 스크립트를 작성하는 방법을 추천합니다.

- **Ray Tune**: 타임프레임 조합을 `tune.grid_search`로 고정하고 나머지 연속 파라미터를 `OptunaSearch`가 관리하도록 구성할 수 있습니다.
  ```python
  from ray import tune
  from ray.tune.search.optuna import OptunaSearch

  search_alg = OptunaSearch()
  tuner = tune.Tuner(
      tune.with_resources(train_fn, {"cpu": 2}),
      tune_config=tune.TuneConfig(search_alg=search_alg, num_samples=1000),
      param_space={"ltf": tune.grid_search(["1m", "3m", "5m"]), "htf": tune.grid_search(["15m", "1h"])}
  )
  tuner.fit()
  ```

## 4. 추가 체크리스트

- 샘플러: 단일목표면 `TPESampler(multivariate=True, group=True)`, 다목표(`multi_objective: true`)가 활성화되어 있으면 NSGA-II 샘플러가 기본 선택됩니다. `search.nsga_params.population_size`/`crossover_prob` 등으로 개체 수와 교차 확률을 조정할 수 있습니다.
- 프루너: `MedianPruner`, `HyperbandPruner`, 혹은 `SuccessiveHalvingPruner(asha)`.
- NaN/Inf 처리: 백테스트 결과에 NaN이 발생하면 큰 패널티를 줘서 트라이얼을 빠르게 가지치기하세요.
- 재현성: 시드 고정(`search.seed`), 데이터 스냅샷 보관, 수수료·슬리피지 값을 명시합니다.
- 검증: 워크포워드(`walk_forward`) 또는 퍼지드 K-fold(`--cv purged-kfold`) 옵션으로 시계열 특성을 반영합니다.

이 가이드에 따라 외부 RDB를 공유하는 여러 프로세스 또는 클러스터 런처(Dask/Ray)를 붙이면 모든 타임프레임 조합을 1,000회 이상 안정적으로 탐색할 수 있습니다.
