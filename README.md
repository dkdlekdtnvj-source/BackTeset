# Pine Strategy Optimisation Toolkit

This repository provides a reproducible pipeline to optimise a Pine Script strategy with
Binance market data. It includes a deterministic Pine script profile, a Python backtest
model, Optuna-based search utilities, reporting helpers, and walk-forward evaluation.

## Project layout

```
strategy/strategy.pine         # Pine reference strategy (optimiser-ready inputs)
config/params.yaml             # Single profile configuration
config/backtest.yaml           # Batch/sweep configuration
optimize/run.py                # CLI entry point
optimize/strategy_model.py     # Python equivalent of the Pine logic
optimize/metrics.py            # Performance metrics and scoring
optimize/search_spaces.py      # YAML → Optuna helper
optimize/wf.py                 # Walk-forward analysis
optimize/report.py             # CSV/JSON/heatmap reporting
optimize/llm.py                # Gemini 기반 후보 제안 도우미
datafeed/binance_client.py     # Binance downloader with retries
datafeed/cache.py              # CSV cache layer
reports/                       # Optimisation output (ignored by git)
studies/                       # Optuna SQLite storage (auto-created)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Ensure Binance OHLCV data is cached under `data/` (the CLI will download missing
   ranges automatically).
2. Adjust optimisation inputs in `config/params.yaml`. The sample profile now sweeps
   the squeeze-momentum core (oscillator length, signal length, BB/KC lengths & multipliers),
   directional-flux smoothing, dynamic threshold options, higher-timeframe confirmation,
   and exit modules (ATR stop, fixed % stop, swing/pivot stops, ATR trail,
   breakeven, time stop). You can also pre-define `overrides` to pin any parameter
   on/off before a run and enable Top-K walk-forward re-ranking via `search.top_k`.
3. Configure the sweep universe in `config/backtest.yaml`. By default it contains
  nine Binance USDT perpetual pairs (ENA, ETH, BTC, SOL, **XPLA**, **ASTER**, DOGE,
  XRP, SUI) with lower timeframes 1m/3m/5m, higher timeframes 15m/1h, and a single
  2024-01-01 → 2025-09-25 창. The optimiser now treats the LTF/HTF selections as
  categorical parameters, so each Optuna trial chooses one combination while the
  reports highlight which pairing delivers the strongest Sortino/Profit Factor.
  The `symbol_aliases` mapping lets you keep the newly-listed ticker names you
  prefer (`XPLUSDT`, `ASTERUSDT`) while the optimiser automatically fetches data
  using Binance's current instruments (`XPLAUSDT`, `ASTRUSDT`).
4. Launch the optimiser non-interactively or with prompts:

   ```bash
   python -m optimize.run --params config/params.yaml --backtest config/backtest.yaml
   ```

   or

  ```bash
  ./시작
  ```

  또는 기존 플래그를 그대로 사용하려면 아래처럼 호출할 수 있습니다.

  ```bash
  python -m optimize.run --interactive
  ```

   The interactive mode lets you pick the symbol, evaluation window, leverage,
   position size, and the boolean filters (HTF sync, ATR trail, pivot stops,
   breakeven, etc.) from the terminal. Command-line overrides are also
   available:

   - `--symbol`, `--timeframe`, `--htf`, `--start`, `--end`
   - `--timeframe-grid 1m@15m,3m@1h` 으로 여러 LTF/HTF 조합을 일괄 실행 (필요 시 `--study-template`, `--run-tag-template` 으로 이름 규칙 지정)
   - `--leverage`, `--qty-pct`
   - `--n-trials`
  - `--n-jobs 4` 처럼 Optuna 병렬 worker 수를 지정해 멀티코어를 활용할 수 있습니다. 기본값은 PostgreSQL 스토리지일 때 CPU 코어 수 전체를 활용하도록 자동 설정됩니다.
   - `--enable name1,name2`, `--disable name3`
   - `--top-k 10` to re-rank the best Optuna trials by walk-forward out-of-sample
     performance.
  - `--storage-url-env OPTUNA_STORAGE` 로 YAML 설정 없이도 Optuna 스토리지 환경 변수를 바꿔 외부 RDB를 가리킬 수 있습니다.
  - 실행이 한 번 끝나면 `studies/<심볼>_<ltf>_<htf>/storage.json` 파일에 사용된 스토리지 백엔드가 기록됩니다. 이후에는 YAML에 따로 URL을 넣지 않아도 동일한 심볼/타임프레임 조합으로 실행할 때 자동으로 PostgreSQL 설정을 불러옵니다.

  Outputs are written to `reports/` (`results.csv`, `results_datasets.csv`,
  `results_timeframe_summary.csv`, `results_timeframe_rankings.csv`, `best.json`,
  `heatmap.png`) along with a machine-readable `trials/` 폴더(`trials.jsonl`,
  `best.yaml`, `trials_final.csv`). These files are flushed after **every** trial so
  you still keep the trail even if the run is interrupted. The `results_datasets.csv`
  file is especially useful for answering
  “어떤 LTF/HTF 조합이 가장 좋은가요?” because every dataset row lists the symbol,
  LTF, HTF, and the full metric set (Net Profit, Sortino, Profit Factor, MaxDD,
  Win Rate, Weekly Net Profit, 등). The automatically generated
  `results_timeframe_summary.csv`/`results_timeframe_rankings.csv` pair then pivots
  those rows into 평균·중앙값 테이블과 정렬 리스트 so you can immediately compare
  1m/3m/5m vs 15m/60m 조합. The `best.json` payload also includes the Top-K candidate
  summary with their walk-forward scores.

### Quick-start helper

If you would like a guided experience without memorising the CLI switches, run:

```bash
python -m optimize.quickstart
```

The helper will prompt for the symbol, backtest window, leverage, position size,
boolean filter toggles, and trial count. Once the questions are answered it
forwards the selections to `optimize.run` and the
reports appear under `reports/` just like the direct CLI entry point.

### vectorbt 기반 고속 백테스트 (선택)

PostgreSQL 등 외부 RDB 스토리지를 연결하고 멀티코어 환경을 충분히 활용하고
싶다면 `vectorbt` 엔진으로 백테스트를 실행할 수 있습니다. `params` 프로필에
`altEngine: vectorbt` 값을 추가하거나 Optuna 트라이얼 파라미터로 동일한 키를
전달하면, 기본 파이썬 구현 대신 vectorbt 포트폴리오 시뮬레이터가 호출됩니다.

```yaml
overrides:
  altEngine: vectorbt
```

- vectorbt는 선택적 의존성이므로 직접 설치해야 합니다. (`pip install vectorbt`)
- 현재 호환 레이어는 기본 모멘텀·플럭스·동적 임계값·`exitOpposite` 규칙을 빠르게
  재현하는 데 집중했습니다. `useStopLoss`, `useAtrTrail`, `useMomFade`,
  `useStructureGate` 등 고급 기능이 활성화된 경우에는 아직 기본 엔진으로 자동
  폴백됩니다.
- PyBroker 호환 경로는 향후 버전에서 제공될 예정입니다.

## LLM 보조(선택)

`config/params.yaml` 의 `llm` 블록은 기본값이 `enabled: true` 로 바뀌었으며,
일정 수(`initial_trials`) 만큼의 Optuna 트라이얼을 먼저 수행한 뒤 Gemini API에
"탑 트라이얼 요약 + 탐색 공간"을 전달합니다. 응답은 다음 두 가지 정보를 포함한
단일 JSON 객체여야 하며, 유효성 검사를 통과하면 큐에 등록됩니다.

- `candidates`: 파라미터 이름을 key 로 가지는 제안 목록. 경계·스텝·카테고리 규칙을
  통과한 값만 자동으로 `study.enqueue_trial()` 에 추가됩니다.
- `insights`: 상위 트라이얼을 바탕으로 도출한 전략/탐색 아이디어. 수신된 텍스트는
  `reports/<timestamp>/logs/gemini_insights.md` 에 타임스탬프와 함께 기록되고 로그에도
  출력되어 후속 튜닝 방향을 빠르게 파악할 수 있습니다.

- API 키는 다음 우선순위로 탐색합니다.
  1. `llm.api_key`
  2. `llm.api_key_file`/`llm.api_key_path` 로 지정한 파일
  3. `llm.api_key_env` (기본 `GEMINI_API_KEY`) 환경 변수
  4. 현재 작업 디렉터리·저장소 루트·`config/` 아래 `.env` 파일에 정의된 동일한 이름의 키
  어떤 경로에서도 키를 찾지 못하면 실행 시 경고가 출력되고 LLM 단계는 생략됩니다.
- 기본 모델은 `gemini-2.0-flash-exp` 이며 `top_n`/`count` 값으로 참고할 트라이얼
  수와 제안 받을 후보 수를 제어할 수 있습니다.
- `google-genai` 패키지가 설치돼 있지 않으면 경고만 출력하고 LLM 단계를 건너뜁니다.
- CLI `--llm`/`--no-llm` 플래그 또는 인터랙티브 모드 질문으로 실행 중에도 손쉽게
  활성/비활성 전환이 가능합니다.

예시:

```bash
echo "GEMINI_API_KEY=YOUR_API_KEY" > .env  # 또는 export GEMINI_API_KEY=...
python -m optimize.run --params config/params.yaml --backtest config/backtest.yaml
```

실행 후 `reports/<timestamp>.../trials/trials.jsonl` 에서는 각 트라이얼의 상태, 점수,
파라미터를 줄 단위 JSON 으로 확인할 수 있고 `trials_final.csv` 는 Excel/BI 도구에서
바로 열 수 있도록 준비됩니다.

## 병렬/대규모 최적화

- `config/params.yaml` 의 `search.study_name` 으로 스터디 이름을 고정하면 여러 프로세스가 같은 스터디를 공유할 수 있습니다. 이름을 지정하지 않으면 자동으로 `심볼_LTF_HTF_해시` 형태가 생성돼 배치 실행 시 충돌을 방지합니다.
- `search.storage_url_env`(기본값 `OPTUNA_STORAGE`), CLI `--storage-url-env`, `--storage-url` 로 RDB 접속 정보를 지정하면 Optuna가 프로세스/노드 병렬을 지원합니다. 환경 변수가 없으면 자동으로 `studies/` 아래 SQLite 파일을 사용합니다.
- 스터디를 처음 생성하면 `studies/<slug>/storage.json` 포인터가 같이 생성됩니다. 여기에는 최근 실행 시 사용된 스토리지 URL(비밀번호 등 민감 정보는 마스킹), 환경 변수 이름, 풀 설정이 기록되며, 다음 실행에서 `search.storage_url`/`storage_url_env` 값이 비어 있을 경우 자동으로 재사용됩니다.
- CLI `--study-name`/`--storage-url` 플래그는 YAML 설정을 일시적으로 덮어쓰는 용도로 사용할 수 있습니다.
- `--timeframe-grid` 를 사용하면 여러 타임프레임 조합을 한 번에 실행하면서 각 조합마다 독립된 리포트/스터디가 생성되며, 필요 시 `--study-template`, `--run-tag-template` 로 이름 규칙을 조정할 수 있습니다.
- 기본 프로필은 다목표(`NetProfit`, `Sortino`, `ProfitFactor`, `MaxDD`) 최적화를 활성화하고 Optuna NSGA-II 샘플러(population 120, crossover 0.9)를 자동 선택합니다. 파라미터는 `search.nsga_params` 로 세부 조정 가능합니다.
- 타임프레임 조합별 1,000회 실행, Dask/Ray 연동 방법 등 자세한 절차는 [`docs/optuna_parallel.md`](docs/optuna_parallel.md) 를 참고하세요.

## Testing

```bash
pytest
```

## Notes

- The Pine script uses `process_orders_on_close=true`, `request.security(..., lookahead_off)`,
  and confirmed-bar guards to prevent repainting. Commission, slippage, leverage, minimum
  tradable capital, and liquidation buffers are exposed as optimiser-friendly inputs.
- Walk-forward parameters (train/test window length and step) can be configured under
  `walk_forward` in `config/params.yaml`. The Python model now mirrors the Pine exits more
  closely with ATR trailing, pivot stops, breakeven, time-stop handling, minimum hold
  enforcement, and optional event-window gating.
- Metrics include Net Profit, Max Drawdown, Sortino, Sharpe, Profit Factor, Win Rate,
  weekly net profit, expectancy, RR, average MFE/MAE, and average holding period. The
  optimiser combines weighted objectives with penalties for breaching the risk gates and
  can optionally re-score the top trials by walk-forward OOS mean.
- Optimisation state is stored in `studies/<symbol>_<ltf>_<htf>.db` (SQLite + heartbeat)
  so 중단 후 재실행 시 자동으로 이어달리기(warm start)가 됩니다. JSONL/YAML 로그는
  최적화 도중 예기치 못한 종료가 발생해도 남도록 `trials/` 폴더에 즉시 기록됩니다.
