"""Command line interface for running parameter optimisation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence as AbcSequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import optuna
import optuna.storages
import pandas as pd
import yaml
import multiprocessing
import ccxt
import sqlalchemy
from sqlalchemy import event
from sqlalchemy.engine import make_url
from optuna.trial import TrialState

from datafeed.cache import DataCache
from optimize.metrics import (
    EPS,
    LOSSLESS_ANOMALY_FLAG,
    LOSSLESS_GROSS_LOSS_PCT,
    MICRO_LOSS_ANOMALY_FLAG,
    ObjectiveSpec,
    Trade,
    apply_lossless_anomaly,
    aggregate_metrics,
    equity_curve_from_returns,
    evaluate_objective_values,
    normalise_objectives,
)
from optimize.report import generate_reports, write_bank_file, write_trials_dataframe
from optimize.search_spaces import build_space, grid_choices, mutate_around, sample_parameters
from optimize.strategy_model import run_backtest
from optimize.wf import run_purged_kfold, run_walk_forward
from optimize.regime import detect_regime_label, summarise_regime_performance
from optimize.llm import LLMSuggestions, generate_llm_candidates
from optuna.exceptions import StorageInternalError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


HTF_ENABLED = False

def fetch_top_usdt_perp_symbols(
    limit: int = 50,
    exclude_symbols: Optional[Sequence[str]] = None,
    exclude_keywords: Optional[Sequence[str]] = None,
    min_price: Optional[float] = None,
) -> List[str]:
    """Binance USDT-M Perp 선물에서 24h quote volume 상위 심볼을 반환합니다."""

    ex = ccxt.binanceusdm(
        {
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
        }
    )
    ex.load_markets()
    tickers = ex.fetch_tickers()

    exclude_symbols_set = set(exclude_symbols or [])
    exclude_keywords = list(exclude_keywords or [])
    keyword_pattern = (
        re.compile("|".join(re.escape(k) for k in exclude_keywords))
        if exclude_keywords
        else None
    )

    rows: List[Tuple[str, float]] = []
    for sym, ticker in tickers.items():
        market = ex.market(sym)
        if not market.get("swap", False):
            continue
        if market.get("quote") != "USDT":
            continue

        unified = market.get("id", "")
        if unified in exclude_symbols_set:
            continue
        if keyword_pattern and keyword_pattern.search(unified):
            continue

        last = ticker.get("last")
        if min_price is not None:
            if last is None or float(last) < float(min_price):
                continue

        quote_volume = ticker.get("quoteVolume")
        if quote_volume is None:
            base_volume = ticker.get("baseVolume") or 0
            last_price = ticker.get("last") or 0
            quote_volume = base_volume * last_price

        try:
            rows.append((unified, float(quote_volume)))
        except (TypeError, ValueError):
            continue

    rows.sort(key=lambda item: item[1], reverse=True)
    return [f"BINANCE:{symbol}" for symbol, _ in rows[:limit]]


LOGGER = logging.getLogger("optimize")

# ---------------------------------------------------------------------------
# Warning management
#
# Optuna exposes a number of experimental features (e.g. heartbeat_interval,
# multivariate sampling) that trigger `ExperimentalWarning` on import or
# configuration.  These warnings are not actionable for end users of this
# framework and clutter the console.  Filter them globally so they do not
# interfere with logging output.  Likewise, other libraries may raise
# `UserWarning` when a supplied option has no effect; these should be handled
# at the call site (see optimize/alternative_engine.py).
import warnings
try:
    from optuna.exceptions import ExperimentalWarning  # type: ignore
except Exception:
    ExperimentalWarning = None  # type: ignore

if ExperimentalWarning is not None:
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Set the number of threads for OpenMP/BLAS libraries to the number of
# available CPU cores.  This improves performance of underlying numerical
# routines by utilising multiple cores during heavy vectorized operations.
_threads = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", _threads)
os.environ.setdefault("MKL_NUM_THREADS", _threads)

CPU_COUNT = os.cpu_count() or 4
DEFAULT_OPTUNA_JOBS = max(1, CPU_COUNT)
DEFAULT_DATASET_JOBS = max(1, CPU_COUNT)
SQLITE_SAFE_OPTUNA_JOBS = max(1, CPU_COUNT // 2)
SQLITE_SAFE_DATASET_JOBS = max(1, CPU_COUNT // 2)
DEFAULT_STORAGE_ENV_KEY = "OPTUNA_STORAGE"
DEFAULT_POSTGRES_STORAGE_URL = (
    "postgresql://postgres:5432@127.0.0.1:5432/optuna"
)
POSTGRES_PREFIXES = ("postgresql://", "postgresql+psycopg://")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = Path("reports")
STUDY_ROOT = Path("studies")
NON_FINITE_PENALTY = -1e12
PF_ANOMALY_THRESHOLD = 50.0
MIN_VOLUME_THRESHOLD = 100.0
# 최소 트레이드 수 요구를 비활성화합니다. 원본에서는 적은 트레이드 수로 인해 과도한 패널티가 발생했습니다.
MIN_TRADES_ENFORCED = 0
PROFIT_FACTOR_CHECK_LABEL = "체크 필요"
# When logging individual trial progress to CSV, record Sortino before
# ProfitFactor and maintain parameters in the order they were provided.  The
# ordering here dictates the column order of the CSV.  A field for Sortino
# has been added to capture this risk-adjusted metric per trial.
TRIAL_PROGRESS_FIELDS = [
    "number",
    "sortino",
    "profit_factor",
    "lossless_profit_factor_value",
    "score",
    "value",
    "state",
    "trades",
    "win_rate",
    "max_dd",
    "valid",
    "timeframe",
    "htf_timeframe",
    "pruned",
    "params",
    "skipped_datasets",
    "datetime_complete",
]


TRIAL_LOG_WRITE_LOCK = Lock()


def _mask_storage_url(url: str) -> str:
    if not url:
        return ""
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return url


def _make_sqlite_storage(
    url: str,
    *,
    timeout_sec: int = 120,
    heartbeat_interval: Optional[int] = None,
    grace_period: Optional[int] = None,
) -> optuna.storages.RDBStorage:
    """SQLite 스토리지에 WAL 모드와 타임아웃을 적용해 생성합니다."""

    connect_args = {"timeout": timeout_sec, "check_same_thread": False}
    engine_kwargs = {"connect_args": connect_args, "pool_pre_ping": True}
    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=heartbeat_interval or None,
        grace_period=grace_period or None,
    )

    @event.listens_for(storage.engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:  # type: ignore[unused-ignore]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout=60000;")

            wal_pragmas = (
                "PRAGMA journal_mode=WAL;",
                "PRAGMA synchronous=NORMAL;",
                "PRAGMA temp_store=MEMORY;",
            )

            for attempt in range(5):
                try:
                    for pragma in wal_pragmas:
                        cursor.execute(pragma)
                except sqlite3.OperationalError as exc:  # pragma: no cover - 환경 의존
                    is_locked = "database is locked" in str(exc).lower()
                    if is_locked and attempt < 4:
                        time.sleep(0.2 * (attempt + 1))
                        continue

                    LOGGER.warning(
                        "SQLite PRAGMA 설정 중 오류가 발생했습니다 (WAL 미적용 가능성): %s",
                        exc,
                    )
                else:
                    break
        finally:
            cursor.close()

    return storage


def _make_rdb_storage(
    url: str,
    *,
    heartbeat_interval: Optional[int] = None,
    grace_period: Optional[int] = None,
    pool_size: Optional[int] = None,
    max_overflow: Optional[int] = None,
    pool_timeout: Optional[int] = None,
    pool_recycle: Optional[int] = None,
    isolation_level: Optional[str] = None,
    connect_timeout: Optional[int] = None,
    statement_timeout_ms: Optional[int] = None,
) -> optuna.storages.RDBStorage:
    """PostgreSQL 등 외부 RDB 용도의 Optuna 스토리지를 생성합니다."""

    engine_kwargs: Dict[str, object] = {"pool_pre_ping": True}

    if pool_size is not None:
        engine_kwargs["pool_size"] = pool_size
    if max_overflow is not None:
        engine_kwargs["max_overflow"] = max_overflow
    if pool_timeout is not None:
        engine_kwargs["pool_timeout"] = pool_timeout
    if pool_recycle is not None:
        engine_kwargs["pool_recycle"] = pool_recycle
    if isolation_level:
        engine_kwargs["isolation_level"] = isolation_level

    connect_args: Dict[str, object] = {}
    if connect_timeout is not None:
        connect_args["connect_timeout"] = connect_timeout

    options_parts: List[str] = []
    if statement_timeout_ms is not None:
        options_parts.append(f"-c statement_timeout={statement_timeout_ms}")

    url_info = None
    try:
        url_info = make_url(url)
    except Exception:
        url_info = None

    is_postgres = bool(url_info and url_info.drivername.startswith("postgresql"))
    if is_postgres:
        engine_kwargs.setdefault("pool_size", 5)
        engine_kwargs.setdefault("max_overflow", 10)
        engine_kwargs.setdefault("pool_recycle", 1800)
        if connect_timeout is None:
            connect_args.setdefault("connect_timeout", 10)
        options_parts.append("-c timezone=UTC")

    if options_parts:
        existing_options = str(connect_args.get("options", "")).strip()
        if existing_options:
            options_parts.insert(0, existing_options)
        connect_args["options"] = " ".join(part for part in options_parts if part)

    if connect_args:
        engine_kwargs["connect_args"] = connect_args

    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=heartbeat_interval or None,
        grace_period=grace_period or None,
    )

    return storage


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=3),
    retry=(
        retry_if_exception_type(sqlalchemy.exc.OperationalError)
        | retry_if_exception_type(StorageInternalError)
    ),
)
def _safe_sample_parameters(
    trial: optuna.trial.Trial, space: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """SQLite 잠금 오류 발생 시 짧게 재시도하며 파라미터를 샘플링합니다."""

    return sample_parameters(trial, space)


# 단순 메트릭 계산 경로 사용 여부 (CLI 인자/설정으로 갱신됩니다).
simple_metrics_enabled: bool = False


_INITIAL_BALANCE_KEYS = (
    "InitialCapital",
    "InitialEquity",
    "InitialBalance",
    "StartingBalance",
)


def _apply_lossless_anomaly(target: Dict[str, float]) -> Optional[Tuple[str, float, float, float, float]]:
    return apply_lossless_anomaly(target)

# 기본 팩터 최적화에 사용할 파라미터 키 집합입니다.
# 복잡한 보호 장치·부가 필터 대신 핵심 진입 로직과 직접 관련된 항목만 남겨
# 탐색 공간을 크게 줄이고 수렴 속도를 높입니다.
# Keys used when `--basic-factors-only` is enabled.  This set should mirror
# the parameters defined in `config/params.yaml` so that only the core
# oscillator, volatility channel, flux and exit parameters are swept.  If you
# modify `params.yaml` or add new tuneable inputs, update this set
# accordingly.  Removing unused or unsupported names from this set prevents
# accidental sampling of unrelated stop/trail/slippage options.
BASIC_FACTOR_KEYS = {
    # Oscillator & signal lengths
    "oscLen",
    "signalLen",
    # Bollinger/Keltner channels
    "bbLen",
    "bbMult",
    "kcLen",
    "kcMult",
    # Directional flux
    "fluxLen",
    "fluxSmoothLen",
    "useFluxHeikin",
    # Dynamic threshold & gates
    "useDynamicThresh",
    "useSymThreshold",
    "statThreshold",
    "buyThreshold",
    "sellThreshold",
    "dynLen",
    "dynMult",
    # Exit logic
    "exitOpposite",
    "useMomFade",
    "momFadeRegLen",
    "momFadeBbLen",
    "momFadeKcLen",
    "momFadeBbMult",
    "momFadeKcMult",
}


def _utcnow_isoformat() -> str:
    """현재 UTC 시각을 ISO8601 ``Z`` 표기 문자열로 반환합니다."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _register_study_reference(
    study_storage: Optional[Path],
    *,
    storage_meta: Dict[str, object],
    study_name: Optional[str] = None,
) -> None:
    """Persist study storage metadata for later reuse."""

    if study_storage is None:
        return

    backend = str(storage_meta.get("backend") or "none").lower()
    if backend in {"", "none"}:
        return

    registry_dir = _study_registry_dir(study_storage)
    registry_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = registry_dir / "storage.json"

    payload: Dict[str, object] = {
        "updated_at": _utcnow_isoformat(),
        "backend": backend,
        "study_name": study_name,
        "storage_url_env": storage_meta.get("env_key"),
        "env_value_present": storage_meta.get("env_value_present"),
    }

    url_value = storage_meta.get("url")
    if isinstance(url_value, str) and url_value:
        payload["storage_url"] = url_value
        try:
            payload["storage_url_masked"] = make_url(url_value).render_as_string(
                hide_password=True
            )
        except Exception:
            payload["storage_url_masked"] = url_value

    if backend == "sqlite":
        payload["sqlite_path"] = storage_meta.get("path") or str(study_storage)
        payload["allow_parallel"] = storage_meta.get("allow_parallel")
    else:
        pool_meta = storage_meta.get("pool")
        if isinstance(pool_meta, dict) and pool_meta:
            payload["pool"] = pool_meta
        if storage_meta.get("connect_timeout") is not None:
            payload["connect_timeout"] = storage_meta.get("connect_timeout")
        if storage_meta.get("isolation_level"):
            payload["isolation_level"] = storage_meta.get("isolation_level")
        if storage_meta.get("statement_timeout_ms") is not None:
            payload["statement_timeout_ms"] = storage_meta.get("statement_timeout_ms")

    pointer_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _sanitise_storage_meta(storage_meta: Dict[str, object]) -> Dict[str, object]:
    if not storage_meta:
        return {}

    cleaned = copy.deepcopy(storage_meta)
    url_value = cleaned.get("url")
    if isinstance(url_value, str) and url_value:
        try:
            cleaned["url"] = make_url(url_value).render_as_string(hide_password=True)
        except Exception:
            cleaned["url"] = "***invalid-url***"
    return cleaned


def _slugify_symbol(symbol: str) -> str:
    text = symbol.split(":")[-1]
    return text.replace("/", "").replace(" ", "")


def _slugify_timeframe(timeframe: Optional[str]) -> str:
    if not timeframe:
        return ""
    return str(timeframe).replace("/", "_").replace(" ", "")


def _space_hash(space: Dict[str, object]) -> str:
    payload = json.dumps(space or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _restrict_to_basic_factors(
    space: Dict[str, Dict[str, object]], *, enabled: bool = True
) -> Dict[str, Dict[str, object]]:
    """기본 팩터만 남긴 탐색 공간 사본을 반환합니다."""

    if not space:
        return {}

    if not enabled:
        return {name: dict(spec) for name, spec in space.items()}

    filtered: Dict[str, Dict[str, object]] = {}
    for name, spec in space.items():
        if name in BASIC_FACTOR_KEYS:
            filtered[name] = dict(spec)
    return filtered


def _filter_basic_factor_params(
    params: Dict[str, object], *, enabled: bool = True
) -> Dict[str, object]:
    """기본 팩터 키만 남겨 파라미터 딕셔너리를 정리합니다."""

    if not params:
        return {}
    if not enabled:
        return dict(params)
    return {key: value for key, value in params.items() if key in BASIC_FACTOR_KEYS}


def _order_mapping(
    payload: Mapping[str, object],
    preferred_order: Optional[Sequence[str]] = None,
    *,
    priority: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """주어진 참조 순서에 맞춰 딕셔너리 순서를 재정렬합니다."""

    if not isinstance(payload, Mapping):
        return {}

    ordered: "OrderedDict[str, object]" = OrderedDict()

    for key in priority or ():
        if key in payload and key not in ordered:
            ordered[key] = payload[key]

    if preferred_order:
        for key in preferred_order:
            if key in payload and key not in ordered:
                ordered[key] = payload[key]

    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value

    return dict(ordered)


def _git_revision() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.parent / f"{path.name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(handler)


def _build_run_tag(
    datasets: Sequence["DatasetSpec"],
    params_cfg: Dict[str, object],
    run_tag: Optional[str],
) -> Tuple[str, str, str, str]:
    symbol = params_cfg.get("symbol") or (datasets[0].symbol if datasets else "unknown")
    timeframe = (
        params_cfg.get("timeframe")
        or (datasets[0].timeframe if datasets else "multi")
    )
    htf = None
    if HTF_ENABLED:
        htf = (
            params_cfg.get("htf_timeframe")
            or params_cfg.get("htf")
            or (datasets[0].htf_timeframe if datasets and datasets[0].htf_timeframe else "nohtf")
        )
        if not htf:
            htf = "nohtf"
    symbol_slug = _slugify_symbol(str(symbol))
    timeframe_slug = str(timeframe).replace("/", "_")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    parts = [timestamp, symbol_slug, timeframe_slug]
    if run_tag:
        parts.append(run_tag)
    return timestamp, symbol_slug, timeframe_slug, "_".join(filter(None, parts))


def _coerce_min_trades_value(value: object) -> Optional[int]:
    """Convert ``value`` to a non-negative integer if possible."""

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "na"}:
            return None
        value = text

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric):
        return None

    return max(0, int(round(numeric)))


def _coerce_config_int(value: object, *, minimum: int, name: str) -> Optional[int]:
    """설정값을 정수로 강제 변환하며 하한선을 검증합니다."""

    if value is None:
        return None

    raw_value = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        raw_value = text

    try:
        numeric = int(float(raw_value))
    except (TypeError, ValueError):
        LOGGER.warning(
            "%s 값 '%s' 을(를) 정수로 변환할 수 없어 무시합니다.",
            name,
            raw_value,
        )
        return None

    if numeric < minimum:
        LOGGER.warning(
            "%s 값 %d 이(가) %d 보다 작아 무시합니다.",
            name,
            numeric,
            minimum,
        )
        return None

    return numeric


def _timeframe_lookup_keys(timeframe: Optional[str], htf: Optional[str]) -> List[str]:
    """Return candidate keys for matching timeframe-specific constraints."""

    keys: List[str] = []

    def _normalise(token: str) -> List[str]:
        variants = [token]
        variants.append(token.lower())
        variants.append(token.upper())
        compact = token.replace("/", "").replace(" ", "")
        if compact and compact not in variants:
            variants.append(compact)
            variants.append(compact.lower())
            variants.append(compact.upper())
        return list(dict.fromkeys(variants))

    if timeframe:
        keys.extend(_normalise(timeframe))

    if timeframe and htf:
        keys.extend(_normalise(f"{timeframe}@{htf}"))

    return list(dict.fromkeys(keys))


def _extract_min_trades_from_mapping(entry: object) -> Optional[int]:
    """Extract ``min_trades`` requirement from an arbitrary mapping entry."""

    if entry is None:
        return None

    if isinstance(entry, (int, float, str)):
        return _coerce_min_trades_value(entry)

    if not isinstance(entry, dict):
        return None

    priority_keys = [
        "min_trades_test",
        "minTradesTest",
        "min_trades",
        "minTrades",
        "oos",
        "OOS",
        "test",
        "Test",
        "value",
        "Value",
        "default",
        "Default",
    ]

    for key in priority_keys:
        if key not in entry:
            continue
        candidate = entry[key]
        if isinstance(candidate, dict):
            resolved = _extract_min_trades_from_mapping(candidate)
        else:
            resolved = _coerce_min_trades_value(candidate)
        if resolved is not None:
            return resolved

    # Fallback: inspect nested mappings for a usable value.
    for candidate in entry.values():
        resolved = _extract_min_trades_from_mapping(candidate)
        if resolved is not None:
            return resolved

    return None


def _resolve_dataset_min_trades(
    dataset: "DatasetSpec",
    *,
    constraints: Optional[Dict[str, object]] = None,
    risk: Optional[Dict[str, object]] = None,
    explicit: Optional[object] = None,
) -> Optional[int]:
    """Resolve the minimum trade requirement for a dataset."""

    constraints = constraints or {}
    candidates: List[Optional[int]] = []

    candidates.append(_coerce_min_trades_value(explicit))

    lookup_keys = _timeframe_lookup_keys(dataset.timeframe, dataset.htf_timeframe)
    timeframe_rule_keys = [
        "timeframes",
        "per_timeframe",
        "perTimeframe",
        "timeframe_rules",
        "timeframeRules",
        "min_trades_by_timeframe",
        "minTradesByTimeframe",
    ]

    for container_key in timeframe_rule_keys:
        rules = constraints.get(container_key)
        if not isinstance(rules, dict):
            continue
        for key in lookup_keys:
            if key in rules:
                candidates.append(_extract_min_trades_from_mapping(rules[key]))
                break

    candidates.append(_coerce_min_trades_value(constraints.get("min_trades_test")))

    if isinstance(risk, dict):
        candidates.append(_coerce_min_trades_value(risk.get("min_trades")))

    for candidate in candidates:
        if candidate is not None:
            return candidate

    return None


def _run_dataset_backtest_task(
    dataset_ref: object,
    params: Dict[str, object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    """Execute ``run_backtest`` for a single dataset.

    ``dataset_ref`` may be a :class:`DatasetSpec` (thread executor) or 문자열 ID
    (process executor). When a string ID is provided, 워커 초기화 시 등록된 전역
    캐시를 통해 DataFrame을 최초 한 번만 로드합니다. 이 함수는 모듈 레벨에
    존재해야 ``ProcessPoolExecutor``에서 피클링할 수 있습니다.
    """

    # Determine if an alternative engine is requested.  The ``engine``
    # parameter can be provided in the ``params`` dict as ``altEngine``
    # (e.g. "vectorbt" or "pybroker").  If specified and an alternative
    # engine is available, attempt to delegate the backtest accordingly.  In
    # the event of missing dependencies or unimplemented integration, fall
    # back to the native run_backtest implementation.
    dataset = _resolve_dataset_reference(dataset_ref)

    engine = None
    try:
        engine = params.get("altEngine") or params.get("engine")
    except Exception:
        engine = None
    if engine:
        try:
            from .alternative_engine import run_backtest_alternative
            return run_backtest_alternative(
                dataset.df,
                params,
                fees,
                risk,
                htf_df=dataset.htf,
                min_trades=min_trades,
                engine=str(engine),
            )
        except Exception as exc:
            # Log but continue with the default implementation
            LOGGER.warning(
                "Alternative engine '%s' failed (%s); falling back to native backtest.",
                engine,
                exc,
            )
    # Default: use native Python backtest
    return run_backtest(
        dataset.df,
        params,
        fees,
        risk,
        htf_df=dataset.htf,
        min_trades=min_trades,
    )


def _resolve_output_directory(
    base: Optional[Path],
    datasets: Sequence["DatasetSpec"],
    params_cfg: Dict[str, object],
    run_tag: Optional[str],
) -> Tuple[Path, Dict[str, str]]:
    ts, symbol_slug, timeframe_slug, tag = _build_run_tag(datasets, params_cfg, run_tag)
    if base is None:
        root = DEFAULT_REPORT_ROOT
        output = root / tag
    else:
        output = base
    output = _next_available_dir(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "timestamp": ts,
        "symbol": symbol_slug,
        "timeframe": timeframe_slug,
        "tag": tag,
    }
    if HTF_ENABLED:
        manifest["htf_timeframe"] = _slugify_timeframe(_extract_primary_htf(params_cfg, datasets))
    return output, manifest


def _write_manifest(
    output_dir: Path,
    *,
    manifest: Dict[str, object],
) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _load_json(path: Path) -> Dict[str, object]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _parse_timeframe_grid(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    combos: List[str] = []
    text = str(raw).replace("\n", ",").replace(";", ",")
    for token in text.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        if "@" in candidate:
            ltf, _ = candidate.split("@", 1)
        elif ":" in candidate:
            ltf, _ = candidate.split(":", 1)
        else:
            ltf = candidate
        ltf = ltf.strip()
        if not ltf:
            continue
        combos.append(ltf)
    return combos


def _format_batch_value(
    template: Optional[str],
    base: Optional[str],
    suffix: str,
    context: Dict[str, object],
) -> Optional[str]:
    if template:
        try:
            return template.format(**context)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(f"Unknown placeholder '{missing}' in template {template!r}") from exc
    if base:
        return f"{base}_{suffix}" if suffix else base
    return suffix or None


def _resolve_study_storage(
    params_cfg: Dict[str, object],
    datasets: Sequence["DatasetSpec"],
) -> Optional[Path]:
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    _, symbol_slug, timeframe_slug, _ = _build_run_tag(datasets, params_cfg, None)
    return STUDY_ROOT / f"{symbol_slug}_{timeframe_slug}.db"


def _study_registry_dir(storage_path: Path) -> Path:
    """Return the directory that holds study registry metadata."""

    if storage_path.suffix:
        return storage_path.with_suffix("")
    return storage_path


def _study_registry_payload_path(storage_path: Path) -> Path:
    return _study_registry_dir(storage_path) / "storage.json"


def _load_study_registry(
    study_storage: Optional[Path],
) -> Tuple[Dict[str, object], Optional[Path]]:
    if study_storage is None:
        return {}, None

    pointer_path = _study_registry_payload_path(study_storage)
    if not pointer_path.exists():
        return {}, pointer_path

    return _load_json(pointer_path), pointer_path


def _apply_study_registry_defaults(
    search_cfg: Dict[str, object], study_storage: Optional[Path]
) -> None:
    """Apply stored storage settings when explicit configuration is missing."""

    payload, pointer_path = _load_study_registry(study_storage)
    if not payload:
        return

    backend = str(payload.get("backend") or "none").lower()
    if backend in {"", "none"}:
        return

    applied: List[str] = []

    if backend == "sqlite":
        stored_url = payload.get("storage_url") or payload.get("sqlite_url")
        if stored_url and not search_cfg.get("storage_url"):
            search_cfg["storage_url"] = stored_url
            applied.append("storage_url")
    else:
        env_key = payload.get("storage_url_env")
        if env_key and not search_cfg.get("storage_url_env"):
            search_cfg["storage_url_env"] = env_key
            applied.append("storage_url_env")
        stored_url = payload.get("storage_url")
        if stored_url and not search_cfg.get("storage_url"):
            search_cfg["storage_url"] = stored_url
            applied.append("storage_url")

    if applied and pointer_path is not None:
        LOGGER.info(
            "스터디 레지스트리(%s)에서 %s 설정을 불러왔습니다.",
            pointer_path,
            ", ".join(applied),
        )


def _extract_primary_htf(
    params_cfg: Dict[str, object],
    datasets: Sequence["DatasetSpec"],
) -> Optional[str]:
    if not HTF_ENABLED:
        return None
    raw = params_cfg.get("htf_timeframes")
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        return str(raw[0])
    direct = params_cfg.get("htf_timeframe") or params_cfg.get("htf")
    if direct:
        return str(direct)
    if datasets and getattr(datasets[0], "htf_timeframe", None):
        return str(datasets[0].htf_timeframe)
    return None


def _default_study_name(
    params_cfg: Dict[str, object],
    datasets: Sequence["DatasetSpec"],
    space_hash: Optional[str] = None,
) -> str:
    _, symbol_slug, timeframe_slug, _ = _build_run_tag(datasets, params_cfg, None)
    suffix = f"_{space_hash[:6]}" if space_hash else ""
    return f"{symbol_slug}_{timeframe_slug}{suffix}"


def _discover_bank_path(
    current_output: Path,
    tag_info: Dict[str, str],
    space_hash: str,
) -> Optional[Path]:
    root = current_output.parent
    if not root.exists():
        return None
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p != current_output],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        bank_path = candidate / "bank.json"
        if not bank_path.exists():
            continue
        payload = _load_json(bank_path)
        metadata = payload.get("metadata", {})
        if payload.get("space_hash") != space_hash:
            continue
        if metadata.get("symbol") != tag_info.get("symbol"):
            continue
        if metadata.get("timeframe") != tag_info.get("timeframe"):
            continue
        metadata_htf = metadata.get("htf_timeframe") or "nohtf"
        target_htf = tag_info.get("htf_timeframe") or "nohtf"
        if metadata_htf != target_htf:
            continue
        return bank_path
    return None


def _load_seed_trials(
    bank_path: Optional[Path],
    space: Dict[str, object],
    space_hash: str,
    regime_label: Optional[str] = None,
    max_seeds: int = 20,
    *,
    basic_filter_enabled: bool = True,
) -> List[Dict[str, object]]:
    if bank_path is None:
        return []
    payload = _load_json(bank_path)
    if not payload or payload.get("space_hash") != space_hash:
        return []

    entries = payload.get("entries", [])
    if regime_label:
        filtered = [entry for entry in entries if entry.get("regime", {}).get("label") == regime_label]
        if filtered:
            entries = filtered

    seeds: List[Dict[str, object]] = []
    rng = np.random.default_rng()
    for entry in entries[:max_seeds]:
        params = entry.get("params")
        if not isinstance(params, dict):
            continue
        filtered_params = _filter_basic_factor_params(
            dict(params), enabled=basic_filter_enabled
        )
        if not filtered_params:
            continue
        seeds.append(filtered_params)
        mutated = mutate_around(
            filtered_params,
            space,
            scale=float(payload.get("mutation_scale", 0.1)),
            rng=rng,
        )
        mutated_filtered = _filter_basic_factor_params(
            mutated, enabled=basic_filter_enabled
        )
        if mutated_filtered:
            seeds.append(mutated_filtered)
    return seeds


def _build_bank_payload(
    *,
    tag_info: Dict[str, str],
    space_hash: str,
    entries: List[Dict[str, object]],
    regime_summary,
    mutation_scale: float = 0.1,
) -> Dict[str, object]:
    payload_entries: List[Dict[str, object]] = []
    for entry in entries:
        regime_info = summarise_regime_performance(entry, regime_summary)
        payload_entries.append({**entry, "regime": regime_info})

    return {
        "created_at": _utcnow_isoformat(),
        "metadata": {
            "symbol": tag_info.get("symbol"),
            "timeframe": tag_info.get("timeframe"),
            "htf_timeframe": tag_info.get("htf_timeframe"),
            "tag": tag_info.get("tag"),
        },
        "space_hash": space_hash,
        "mutation_scale": mutation_scale,
        "entries": payload_entries,
    }


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _prompt_choice(label: str, choices: List[str], default: Optional[str] = None) -> Optional[str]:
    if not choices:
        return default
    while True:
        print(f"\n{label}:")
        for idx, value in enumerate(choices, start=1):
            marker = " (default)" if default == value else ""
            print(f"  {idx}. {value}{marker}")
        raw = input("Select option (press Enter for default): ").strip()
        if not raw:
            return default or (choices[0] if choices else None)
        if raw.isdigit():
            sel = int(raw)
            if 1 <= sel <= len(choices):
                return choices[sel - 1]
        print("Invalid selection. Please try again.")


def _prompt_bool(label: str, default: Optional[bool] = None) -> Optional[bool]:
    suffix = " [y/n]" if default is None else (" [Y/n]" if default else " [y/N]")
    while True:
        raw = input(f"{label}{suffix}: ").strip().lower()
        if not raw and default is not None:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        if not raw:
            return default
        print("Please answer 'y' or 'n'.")


@dataclass(frozen=True)
class LTFPromptResult:
    timeframe: Optional[str]
    use_all: bool = False


def _prompt_ltf_selection() -> LTFPromptResult:
    """사용자가 선호하는 LTF 조합(1, 3, 5분봉 또는 전체)을 선택하도록 안내합니다."""

    options = {"1": "1m", "3": "3m", "5": "5m"}
    if not sys.stdin or not sys.stdin.isatty():
        LOGGER.info("비대화형 환경이 감지되어 기본 1m LTF를 사용합니다.")
        return LTFPromptResult("1m")

    while True:
        print("\n작업을 시작 하기 전에 LTF를 선택해주세요.")
        print("  1) 1분봉")
        print("  3) 3분봉")
        print("  5) 5분봉")
        print("  7) 1/3/5 전체 (혼합 실행)")
        raw = input("선택 (1/3/5/7): ").strip()
        if raw in options:
            selection = options[raw]
            print(f"{raw}분봉을 선택했습니다.")
            return LTFPromptResult(selection)
        if raw == "7":
            print("1, 3, 5분봉을 모두 활용해 순차 실행합니다.")
            return LTFPromptResult(None, use_all=True)
        print("잘못된 입력입니다. 1, 3, 5, 7 중 하나를 입력해주세요.")


def _apply_ltf_override_to_datasets(backtest_cfg: Dict[str, object], timeframe: str) -> None:
    entries = backtest_cfg.get("datasets")
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry["ltf"] = [timeframe]
        entry["ltfs"] = [timeframe]
        entry["timeframes"] = [timeframe]


def _coerce_bool_or_none(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan"}:
            return None
        if text in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "f", "0", "no", "n", "off"}:
            return False
    return None


def _collect_tokens(items: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    for item in items:
        if not item:
            continue
        for token in item.split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def _collect_ltf_candidates(*configs: Mapping[str, object]) -> List[str]:
    seen: "OrderedDict[str, None]" = OrderedDict()

    def _register(value: object) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        seen.setdefault(text, None)

    for cfg in configs:
        if not isinstance(cfg, Mapping):
            continue
        datasets = cfg.get("datasets")
        if isinstance(datasets, list):
            for entry in datasets:
                if not isinstance(entry, Mapping):
                    continue
                for key in ("ltf", "ltfs", "timeframes"):
                    raw = entry.get(key)
                    if isinstance(raw, (list, tuple)):
                        for item in raw:
                            _register(item)
                    elif raw is not None:
                        _register(raw)

        timeframes = cfg.get("timeframes")
        if isinstance(timeframes, (list, tuple)):
            for tf in timeframes:
                _register(tf)
        elif timeframes is not None:
            _register(timeframes)

    return list(seen.keys())


def _ensure_dict(root: Dict[str, object], key: str) -> Dict[str, object]:
    value = root.get(key)
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


@dataclass(frozen=True)
class DatasetCacheInfo:
    root: Path
    futures: bool = False

    def serialise(self) -> Dict[str, object]:
        return {"root": str(self.root), "futures": self.futures}


@dataclass
class DatasetSpec:
    symbol: str
    timeframe: str
    start: str
    end: str
    df: pd.DataFrame
    htf: Optional[pd.DataFrame]
    htf_timeframe: Optional[str] = None
    source_symbol: Optional[str] = None
    cache_info: Optional[DatasetCacheInfo] = None

    @property
    def name(self) -> str:
        parts = [self.symbol, self.timeframe]
        if self.htf_timeframe:
            parts.append(f"htf{self.htf_timeframe}")
        parts.extend([self.start, self.end])
        return "_".join(parts)

    @property
    def meta(self) -> Dict[str, str]:
        return {
            "symbol": self.symbol,
            "source_symbol": self.source_symbol or self.symbol,
            "timeframe": self.timeframe,
            "from": self.start,
            "to": self.end,
            "htf_timeframe": self.htf_timeframe or "",
        }


_PROCESS_DATASET_REGISTRY: Dict[str, Dict[str, object]] = {}
_PROCESS_DATASET_OBJECTS: Dict[str, DatasetSpec] = {}
_PROCESS_DATASET_CACHES: Dict[Tuple[str, bool], DataCache] = {}
_PROCESS_DATASET_LOCK = Lock()


def _register_process_datasets(handles: Sequence[Dict[str, object]]) -> None:
    global _PROCESS_DATASET_REGISTRY, _PROCESS_DATASET_OBJECTS, _PROCESS_DATASET_CACHES
    _PROCESS_DATASET_REGISTRY = {entry["id"]: entry for entry in handles}
    _PROCESS_DATASET_OBJECTS = {}
    _PROCESS_DATASET_CACHES = {}


def _process_pool_initializer(handles: Sequence[Dict[str, object]]) -> None:
    _register_process_datasets(handles)


def _resolve_process_cache(root: str, futures: bool) -> DataCache:
    key = (root, futures)
    cache = _PROCESS_DATASET_CACHES.get(key)
    if cache is not None:
        return cache

    with _PROCESS_DATASET_LOCK:
        cache = _PROCESS_DATASET_CACHES.get(key)
        if cache is not None:
            return cache
        cache = DataCache(Path(root), futures=futures)
        _PROCESS_DATASET_CACHES[key] = cache
        return cache


def _load_process_dataset(dataset_id: str) -> DatasetSpec:
    handle = _PROCESS_DATASET_REGISTRY.get(dataset_id)
    if handle is None:
        raise KeyError(f"등록되지 않은 데이터셋 ID: {dataset_id}")

    cache_root = handle.get("cache_root")
    if not cache_root:
        raise RuntimeError("process executor에서 데이터셋을 로드할 캐시 경로가 설정되지 않았습니다.")
    futures_flag = bool(handle.get("cache_futures", False))
    cache = _resolve_process_cache(str(cache_root), futures_flag)

    source_symbol = str(handle.get("source_symbol"))
    timeframe = str(handle.get("timeframe"))
    start = str(handle.get("start"))
    end = str(handle.get("end"))
    df = cache.get(source_symbol, timeframe, start, end)

    htf_timeframe = handle.get("htf_timeframe") or None
    htf_df: Optional[pd.DataFrame] = None
    if htf_timeframe:
        try:
            htf_df = cache.get(source_symbol, str(htf_timeframe), start, end)
        except Exception as exc:
            LOGGER.warning("HTF 데이터 로드 실패(%s): %s", dataset_id, exc)
            htf_df = None

    dataset = DatasetSpec(
        symbol=str(handle.get("symbol")),
        timeframe=timeframe,
        start=start,
        end=end,
        df=df,
        htf=htf_df,
        htf_timeframe=htf_timeframe,
        source_symbol=source_symbol,
        cache_info=DatasetCacheInfo(Path(str(cache_root)), futures=futures_flag),
    )
    _PROCESS_DATASET_OBJECTS[dataset_id] = dataset
    return dataset


def _resolve_dataset_reference(dataset_ref: object) -> DatasetSpec:
    if isinstance(dataset_ref, DatasetSpec):
        return dataset_ref
    if isinstance(dataset_ref, str):
        cached = _PROCESS_DATASET_OBJECTS.get(dataset_ref)
        if cached is not None:
            return cached
        with _PROCESS_DATASET_LOCK:
            cached = _PROCESS_DATASET_OBJECTS.get(dataset_ref)
            if cached is not None:
                return cached
            return _load_process_dataset(dataset_ref)
    raise TypeError(f"dataset_ref 타입을 처리할 수 없습니다: {type(dataset_ref)!r}")


def _serialise_datasets_for_process(datasets: Sequence[DatasetSpec]) -> List[Dict[str, object]]:
    handles: List[Dict[str, object]] = []
    for dataset in datasets:
        if not dataset.cache_info:
            raise RuntimeError(
                "process executor를 사용하려면 DatasetSpec.cache_info 가 필요합니다."
            )
        cache_data = dataset.cache_info.serialise()
        handles.append(
            {
                "id": dataset.name,
                "symbol": dataset.symbol,
                "source_symbol": dataset.source_symbol or dataset.symbol,
                "timeframe": dataset.timeframe,
                "start": dataset.start,
                "end": dataset.end,
                "htf_timeframe": dataset.htf_timeframe,
                "cache_root": cache_data["root"],
                "cache_futures": cache_data["futures"],
            }
        )
    return handles


def _dataset_total_volume(dataset: "DatasetSpec") -> float:
    """Return a finite total volume for the given dataset."""

    if dataset.df is None or "volume" not in dataset.df.columns:
        return 0.0

    volume_series = pd.to_numeric(dataset.df["volume"], errors="coerce")
    total = float(np.nansum(volume_series.to_numpy(dtype=float)))
    if not np.isfinite(total):
        return 0.0
    return total


def _has_sufficient_volume(dataset: "DatasetSpec", threshold: float) -> Tuple[bool, float]:
    """Return whether the dataset meets the minimum volume requirement."""

    total = _dataset_total_volume(dataset)
    return total >= threshold, total


def _normalise_timeframe_value(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_htf_value(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "na", "off", "0"}:
        return None
    return text


def _group_datasets(
    datasets: Sequence[DatasetSpec],
) -> Tuple[Dict[Tuple[str, Optional[str]], List[DatasetSpec]], Dict[str, List[DatasetSpec]], Tuple[str, Optional[str]]]:
    groups: Dict[Tuple[str, Optional[str]], List[DatasetSpec]] = {}
    timeframe_groups: Dict[str, List[DatasetSpec]] = {}
    for dataset in datasets:
        key = (dataset.timeframe, dataset.htf_timeframe or None)
        groups.setdefault(key, []).append(dataset)
        timeframe_groups.setdefault(dataset.timeframe, []).append(dataset)

    if not groups:
        raise RuntimeError("No datasets available for optimisation")

    default_key = next(iter(groups))
    return groups, timeframe_groups, default_key


def _configure_parallel_workers(
    search_cfg: Dict[str, object],
    dataset_groups: Mapping[Tuple[str, Optional[str]], List[DatasetSpec]],
    *,
    available_cpu: int,
    n_jobs: int,
) -> Tuple[int, int, str, Optional[str]]:
    dataset_executor = str(search_cfg.get("dataset_executor", "thread") or "thread").lower()
    if dataset_executor not in {"thread", "process"}:
        LOGGER.warning(
            "알 수 없는 dataset_executor '%s' 가 지정되어 thread 모드로 대체합니다.",
            dataset_executor,
        )
        dataset_executor = "thread"

    dataset_start_method_raw = search_cfg.get("dataset_start_method")
    dataset_start_method = (
        str(dataset_start_method_raw).lower() if dataset_start_method_raw else None
    )

    max_parallel_datasets = max((len(group) for group in dataset_groups.values()), default=1)
    auto_dataset_jobs = min(max_parallel_datasets, max(1, available_cpu))

    legacy_dataset_jobs = search_cfg.get("dataset_n_jobs")
    if legacy_dataset_jobs is not None and "dataset_jobs" not in search_cfg:
        search_cfg["dataset_jobs"] = legacy_dataset_jobs

    search_cfg.setdefault("dataset_jobs", auto_dataset_jobs)

    raw_dataset_jobs = search_cfg.get("dataset_jobs")
    try:
        dataset_jobs = max(1, int(raw_dataset_jobs))
    except (TypeError, ValueError):
        LOGGER.warning(
            "search.dataset_jobs 값 '%s' 을 해석할 수 없어 %d로 대체합니다.",
            raw_dataset_jobs,
            auto_dataset_jobs,
        )
        dataset_jobs = auto_dataset_jobs

    dataset_jobs = min(dataset_jobs, max(1, available_cpu))

    dataset_parallel_capable = max_parallel_datasets > 1
    if not dataset_parallel_capable:
        if dataset_jobs != 1:
            LOGGER.info("단일 티커 구성으로 dataset_jobs %d→1로 비활성화합니다.", dataset_jobs)
        dataset_jobs = 1
    else:
        if dataset_jobs <= 1 and auto_dataset_jobs > 1:
            dataset_jobs = auto_dataset_jobs
            LOGGER.info(
                "데이터셋 병렬 worker %d개 자동 설정 (가용 CPU=%d, 최대 병렬=%d)",
                dataset_jobs,
                available_cpu,
                max_parallel_datasets,
            )
        elif dataset_jobs > auto_dataset_jobs:
            LOGGER.info(
                "데이터셋 병렬 worker 수를 %d→%d로 제한합니다. (최대 병렬=%d)",
                dataset_jobs,
                auto_dataset_jobs,
                max_parallel_datasets,
            )
            dataset_jobs = auto_dataset_jobs

    dataset_jobs = max(1, dataset_jobs)
    search_cfg["dataset_jobs"] = dataset_jobs

    if dataset_parallel_capable and dataset_jobs > 1:
        optuna_budget = max(1, available_cpu // dataset_jobs)
        if optuna_budget < n_jobs:
            LOGGER.info(
                "데이터셋 병렬(%d worker) 활성화로 Optuna worker %d→%d 조정",
                dataset_jobs,
                n_jobs,
                optuna_budget,
            )
            n_jobs = optuna_budget
            search_cfg["n_jobs"] = n_jobs
        LOGGER.info(
            "보조 데이터셋 병렬 worker %d개 (%s) 사용",
            dataset_jobs,
            dataset_executor,
        )
        if dataset_executor == "process" and dataset_start_method:
            LOGGER.info("프로세스 start method=%s", dataset_start_method)
    elif not dataset_parallel_capable:
        LOGGER.info(
            "단일 티커/데이터셋 구성이라 데이터셋 병렬화를 비활성화하고 Optuna worker %d개를 유지합니다.",
            n_jobs,
        )
        dataset_jobs = 1
    else:
        LOGGER.info(
            "설정상 데이터셋 병렬 worker 1개라 Optuna 병렬(worker=%d)만 사용합니다.",
            n_jobs,
        )
        dataset_jobs = 1

    LOGGER.info(
        "최종 병렬 전략: Optuna worker=%d (우선), 데이터셋 worker=%d (%s, 보조)",
        n_jobs,
        dataset_jobs,
        dataset_executor,
    )

    search_cfg["dataset_jobs"] = dataset_jobs

    return n_jobs, dataset_jobs, dataset_executor, dataset_start_method


def _select_datasets_for_params(
    params_cfg: Dict[str, object],
    dataset_groups: Dict[Tuple[str, Optional[str]], List[DatasetSpec]],
    timeframe_groups: Dict[str, List[DatasetSpec]],
    default_key: Tuple[str, Optional[str]],
    params: Dict[str, object],
) -> Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]:
    def _match(tf: str, htf: Optional[str]) -> Optional[Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]]:
        tf_lower = tf.lower()
        htf_lower = (htf or "").lower()
        for key, group in dataset_groups.items():
            key_tf, key_htf = key
            if key_tf.lower() != tf_lower:
                continue
            key_htf_lower = (key_htf or "").lower()
            if key_htf_lower == htf_lower:
                return key, group
        return None

    timeframe_value = (
        _normalise_timeframe_value(params.get("timeframe"))
        or _normalise_timeframe_value(params.get("ltf"))
        or _normalise_timeframe_value(params_cfg.get("timeframe"))
    )

    htf_value = (
        _normalise_htf_value(params.get("htf"))
        or _normalise_htf_value(params.get("htf_timeframe"))
    )

    if htf_value is None:
        cfg_htf = params_cfg.get("htf_timeframe")
        if cfg_htf:
            htf_value = _normalise_htf_value(cfg_htf)
        elif isinstance(params_cfg.get("htf_timeframes"), list) and len(params_cfg["htf_timeframes"]) == 1:
            htf_value = _normalise_htf_value(params_cfg["htf_timeframes"][0])

    if timeframe_value is None:
        timeframe_value = default_key[0]

    selected = None
    if timeframe_value:
        selected = _match(timeframe_value, htf_value)
        if selected is None and htf_value is not None:
            selected = _match(timeframe_value, None)
        if selected is None:
            for key, group in dataset_groups.items():
                if key[0].lower() == timeframe_value.lower():
                    selected = (key, group)
                    break

        if selected is None:
            for tf, group in timeframe_groups.items():
                if tf.lower() == timeframe_value.lower():
                    key = (group[0].timeframe, group[0].htf_timeframe or None)
                    selected = (key, group)
                    break

    if selected is None:
        selected = (default_key, dataset_groups[default_key])

    return selected


def _pick_primary_dataset(datasets: Sequence[DatasetSpec]) -> DatasetSpec:
    return max(datasets, key=lambda item: len(item.df))


def _resolve_symbol_entry(entry: object, alias_map: Dict[str, str]) -> Tuple[str, str]:
    """Normalise a symbol entry to a display name and a Binance fetch symbol."""

    if isinstance(entry, dict):
        alias = entry.get("alias") or entry.get("name") or entry.get("symbol") or entry.get("id") or ""
        resolved = entry.get("symbol") or entry.get("id") or alias
        alias = str(alias) if alias else str(resolved)
        resolved = str(resolved) if resolved else alias
    else:
        alias = str(entry)
        resolved = alias

    resolved = alias_map.get(alias, alias_map.get(resolved, resolved))
    if not alias:
        alias = resolved
    if not resolved:
        resolved = alias
    return alias, resolved


def _normalise_periods(
    periods_cfg: Optional[Iterable[Dict[str, object]]],
    base_period: Dict[str, object],
) -> List[Dict[str, str]]:
    periods: List[Dict[str, str]] = []
    if periods_cfg:
        for idx, raw in enumerate(periods_cfg):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Period entry #{idx + 1} must be a mapping with 'from'/'to' keys, got {type(raw).__name__}."
                )
            start = raw.get("from")
            end = raw.get("to")
            if not start or not end:
                raise ValueError(
                    f"Period entry #{idx + 1} is missing required 'from'/'to' values: {raw}."
                )
            periods.append({"from": str(start), "to": str(end)})

    if not periods:
        start = base_period.get("from") if isinstance(base_period, dict) else None
        end = base_period.get("to") if isinstance(base_period, dict) else None
        if start and end:
            periods.append({"from": str(start), "to": str(end)})

    return periods


def prepare_datasets(
    params_cfg: Dict[str, object],
    backtest_cfg: Dict[str, object],
    data_dir: Path,
) -> List[DatasetSpec]:
    data_cfg = backtest_cfg.get("data") if isinstance(backtest_cfg.get("data"), dict) else {}
    cache_root = Path(data_dir).expanduser()
    futures_flag = bool(backtest_cfg.get("futures", False))
    if data_cfg:
        market_text = str(data_cfg.get("market", "")).lower()
        if market_text == "futures":
            futures_flag = True
        elif market_text == "spot":
            futures_flag = False
        if "futures" in data_cfg:
            futures_flag = bool(data_cfg.get("futures"))
        cache_override = data_cfg.get("cache_dir")
        if cache_override:
            cache_root = Path(cache_override).expanduser()
    cache = DataCache(cache_root, futures=futures_flag)
    cache_info = DatasetCacheInfo(root=cache_root, futures=futures_flag)

    base_symbol = str(params_cfg.get("symbol")) if params_cfg.get("symbol") else ""
    base_timeframe = str(params_cfg.get("timeframe")) if params_cfg.get("timeframe") else ""
    base_period = params_cfg.get("backtest", {}) or {}

    alias_map: Dict[str, str] = {}
    for source in (backtest_cfg.get("symbol_aliases"), params_cfg.get("symbol_aliases")):
        if isinstance(source, dict):
            for key, value in source.items():
                if key and value:
                    alias_map[str(key)] = str(value)

    def _to_list(value: Optional[object]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if v]
        text = str(value)
        return [text] if text else []

    dataset_entries = backtest_cfg.get("datasets")
    if isinstance(dataset_entries, list) and dataset_entries:
        datasets: List[DatasetSpec] = []
        for entry in dataset_entries:
            if not isinstance(entry, dict):
                continue
            symbol_value = (
                entry.get("symbol")
                or entry.get("name")
                or entry.get("id")
                or entry.get("ticker")
            )
            if not symbol_value:
                raise ValueError("datasets 항목에 symbol 키가 필요합니다.")
            display_symbol, source_symbol = _resolve_symbol_entry(str(symbol_value), alias_map)

            ltf_candidates = _to_list(entry.get("ltf") or entry.get("ltfs") or entry.get("timeframes"))
            if not ltf_candidates:
                raise ValueError(f"{symbol_value} 데이터셋에 최소 하나의 ltf/timeframe 이 필요합니다.")

            start_value = entry.get("start") or entry.get("from") or base_period.get("from")
            end_value = entry.get("end") or entry.get("to") or base_period.get("to")
            if not start_value or not end_value:
                raise ValueError(f"{symbol_value} 데이터셋에 start/end 구간이 필요합니다.")
            start = str(start_value)
            end = str(end_value)

            symbol_log = (
                display_symbol if display_symbol == source_symbol else f"{display_symbol}→{source_symbol}"
            )
            for timeframe in ltf_candidates:
                timeframe_text = str(timeframe)
                LOGGER.info(
                    "Preparing dataset %s %s %s→%s (LTF only)",
                    symbol_log,
                    timeframe_text,
                    start,
                    end,
                )
                df = cache.get(source_symbol, timeframe_text, start, end)
                datasets.append(
                    DatasetSpec(
                        symbol=display_symbol,
                        timeframe=timeframe_text,
                        start=start,
                        end=end,
                        df=df,
                        htf=None,
                        htf_timeframe=None,
                        source_symbol=source_symbol,
                        cache_info=cache_info,
                    )
                )
        if not datasets:
            raise ValueError("backtest.datasets 설정에서 어떤 데이터셋도 생성되지 않았습니다.")
        return datasets

    symbols = backtest_cfg.get("symbols") or ([base_symbol] if base_symbol else [])
    timeframes = backtest_cfg.get("timeframes") or ([base_timeframe] if base_timeframe else [])
    if timeframes:

        def _tf_priority(tf: str) -> Tuple[int, float]:
            text = str(tf).strip().lower()
            if text == "1m":
                return (0, 1.0)
            if text.endswith("m"):
                try:
                    minutes = float(text[:-1])
                except ValueError:
                    minutes = float("inf")
                return (1, minutes)
            return (2, float("inf"))

        timeframes = sorted(dict.fromkeys(timeframes), key=_tf_priority)
    periods = _normalise_periods(backtest_cfg.get("periods"), base_period)

    if not symbols or not timeframes or not periods:
        raise ValueError(
            "Backtest configuration must specify symbol(s), timeframe(s), and at least one period with 'from'/'to' dates."
        )

    symbol_pairs = [_resolve_symbol_entry(symbol, alias_map) for symbol in symbols]

    datasets: List[DatasetSpec] = []
    for (display_symbol, source_symbol), timeframe, period in product(
        symbol_pairs, timeframes, periods
    ):
        start = str(period["from"])
        end = str(period["to"])
        symbol_log = (
            display_symbol if display_symbol == source_symbol else f"{display_symbol}→{source_symbol}"
        )
        LOGGER.info(
            "Preparing dataset %s %s %s→%s (LTF only)",
            symbol_log,
            timeframe,
            start,
            end,
        )
        df = cache.get(source_symbol, timeframe, start, end)
        datasets.append(
            DatasetSpec(
                symbol=display_symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                df=df,
                htf=None,
                htf_timeframe=None,
                source_symbol=source_symbol,
                cache_info=cache_info,
            )
        )
    return datasets


def combine_metrics(
    metric_list: List[Dict[str, float]], *, simple_override: Optional[bool] = None
) -> Dict[str, float]:
    if not metric_list:
        return {}

    simple_mode = bool(simple_override)

    if simple_override is None:
        for metrics in metric_list:
            if bool(metrics.get("SimpleMetricsOnly")):
                simple_mode = True
                break

    combined_returns: List[pd.Series] = []
    combined_trades: List[Trade] = [] if not simple_mode else []
    anomaly_flags: List[str] = []
    lossless_detected = False

    def _flag_lossless(target: Dict[str, float]) -> None:
        nonlocal lossless_detected, anomaly_flags
        result = _apply_lossless_anomaly(target)
        if not result:
            return

        flag, trades_val, wins_val, abs_loss, threshold = result
        lossless_detected = True
        if flag not in anomaly_flags:
            anomaly_flags.append(flag)
            if flag == LOSSLESS_ANOMALY_FLAG:
                LOGGER.info(
                    "손실 거래가 없는 결과(trades=%d, wins=%d)로 ProfitFactor='overfactor' 및 DisplayedProfitFactor=0으로 표기합니다.",
                    int(trades_val),
                    int(wins_val),
                )
            elif flag == MICRO_LOSS_ANOMALY_FLAG:
                LOGGER.warning(
                    "미세 손실 %.6g (임계값 %.6g 이하)로 DisplayedProfitFactor=0으로 고정합니다. trades=%d, wins=%d",
                    abs_loss,
                    threshold,
                    int(trades_val),
                    int(wins_val),
                )
            else:
                LOGGER.warning(
                    "DisplayedProfitFactor=0으로 처리한 특이 케이스(flag=%s)를 감지했습니다. trades=%d, wins=%d",
                    flag,
                    int(trades_val),
                    int(wins_val),
                )

    def _coerce_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    total_gross_profit = 0.0
    total_gross_loss = 0.0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    hold_weight_sum = 0.0
    max_consecutive_losses = 0
    valid_flag = True
    initial_balance_value: Optional[float] = None
    threshold_override: Optional[float] = None

    for metrics in metric_list:
        returns = metrics.get("Returns")
        if isinstance(returns, pd.Series):
            combined_returns.append(returns)

        valid_flag = valid_flag and bool(metrics.get("Valid", True))

        flags_value = metrics.get("AnomalyFlags")
        if isinstance(flags_value, str):
            tokens = [token.strip() for token in flags_value.split(",") if token.strip()]
        elif isinstance(flags_value, (list, tuple)):
            tokens = [str(token) for token in flags_value if str(token)]
        else:
            tokens = []
        for token in tokens:
            if token not in anomaly_flags:
                anomaly_flags.append(token)

        if initial_balance_value is None:
            for key in _INITIAL_BALANCE_KEYS:
                candidate = metrics.get(key)
                if candidate is None:
                    continue
                try:
                    numeric = float(candidate)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric) or numeric == 0:
                    continue
                initial_balance_value = float(numeric)
                break

        if threshold_override is None and metrics.get("LosslessGrossLossThreshold") is not None:
            try:
                candidate_threshold = float(metrics["LosslessGrossLossThreshold"])
            except (TypeError, ValueError):
                candidate_threshold = None
            else:
                if np.isfinite(candidate_threshold) and candidate_threshold > 0:
                    threshold_override = float(candidate_threshold)

        if simple_mode:
            trades_count = int(_coerce_float(metrics.get("Trades")))
            wins_count = int(_coerce_float(metrics.get("Wins")))
            losses_count = int(_coerce_float(metrics.get("Losses")))
            gross_profit = _coerce_float(metrics.get("GrossProfit"))
            gross_loss = _coerce_float(metrics.get("GrossLoss"))
            avg_hold = _coerce_float(metrics.get("AvgHoldBars"))
            streak = int(_coerce_float(metrics.get("MaxConsecutiveLosses")))

            trades_list = metrics.get("TradesList")
            if isinstance(trades_list, list) and trades_list:
                local_wins = 0
                local_losses = 0
                local_gross_profit = 0.0
                local_gross_loss = 0.0
                hold_sum = 0.0
                current_streak = 0
                worst_streak = 0
                for trade in trades_list:
                    profit = _coerce_float(getattr(trade, "profit", 0.0))
                    if profit > 0:
                        local_gross_profit += profit
                        local_wins += 1
                        current_streak = 0
                    elif profit < 0:
                        local_gross_loss += profit
                        local_losses += 1
                        current_streak += 1
                        worst_streak = max(worst_streak, current_streak)
                    else:
                        current_streak = 0
                    hold_sum += _coerce_float(getattr(trade, "bars_held", 0))

                if trades_count == 0:
                    trades_count = len(trades_list)
                if wins_count == 0 and local_wins:
                    wins_count = local_wins
                if losses_count == 0 and local_losses:
                    losses_count = local_losses
                if gross_profit == 0.0 and local_gross_profit:
                    gross_profit = local_gross_profit
                if gross_loss == 0.0 and local_gross_loss:
                    gross_loss = local_gross_loss
                if avg_hold == 0.0 and trades_count:
                    avg_hold = hold_sum / trades_count if trades_count else 0.0
                if streak == 0 and worst_streak:
                    streak = worst_streak

            total_trades += trades_count
            total_wins += wins_count
            total_losses += losses_count
            total_gross_profit += gross_profit
            total_gross_loss += gross_loss
            hold_weight_sum += avg_hold * trades_count
            max_consecutive_losses = max(max_consecutive_losses, streak)
        else:
            trades = metrics.get("TradesList")
            if isinstance(trades, list):
                combined_trades.extend(trades)

    merged_returns = (
        pd.concat(combined_returns, axis=0).sort_index() if combined_returns else pd.Series(dtype=float)
    )

    if simple_mode:
        returns_clean = merged_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if returns_clean.empty:
            net_profit = 0.0
        else:
            equity = equity_curve_from_returns(returns_clean, initial=1.0)
            net_profit = (
                float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0])
                if len(equity) > 1
                else 0.0
            )
        if returns_clean.empty and not net_profit:
            fallback = [
                metrics.get("NetProfit") if metrics.get("NetProfit") is not None else metrics.get("TotalReturn")
                for metrics in metric_list
            ]
            fallback = [float(value) for value in fallback if value is not None]
            if fallback:
                net_profit = float(np.mean(fallback))

        denominator = max(abs(total_gross_loss), EPS)
        profit_factor_value = float(total_gross_profit / denominator) if denominator else 0.0
        if not np.isfinite(profit_factor_value):
            profit_factor_value = 0.0

        if total_trades > 0 and hold_weight_sum > 0:
            avg_hold = hold_weight_sum / total_trades
        else:
            avg_hold = 0.0

        win_rate = float(total_wins / total_trades) if total_trades else 0.0

        aggregated: Dict[str, float] = {
            "NetProfit": net_profit,
            "TotalReturn": net_profit,
            "ProfitFactor": profit_factor_value,
            "Trades": float(total_trades),
            "Wins": float(total_wins),
            "Losses": float(total_losses),
            "GrossProfit": float(total_gross_profit),
            "GrossLoss": float(total_gross_loss),
            "AvgHoldBars": float(avg_hold),
            "WinRate": win_rate,
            "MaxConsecutiveLosses": float(max_consecutive_losses),
        }
        aggregated["SimpleMetricsOnly"] = True
        effective_threshold = None
        if initial_balance_value is not None:
            aggregated["InitialCapital"] = float(initial_balance_value)
            aggregated.setdefault("InitialEquity", float(initial_balance_value))
            base_threshold = abs(float(initial_balance_value)) * LOSSLESS_GROSS_LOSS_PCT
            effective_threshold = base_threshold
            if threshold_override is not None:
                effective_threshold = max(float(threshold_override), base_threshold)
        elif threshold_override is not None:
            effective_threshold = float(threshold_override)
        if effective_threshold is not None:
            aggregated["LosslessGrossLossThreshold"] = float(effective_threshold)
        _flag_lossless(aggregated)
    else:
        combined_trades.sort(
            key=lambda trade: (
                getattr(trade, "entry_time", None),
                getattr(trade, "exit_time", None),
            )
        )
        aggregated = aggregate_metrics(combined_trades, merged_returns, simple=False)
        effective_threshold = None
        if initial_balance_value is not None:
            aggregated["InitialCapital"] = float(initial_balance_value)
            aggregated.setdefault("InitialEquity", float(initial_balance_value))
            base_threshold = abs(float(initial_balance_value)) * LOSSLESS_GROSS_LOSS_PCT
            effective_threshold = base_threshold
            if threshold_override is not None:
                effective_threshold = max(float(threshold_override), base_threshold)
        elif threshold_override is not None:
            effective_threshold = float(threshold_override)
        if effective_threshold is not None:
            aggregated["LosslessGrossLossThreshold"] = float(effective_threshold)
        _flag_lossless(aggregated)

    aggregated["Trades"] = int(aggregated.get("Trades", 0))
    aggregated["Wins"] = int(aggregated.get("Wins", 0))
    aggregated["Losses"] = int(aggregated.get("Losses", 0))

    def _first_finite_value(key: str) -> Optional[float]:
        for metrics in metric_list:
            if key not in metrics:
                continue
            try:
                value = float(metrics[key])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            return float(value)
        return None

    penalty_keys = {"TradePenalty", "HoldPenalty", "ConsecutiveLossPenalty"}
    requirement_keys = {"MinTrades", "MinHoldBars", "MaxConsecutiveLossLimit"}

    for key in sorted(penalty_keys | requirement_keys):
        value = _first_finite_value(key)
        if value is None:
            if key in penalty_keys or key in requirement_keys:
                aggregated.setdefault(key, 0.0)
            continue
        if key in penalty_keys:
            value = abs(value)
        aggregated[key] = float(max(0.0, value))

    aggregated["Valid"] = valid_flag and not lossless_detected
    if anomaly_flags:
        aggregated["AnomalyFlags"] = list(dict.fromkeys(anomaly_flags))
    return aggregated


def compute_score_pf_basic(
    metrics: Dict[str, object], constraints: Optional[Dict[str, object]] = None
) -> float:
    """ProfitFactor 중심 기본 점수를 계산합니다."""

    constraints = constraints or {}

    def _as_float(value: object, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(number):
            return default
        return number

    pf_source = metrics.get("ProfitFactor")
    pf = _as_float(pf_source, 0.0)
    if isinstance(pf_source, str):
        pf = _as_float(metrics.get("DisplayedProfitFactor"), pf)

    dd_raw = metrics.get("MaxDD")
    if dd_raw is None:
        dd_raw = metrics.get("MaxDrawdown")
    dd_value = abs(_as_float(dd_raw, 0.0))
    dd_pct = dd_value * 100.0 if dd_value <= 1.0 else dd_value

    trades_raw = metrics.get("Trades")
    if trades_raw is None:
        trades_raw = metrics.get("TotalTrades")
    trades = int(round(_as_float(trades_raw, 0.0)))

    min_trades = int(round(_as_float(constraints.get("min_trades_test"), 12.0)))
    max_dd = _as_float(constraints.get("max_dd_pct"), 70.0)

    base = pf

    if trades < min_trades:
        base -= (min_trades - trades) * 0.2

    if dd_pct > max_dd:
        base -= (dd_pct - max_dd) * 0.05

    return base


def _clean_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    clean: Dict[str, object] = {}
    for key, value in metrics.items():
        if key == "DisplayedProfitFactor":
            continue
        if isinstance(value, (int, float, bool, str)):
            if key in {"ProfitFactor", "LosslessProfitFactorValue"} and isinstance(value, (int, float)):
                clean[key] = f"{float(value):.3f}"
            else:
                clean[key] = value
        elif isinstance(value, (list, tuple)):
            if all(isinstance(item, (int, float, bool, str)) for item in value):
                clean[key] = ", ".join(str(item) for item in value)
    return clean


def _create_pruner(name: str, params: Dict[str, object]) -> optuna.pruners.BasePruner:
    name = (name or "asha").lower()
    params = params or {}
    if name in {"none", "nop", "off"}:
        return optuna.pruners.NopPruner()
    if name in {"median", "medianpruner"}:
        return optuna.pruners.MedianPruner(**params)
    if name in {"hyperband"}:
        return optuna.pruners.HyperbandPruner(**params)
    if name in {"threshold", "thresholdpruner"}:
        return optuna.pruners.ThresholdPruner(**params)
    if name in {"patient", "patientpruner"}:
        patience = int(params.get("patience", 10))
        wrapped = _create_pruner(params.get("wrapped", "nop"), params.get("wrapped_params", {}))
        return optuna.pruners.PatientPruner(wrapped, patience=patience)
    if name in {"wilcoxon", "wilcoxonpruner"}:
        return optuna.pruners.WilcoxonPruner(**params)
    # Default to ASHA / successive halving
    return optuna.pruners.SuccessiveHalvingPruner(**params)


def optimisation_loop(
    datasets: List[DatasetSpec],
    params_cfg: Dict[str, object],
    objectives: Iterable[object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    forced_params: Optional[Dict[str, object]] = None,
    *,
    study_storage: Optional[Path] = None,
    space_hash: Optional[str] = None,
    seed_trials: Optional[List[Dict[str, object]]] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, object]:
    search_cfg = params_cfg.get("search", {})
    objective_specs: List[ObjectiveSpec] = normalise_objectives(objectives)
    if not objective_specs:
        objective_specs = [ObjectiveSpec(name="NetProfit")]
    multi_objective = bool(search_cfg.get("multi_objective", False)) and len(objective_specs) > 1
    directions = [spec.direction for spec in objective_specs]
    original_space = build_space(params_cfg.get("space", {}))

    basic_profile_flag = _coerce_bool_or_none(search_cfg.get("basic_factor_profile"))
    if basic_profile_flag is None:
        basic_profile_flag = _coerce_bool_or_none(search_cfg.get("use_basic_factors"))
    use_basic_factors = True if basic_profile_flag is None else basic_profile_flag

    allow_sqlite_parallel_flag = _coerce_bool_or_none(
        search_cfg.get("allow_sqlite_parallel")
    )
    allow_sqlite_parallel = (
        bool(allow_sqlite_parallel_flag)
        if allow_sqlite_parallel_flag is not None
        else False
    )

    space = _restrict_to_basic_factors(original_space, enabled=use_basic_factors)
    param_order = list(space.keys())
    if use_basic_factors:
        if len(space) != len(original_space):
            LOGGER.info(
                "기본 팩터 프로파일: %d→%d개 파라미터로 탐색 공간을 축소합니다.",
                len(original_space),
                len(space),
            )
            if not space:
                LOGGER.warning(
                    "기본 팩터 집합에 해당하는 항목이 없어 탐색 공간이 비었습니다."
                    " space 설정을 점검하세요."
                )
    else:
        LOGGER.info(
            "기본 팩터 프로파일 비활성화: 전체 %d개 파라미터 탐색", len(space)
        )

    params_cfg["space"] = space

    dataset_groups, timeframe_groups, default_key = _group_datasets(datasets)

    available_cpu = max(1, multiprocessing.cpu_count())

    raw_n_jobs = search_cfg.get("n_jobs", 1)
    try:
        n_jobs = max(1, int(raw_n_jobs))
    except (TypeError, ValueError):
        LOGGER.warning("search.n_jobs 값 '%s' 을 해석할 수 없어 1로 대체합니다.", raw_n_jobs)
        n_jobs = 1
    force_sqlite_serial = bool(search_cfg.get("force_sqlite_serial"))
    if force_sqlite_serial and n_jobs != 1:
        LOGGER.info("SQLite 직렬 강제 옵션으로 Optuna worker %d→1개 조정", n_jobs)
        n_jobs = 1
        search_cfg["n_jobs"] = n_jobs
    if n_trials := int(search_cfg.get("n_trials", 0) or 0):
        auto_jobs = max(1, min(available_cpu, n_trials))
    else:
        auto_jobs = max(1, available_cpu)
    if not force_sqlite_serial and n_jobs <= 1 and auto_jobs > n_jobs:
        n_jobs = auto_jobs
        search_cfg["n_jobs"] = n_jobs
        LOGGER.info("가용 자원에 맞춰 Optuna worker %d개를 자동 할당했습니다.", n_jobs)

    if n_jobs > 1:
        LOGGER.info("Optuna 병렬 worker %d개를 사용합니다.", n_jobs)
    (
        n_jobs,
        dataset_jobs,
        dataset_executor,
        dataset_start_method,
    ) = _configure_parallel_workers(
        search_cfg,
        dataset_groups,
        available_cpu=available_cpu,
        n_jobs=n_jobs,
    )

    algo_raw = search_cfg.get("algo", "bayes")
    algo = str(algo_raw or "bayes").lower()
    seed = search_cfg.get("seed")
    n_trials = int(search_cfg.get("n_trials", 50))
    forced_params = forced_params or {}
    log_dir_path: Optional[Path] = Path(log_dir) if log_dir else None
    trial_log_path: Optional[Path] = None
    best_yaml_path: Optional[Path] = None
    final_csv_path: Optional[Path] = None
    trial_csv_path: Optional[Path] = None
    if log_dir_path:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        trial_log_path = log_dir_path / "trials.jsonl"
        best_yaml_path = log_dir_path / "best.yaml"
        final_csv_path = log_dir_path / "trials_final.csv"
        trial_csv_path = log_dir_path / "trials_progress.csv"
        for candidate in (trial_log_path, best_yaml_path, final_csv_path, trial_csv_path):
            if candidate.exists():
                candidate.unlink()
    non_finite_penalty = float(search_cfg.get("non_finite_penalty", NON_FINITE_PENALTY))
    constraints_raw = params_cfg.get("constraints")
    constraints_cfg = dict(constraints_raw) if isinstance(constraints_raw, dict) else {}
    if not constraints_cfg:
        backtest_constraints = backtest_cfg.get("constraints")
        if isinstance(backtest_constraints, dict):
            constraints_cfg = dict(backtest_constraints)
    llm_cfg = params_cfg.get("llm", {}) if isinstance(params_cfg.get("llm"), dict) else {}

    nsga_params_cfg = search_cfg.get("nsga_params") or {}
    nsga_kwargs: Dict[str, object] = {}
    population_override = nsga_params_cfg.get("population_size") or search_cfg.get("nsga_population")
    if population_override is not None:
        try:
            nsga_kwargs["population_size"] = int(population_override)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid nsga population size '%s'; using Optuna default", population_override)
    elif multi_objective:
        space_size = len(space) if hasattr(space, "__len__") else 0
        nsga_kwargs["population_size"] = max(64, (space_size or 0) * 2 or 64)
    if nsga_params_cfg.get("mutation_prob") is not None:
        try:
            nsga_kwargs["mutation_prob"] = float(nsga_params_cfg["mutation_prob"])
        except (TypeError, ValueError):
            LOGGER.warning("Invalid nsga mutation_prob '%s'; ignoring", nsga_params_cfg["mutation_prob"])
    if nsga_params_cfg.get("crossover_prob") is not None:
        try:
            nsga_kwargs["crossover_prob"] = float(nsga_params_cfg["crossover_prob"])
        except (TypeError, ValueError):
            LOGGER.warning("Invalid nsga crossover_prob '%s'; ignoring", nsga_params_cfg["crossover_prob"])
    if nsga_params_cfg.get("swap_step") is not None:
        try:
            nsga_kwargs["swap_step"] = int(nsga_params_cfg["swap_step"])
        except (TypeError, ValueError):
            LOGGER.warning("Invalid nsga swap_step '%s'; ignoring", nsga_params_cfg["swap_step"])
    if seed is not None:
        try:
            nsga_kwargs["seed"] = int(seed)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid seed '%s'; ignoring for NSGA-II", seed)

    use_nsga = algo in {"nsga", "nsga2", "nsgaii"}
    if multi_objective and not use_nsga and algo in {"bayes", "tpe", "default", "auto"}:
        use_nsga = True

    if algo == "grid":
        sampler = optuna.samplers.GridSampler(grid_choices(space))
    elif algo == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif algo in {"cmaes", "cma-es", "cma"}:
        sampler = optuna.samplers.CmaEsSampler(seed=seed, consider_pruned_trials=True)
    elif use_nsga:
        sampler = optuna.samplers.NSGAIISampler(**nsga_kwargs)
    else:
        # Instantiate TPESampler without experimental arguments.  Passing
        # multivariate=True or group=True triggers ExperimentalWarning and may
        # change algorithm behaviour.  Use default settings to avoid
        # experimental features and suppress related warnings.
        sampler = optuna.samplers.TPESampler(seed=seed)

    pruner_cfg = str(search_cfg.get("pruner", "asha"))
    pruner_params = search_cfg.get("pruner_params", {})
    pruner = _create_pruner(pruner_cfg, pruner_params or {})

    storage_cfg = search_cfg.get("storage_url")
    storage_env_key = search_cfg.get("storage_url_env")
    storage_env_value = os.getenv(str(storage_env_key)) if storage_env_key else None

    storage_url = None
    if storage_env_value:
        storage_url = str(storage_env_value)
    elif storage_cfg:
        storage_url = str(storage_cfg)
    elif study_storage is not None:
        study_storage.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{study_storage}"

    study_name = search_cfg.get("study_name") or (space_hash[:12] if space_hash else None)

    storage: Optional[optuna.storages.RDBStorage]
    storage = None
    storage_meta = {
        "backend": None,
        "url": None,
        "path": None,
        "env_key": storage_env_key,
        "env_value_present": bool(storage_env_value),
    }
    if storage_url:
        # When creating storages, avoid setting heartbeat_interval or grace_period
        # to prevent Optuna experimental warnings.  Leave these as None so that
        # Optuna uses its defaults without emitting ExperimentalWarning.
        heartbeat_interval = None
        heartbeat_grace = None
        if storage_url.startswith("sqlite:///"):
            timeout_raw = search_cfg.get("sqlite_timeout", 120)
            try:
                sqlite_timeout = max(1, int(timeout_raw))
            except (TypeError, ValueError):
                LOGGER.warning(
                    "sqlite_timeout 값 '%s' 을 정수로 변환할 수 없어 120초로 대체합니다.",
                    timeout_raw,
                )
                sqlite_timeout = 120
            storage = _make_sqlite_storage(
                storage_url,
                timeout_sec=sqlite_timeout,
                heartbeat_interval=None,
                grace_period=None,
            )
            storage_meta["backend"] = "sqlite"
            storage_meta["url"] = storage_url
            storage_meta["allow_parallel"] = allow_sqlite_parallel
            try:
                storage_path = make_url(storage_url).database
            except Exception:
                storage_path = None
            if storage_path:
                storage_meta["path"] = storage_path
            elif study_storage is not None:
                storage_meta["path"] = str(study_storage)
            if n_jobs > 1:
                if allow_sqlite_parallel:
                    LOGGER.warning(
                        "SQLite 병렬 허용 옵션이 활성화되었습니다. Optuna worker %d개를 유지하지만 잠금"
                        " 충돌이 발생할 수 있습니다.",
                        n_jobs,
                    )
                else:
                    LOGGER.warning(
                        "SQLite 스토리지에서 Optuna worker %d개를 병렬로 실행합니다. 잠금 충돌 시 자동 재시도로 복구를 시도하니 모니터링하세요.",
                        n_jobs,
                    )
        else:
            pool_size = _coerce_config_int(
                search_cfg.get("storage_pool_size"),
                minimum=1,
                name="storage_pool_size",
            )
            max_overflow = _coerce_config_int(
                search_cfg.get("storage_max_overflow"),
                minimum=0,
                name="storage_max_overflow",
            )
            pool_timeout = _coerce_config_int(
                search_cfg.get("storage_pool_timeout"),
                minimum=0,
                name="storage_pool_timeout",
            )
            pool_recycle = _coerce_config_int(
                search_cfg.get("storage_pool_recycle"),
                minimum=0,
                name="storage_pool_recycle",
            )
            connect_timeout = _coerce_config_int(
                search_cfg.get("storage_connect_timeout"),
                minimum=1,
                name="storage_connect_timeout",
            )
            statement_timeout_ms = _coerce_config_int(
                search_cfg.get("storage_statement_timeout_ms"),
                minimum=1,
                name="storage_statement_timeout_ms",
            )

            isolation_level_raw = search_cfg.get("storage_isolation_level")
            isolation_level = None
            if isinstance(isolation_level_raw, str):
                isolation_level = isolation_level_raw.strip() or None
            elif isolation_level_raw is not None:
                isolation_level = str(isolation_level_raw).strip() or None

            storage = _make_rdb_storage(
                storage_url,
                heartbeat_interval=None,
                grace_period=None,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                isolation_level=isolation_level,
                connect_timeout=connect_timeout,
                statement_timeout_ms=statement_timeout_ms,
            )
            storage_meta["backend"] = "rdb"
            storage_meta["url"] = storage_url
            pool_meta = {}
            if pool_size is not None:
                pool_meta["size"] = pool_size
            if max_overflow is not None:
                pool_meta["max_overflow"] = max_overflow
            if pool_timeout is not None:
                pool_meta["timeout"] = pool_timeout
            if pool_recycle is not None:
                pool_meta["recycle"] = pool_recycle
            if pool_meta:
                storage_meta["pool"] = pool_meta
            if connect_timeout is not None:
                storage_meta["connect_timeout"] = connect_timeout
            if isolation_level:
                storage_meta["isolation_level"] = isolation_level
            if statement_timeout_ms is not None:
                storage_meta["statement_timeout_ms"] = statement_timeout_ms
    else:
        storage_meta["backend"] = "none"
    storage_arg = storage if storage is not None else storage_url

    study_kwargs = dict(
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage_arg,
        load_if_exists=bool(storage_arg),
    )
    if multi_objective:
        study = optuna.create_study(directions=directions, **study_kwargs)
    else:
        study = optuna.create_study(direction="maximize", **study_kwargs)
    if space_hash:
        study.set_user_attr("space_hash", space_hash)

    for params in seed_trials or []:
        trial_params = dict(params)
        trial_params.update(forced_params)
        try:
            study.enqueue_trial(trial_params, skip_if_exists=True)
        except Exception:
            continue

    results: List[Dict[str, object]] = []
    results_lock = Lock()

    def _to_native(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        with TRIAL_LOG_WRITE_LOCK:
            def _normalise_value(value: object) -> Optional[object]:
                if value is None:
                    return None
                if isinstance(value, AbcSequence) and not isinstance(value, (str, bytes, bytearray)):
                    normalised: List[float] = []
                    for item in value:
                        try:
                            normalised.append(float(item))
                        except Exception:
                            return None
                    return normalised
                try:
                    return float(value)
                except Exception:
                    return None

            trial_value = _normalise_value(trial.value)
            record = {
                "number": trial.number,
                "value": trial_value,
                "state": str(trial.state),
                "params": {key: _to_native(val) for key, val in trial.params.items()},
                "datetime_complete": str(trial.datetime_complete) if trial.datetime_complete else None,
            }

            if trial_log_path is not None:
                trial_log_path.parent.mkdir(parents=True, exist_ok=True)
                with trial_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            metrics_attr = trial.user_attrs.get("metrics")
            metrics = metrics_attr if isinstance(metrics_attr, dict) else {}

            def _metric_value(name: str) -> Optional[object]:
                if name not in metrics:
                    return None
                return _to_native(metrics.get(name))

            dataset_key = trial.user_attrs.get("dataset_key")
            dataset_meta = dataset_key if isinstance(dataset_key, dict) else {}
            skipped_attr = trial.user_attrs.get("skipped_datasets")
            if isinstance(skipped_attr, list):
                skipped_serialisable = skipped_attr
            else:
                skipped_serialisable = [skipped_attr] if skipped_attr else []

            max_dd_value = _metric_value("MaxDD")
            if max_dd_value is None:
                max_dd_value = _metric_value("MaxDrawdown")

            value_field: object
            if isinstance(trial_value, list):
                value_field = json.dumps(trial_value, ensure_ascii=False)
            else:
                value_field = trial_value

            params_json = json.dumps(record["params"], ensure_ascii=False, sort_keys=True)
            skipped_json = (
                json.dumps(skipped_serialisable, ensure_ascii=False)
                if skipped_serialisable
                else ""
            )

            # Extract metrics for the trial.  Sortino and ProfitFactor are
            # explicitly captured to support ordered CSV logging.
            sortino_val = _metric_value("Sortino")
            if sortino_val is None:
                sortino_val = trial.user_attrs.get("sortino")
            pf_val = _metric_value("ProfitFactor")
            if pf_val is None:
                pf_val = trial.user_attrs.get("profit_factor")
            lossless_pf_val = _metric_value("LosslessProfitFactorValue")
            if lossless_pf_val is None:
                lossless_pf_val = trial.user_attrs.get("lossless_profit_factor_value")

            row = {
                "number": trial.number,
                "state": str(trial.state),
                "value": value_field,
                "score": trial.user_attrs.get("score"),
                "sortino": sortino_val,
                "profit_factor": pf_val,
                "lossless_profit_factor_value": lossless_pf_val,
                "trades": _metric_value("Trades"),
                "win_rate": _metric_value("WinRate"),
                "max_dd": max_dd_value,
                "valid": trial.user_attrs.get("valid"),
                "timeframe": dataset_meta.get("timeframe"),
                "htf_timeframe": dataset_meta.get("htf_timeframe"),
                "pruned": trial.user_attrs.get("pruned"),
                "params": params_json,
                "skipped_datasets": skipped_json,
                "datetime_complete": record["datetime_complete"],
            }

            if trial_csv_path is not None:
                file_exists = trial_csv_path.exists()
                trial_csv_path.parent.mkdir(parents=True, exist_ok=True)
                with trial_csv_path.open("a", encoding="utf-8", newline="") as csv_handle:
                    writer = csv.DictWriter(csv_handle, fieldnames=TRIAL_PROGRESS_FIELDS)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)

            try:
                trials_snapshot = study.get_trials(deepcopy=False)
            except TypeError:
                trials_snapshot = study.trials
            completed = sum(
                1
                for item in trials_snapshot
                if item.state in {TrialState.COMPLETE, TrialState.PRUNED}
            )
            total_display: object = n_trials if n_trials else len(trials_snapshot)
            # Display Sortino and ProfitFactor for progress logs in order of importance
            sortino_display = row.get("sortino")
            if sortino_display in {None, ""}:
                sortino_display = trial.user_attrs.get("sortino")
            if sortino_display in {None, ""}:
                sortino_display = "-"

            pf_display = row.get("profit_factor")
            if pf_display in {None, ""}:
                pf_display = trial.user_attrs.get("profit_factor")
            if pf_display in {None, ""}:
                pf_display = "-"
            if (
                isinstance(pf_display, str)
                and pf_display.strip().lower() == "overfactor"
                and lossless_pf_val not in {None, ""}
            ):
                pf_display = f"overfactor(원본={lossless_pf_val})"
            elif pf_display in {None, ""} and lossless_pf_val not in {None, ""}:
                pf_display = lossless_pf_val
            trades_display = row.get("trades") if row.get("trades") not in {None, ""} else "-"
            score_display = row.get("score") if row.get("score") not in {None, ""} else "-"
            LOGGER.info(
                "작업 진행상황 ＝＝＝＝＝＝ %d/%s (Trial %d %s) Sortino=%s, ProfitFactor=%s, Score=%s, Trades=%s",
                completed,
                total_display,
                trial.number,
                row.get("state"),
                sortino_display,
                pf_display,
                score_display,
                trades_display,
            )

            if best_yaml_path is None:
                return
            best_yaml_path.parent.mkdir(parents=True, exist_ok=True)

            selected_trial: Optional[optuna.trial.FrozenTrial]
            if multi_objective:
                try:
                    pareto_trials = list(study.best_trials)
                except ValueError:
                    return
                if not pareto_trials:
                    return
                selected_trial = next(
                    (best_trial for best_trial in pareto_trials if best_trial.number == trial.number),
                    None,
                )
                if selected_trial is None:
                    return
            else:
                try:
                    selected_trial = study.best_trial
                except ValueError:
                    return
                if selected_trial.number != trial.number:
                    return

            best_value = _normalise_value(selected_trial.value)
            best_params_full = {key: _to_native(val) for key, val in selected_trial.params.items()}
            snapshot = {
                "best_value": best_value,
                "best_params": best_params_full,
            }
            if use_basic_factors:
                snapshot["basic_params"] = {
                    key: value for key, value in best_params_full.items() if key in BASIC_FACTOR_KEYS
                }
            else:
                snapshot["basic_params"] = dict(best_params_full)
            with best_yaml_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)
    callbacks: List = [_log_trial]

    def objective(trial: optuna.Trial) -> float:
        params = _safe_sample_parameters(trial, space)
        params.update(forced_params)
        key, selected_datasets = _select_datasets_for_params(
            params_cfg, dataset_groups, timeframe_groups, default_key, params
        )
        trial.set_user_attr(
            "dataset_key",
            {"timeframe": key[0], "htf_timeframe": key[1]},
        )
        dataset_metrics: List[Dict[str, object]] = []
        numeric_metrics: List[Dict[str, float]] = []
        dataset_scores: List[float] = []
        skipped_dataset_records: List[Dict[str, object]] = []

        def _safe_float(value: object) -> Optional[float]:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric):
                return None
            return float(numeric)

        def _coerce_int(value: object) -> Optional[int]:
            numeric = _safe_float(value)
            if numeric is None:
                return None
            return int(round(numeric))

        def _resolve_min_volume_threshold() -> float:
            candidates: List[object] = [
                params.get("min_volume"),
                params.get("minVolume"),
            ]
            if isinstance(risk, dict):
                candidates.extend([risk.get("min_volume"), risk.get("minVolume")])
            candidates.append(constraints_cfg.get("min_volume"))

            for candidate in candidates:
                numeric = _safe_float(candidate)
                if numeric is None or numeric < 0:
                    continue
                return float(numeric)
            return MIN_VOLUME_THRESHOLD

        def _sanitise(value: float, stage: str) -> float:
            try:
                numeric = float(value)
            except Exception:
                numeric = non_finite_penalty
            if not np.isfinite(numeric):
                LOGGER.warning(
                    "Non-finite %s score detected for trial %s; applying penalty %.0e",
                    stage,
                    trial.number,
                    non_finite_penalty,
                )
                return non_finite_penalty
            return numeric

        def _consume_dataset(
            idx: int,
            dataset: DatasetSpec,
            metrics: Dict[str, float],
            *,
            simple_override: bool = False,
        ) -> None:
            cleaned_metrics = _clean_metrics(metrics)
            dataset_entry: Dict[str, object] = {
                "name": dataset.name,
                "meta": dataset.meta,
                "metrics": cleaned_metrics,
            }

            lossless_info = _apply_lossless_anomaly(metrics)
            if lossless_info:
                _apply_lossless_anomaly(cleaned_metrics)
                flag, trades_val, wins_val, abs_loss, threshold = lossless_info
                if flag == LOSSLESS_ANOMALY_FLAG:
                    LOGGER.info(
                        "데이터셋 %s 에서 손실 거래가 없는 결과(trades=%d, wins=%d)로 ProfitFactor='overfactor' 및 DisplayedProfitFactor=0으로 표기합니다.",
                        dataset.name,
                        int(trades_val),
                        int(wins_val),
                    )
                elif flag == MICRO_LOSS_ANOMALY_FLAG:
                    LOGGER.warning(
                        "데이터셋 %s 에서 미세 손실 %.6g (임계값 %.6g 이하)로 DisplayedProfitFactor=0으로 고정합니다. trades=%d, wins=%d",
                        dataset.name,
                        abs_loss,
                        threshold,
                        int(trades_val),
                        int(wins_val),
                    )
                else:
                    LOGGER.warning(
                        "데이터셋 %s 에서 DisplayedProfitFactor=0으로 처리한 특이 케이스(flag=%s)를 감지했습니다. trades=%d, wins=%d",
                        dataset.name,
                        flag,
                        int(trades_val),
                        int(wins_val),
                    )

            trades_value = _coerce_int(metrics.get("Trades") or metrics.get("TotalTrades"))
            if trades_value is None:
                trades_value = _coerce_int(cleaned_metrics.get("Trades"))
            if trades_value is not None:
                cleaned_metrics["Trades"] = trades_value
            if trades_value is not None and trades_value < MIN_TRADES_ENFORCED:
                dataset_entry["skipped"] = True
                dataset_entry["skip_reason"] = "trades_threshold"
                dataset_entry["skip_metric"] = trades_value
                dataset_entry["skip_threshold"] = MIN_TRADES_ENFORCED
                dataset_metrics.append(dataset_entry)
                skipped_dataset_records.append(
                    {
                        "name": dataset.name,
                        "timeframe": dataset.timeframe,
                        "htf_timeframe": dataset.htf_timeframe,
                        "trades": trades_value,
                        "min_trades": MIN_TRADES_ENFORCED,
                    }
                )
                LOGGER.warning(
                    "데이터셋 %s 의 트레이드 수 %d 가 최소 요구치 %d 미만이라 제외합니다.",
                    dataset.name,
                    trades_value,
                    MIN_TRADES_ENFORCED,
                )
                return

            pf_value = _safe_float(cleaned_metrics.get("ProfitFactor"))
            if pf_value is None:
                pf_value = _safe_float(cleaned_metrics.get("LosslessProfitFactorValue"))
            if pf_value is None:
                pf_value = _safe_float(cleaned_metrics.get("DisplayedProfitFactor"))
            if pf_value is not None and pf_value >= PF_ANOMALY_THRESHOLD:
                # Instead of skipping the dataset entirely when the profit factor
                # exceeds the anomaly threshold, mark the profit factor as a
                # sentinel string.  This avoids injecting an arbitrarily large
                # numeric value into the optimisation whilst preserving other
                # metrics for learning.  Record the anomaly for reporting.
                cleaned_metrics["ProfitFactor"] = "overfactor"
                dataset_entry["profit_factor_anomaly"] = {
                    "value": f"{pf_value:.3f}",
                    "threshold": PF_ANOMALY_THRESHOLD,
                }
                skipped_dataset_records.append(
                    {
                        "name": dataset.name,
                        "timeframe": dataset.timeframe,
                        "htf_timeframe": dataset.htf_timeframe,
                        "profit_factor": f"{pf_value:.3f}",
                        "status": "overfactor",
                    }
                )
                LOGGER.warning(
                    "데이터셋 %s 의 ProfitFactor %.2f 가 %.2f 이상으로 감지되어 'overfactor'로 표시합니다.",
                    dataset.name,
                    pf_value,
                    PF_ANOMALY_THRESHOLD,
                )

            numeric_metrics.append(metrics)
            dataset_metrics.append(dataset_entry)

            dataset_score = compute_score_pf_basic(metrics, constraints_cfg)
            dataset_score = _sanitise(dataset_score, f"dataset@{idx}")
            dataset_scores.append(dataset_score)

            partial_score = sum(dataset_scores) / max(1, len(dataset_scores))
            partial_score = _sanitise(partial_score, f"partial@{idx}")

            partial_metrics = combine_metrics(
                numeric_metrics, simple_override=simple_override
            )
            partial_objectives: Optional[Tuple[float, ...]] = (
                evaluate_objective_values(partial_metrics, objective_specs, non_finite_penalty)
                if multi_objective
                else None
            )
            trial.report(partial_score, step=idx)
            if trial.should_prune():
                cleaned_partial = _clean_metrics(partial_metrics)
                pruned_record = {
                    "trial": trial.number,
                    "params": params,
                    "metrics": cleaned_partial,
                    "datasets": dataset_metrics,
                    "score": partial_score,
                    "valid": cleaned_partial.get("Valid", True),
                    "dataset_key": {"timeframe": key[0], "htf_timeframe": key[1]},
                    "pruned": True,
                    "skipped_datasets": list(skipped_dataset_records),
                }
                if partial_objectives is not None:
                    pruned_record["objective_values"] = list(partial_objectives)
                with results_lock:
                    results.append(pruned_record)
                trial.set_user_attr("score", float(partial_score))
                trial.set_user_attr("metrics", cleaned_partial)
                pf_attr = _safe_float(cleaned_partial.get("ProfitFactor"))
                if pf_attr is None:
                    pf_attr = _safe_float(cleaned_partial.get("DisplayedProfitFactor"))
                trial.set_user_attr("profit_factor", pf_attr)
                if "LosslessProfitFactorValue" in cleaned_partial:
                    trial.set_user_attr(
                        "lossless_profit_factor_value",
                        cleaned_partial.get("LosslessProfitFactorValue"),
                    )
                trial.set_user_attr("trades", _coerce_int(cleaned_partial.get("Trades")))
                trial.set_user_attr("valid", bool(cleaned_partial.get("Valid", True)))
                trial.set_user_attr("pruned", True)
                trial.set_user_attr("skipped_datasets", list(skipped_dataset_records))
                raise optuna.TrialPruned()

        min_volume_threshold = _resolve_min_volume_threshold()
        eligible_datasets: List[DatasetSpec] = []
        for dataset in selected_datasets:
            meets_volume, total_volume = _has_sufficient_volume(dataset, min_volume_threshold)
            if meets_volume:
                eligible_datasets.append(dataset)
                continue

            dataset_entry: Dict[str, object] = {
                "name": dataset.name,
                "meta": dataset.meta,
                "metrics": {},
                "skipped": True,
                "skip_reason": "volume_threshold",
                "skip_metric": total_volume,
                "skip_threshold": min_volume_threshold,
            }
            dataset_metrics.append(dataset_entry)
            skipped_dataset_records.append(
                {
                    "name": dataset.name,
                    "timeframe": dataset.timeframe,
                    "htf_timeframe": dataset.htf_timeframe,
                    "total_volume": total_volume,
                    "min_volume": min_volume_threshold,
                }
            )
            LOGGER.warning(
                "데이터셋 %s 의 총 거래량 %.2f 이 최소 요구치 %.2f 미만이라 제외합니다.",
                dataset.name,
                total_volume,
                min_volume_threshold,
            )

        selected_datasets = eligible_datasets

        parallel_enabled = dataset_jobs > 1 and len(selected_datasets) > 1
        if parallel_enabled:
            executor_kwargs: Dict[str, object] = {"max_workers": dataset_jobs}
            if dataset_executor == "process":
                handles = _serialise_datasets_for_process(selected_datasets)
                try:
                    ctx = (
                        multiprocessing.get_context(dataset_start_method)
                        if dataset_start_method
                        else multiprocessing.get_context("spawn")
                    )
                except ValueError:
                    LOGGER.warning(
                        "dataset_start_method '%s' 을 사용할 수 없어 기본 spawn 을 사용합니다.",
                        dataset_start_method,
                    )
                    ctx = multiprocessing.get_context("spawn")
                executor_cls = ProcessPoolExecutor
                executor_kwargs["mp_context"] = ctx
                executor_kwargs["initializer"] = _process_pool_initializer
                executor_kwargs["initargs"] = (handles,)
                dataset_refs: Sequence[object] = [handle["id"] for handle in handles]
            else:
                executor_cls = ThreadPoolExecutor
                dataset_refs = list(selected_datasets)

            futures = []
            with executor_cls(**executor_kwargs) as executor:
                for dataset, dataset_ref in zip(selected_datasets, dataset_refs):
                    min_trades_requirement = _resolve_dataset_min_trades(
                        dataset,
                        constraints=constraints_cfg,
                        risk=risk,
                    )
                    futures.append(
                        executor.submit(
                            _run_dataset_backtest_task,
                            dataset_ref,
                            params,
                            fees,
                            risk,
                            min_trades_requirement,
                        )
                    )

                for idx, (dataset, future) in enumerate(zip(selected_datasets, futures), start=1):
                    try:
                        metrics = future.result()
                    except Exception:
                        for pending in futures[idx:]:
                            pending.cancel()
                        LOGGER.exception(
                            "백테스트 실행 중 오류 발생 (dataset=%s, timeframe=%s, htf=%s)",
                            dataset.name,
                            dataset.timeframe,
                            dataset.htf_timeframe,
                        )
                        raise
                    try:
                        _consume_dataset(
                            idx, dataset, metrics, simple_override=simple_metrics_enabled
                        )
                    except optuna.TrialPruned:
                        for pending in futures[idx:]:
                            pending.cancel()
                        raise
        else:
            for idx, dataset in enumerate(selected_datasets, start=1):
                try:
                    min_trades_requirement = _resolve_dataset_min_trades(
                        dataset,
                        constraints=constraints_cfg,
                        risk=risk,
                    )
                    # Delegate backtest to the helper which supports alternative engines
                    metrics = _run_dataset_backtest_task(
                        dataset,
                        params,
                        fees,
                        risk,
                        min_trades=min_trades_requirement,
                    )
                except Exception:
                    LOGGER.exception(
                        "백테스트 실행 중 오류 발생 (dataset=%s, timeframe=%s, htf=%s)",
                        dataset.name,
                        dataset.timeframe,
                        dataset.htf_timeframe,
                    )
                    raise
                _consume_dataset(
                    idx, dataset, metrics, simple_override=simple_metrics_enabled
                )

        aggregated = combine_metrics(
            numeric_metrics, simple_override=simple_metrics_enabled
        )
        if not aggregated:
            aggregated = {"Valid": False}
        if dataset_scores:
            score = sum(dataset_scores) / len(dataset_scores)
        else:
            score = non_finite_penalty
        score = _sanitise(score, "final")
        objective_values = (
            evaluate_objective_values(aggregated, objective_specs, non_finite_penalty)
            if multi_objective
            else None
        )

        cleaned_aggregated = _clean_metrics(aggregated)
        valid_status = bool(aggregated.get("Valid", bool(numeric_metrics)))

        pf_anomaly = False
        anomaly_info: Optional[Dict[str, object]] = None
        final_pf = _safe_float(cleaned_aggregated.get("ProfitFactor"))
        if final_pf is None:
            final_pf = _safe_float(cleaned_aggregated.get("DisplayedProfitFactor"))
        if final_pf is not None and final_pf >= PF_ANOMALY_THRESHOLD:
            # Mark the profit factor anomaly rather than invalidating the entire trial.
            pf_anomaly = True
            anomaly_info = {
                "type": "profit_factor_threshold",
                "value": final_pf,
                "threshold": PF_ANOMALY_THRESHOLD,
            }
            LOGGER.warning(
                "트라이얼 %d ProfitFactor %.2f 가 %.2f 이상으로 감지되어 'overfactor'로 표시합니다.",
                trial.number,
                final_pf,
                PF_ANOMALY_THRESHOLD,
            )
            # Replace the profit factor with the sentinel so that downstream
            # objective evaluation treats it neutrally.  Do not penalise
            # the overall score; other metrics should still influence the
            # optimisation.
            cleaned_aggregated["ProfitFactor"] = "overfactor"

        trades_total = _coerce_int(cleaned_aggregated.get("Trades"))
        trade_anomaly = False
        if trades_total is not None:
            cleaned_aggregated["Trades"] = trades_total
            if trades_total < MIN_TRADES_ENFORCED:
                trade_anomaly = True
                LOGGER.warning(
                    "트라이얼 %d 의 총 트레이드 수 %d 가 최소 요구치 %d 미만이라 결과를 무효 처리합니다.",
                    trial.number,
                    trades_total,
                    MIN_TRADES_ENFORCED,
                )
                score = non_finite_penalty
                valid_status = False
                cleaned_aggregated["Valid"] = False
                if multi_objective and objective_values is not None:
                    objective_values = tuple(non_finite_penalty for _ in objective_values)
                trade_info = {
                    "type": "trades_threshold",
                    "value": trades_total,
                    "threshold": MIN_TRADES_ENFORCED,
                }
                if anomaly_info is None:
                    anomaly_info = trade_info
                elif isinstance(anomaly_info, dict):
                    related = anomaly_info.setdefault("related", [])
                    if isinstance(related, list):
                        related.append(trade_info)

        record = {
            "trial": trial.number,
            "params": params,
            "metrics": cleaned_aggregated,
            "datasets": dataset_metrics,
            "score": score,
            "valid": valid_status,
            "dataset_key": {"timeframe": key[0], "htf_timeframe": key[1]},
            "pruned": False,
            "skipped_datasets": list(skipped_dataset_records),
        }
        if anomaly_info is not None:
            record["anomaly"] = anomaly_info
        if objective_values is not None:
            record["objective_values"] = list(objective_values)
        with results_lock:
            results.append(record)
        trial.set_user_attr("score", float(score))
        trial.set_user_attr("metrics", cleaned_aggregated)
        pf_display = cleaned_aggregated.get("ProfitFactor")
        if pf_display is None and pf_anomaly:
            pf_display = PROFIT_FACTOR_CHECK_LABEL
        lossless_pf_attr = cleaned_aggregated.get("LosslessProfitFactorValue")
        if (
            isinstance(pf_display, str)
            and pf_display.strip().lower() == "overfactor"
            and lossless_pf_attr not in {None, ""}
        ):
            pf_user_attr = f"overfactor(원본={lossless_pf_attr})"
        else:
            pf_user_attr = pf_display
        trial.set_user_attr("profit_factor", pf_user_attr)
        if lossless_pf_attr is not None:
            trial.set_user_attr("lossless_profit_factor_value", lossless_pf_attr)
        trial.set_user_attr("trades", _coerce_int(cleaned_aggregated.get("Trades")))
        trial.set_user_attr("valid", valid_status)
        trial.set_user_attr("pruned", False)
        trial.set_user_attr("skipped_datasets", list(skipped_dataset_records))
        trial.set_user_attr("profit_factor_anomaly", pf_anomaly)
        trial.set_user_attr("min_trades_enforced", trade_anomaly)
        if pf_anomaly and anomaly_info is not None:
            trial.set_user_attr("anomaly_reason", anomaly_info.get("type"))
        if multi_objective and objective_values is not None:
            return objective_values
        return score

    def _run_optuna(batch: int) -> None:
        if batch <= 0:
            return
        study.optimize(
            objective,
            n_trials=batch,
            n_jobs=n_jobs,
            show_progress_bar=False,
            callbacks=callbacks,
            gc_after_trial=True,
            catch=(sqlalchemy.exc.OperationalError, StorageInternalError),
        )

    use_llm = bool(llm_cfg.get("enabled"))
    llm_count = int(llm_cfg.get("count", 0)) if use_llm else 0
    llm_initial = int(llm_cfg.get("initial_trials", max(10, n_trials // 2))) if use_llm else 0
    llm_initial = max(0, min(llm_initial, n_trials))
    llm_insights: List[str] = []

    try:
        if use_llm and llm_count > 0 and 0 < llm_initial < n_trials:
            _run_optuna(llm_initial)
            llm_bundle: LLMSuggestions = generate_llm_candidates(space, study.trials, llm_cfg)
            llm_insights = list(llm_bundle.insights)
            for candidate in llm_bundle.candidates[:llm_count]:
                trial_params = _filter_basic_factor_params(
                    dict(candidate), enabled=use_basic_factors
                )
                if not trial_params:
                    continue
                trial_params.update(forced_params)
                try:
                    study.enqueue_trial(trial_params, skip_if_exists=True)
                except Exception as exc:
                    LOGGER.debug("Failed to enqueue LLM candidate %s: %s", candidate, exc)
            remaining = n_trials - llm_initial
            _run_optuna(remaining)
        else:
            _run_optuna(n_trials)
    finally:
        if final_csv_path is not None:
            try:
                df = study.trials_dataframe()
            except Exception:
                df = None
            if df is not None:
                final_csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(final_csv_path, index=False)

    if not results:
        raise RuntimeError("No completed trials were produced during optimisation.")

    def _profit_factor_key(record: Dict[str, object]) -> float:
        metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
        if not record.get("valid", True):
            return float("-inf")
        try:
            value_obj = metrics.get("DisplayedProfitFactor")
            if value_obj is None:
                value_obj = metrics.get("ProfitFactor", float("-inf"))
            value = float(value_obj)
        except (TypeError, ValueError):
            value = float("-inf")
        if not np.isfinite(value) or value <= 0:
            return float("-inf")
        return value

    if multi_objective:
        best_record = max(results, key=lambda res: res.get("score", float("-inf")))
    else:
        best_record = max(results, key=_profit_factor_key)
        if not np.isfinite(_profit_factor_key(best_record)):
            best_trial = study.best_trial.number
            best_record = next(res for res in results if res["trial"] == best_trial)
    return {
        "study": study,
        "results": results,
        "best": best_record,
        "multi_objective": multi_objective,
        "storage": storage_meta,
        "basic_factor_profile": use_basic_factors,
        "llm_insights": llm_insights,
        "param_order": param_order,
    }


def merge_dicts(primary: Dict[str, float], secondary: Dict[str, float]) -> Dict[str, float]:
    merged = dict(primary)
    merged.update({k: v for k, v in secondary.items() if v is not None})
    return merged


def build_parser() -> argparse.ArgumentParser:
    """Build an :class:`argparse.ArgumentParser` for optimisation commands."""

    parser = argparse.ArgumentParser(description="Run Pine strategy optimisation")
    parser.add_argument("--params", type=Path, default=Path("config/params.yaml"))
    parser.add_argument("--backtest", type=Path, default=Path("config/backtest.yaml"))
    parser.add_argument("--output", type=Path, help="Custom output directory (defaults to timestamped folder)")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--symbol", type=str, help="Override symbol to optimise")
    parser.add_argument(
        "--list-top50",
        action="store_true",
        help="USDT-Perp 24h 거래대금 상위 50개 심볼을 번호와 함께 출력 후 종료",
    )
    parser.add_argument(
        "--pick-top50",
        type=int,
        default=0,
        help="USDT-Perp 상위 50 리스트에서 번호로 선택(1~50). 선택된 심볼만 백테스트",
    )
    parser.add_argument(
        "--pick-symbol",
        type=str,
        default="",
        help="직접 심볼 지정 (예: BINANCE:ETHUSDT). 지정 시 top50 무시",
    )
    parser.add_argument("--timeframe", type=str, help="Override lower timeframe")
    parser.add_argument(
        "--timeframe-grid",
        type=str,
        help="쉼표/세미콜론으로 구분된 다중 LTF 목록을 일괄 실행 (예: '1m,3m,5m')",
    )
    parser.add_argument("--start", type=str, help="Override backtest start date (ISO8601)")
    parser.add_argument("--end", type=str, help="Override backtest end date (ISO8601)")
    parser.add_argument("--leverage", type=float, help="Override leverage setting")
    parser.add_argument("--qty-pct", type=float, help="Override quantity percent")
    parser.add_argument(
        "--full-space",
        action="store_true",
        help="기본 팩터 필터를 비활성화하고 원본 탐색 공간을 그대로 사용합니다",
    )
    parser.add_argument(
        "--basic-factors-only",
        action="store_true",
        help="기본 팩터 필터를 강제로 활성화합니다",
    )
    parser.add_argument(
        "--llm",
        dest="llm",
        action="store_true",
        help="Gemini 기반 LLM 후보 제안과 전략 인사이트를 활성화합니다",
    )
    parser.add_argument(
        "--no-llm",
        dest="llm",
        action="store_false",
        help="Gemini 기반 제안을 비활성화합니다",
    )
    parser.set_defaults(llm=None)
    parser.add_argument("--interactive", action="store_true", help="Prompt for dataset and toggle selections")
    parser.add_argument("--enable", action="append", default=[], help="Force-enable boolean parameters (comma separated)")
    parser.add_argument("--disable", action="append", default=[], help="Force-disable boolean parameters (comma separated)")
    parser.add_argument("--top-k", type=int, default=0, help="Re-rank top-K trials by walk-forward OOS mean")
    parser.add_argument("--n-trials", type=int, help="Override Optuna trial count")
    parser.add_argument("--n-jobs", type=int, help="Optuna 병렬 worker 수 (기본 1)")
    parser.add_argument(
        "--optuna-jobs",
        type=int,
        default=DEFAULT_OPTUNA_JOBS,
        help="Optuna 트라이얼 병렬 n_jobs. 기본=코어수 절반",
    )
    parser.add_argument(
        "--dataset-jobs",
        type=int,
        default=DEFAULT_DATASET_JOBS,
        help="데이터셋 병렬 워커 수. 기본=코어수 절반",
    )
    parser.add_argument(
        "--dataset-executor",
        choices=["thread", "process"],
        help="데이터셋 병렬 백테스트 실행기를 지정합니다 (기본 thread)",
    )
    parser.add_argument(
        "--dataset-start-method",
        type=str,
        help="Process executor 사용 시 multiprocessing start method 를 지정합니다",
    )
    parser.add_argument(
        "--auto-workers",
        action="store_true",
        help="가용 CPU 기반으로 Optuna worker 와 데이터셋 병렬 구성을 자동 조정합니다",
    )
    parser.add_argument("--study-name", type=str, help="Override Optuna study name")
    parser.add_argument(
        "--study-template",
        type=str,
        help="--timeframe-grid 사용 시 스터디 이름 템플릿 (예: '{symbol_slug}_{ltf_slug}')",
    )
    parser.add_argument("--storage-url", type=str, help="Override Optuna storage URL (sqlite:/// or RDB)")
    parser.add_argument(
        "--storage-url-env",
        type=str,
        help="Optuna 스토리지 URL을 읽어올 환경 변수 이름을 덮어씁니다",
    )
    parser.add_argument(
        "--allow-sqlite-parallel",
        action="store_true",
        help="SQLite 스토리지에서도 Optuna 병렬 worker를 유지합니다 (잠금 충돌 가능)",
    )
    parser.add_argument(
        "--force-sqlite-serial",
        action="store_true",
        help="SQLite 스토리지 사용 시 Optuna worker를 1개로 강제합니다",
    )
    parser.add_argument("--run-tag", type=str, help="Additional suffix for the output directory name")
    parser.add_argument(
        "--run-tag-template",
        type=str,
        help="--timeframe-grid 실행 시 출력 디렉터리 태그 템플릿",
    )
    parser.add_argument("--resume-from", type=Path, help="Path to a bank.json file for warm-start seeding")
    parser.add_argument("--pruner", type=str, help="Override pruner selection (asha, hyperband, median, threshold, patient, wilcoxon, none)")
    parser.add_argument("--cv", type=str, choices=["purged-kfold", "none"], help="Enable auxiliary cross-validation scoring")
    parser.add_argument("--cv-k", type=int, help="Number of folds for Purged K-Fold validation")
    parser.add_argument("--cv-embargo", type=float, help="Embargo fraction for Purged K-Fold validation")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments and return a populated :class:`Namespace`."""

    args = build_parser().parse_args(argv)
    global simple_metrics_enabled
    simple_metrics_enabled = False
    return args


def _execute_single(
    args: argparse.Namespace,
    params_cfg: Dict[str, object],
    backtest_cfg: Dict[str, object],
    argv: Optional[Sequence[str]] = None,
) -> None:
    params_cfg = copy.deepcopy(params_cfg)
    backtest_cfg = copy.deepcopy(backtest_cfg)
    params_cfg.setdefault("space", {})
    backtest_cfg.setdefault("symbols", backtest_cfg.get("symbols", []))
    backtest_cfg.setdefault("timeframes", backtest_cfg.get("timeframes", []))

    cli_tokens = list(argv or [])

    def _has_flag(flag: str) -> bool:
        return any(token == flag or token.startswith(f"{flag}=") for token in cli_tokens)

    search_cfg = _ensure_dict(params_cfg, "search")

    search_cfg.setdefault("dataset_executor", "thread")
    search_cfg.setdefault("allow_sqlite_parallel", False)

    batch_ctx = getattr(args, "_batch_context", None)
    if batch_ctx:
        suffix = batch_ctx.get("suffix") or batch_ctx.get("ltf_slug") or ""
        try:
            args.run_tag = _format_batch_value(
                batch_ctx.get("run_tag_template"),
                batch_ctx.get("base_run_tag"),
                suffix,
                batch_ctx,
            )
            base_study = batch_ctx.get("base_study_name")
            study_template = batch_ctx.get("study_template")
            if base_study or study_template:
                args.study_name = _format_batch_value(
                    study_template,
                    base_study,
                    suffix,
                    batch_ctx,
                )
        except ValueError as exc:
            raise ValueError(f"배치 템플릿 해석에 실패했습니다: {exc}") from exc

    if args.n_trials is not None:
        search_cfg["n_trials"] = int(args.n_trials)

    if args.n_jobs is not None:
        try:
            search_cfg["n_jobs"] = max(1, int(args.n_jobs))
        except (TypeError, ValueError):
            LOGGER.warning("--n-jobs 값 '%s' 이 올바르지 않아 1로 설정합니다.", args.n_jobs)
            search_cfg["n_jobs"] = 1

    if _has_flag("--optuna-jobs"):
        try:
            search_cfg["n_jobs"] = max(1, int(args.optuna_jobs))
        except (TypeError, ValueError):
            LOGGER.warning(
                "--optuna-jobs 값 '%s' 이 올바르지 않아 %d로 설정합니다.",
                args.optuna_jobs,
                DEFAULT_OPTUNA_JOBS,
            )
            search_cfg["n_jobs"] = DEFAULT_OPTUNA_JOBS

    if _has_flag("--dataset-jobs"):
        try:
            search_cfg["dataset_jobs"] = max(1, int(args.dataset_jobs))
        except (TypeError, ValueError):
            LOGGER.warning(
                "--dataset-jobs 값 '%s' 이 올바르지 않아 1로 설정합니다.",
                args.dataset_jobs,
            )
            search_cfg["dataset_jobs"] = 1

    if args.dataset_executor:
        search_cfg["dataset_executor"] = args.dataset_executor

    if args.dataset_start_method:
        search_cfg["dataset_start_method"] = args.dataset_start_method

    if getattr(args, "full_space", False):
        search_cfg["basic_factor_profile"] = False
    elif getattr(args, "basic_factors_only", False):
        search_cfg["basic_factor_profile"] = True

    if args.study_name:
        search_cfg["study_name"] = args.study_name

    if args.storage_url:
        search_cfg["storage_url"] = args.storage_url

    if args.storage_url_env:
        search_cfg["storage_url_env"] = args.storage_url_env

    if getattr(args, "allow_sqlite_parallel", False):
        search_cfg["allow_sqlite_parallel"] = True

    if getattr(args, "force_sqlite_serial", False):
        search_cfg["allow_sqlite_parallel"] = False
        search_cfg["force_sqlite_serial"] = True
        LOGGER.info("CLI --force-sqlite-serial 지정: Optuna worker를 1개로 강제합니다.")

    if args.pruner:
        search_cfg["pruner"] = args.pruner

    forced_params: Dict[str, object] = dict(params_cfg.get("overrides", {}))
    auto_workers = bool(getattr(args, "auto_workers", False))
    for name in _collect_tokens(args.enable):
        forced_params[name] = True
    for name in _collect_tokens(args.disable):
        forced_params[name] = False

    symbol_choices = list(dict.fromkeys(backtest_cfg.get("symbols") or ([params_cfg.get("symbol")] if params_cfg.get("symbol") else [])))

    selected_symbol = args.symbol or params_cfg.get("symbol") or (symbol_choices[0] if symbol_choices else None)
    selected_timeframe: Optional[str] = args.timeframe
    timeframe_overridden = args.timeframe is not None
    all_timeframes_requested = False

    if (
        not timeframe_overridden
        and not getattr(args, "timeframe_grid", None)
        and batch_ctx is None
    ):
        prompt_selection = _prompt_ltf_selection()
        if prompt_selection.use_all:
            all_timeframes_requested = True
            selected_timeframe = None
            timeframe_overridden = False
        else:
            selected_timeframe = prompt_selection.timeframe
            timeframe_overridden = True

    if args.interactive and symbol_choices:
        selected_symbol = _prompt_choice("Select symbol", symbol_choices, selected_symbol)

    if selected_symbol:
        params_cfg["symbol"] = selected_symbol
        backtest_cfg["symbols"] = [selected_symbol]
    if timeframe_overridden and selected_timeframe:
        params_cfg["timeframe"] = selected_timeframe
        backtest_cfg["timeframes"] = [selected_timeframe]
        _apply_ltf_override_to_datasets(backtest_cfg, selected_timeframe)
    for key in ("htf", "htf_timeframe", "htf_timeframes"):
        params_cfg.pop(key, None)
        backtest_cfg.pop(key, None)

    backtest_periods = backtest_cfg.get("periods") or []
    params_backtest = _ensure_dict(params_cfg, "backtest")
    if args.start or args.end:
        start = args.start or params_backtest.get("from") or (backtest_periods[0]["from"] if backtest_periods else None)
        end = args.end or params_backtest.get("to") or (backtest_periods[0]["to"] if backtest_periods else None)
        if start and end:
            params_backtest["from"] = start
            params_backtest["to"] = end
            backtest_cfg["periods"] = [{"from": start, "to": end}]
    elif args.interactive and backtest_periods:
        labels = [f"{p['from']} → {p['to']}" for p in backtest_periods]
        default_label = f"{params_backtest.get('from')} → {params_backtest.get('to')}" if params_backtest.get("from") and params_backtest.get("to") else labels[0]
        choice = _prompt_choice("Select backtest period", labels, default_label)
        if choice and choice in labels:
            selected = dict(backtest_periods[labels.index(choice)])
            params_backtest["from"] = selected["from"]
            params_backtest["to"] = selected["to"]
            backtest_cfg["periods"] = [selected]
    elif params_backtest.get("from") and params_backtest.get("to"):
        backtest_cfg["periods"] = [{"from": params_backtest["from"], "to": params_backtest["to"]}]

    risk_cfg = _ensure_dict(params_cfg, "risk")
    backtest_risk = _ensure_dict(backtest_cfg, "risk")
    if args.leverage is not None:
        risk_cfg["leverage"] = args.leverage
        backtest_risk["leverage"] = args.leverage
    if args.qty_pct is not None:
        risk_cfg["qty_pct"] = args.qty_pct
        backtest_risk["qty_pct"] = args.qty_pct

    if args.interactive:
        leverage_default = risk_cfg.get("leverage")
        qty_default = risk_cfg.get("qty_pct")
        lev_input = input(f"Leverage [{leverage_default}]: ").strip()
        if lev_input:
            try:
                leverage_val = float(lev_input)
                risk_cfg["leverage"] = leverage_val
                backtest_risk["leverage"] = leverage_val
            except ValueError:
                print("Invalid leverage value, keeping previous setting.")
        qty_input = input(f"Position size %% [{qty_default}]: ").strip()
        if qty_input:
            try:
                qty_val = float(qty_input)
                risk_cfg["qty_pct"] = qty_val
                backtest_risk["qty_pct"] = qty_val
            except ValueError:
                print("Invalid quantity percentage, keeping previous setting.")

        bool_candidates = [name for name, spec in params_cfg.get("space", {}).items() if spec.get("type") == "bool"]
        for name in bool_candidates:
            default_val = forced_params.get(name)
            decision = _prompt_bool(f"Enable {name}", default_val)
            if decision is not None:
                forced_params[name] = decision

    llm_cfg = _ensure_dict(params_cfg, "llm")
    if args.llm is not None:
        llm_cfg["enabled"] = bool(args.llm)
    elif args.interactive:
        llm_default = _coerce_bool_or_none(llm_cfg.get("enabled"))
        llm_choice = _prompt_bool(
            "Gemini 후보/전략 인사이트를 사용할까요?", llm_default
        )
        if llm_choice is not None:
            llm_cfg["enabled"] = llm_choice

    if llm_cfg.get("enabled"):
        api_key_env = str(llm_cfg.get("api_key_env", "GEMINI_API_KEY"))
        if not llm_cfg.get("api_key") and not os.environ.get(api_key_env):
            LOGGER.warning(
                "Gemini 활성화 상태지만 API 키가 설정되지 않았습니다. %s 환경 변수를 확인하세요.",
                api_key_env,
            )

    def _resolve_simple_metrics_flag() -> bool:
        for candidate in (
            forced_params.get("simpleMetricsOnly"),
            forced_params.get("simpleProfitOnly"),
            risk_cfg.get("simpleMetricsOnly"),
            risk_cfg.get("simpleProfitOnly"),
            backtest_risk.get("simpleMetricsOnly"),
            backtest_risk.get("simpleProfitOnly"),
        ):
            coerced = _coerce_bool_or_none(candidate)
            if coerced is not None:
                return coerced
        return False

    simple_metrics_state = _resolve_simple_metrics_flag()
    if simple_metrics_state:
        LOGGER.info("단순 ProfitFactor 기반 계산 경로가 구성에서 활성화되어 있습니다.")
    else:
        for key in ("simpleMetricsOnly", "simpleProfitOnly"):
            forced_params.pop(key, None)
            risk_cfg.pop(key, None)
            backtest_risk.pop(key, None)
        LOGGER.info("전체 지표 계산 경로를 사용합니다.")

    global simple_metrics_enabled
    simple_metrics_enabled = simple_metrics_state

    params_cfg["overrides"] = forced_params

    datasets = prepare_datasets(params_cfg, backtest_cfg, args.data)
    if not datasets:
        raise RuntimeError("No datasets prepared for optimisation")

    if auto_workers:
        available_cpu = max(multiprocessing.cpu_count(), 1)
        search_cfg = _ensure_dict(params_cfg, "search")
        current_n_jobs = int(search_cfg.get("n_jobs", 1) or 1)
        if current_n_jobs <= 1 and available_cpu > 1:
            recommended_trials = min(available_cpu, max(2, available_cpu // 2))
            if recommended_trials > 1:
                search_cfg["n_jobs"] = recommended_trials
                LOGGER.info(
                    "Auto workers: Optuna n_jobs=%d (available CPU=%d)",
                    recommended_trials,
                    available_cpu,
                )
        dataset_jobs_current = int(search_cfg.get("dataset_jobs", 1) or 1)
        if len(datasets) > 1 and dataset_jobs_current <= 1:
            max_parallel = min(len(datasets), max(1, available_cpu))
            if max_parallel > 1:
                search_cfg["dataset_jobs"] = max_parallel
                LOGGER.info(
                    "Auto workers: dataset_jobs=%d (datasets=%d, CPU=%d)",
                    max_parallel,
                    len(datasets),
                    available_cpu,
                )
                adjusted_n_jobs = max(1, available_cpu // max_parallel)
                if adjusted_n_jobs < current_n_jobs:
                    search_cfg["n_jobs"] = adjusted_n_jobs
                    LOGGER.info(
                        "Auto workers: Optuna n_jobs=%d (dataset 병렬 보정)",
                        adjusted_n_jobs,
                    )


    output_dir, tag_info = _resolve_output_directory(args.output, datasets, params_cfg, args.run_tag)
    _configure_logging(output_dir / "logs")
    LOGGER.info("Writing outputs to %s", output_dir)

    fees = merge_dicts(params_cfg.get("fees", {}), backtest_cfg.get("fees", {}))
    risk = merge_dicts(params_cfg.get("risk", {}), backtest_cfg.get("risk", {}))

    objectives_raw = params_cfg.get("objectives")
    if not objectives_raw:
        objectives_raw = params_cfg.get("objective")
    if objectives_raw is None:
        objectives_raw = []
    if isinstance(objectives_raw, (list, tuple)):
        objectives_config: List[object] = list(objectives_raw)
    elif objectives_raw:
        objectives_config = [objectives_raw]
    else:
        objectives_config = []
    objective_specs = normalise_objectives(objectives_config)
    space_hash = _space_hash(params_cfg.get("space", {}))
    search_cfg = _ensure_dict(params_cfg, "search")
    if not search_cfg.get("study_name"):
        search_cfg["study_name"] = _default_study_name(params_cfg, datasets, space_hash)
    primary_for_regime = _pick_primary_dataset(datasets)
    regime_summary = detect_regime_label(primary_for_regime.df)

    resume_bank_path = args.resume_from
    if resume_bank_path is None:
        resume_bank_path = _discover_bank_path(output_dir, tag_info, space_hash)

    search_cfg_effective = params_cfg.get("search", {})
    basic_flag = _coerce_bool_or_none(search_cfg_effective.get("basic_factor_profile"))
    if basic_flag is None:
        basic_flag = _coerce_bool_or_none(search_cfg_effective.get("use_basic_factors"))
    basic_filter_enabled = True if basic_flag is None else basic_flag

    seed_trials = _load_seed_trials(
        resume_bank_path,
        params_cfg.get("space", {}),
        space_hash,
        regime_label=regime_summary.label,
        basic_filter_enabled=basic_filter_enabled,
    )

    study_storage = _resolve_study_storage(params_cfg, datasets)
    _apply_study_registry_defaults(search_cfg, study_storage)

    storage_env_key = str(search_cfg.get("storage_url_env") or DEFAULT_STORAGE_ENV_KEY)
    if not storage_env_key:
        storage_env_key = DEFAULT_STORAGE_ENV_KEY
    search_cfg["storage_url_env"] = storage_env_key

    if not search_cfg.get("storage_url"):
        search_cfg["storage_url"] = DEFAULT_POSTGRES_STORAGE_URL

    storage_env_value = os.getenv(storage_env_key) if storage_env_key else None
    effective_storage_url = str(
        storage_env_value or search_cfg.get("storage_url") or ""
    )
    using_sqlite = effective_storage_url.startswith("sqlite:///")
    is_postgres = effective_storage_url.startswith(POSTGRES_PREFIXES)
    masked_storage_url = _mask_storage_url(effective_storage_url) if effective_storage_url else ""
    if storage_env_value:
        storage_source = f"환경변수 {storage_env_key}"
    elif effective_storage_url:
        storage_source = "설정값"
    else:
        storage_source = "기본값"
    backend_label = (
        "PostgreSQL"
        if is_postgres
        else "SQLite"
        if using_sqlite
        else "기타 RDB"
        if effective_storage_url
        else "비활성"
    )
    LOGGER.info(
        "Optuna 스토리지 백엔드: %s (%s, URL=%s)",
        backend_label,
        storage_source,
        masked_storage_url or "(없음)",
    )

    default_optuna_jobs = (
        SQLITE_SAFE_OPTUNA_JOBS if using_sqlite else DEFAULT_OPTUNA_JOBS
    )
    default_dataset_jobs = (
        SQLITE_SAFE_DATASET_JOBS if using_sqlite else DEFAULT_DATASET_JOBS
    )

    if not search_cfg.get("n_jobs"):
        search_cfg["n_jobs"] = default_optuna_jobs
    if not search_cfg.get("dataset_jobs"):
        search_cfg["dataset_jobs"] = default_dataset_jobs

    if using_sqlite and not search_cfg.get("allow_sqlite_parallel"):
        search_cfg.setdefault("force_sqlite_serial", True)

    trials_log_dir = output_dir / "trials"

    optimisation = optimisation_loop(
        datasets,
        params_cfg,
        objective_specs,
        fees,
        risk,
        forced_params,
        study_storage=study_storage,
        space_hash=space_hash,
        seed_trials=seed_trials,
        log_dir=trials_log_dir,
    )

    llm_insights_logged: List[str] = []
    raw_llm_insights = optimisation.get("llm_insights")
    if isinstance(raw_llm_insights, (list, tuple)):
        llm_insights_logged = [str(item) for item in raw_llm_insights if str(item).strip()]
    if llm_insights_logged:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        insight_file = logs_dir / "gemini_insights.md"
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
        with insight_file.open("a", encoding="utf-8") as handle:
            handle.write(f"## {timestamp}\n")
            for insight in llm_insights_logged:
                handle.write(f"- {insight}\n")
            handle.write("\n")
        for insight in llm_insights_logged:
            LOGGER.info("[Gemini Insight] %s", insight)

    study = optimisation.get("study")
    if study is not None:
        write_trials_dataframe(
            study,
            output_dir,
            param_order=optimisation.get("param_order"),
        )
    else:
        LOGGER.warning("No Optuna study returned; skipping trials export")

    walk_cfg = (
        params_cfg.get("walk_forward")
        or backtest_cfg.get("walk_forward")
        or {"train_bars": 5000, "test_bars": 2000, "step": 2000}
    )
    dataset_groups, timeframe_groups, default_key = _group_datasets(datasets)

    def _resolve_record_dataset(record: Dict[str, object]) -> Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]:
        key_info = record.get("dataset_key") if isinstance(record, dict) else None
        if isinstance(key_info, dict):
            candidate_key = (key_info.get("timeframe"), key_info.get("htf_timeframe"))
            if candidate_key in dataset_groups:
                return candidate_key, dataset_groups[candidate_key]
        return _select_datasets_for_params(
            params_cfg,
            dataset_groups,
            timeframe_groups,
            default_key,
            record.get("params", {}),
        )

    best_record = optimisation["best"]
    best_key, best_group = _resolve_record_dataset(best_record)
    primary_dataset = _pick_primary_dataset(best_group)

    wf_min_trades_override = _coerce_min_trades_value(walk_cfg.get("min_trades"))

    def _min_trades_for_dataset(dataset: DatasetSpec) -> Optional[int]:
        return _resolve_dataset_min_trades(
            dataset,
            constraints=constraints_cfg,
            risk=risk,
            explicit=wf_min_trades_override,
        )

    primary_min_trades = _min_trades_for_dataset(primary_dataset)
    param_order = optimisation.get("param_order")

    def _ordered_params_view(raw_params: object) -> Dict[str, object]:
        if isinstance(raw_params, Mapping):
            return _order_mapping(raw_params, param_order)
        return {}

    wf_summary = run_walk_forward(
        primary_dataset.df,
        best_record["params"],
        fees,
        risk,
        train_bars=int(walk_cfg.get("train_bars", 5000)),
        test_bars=int(walk_cfg.get("test_bars", 2000)),
        step=int(walk_cfg.get("step", 2000)),
        htf_df=primary_dataset.htf,
        min_trades=primary_min_trades,
    )

    cv_summary = None
    cv_manifest: Dict[str, object] = {}
    cv_choice = (args.cv or str(params_cfg.get("validation", {}).get("type", ""))).lower()
    if cv_choice == "purged-kfold":
        cv_k = args.cv_k or int(params_cfg.get("validation", {}).get("k", 5))
        cv_embargo = args.cv_embargo or float(params_cfg.get("validation", {}).get("embargo", 0.01))
        cv_summary = run_purged_kfold(
            primary_dataset.df,
            best_record["params"],
            fees,
            risk,
            k=cv_k,
            embargo=cv_embargo,
            htf_df=primary_dataset.htf,
            min_trades=primary_min_trades,
        )
        wf_summary["purged_kfold"] = cv_summary
        cv_manifest = {"type": "purged-kfold", "k": cv_k, "embargo": cv_embargo}
    elif cv_choice and cv_choice != "none":
        cv_manifest = {"type": cv_choice}

    def _profit_factor_value(record: Dict[str, object]) -> float:
        metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
        if not record.get("valid", True):
            return float("-inf")
        try:
            value_obj = metrics.get("DisplayedProfitFactor")
            if value_obj is None:
                value_obj = metrics.get("ProfitFactor", float("-inf"))
            value = float(value_obj)
        except (TypeError, ValueError):
            value = float("-inf")
        if not np.isfinite(value) or value <= 0:
            return float("-inf")
        return value

    candidate_summaries = [
        {
            "trial": best_record["trial"],
            "score": best_record.get("score"),
            "oos_mean": wf_summary.get("oos_mean"),
            "params": _ordered_params_view(best_record.get("params")),
            "timeframe": best_key[0],
            "htf_timeframe": best_key[1],
        }
    ]

    top_k = args.top_k or int(params_cfg.get("search", {}).get("top_k", 0))
    if top_k > 0:
        sorted_results = sorted(optimisation["results"], key=_profit_factor_value, reverse=True)
        best_oos = wf_summary.get("oos_mean", float("-inf"))
        wf_cache = {best_record["trial"]: wf_summary}
        for record in sorted_results[:top_k]:
            if record["trial"] == best_record["trial"]:
                continue
            candidate_key, candidate_group = _resolve_record_dataset(record)
            candidate_dataset = _pick_primary_dataset(candidate_group)
            candidate_min_trades = _min_trades_for_dataset(candidate_dataset)

            candidate_wf = run_walk_forward(
                candidate_dataset.df,
                record["params"],
                fees,
                risk,
                train_bars=int(walk_cfg.get("train_bars", 5000)),
                test_bars=int(walk_cfg.get("test_bars", 2000)),
                step=int(walk_cfg.get("step", 2000)),
                htf_df=candidate_dataset.htf,
                min_trades=candidate_min_trades,
            )
            wf_cache[record["trial"]] = candidate_wf
            candidate_summaries.append(
                {
                    "trial": record["trial"],
                    "score": record.get("score"),
                    "oos_mean": candidate_wf.get("oos_mean"),
                    "params": _ordered_params_view(record.get("params")),
                    "timeframe": candidate_key[0],
                    "htf_timeframe": candidate_key[1],
                }
            )
            candidate_oos = candidate_wf.get("oos_mean", float("-inf"))
            if candidate_oos > best_oos or (
                candidate_oos == best_oos
                and _profit_factor_value(record) > _profit_factor_value(best_record)
            ):
                best_oos = candidate_oos
                best_record = record
                best_key = candidate_key
                best_group = candidate_group
                primary_dataset = candidate_dataset
                wf_summary = candidate_wf
        optimisation["best"] = best_record

    candidate_summaries[0] = {
        "trial": best_record["trial"],
        "score": best_record.get("score"),
        "oos_mean": wf_summary.get("oos_mean"),
        "params": _ordered_params_view(best_record.get("params")),
        "timeframe": best_key[0],
        "htf_timeframe": best_key[1],
    }

    wf_summary["candidates"] = candidate_summaries

    trial_index = {record["trial"]: record for record in optimisation["results"]}
    bank_entries: List[Dict[str, object]] = []
    for item in candidate_summaries:
        trial_record = trial_index.get(item["trial"], {})
        filtered_params = _filter_basic_factor_params(
            item.get("params") or {}, enabled=optimisation.get("basic_factor_profile", True)
        )
        ordered_params = _order_mapping(filtered_params, param_order)
        raw_metrics = trial_record.get("metrics") if isinstance(trial_record, dict) else {}
        if isinstance(raw_metrics, Mapping):
            metrics_payload: object = _order_mapping(
                raw_metrics,
                None,
                priority=("ProfitFactor", "Sortino"),
            )
        else:
            metrics_payload = raw_metrics
        entry = {
            "trial": item["trial"],
            "score": item.get("score"),
            "oos_mean": item.get("oos_mean"),
            "params": ordered_params,
            "metrics": metrics_payload,
            "timeframe": item.get("timeframe"),
            "htf_timeframe": item.get("htf_timeframe"),
        }
        if cv_summary:
            entry["cv_mean"] = cv_summary.get("mean")
        bank_entries.append(entry)

    bank_payload = _build_bank_payload(
        tag_info=tag_info,
        space_hash=space_hash,
        entries=bank_entries,
        regime_summary=regime_summary,
    )

    validation_manifest = dict(cv_manifest)
    if cv_summary:
        validation_manifest["summary"] = cv_summary

    storage_meta = optimisation.get("storage", {}) or {}
    _register_study_reference(
        study_storage,
        storage_meta=storage_meta,
        study_name=str(search_cfg.get("study_name")) if search_cfg.get("study_name") else None,
    )
    sanitised_storage_meta = _sanitise_storage_meta(storage_meta)
    registry_dir = _study_registry_dir(study_storage) if study_storage else None
    registry_dir_str = (
        str(registry_dir) if registry_dir and registry_dir.exists() else None
    )
    search_manifest = copy.deepcopy(params_cfg.get("search", {}))
    if "storage_url" in search_manifest:
        url_value = search_manifest.get("storage_url")
        if url_value and not str(url_value).startswith("sqlite:///"):
            search_manifest["storage_url"] = "***redacted***"

    manifest = {
        "created_at": _utcnow_isoformat(),
        "run": tag_info,
        "space_hash": space_hash,
        "symbol": params_cfg.get("symbol"),
        "fees": fees,
        "risk": risk,
        "objectives": [spec.__dict__ for spec in objective_specs],
        "search": search_manifest,
        "basic_factor_profile": optimisation.get("basic_factor_profile", True),
        "resume_bank": str(resume_bank_path) if resume_bank_path else None,
        "study_storage": storage_meta.get("path") if storage_meta.get("backend") == "sqlite" else None,
        "study_registry": registry_dir_str,
        "regime": regime_summary.__dict__,
        "cli": list(argv or []),
    }
    if sanitised_storage_meta:
        manifest["storage"] = sanitised_storage_meta
    if validation_manifest:
        manifest["validation"] = validation_manifest

    _write_manifest(output_dir, manifest=manifest)
    write_bank_file(output_dir, bank_payload)
    (output_dir / "seed.yaml").write_text(
        yaml.safe_dump(
            {
                "params": params_cfg,
                "backtest": backtest_cfg,
                "forced_params": forced_params,
            },
            sort_keys=False,
        )
    )

    generate_reports(
        optimisation["results"],
        optimisation["best"],
        wf_summary,
        objective_specs,
        output_dir,
        param_order=optimisation.get("param_order"),
    )

    LOGGER.info("Run complete. Outputs saved to %s", output_dir)


def execute(args: argparse.Namespace, argv: Optional[Sequence[str]] = None) -> None:
    """Execute one or more optimisation runs based on CLI arguments."""

    params_cfg = load_yaml(args.params)
    backtest_cfg = load_yaml(args.backtest)

    ltf_prompt = getattr(args, "_ltf_prompt_selection", None)
    if (
        ltf_prompt is None
        and not getattr(args, "timeframe", None)
        and not getattr(args, "timeframe_grid", None)
    ):
        ltf_prompt = _prompt_ltf_selection()
        setattr(args, "_ltf_prompt_selection", ltf_prompt)

    all_timeframes_requested = False
    if ltf_prompt:
        if ltf_prompt.use_all:
            all_timeframes_requested = True
            args.timeframe = None
            if not getattr(args, "timeframe_grid", None):
                ltf_candidates = _collect_ltf_candidates(backtest_cfg, params_cfg)
                if not ltf_candidates:
                    ltf_candidates = ["1m", "3m", "5m"]
                args.timeframe_grid = ",".join(ltf_candidates)
        elif ltf_prompt.timeframe and not getattr(args, "timeframe", None):
            args.timeframe = ltf_prompt.timeframe

    auto_list: List[str] = []

    def _load_top_list() -> List[str]:
        return fetch_top_usdt_perp_symbols(
            limit=50,
            exclude_symbols=["BUSDUSDT", "USDCUSDT"],
            exclude_keywords=["UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "5L", "5S"],
            min_price=0.002,
        )

    if args.list_top50:
        auto_list = _load_top_list()
        import csv

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / "top50_usdt_perp.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "symbol"])
            for index, symbol in enumerate(auto_list, start=1):
                writer.writerow([index, symbol])
        print("Saved: reports/top50_usdt_perp.csv")
        print("\n== USDT-Perp 24h 거래대금 상위 50 ==")
        for index, symbol in enumerate(auto_list, start=1):
            print(f"{index:2d}. {symbol}")
        print("\n예) 7번 선택:  python -m optimize.run --pick-top50 7")
        print("    직접 지정:  python -m optimize.run --pick-symbol BINANCE:ETHUSDT")
        return

    selected_symbol = ""
    if args.pick_symbol:
        selected_symbol = args.pick_symbol.strip()
    elif args.pick_top50:
        auto_list = auto_list or _load_top_list()
        if 1 <= args.pick_top50 <= len(auto_list):
            selected_symbol = auto_list[args.pick_top50 - 1]
        else:
            print("\n[ERROR] --pick-top50 인덱스가 범위를 벗어났습니다 (1~50).")
            return
    elif args.symbol:
        selected_symbol = args.symbol.strip()
    else:
        print("\n[ERROR] 심볼이 지정되지 않았습니다.")
        print("   예) 상위50 출력:       python -m optimize.run --list-top50")
        print("       7번 선택(예):      python -m optimize.run --pick-top50 7")
        print("       직접 지정:         python -m optimize.run --pick-symbol BINANCE:ETHUSDT")
        return

    print(f"[INFO] 선택된 심볼: {selected_symbol}")

    args.symbol = selected_symbol
    params_cfg["symbol"] = selected_symbol
    backtest_cfg["symbols"] = [selected_symbol]

    datasets = backtest_cfg.get("datasets")
    if isinstance(datasets, list):
        for dataset in datasets:
            if isinstance(dataset, dict):
                dataset["symbol"] = selected_symbol

    if all_timeframes_requested:
        if getattr(args, "timeframe_grid", None):
            LOGGER.info("사용자가 이미 타임프레임 그리드를 지정해 혼합 실행 요청을 유지합니다: %s", args.timeframe_grid)
        else:
            ltf_candidates = _collect_ltf_candidates(backtest_cfg, params_cfg)
            if not ltf_candidates:
                ltf_candidates = ["1m", "3m", "5m"]
            args.timeframe_grid = ",".join(ltf_candidates)
            LOGGER.info(
                "혼합 LTF 실행을 위해 타임프레임 그리드를 자동 구성했습니다: %s",
                ", ".join(ltf_candidates),
            )

    combos = _parse_timeframe_grid(getattr(args, "timeframe_grid", None))
    if not combos:
        _execute_single(args, params_cfg, backtest_cfg, argv)
        return

    base_symbol: Optional[str]
    if args.symbol:
        base_symbol = args.symbol
    else:
        base_symbol = params_cfg.get("symbol") if params_cfg else None
        if not base_symbol:
            symbols = backtest_cfg.get("symbols") if isinstance(backtest_cfg, dict) else None
            if isinstance(symbols, list) and symbols:
                first = symbols[0]
                if isinstance(first, dict):
                    base_symbol = (
                        first.get("alias")
                        or first.get("name")
                        or first.get("symbol")
                        or first.get("id")
                    )
                else:
                    base_symbol = str(first)

    symbol_text = str(base_symbol) if base_symbol else "study"
    symbol_slug = _slugify_symbol(symbol_text)
    total = len(combos)
    combo_summary = ", ".join(combos)
    LOGGER.info("타임프레임 그리드 %d건 실행: %s", total, combo_summary)

    for index, ltf in enumerate(combos, start=1):
        batch_args = argparse.Namespace(**vars(args))
        batch_args.timeframe = ltf
        suffix = _slugify_timeframe(ltf)
        context = {
            "index": index,
            "total": total,
            "ltf": ltf,
            "htf": None,
            "ltf_slug": _slugify_timeframe(ltf),
            "htf_slug": "",
            "symbol": symbol_text,
            "symbol_slug": symbol_slug,
            "suffix": suffix,
            "base_run_tag": getattr(args, "run_tag", None),
            "base_study_name": getattr(args, "study_name", None),
            "run_tag_template": getattr(args, "run_tag_template", None),
            "study_template": getattr(args, "study_template", None),
        }
        batch_args._batch_context = context  # type: ignore[attr-defined]
        LOGGER.info(
            "(%d/%d) LTF=%s 조합 최적화 시작",
            index,
            total,
            ltf,
        )
        _execute_single(batch_args, params_cfg, backtest_cfg, argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for ``python -m optimize.run``."""

    if argv is None:
        original_argv: List[str] = list(sys.argv[1:])
    else:
        original_argv = list(argv)

    parsed_argv = list(original_argv)
    replaced_interactive = False
    for index, token in enumerate(parsed_argv):
        if token == "시작":
            parsed_argv[index] = "--interactive"
            replaced_interactive = True
            break

    if replaced_interactive and "--interactive" not in parsed_argv:
        parsed_argv.append("--interactive")

    args = parse_args(parsed_argv)
    execute(args, original_argv)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
