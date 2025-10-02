"""Command line interface for running parameter optimisation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import subprocess
from collections.abc import Sequence as AbcSequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import optuna.storages
import pandas as pd
import yaml
import multiprocessing
import ccxt

from datafeed.cache import DataCache
from optimize.metrics import (
    EPS,
    ObjectiveSpec,
    Trade,
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
from optimize.llm import generate_llm_candidates

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
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

CPU_COUNT = os.cpu_count() or 4
DEFAULT_DATASET_JOBS = max(1, CPU_COUNT // 2)
DEFAULT_OPTUNA_JOBS = max(1, CPU_COUNT // 2)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = Path("reports")
STUDY_ROOT = Path("studies")
NON_FINITE_PENALTY = -1e12

# 단순 메트릭 계산 경로 사용 여부 (CLI 인자/설정으로 갱신됩니다).
simple_metrics_enabled: bool = False

# 기본 팩터 최적화에 사용할 파라미터 키 집합입니다.
# 복잡한 보호 장치·부가 필터 대신 핵심 진입 로직과 직접 관련된 항목만 남겨
# 탐색 공간을 크게 줄이고 수렴 속도를 높입니다.
BASIC_FACTOR_KEYS = {
    "bbLen",
    "bbMult",
    "kcLen",
    "kcMultATR",
    "kcBasis",
    "momLen",
    "thr",
    "requireFlux",
    "exitOpposite",
    "useFadeExit",
    "fadeLevel",
    "useSL",
    "slPct",
    "cooldownBars",
}


def _utcnow_isoformat() -> str:
    """현재 UTC 시각을 ISO8601 ``Z`` 표기 문자열로 반환합니다."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _slugify_symbol(symbol: str) -> str:
    text = symbol.split(":")[-1]
    return text.replace("/", "").replace(" ", "")


def _slugify_timeframe(timeframe: Optional[str]) -> str:
    if not timeframe:
        return "nohtf"
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
    htf = (
        params_cfg.get("htf_timeframe")
        or params_cfg.get("htf")
        or (datasets[0].htf_timeframe if datasets and datasets[0].htf_timeframe else "nohtf")
    )
    if not htf:
        htf = "nohtf"
    symbol_slug = _slugify_symbol(str(symbol))
    timeframe_slug = str(timeframe).replace("/", "_")
    htf_slug = str(htf).replace("/", "_")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    parts = [timestamp, symbol_slug, timeframe_slug, htf_slug]
    if run_tag:
        parts.append(run_tag)
    return timestamp, symbol_slug, timeframe_slug, "_".join(filter(None, parts))


def _run_dataset_backtest_task(
    dataset: "DatasetSpec",
    params: Dict[str, object],
    fees: Dict[str, float],
    risk: Dict[str, float],
) -> Dict[str, float]:
    """Execute ``run_backtest`` for a single dataset.

    This helper is defined at module level so it can be pickled when ``ProcessPoolExecutor``
    is used for parallel evaluation.
    """

    return run_backtest(dataset.df, params, fees, risk, htf_df=dataset.htf)


def _resolve_output_directory(
    base: Optional[Path],
    datasets: Sequence["DatasetSpec"],
    params_cfg: Dict[str, object],
    run_tag: Optional[str],
) -> Tuple[Path, Dict[str, str]]:
    ts, symbol_slug, timeframe_slug, tag = _build_run_tag(datasets, params_cfg, run_tag)
    htf_value = _extract_primary_htf(params_cfg, datasets)
    htf_slug = _slugify_timeframe(htf_value)
    if base is None:
        root = DEFAULT_REPORT_ROOT
        output = root / tag
    else:
        output = base
    output = _next_available_dir(output)
    output.mkdir(parents=True, exist_ok=False)
    return output, {
        "timestamp": ts,
        "symbol": symbol_slug,
        "timeframe": timeframe_slug,
        "htf_timeframe": htf_slug,
        "tag": tag,
    }


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


def _parse_timeframe_grid(raw: Optional[str]) -> List[Tuple[str, Optional[str]]]:
    if not raw:
        return []
    combos: List[Tuple[str, Optional[str]]] = []
    text = str(raw).replace("\n", ",").replace(";", ",")
    for token in text.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        if "@" in candidate:
            ltf, htf = candidate.split("@", 1)
        elif ":" in candidate:
            ltf, htf = candidate.split(":", 1)
        else:
            ltf, htf = candidate, None
        ltf = ltf.strip()
        htf = htf.strip() if htf is not None else None
        if not ltf:
            continue
        combos.append((ltf, htf or None))
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
    htf = params_cfg.get("htf_timeframe") or (
        datasets[0].htf_timeframe if datasets and datasets[0].htf_timeframe else "nohtf"
    )
    htf_slug = str(htf or "nohtf").replace("/", "_")
    return STUDY_ROOT / f"{symbol_slug}_{timeframe_slug}_{htf_slug}.db"


def _extract_primary_htf(
    params_cfg: Dict[str, object],
    datasets: Sequence["DatasetSpec"],
) -> Optional[str]:
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
    htf = _extract_primary_htf(params_cfg, datasets)
    htf_slug = _slugify_timeframe(htf)
    suffix = f"_{space_hash[:6]}" if space_hash else ""
    return f"{symbol_slug}_{timeframe_slug}_{htf_slug}{suffix}"


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


def _ensure_dict(root: Dict[str, object], key: str) -> Dict[str, object]:
    value = root.get(key)
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


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

            htf_candidates = _to_list(
                entry.get("htf")
                or entry.get("htfs")
                or entry.get("htf_timeframes")
                or entry.get("htf_timeframe")
            )
            if not htf_candidates:
                htf_candidates = [None]

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
                for htf_tf in htf_candidates or [None]:
                    htf_text = str(htf_tf) if htf_tf else None
                    LOGGER.info(
                        "Preparing dataset %s %s (HTF %s) %s→%s",
                        symbol_log,
                        timeframe_text,
                        htf_text or "-",
                        start,
                        end,
                    )
                    df = cache.get(source_symbol, timeframe_text, start, end)
                    htf_df = (
                        cache.get(source_symbol, htf_text, start, end, allow_partial=True)
                        if htf_text
                        else None
                    )
                    datasets.append(
                        DatasetSpec(
                            symbol=display_symbol,
                            timeframe=timeframe_text,
                            start=start,
                            end=end,
                            df=df,
                            htf=htf_df,
                            htf_timeframe=htf_text,
                            source_symbol=source_symbol,
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

    htf_candidates = _to_list(params_cfg.get("htf_timeframes"))
    if not htf_candidates:
        htf_candidates = _to_list(params_cfg.get("htf_timeframe"))
    if not htf_candidates:
        htf_candidates = _to_list(backtest_cfg.get("htf_timeframes"))
    if not htf_candidates:
        htf_candidates = _to_list(backtest_cfg.get("htf_timeframe"))
    if not htf_candidates:
        htf_candidates = [None]

    symbol_pairs = [_resolve_symbol_entry(symbol, alias_map) for symbol in symbols]

    datasets: List[DatasetSpec] = []
    for (display_symbol, source_symbol), timeframe, period, htf_tf in product(
        symbol_pairs, timeframes, periods, htf_candidates
    ):
        start = str(period["from"])
        end = str(period["to"])
        symbol_log = (
            display_symbol if display_symbol == source_symbol else f"{display_symbol}→{source_symbol}"
        )
        LOGGER.info(
            "Preparing dataset %s %s (HTF %s) %s→%s",
            symbol_log,
            timeframe,
            htf_tf or "-",
            start,
            end,
        )
        df = cache.get(source_symbol, timeframe, start, end)
        htf = (
            cache.get(source_symbol, str(htf_tf), start, end, allow_partial=True)
            if htf_tf
            else None
        )
        datasets.append(
            DatasetSpec(
                symbol=display_symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                df=df,
                htf=htf,
                htf_timeframe=str(htf_tf) if htf_tf else None,
                source_symbol=source_symbol,
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

    for metrics in metric_list:
        returns = metrics.get("Returns")
        if isinstance(returns, pd.Series):
            combined_returns.append(returns)

        valid_flag = valid_flag and bool(metrics.get("Valid", True))

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
    else:
        combined_trades.sort(
            key=lambda trade: (
                getattr(trade, "entry_time", None),
                getattr(trade, "exit_time", None),
            )
        )
        aggregated = aggregate_metrics(combined_trades, merged_returns, simple=False)

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

    aggregated["Valid"] = valid_flag
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

    pf = _as_float(metrics.get("ProfitFactor"), 0.0)

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
        if isinstance(value, (int, float, bool, str)):
            clean[key] = value
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

    space = _restrict_to_basic_factors(original_space, enabled=use_basic_factors)
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
    if n_trials := int(search_cfg.get("n_trials", 0) or 0):
        auto_jobs = max(1, min(available_cpu, n_trials))
    else:
        auto_jobs = max(1, available_cpu)
    if n_jobs <= 1 and auto_jobs > n_jobs:
        n_jobs = auto_jobs
        search_cfg["n_jobs"] = n_jobs
        LOGGER.info("기본 팩터 프로파일: Optuna worker %d개 자동 할당", n_jobs)

    if n_jobs > 1:
        LOGGER.info("Optuna 병렬 worker %d개를 사용합니다.", n_jobs)

    raw_dataset_jobs = search_cfg.get("dataset_jobs") or search_cfg.get("dataset_n_jobs", 1)
    try:
        dataset_jobs = max(1, int(raw_dataset_jobs))
    except (TypeError, ValueError):
        LOGGER.warning(
            "search.dataset_jobs 값 '%s' 을 해석할 수 없어 1로 대체합니다.", raw_dataset_jobs
        )
        dataset_jobs = 1
    if dataset_jobs <= 1:
        max_parallel = min(len(datasets) or 1, max(1, available_cpu))
        if max_parallel > 1:
            dataset_jobs = max_parallel
            search_cfg["dataset_jobs"] = dataset_jobs
            LOGGER.info(
                "기본 팩터 프로파일: dataset_jobs=%d 자동 설정 (총 CPU=%d)",
                dataset_jobs,
                available_cpu,
            )

    dataset_executor = str(search_cfg.get("dataset_executor", "process") or "process").lower()
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

    if dataset_jobs > 1:
        optuna_budget = max(1, available_cpu // max(1, dataset_jobs))
        if optuna_budget < n_jobs:
            LOGGER.info(
                "데이터셋 병렬화를 위해 Optuna worker 수를 %d→%d 로 조정합니다.",
                n_jobs,
                optuna_budget,
            )
            n_jobs = optuna_budget
            search_cfg["n_jobs"] = n_jobs
        LOGGER.info(
            "데이터셋 병렬 백테스트 worker %d개 (%s) 사용", dataset_jobs, dataset_executor
        )
        if dataset_executor == "process" and dataset_start_method:
            LOGGER.info("프로세스 start method=%s", dataset_start_method)

    algo_raw = search_cfg.get("algo", "bayes")
    algo = str(algo_raw or "bayes").lower()
    seed = search_cfg.get("seed")
    n_trials = int(search_cfg.get("n_trials", 50))
    forced_params = forced_params or {}
    log_dir_path: Optional[Path] = Path(log_dir) if log_dir else None
    trial_log_path: Optional[Path] = None
    best_yaml_path: Optional[Path] = None
    final_csv_path: Optional[Path] = None
    if log_dir_path:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        trial_log_path = log_dir_path / "trials.jsonl"
        best_yaml_path = log_dir_path / "best.yaml"
        final_csv_path = log_dir_path / "trials_final.csv"
        for candidate in (trial_log_path, best_yaml_path, final_csv_path):
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
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)

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
        heartbeat_interval = max(int(search_cfg.get("heartbeat_interval", 60)), 0)
        heartbeat_grace = max(int(search_cfg.get("heartbeat_grace_period", 120)), 0)
        engine_kwargs = {}
        if storage_url.startswith("sqlite:///"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs=engine_kwargs or None,
            heartbeat_interval=heartbeat_interval or None,
            grace_period=heartbeat_grace or None,
        )
        storage_meta["backend"] = "sqlite" if storage_url.startswith("sqlite:///") else "rdb"
        if storage_url.startswith("sqlite:///"):
            storage_meta["url"] = storage_url
            storage_meta["path"] = str(study_storage) if study_storage else storage_url[10:]
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

    def _to_native(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial_log_path is None:
            return
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
        with trial_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if best_yaml_path is None:
            return

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

    callbacks: List = []
    if trial_log_path is not None:
        callbacks.append(_log_trial)

    def objective(trial: optuna.Trial) -> float:
        params = sample_parameters(trial, space)
        params.update(forced_params)
        key, selected_datasets = _select_datasets_for_params(
            params_cfg, dataset_groups, timeframe_groups, default_key, params
        )
        dataset_metrics: List[Dict[str, object]] = []
        numeric_metrics: List[Dict[str, float]] = []
        dataset_scores: List[float] = []

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
            numeric_metrics.append(metrics)
            dataset_metrics.append(
                {
                    "name": dataset.name,
                    "meta": dataset.meta,
                    "metrics": _clean_metrics(metrics),
                }
            )

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
                pruned_record = {
                    "trial": trial.number,
                    "params": params,
                    "metrics": _clean_metrics(partial_metrics),
                    "datasets": dataset_metrics,
                    "score": partial_score,
                    "valid": partial_metrics.get("Valid", True),
                    "dataset_key": {"timeframe": key[0], "htf_timeframe": key[1]},
                    "pruned": True,
                }
                if partial_objectives is not None:
                    pruned_record["objective_values"] = list(partial_objectives)
                results.append(pruned_record)
                raise optuna.TrialPruned()

        parallel_enabled = dataset_jobs > 1 and len(selected_datasets) > 1
        if parallel_enabled:
            executor_kwargs: Dict[str, object] = {"max_workers": dataset_jobs}
            if dataset_executor == "process":
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
            else:
                executor_cls = ThreadPoolExecutor

            futures = []
            with executor_cls(**executor_kwargs) as executor:
                for dataset in selected_datasets:
                    futures.append(
                        executor.submit(
                            _run_dataset_backtest_task,
                            dataset,
                            params,
                            fees,
                            risk,
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
                    metrics = run_backtest(dataset.df, params, fees, risk, htf_df=dataset.htf)
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

        record = {
            "trial": trial.number,
            "params": params,
            "metrics": _clean_metrics(aggregated),
            "datasets": dataset_metrics,
            "score": score,
            "valid": aggregated.get("Valid", True),
            "dataset_key": {"timeframe": key[0], "htf_timeframe": key[1]},
            "pruned": False,
        }
        if objective_values is not None:
            record["objective_values"] = list(objective_values)
        results.append(record)
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
        )

    use_llm = bool(llm_cfg.get("enabled"))
    llm_count = int(llm_cfg.get("count", 0)) if use_llm else 0
    llm_initial = int(llm_cfg.get("initial_trials", max(10, n_trials // 2))) if use_llm else 0
    llm_initial = max(0, min(llm_initial, n_trials))

    try:
        if use_llm and llm_count > 0 and 0 < llm_initial < n_trials:
            _run_optuna(llm_initial)
            candidates = generate_llm_candidates(space, study.trials, llm_cfg)
            for candidate in candidates[:llm_count]:
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
                df.to_csv(final_csv_path, index=False)

    if not results:
        raise RuntimeError("No completed trials were produced during optimisation.")

    def _profit_factor_key(record: Dict[str, object]) -> float:
        metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
        if not record.get("valid", True):
            return float("-inf")
        try:
            value = float(metrics.get("ProfitFactor", float("-inf")))
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
    parser.add_argument("--htf", type=str, help="Override higher timeframe for confirmations")
    parser.add_argument(
        "--timeframe-grid",
        type=str,
        help="Comma/semicolon separated LTF@HTF 조합을 일괄 실행 (예: '1m@15m,1m@1h')",
    )
    parser.add_argument("--start", type=str, help="Override backtest start date (ISO8601)")
    parser.add_argument("--end", type=str, help="Override backtest end date (ISO8601)")
    parser.add_argument("--leverage", type=float, help="Override leverage setting")
    parser.add_argument("--qty-pct", type=float, help="Override quantity percent")
    parser.add_argument(
        "--simple-metrics",
        action="store_true",
        help="간단 메트릭만 계산해 빠르게 탐색",
    )
    parser.add_argument(
        "--simple-profit",
        action="store_true",
        help="--simple-metrics 와 동일하게 ProfitFactor 전용 지표 경로를 사용합니다",
    )
    parser.add_argument(
        "--no-simple-metrics",
        action="store_true",
        help="단순 ProfitFactor 경로 강제 설정을 비활성화합니다",
    )
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
        help="데이터셋 병렬 워커 수 (process). 기본=코어수 절반",
    )
    parser.add_argument(
        "--dataset-executor",
        choices=["thread", "process"],
        help="데이터셋 병렬 백테스트 실행기를 지정합니다",
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
        help="--timeframe-grid 사용 시 스터디 이름 템플릿 (예: '{symbol_slug}_{ltf_slug}_{htf_slug}')",
    )
    parser.add_argument("--storage-url", type=str, help="Override Optuna storage URL (sqlite:/// or RDB)")
    parser.add_argument(
        "--storage-url-env",
        type=str,
        help="Optuna 스토리지 URL을 읽어올 환경 변수 이름을 덮어씁니다",
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
    simple_metrics_enabled = bool(
        getattr(args, "simple_metrics", False) or getattr(args, "simple_profit", False)
    )
    if getattr(args, "no_simple_metrics", False):
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
    backtest_cfg.setdefault("htf_timeframes", backtest_cfg.get("htf_timeframes", []))

    cli_tokens = list(argv or [])

    def _has_flag(flag: str) -> bool:
        return any(token == flag or token.startswith(f"{flag}=") for token in cli_tokens)

    search_cfg = _ensure_dict(params_cfg, "search")
    search_cfg.setdefault("n_jobs", DEFAULT_OPTUNA_JOBS)
    search_cfg.setdefault("dataset_jobs", DEFAULT_DATASET_JOBS)
    search_cfg.setdefault("dataset_executor", "process")

    batch_ctx = getattr(args, "_batch_context", None)
    if batch_ctx:
        suffix = batch_ctx.get("suffix") or (
            f"{batch_ctx.get('ltf_slug', '')}_{batch_ctx.get('htf_slug', '')}".strip("_")
        )
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

    if args.pruner:
        search_cfg["pruner"] = args.pruner

    forced_params: Dict[str, object] = dict(params_cfg.get("overrides", {}))
    auto_workers = bool(getattr(args, "auto_workers", False))
    for name in _collect_tokens(args.enable):
        forced_params[name] = True
    for name in _collect_tokens(args.disable):
        forced_params[name] = False

    symbol_choices = list(dict.fromkeys(backtest_cfg.get("symbols") or ([params_cfg.get("symbol")] if params_cfg.get("symbol") else [])))

    def _collect_htfs(cfg: Dict[str, object]) -> List[str]:
        values: List[str] = []
        raw = cfg.get("htf_timeframes")
        if isinstance(raw, (list, tuple)):
            values.extend(str(item) for item in raw if item)
        single = cfg.get("htf_timeframe")
        if single:
            values.append(str(single))
        return values

    htf_choices = list(dict.fromkeys(_collect_htfs(backtest_cfg) or _collect_htfs(params_cfg)))

    selected_symbol = args.symbol or params_cfg.get("symbol") or (symbol_choices[0] if symbol_choices else None)
    selected_timeframe = args.timeframe
    selected_htf: Optional[str] = args.htf if args.htf else None
    timeframe_overridden = args.timeframe is not None
    htf_overridden = args.htf is not None

    if args.interactive and symbol_choices:
        selected_symbol = _prompt_choice("Select symbol", symbol_choices, selected_symbol)

    if selected_symbol:
        params_cfg["symbol"] = selected_symbol
        backtest_cfg["symbols"] = [selected_symbol]
    if timeframe_overridden and selected_timeframe:
        params_cfg["timeframe"] = selected_timeframe
        backtest_cfg["timeframes"] = [selected_timeframe]
    if htf_overridden and selected_htf is not None:
        params_cfg["htf_timeframes"] = [selected_htf]
        backtest_cfg["htf_timeframes"] = [selected_htf]
    elif htf_choices:
        params_cfg["htf_timeframes"] = htf_choices
        backtest_cfg["htf_timeframes"] = htf_choices
    params_cfg.pop("htf_timeframe", None)
    backtest_cfg.pop("htf_timeframe", None)

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

    def _apply_simple_metrics(decision: bool) -> None:
        forced_params["simpleMetricsOnly"] = decision
        forced_params["simpleProfitOnly"] = decision
        risk_cfg["simpleMetricsOnly"] = decision
        risk_cfg["simpleProfitOnly"] = decision
        backtest_risk["simpleMetricsOnly"] = decision
        backtest_risk["simpleProfitOnly"] = decision

    # 기본 모드는 ProfitFactor 중심 단순 지표 경로를 강제해 계산량을 줄입니다.
    _apply_simple_metrics(True)
    if getattr(args, "no_simple_metrics", False):
        _apply_simple_metrics(False)
    elif getattr(args, "simple_metrics", False) or getattr(args, "simple_profit", False):
        _apply_simple_metrics(True)

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

        simple_default: Optional[bool] = None
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
                simple_default = coerced
                break
        decision = _prompt_bool(
            "단순 ProfitFactor 전용 지표 경로 사용", simple_default
        )
        if decision is not None:
            _apply_simple_metrics(decision)

    global simple_metrics_enabled
    simple_metrics_enabled = bool(forced_params.get("simpleMetricsOnly", False))
    if simple_metrics_enabled:
        LOGGER.info("단순 ProfitFactor 기반 계산 경로를 강제 활성화합니다.")
    else:
        LOGGER.info("전체 지표 계산 경로를 사용합니다.")

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

    study = optimisation.get("study")
    if study is not None:
        write_trials_dataframe(study, output_dir)
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

    wf_summary = run_walk_forward(
        primary_dataset.df,
        best_record["params"],
        fees,
        risk,
        train_bars=int(walk_cfg.get("train_bars", 5000)),
        test_bars=int(walk_cfg.get("test_bars", 2000)),
        step=int(walk_cfg.get("step", 2000)),
        htf_df=primary_dataset.htf,
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
            value = float(metrics.get("ProfitFactor", float("-inf")))
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
            "params": best_record.get("params"),
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
            candidate_wf = run_walk_forward(
                candidate_dataset.df,
                record["params"],
                fees,
                risk,
                train_bars=int(walk_cfg.get("train_bars", 5000)),
                test_bars=int(walk_cfg.get("test_bars", 2000)),
                step=int(walk_cfg.get("step", 2000)),
                htf_df=candidate_dataset.htf,
            )
            wf_cache[record["trial"]] = candidate_wf
            candidate_summaries.append(
                {
                    "trial": record["trial"],
                    "score": record.get("score"),
                    "oos_mean": candidate_wf.get("oos_mean"),
                    "params": record.get("params"),
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
        "params": best_record.get("params"),
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
        entry = {
            "trial": item["trial"],
            "score": item.get("score"),
            "oos_mean": item.get("oos_mean"),
            "params": filtered_params,
            "metrics": trial_record.get("metrics"),
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
        "regime": regime_summary.__dict__,
        "cli": list(argv or []),
    }
    if storage_meta:
        manifest["storage"] = storage_meta
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
    )

    LOGGER.info("Run complete. Outputs saved to %s", output_dir)


def execute(args: argparse.Namespace, argv: Optional[Sequence[str]] = None) -> None:
    """Execute one or more optimisation runs based on CLI arguments."""

    params_cfg = load_yaml(args.params)
    backtest_cfg = load_yaml(args.backtest)

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
    combo_summary = ", ".join(f"{ltf}/{htf or '-'}" for ltf, htf in combos)
    LOGGER.info("타임프레임 그리드 %d건 실행: %s", total, combo_summary)

    for index, (ltf, htf) in enumerate(combos, start=1):
        batch_args = argparse.Namespace(**vars(args))
        batch_args.timeframe = ltf
        batch_args.htf = htf
        suffix = f"{_slugify_timeframe(ltf)}_{_slugify_timeframe(htf)}".strip("_")
        context = {
            "index": index,
            "total": total,
            "ltf": ltf,
            "htf": htf,
            "ltf_slug": _slugify_timeframe(ltf),
            "htf_slug": _slugify_timeframe(htf),
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
            "(%d/%d) LTF=%s, HTF=%s 조합 최적화 시작",
            index,
            total,
            ltf,
            htf or "없음",
        )
        _execute_single(batch_args, params_cfg, backtest_cfg, argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for ``python -m optimize.run``."""

    args = parse_args(argv)
    execute(args, argv)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
