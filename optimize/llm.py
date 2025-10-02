"""LLM-assisted parameter suggestion helpers."""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

import optuna

from optimize.search_spaces import SpaceSpec

LOGGER = logging.getLogger("optimize.llm")

try:  # pragma: no cover - optional dependency
    from google import genai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    genai = None

HARDCODED_GEMINI_API_KEY = "AIzaSyDD1i5TbCqfWEMFunoxtvnpnr0VW3XZtsY"


def _extract_text(response: object) -> str:
    if response is None:
        return ""
    parts: List[str] = []
    text = getattr(response, "text", None)
    if isinstance(text, str):
        parts.append(text)
    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            content_parts = getattr(content, "parts", None) if content else None
            if content_parts:
                for part in content_parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str):
                        parts.append(part_text)
    return "\n".join(parts).strip()


def _extract_json_payload(raw: str) -> Optional[object]:
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # drop opening fence
        lines = lines[1:]
        # drop closing fence if present
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = None
    end = None
    for token in ("[", "{"):
        idx = cleaned.find(token)
        if idx != -1 and (start is None or idx < start):
            start = idx
    for token in ("]", "}"):
        idx = cleaned.rfind(token)
        if idx != -1 and (end is None or idx > end):
            end = idx
    if start is not None and end is not None and start < end:
        cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def _coerce_numeric(value: object, *, to_int: bool = False) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if to_int:
        return float(int(round(numeric)))
    return numeric


def _validate_candidate(candidate: Dict[str, object], space: SpaceSpec) -> Optional[Dict[str, object]]:
    validated: Dict[str, object] = {}
    for name, spec in space.items():
        if name not in candidate:
            continue
        value = candidate[name]
        dtype = spec["type"]
        if dtype == "int":
            numeric = _coerce_numeric(value, to_int=True)
            if numeric is None:
                return None
            low = int(spec["min"])
            high = int(spec["max"])
            step = int(spec.get("step", 1))
            as_int = int(numeric)
            if as_int < low or as_int > high:
                return None
            if step:
                offset = as_int - low
                as_int = low + round(offset / step) * step
                as_int = max(low, min(high, as_int))
            validated[name] = int(as_int)
        elif dtype == "float":
            numeric = _coerce_numeric(value)
            if numeric is None:
                return None
            low = float(spec["min"])
            high = float(spec["max"])
            if numeric < low or numeric > high:
                return None
            step = float(spec.get("step", 0.0))
            if step:
                offset = numeric - low
                numeric = low + round(offset / step) * step
                numeric = max(low, min(high, numeric))
            validated[name] = float(numeric)
        elif dtype == "bool":
            if isinstance(value, str):
                normalised = value.strip().lower()
                validated[name] = normalised in {"true", "1", "yes", "on"}
            else:
                validated[name] = bool(value)
        elif dtype in {"choice", "str", "string"}:
            values = list(
                spec.get("values") or spec.get("options") or spec.get("choices") or []
            )
            if not values:
                return None
            if value in values:
                validated[name] = value
            else:
                normalised = str(value).strip().lower()
                match = next((option for option in values if str(option).strip().lower() == normalised), None)
                if match is None:
                    return None
                validated[name] = match
        else:
            # Unsupported type for LLM suggestions.
            return None
    return validated


def _coerce_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _trial_objective_values(trial: optuna.trial.FrozenTrial) -> Optional[List[object]]:
    raw_values = getattr(trial, "values", None)
    if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
        return list(raw_values)
    raw_value = getattr(trial, "value", None)
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        return list(raw_value)
    return None


def _trial_priority(
    trial: optuna.trial.FrozenTrial, config: Dict[str, object]
) -> float:
    direct = _coerce_float(getattr(trial, "value", None))
    if direct is not None:
        return direct

    values = _trial_objective_values(trial)
    if not values:
        return float("-inf")

    # 멀티 목적 최적화 시 우선순위로 사용할 지표 인덱스를 명시적으로 지정할 수 있습니다.
    index = config.get("objective_index")
    if isinstance(index, int) and 0 <= index < len(values):
        indexed = _coerce_float(values[index])
        if indexed is not None:
            return indexed

    for candidate in values:
        numeric = _coerce_float(candidate)
        if numeric is not None:
            return numeric
    return float("-inf")


def _serialise_objective_values(values: Optional[List[object]]) -> Optional[List[object]]:
    if not values:
        return None
    serialised: List[object] = []
    for value in values:
        numeric = _coerce_float(value)
        serialised.append(numeric if numeric is not None else value)
    return serialised


def _has_objective_values(trial: optuna.trial.FrozenTrial) -> bool:
    value = getattr(trial, "value", None)
    if value is not None:
        return True
    values = _trial_objective_values(trial)
    if not values:
        return False
    return any(item is not None for item in values)


def generate_llm_candidates(
    space: SpaceSpec,
    trials: Iterable[optuna.trial.FrozenTrial],
    config: Dict[str, object],
) -> List[Dict[str, object]]:
    if not config or not config.get("enabled"):
        return []
    if genai is None:
        LOGGER.warning("google-genai is not installed; skipping Gemini-guided proposals.")
        return []

    api_key = HARDCODED_GEMINI_API_KEY or config.get("api_key") or os.environ.get(
        str(config.get("api_key_env", "GEMINI_API_KEY"))
    )
    if not api_key:
        LOGGER.warning("Gemini API 키가 설정되지 않아 LLM 제안을 건너뜁니다.")
        return []

    finished_trials: List[optuna.trial.FrozenTrial] = [
        trial
        for trial in trials
        if trial.state.is_finished() and _has_objective_values(trial)
    ]
    if not finished_trials:
        LOGGER.info("아직 완료된 트라이얼이 없어 LLM 제안을 생략합니다.")
        return []

    top_n = max(int(config.get("top_n", 10)), 1)
    count = max(int(config.get("count", 8)), 1)
    sorted_trials = sorted(
        finished_trials,
        key=lambda trial: _trial_priority(trial, config),
        reverse=True,
    )
    top_trials: List[Dict[str, object]] = []
    for trial in sorted_trials[:top_n]:
        priority = _trial_priority(trial, config)
        entry: Dict[str, object] = {
            "number": trial.number,
            "value": priority if math.isfinite(priority) else None,
            "params": trial.params,
        }
        values = _serialise_objective_values(_trial_objective_values(trial))
        if values is not None:
            entry["values"] = values
        top_trials.append(entry)

    client = genai.Client(api_key=api_key)
    model = str(config.get("model", "gemini-2.0-flash-exp"))
    prompt = (
        "You are assisting with hyper-parameter optimisation for a trading strategy.\n"
        "The search space is defined by the following JSON (types: int, float, bool, choice):\n"
        f"{json.dumps(space, indent=2)}\n\n"
        "Here are the top completed trials with their objective values (higher is better):\n"
        f"{json.dumps(top_trials, indent=2)}\n\n"
        f"Propose {count} new parameter sets strictly within the given bounds."
        " Return only a JSON array of objects with keys matching the parameter names."
    )

    # Optional thinking budget / generation config support for Gemini 2.5+ models.
    generation_config = None
    thinking_budget = config.get("thinking_budget")
    if thinking_budget is not None:
        try:
            # Try to build a GenerationConfig if types are available. The
            # google-genai SDK exposes these under genai.types. Not all versions
            # include a ThinkingConfig, so we fall back gracefully on failure.
            budget_int = int(thinking_budget)
            try:
                # pylint: disable=no-member
                GenerationConfig = getattr(genai, "types", None)
                if GenerationConfig is not None:
                    types_module = GenerationConfig
                    # Build thinking_config
                    thinking_cls = getattr(types_module, "ThinkingConfig", None)
                    gen_cfg_cls = getattr(types_module, "GenerationConfig", None)
                    if thinking_cls is not None and gen_cfg_cls is not None:
                        generation_config = gen_cfg_cls(
                            thinking_config=thinking_cls(thinking_budget=budget_int)
                        )
            except Exception:
                generation_config = None
        except Exception:
            generation_config = None

    try:  # pragma: no cover - network side effects
        # Pass the optional generation_config if available. The SDK will ignore
        # unknown kwargs.
        if generation_config is not None:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                generation_config=generation_config,
            )
        else:
            response = client.models.generate_content(model=model, contents=prompt)
    except Exception as exc:  # pragma: no cover - network side effects
        LOGGER.warning("Gemini 호출에 실패했습니다: %s", exc)
        return []

    raw_text = _extract_text(response)
    payload = _extract_json_payload(raw_text)
    if not isinstance(payload, list):
        LOGGER.warning("Gemini 응답에서 유효한 JSON 배열을 찾지 못했습니다.")
        return []

    accepted: List[Dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        validated = _validate_candidate(entry, space)
        if not validated:
            continue
        accepted.append(validated)
        if len(accepted) >= count:
            break

    if accepted:
        LOGGER.info("Gemini가 제안한 %d개의 후보를 큐에 추가합니다.", len(accepted))
    else:
        LOGGER.info("Gemini 제안 중 조건을 만족하는 후보가 없었습니다.")
    return accepted
