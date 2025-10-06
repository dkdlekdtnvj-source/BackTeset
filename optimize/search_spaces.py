"""Helpers for translating YAML search spaces to Optuna."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import optuna

SpaceSpec = Dict[str, Dict[str, object]]


def build_space(space: SpaceSpec) -> SpaceSpec:
    return space


def _requirements_met(params: Dict[str, object], requirement: Union[str, Sequence[object], Dict[str, object]]) -> bool:
    if requirement in (None, ""):
        return True
    if isinstance(requirement, dict):
        name = requirement.get("name") or requirement.get("param") or requirement.get("key")
        expected = requirement.get("equals")
        if expected is None:
            expected = requirement.get("value", True)
        if not name:
            return True
        if name not in params:
            return False
        return params.get(name) == expected
    if isinstance(requirement, (list, tuple, set)):
        return all(_requirements_met(params, item) for item in requirement)
    # Treat plain strings as boolean flags that must evaluate to truthy
    name = str(requirement)
    if name not in params:
        return False
    return bool(params.get(name))


def sample_parameters(trial: optuna.Trial, space: SpaceSpec) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for name, spec in space.items():
        requires = spec.get("requires")
        if requires and not _requirements_met(params, requires):
            if "default" in spec:
                params[name] = spec["default"]
            continue
        dtype = spec["type"]
        if dtype == "int":
            params[name] = trial.suggest_int(name, int(spec["min"]), int(spec["max"]), step=int(spec.get("step", 1)))
        elif dtype == "float":
            params[name] = trial.suggest_float(name, float(spec["min"]), float(spec["max"]), step=float(spec.get("step", 0.1)))
        elif dtype == "bool":
            params[name] = trial.suggest_categorical(name, [True, False])
        elif dtype in {"choice", "str", "string"}:
            values = spec.get("values") or spec.get("options") or spec.get("choices")
            if not values:
                raise ValueError(f"Choice parameter '{name}' requires a non-empty 'values' list.")
            params[name] = trial.suggest_categorical(name, list(values))
        else:
            raise ValueError(f"Unsupported parameter type: {dtype}")
    return params


def grid_choices(space: SpaceSpec) -> Dict[str, List[object]]:
    grid: Dict[str, List[object]] = {}
    for name, spec in space.items():
        dtype = spec["type"]
        if dtype == "int":
            grid[name] = list(range(int(spec["min"]), int(spec["max"]) + 1, int(spec.get("step", 1))))
        elif dtype == "float":
            step = float(spec.get("step", 0.1))
            values = np.arange(float(spec["min"]), float(spec["max"]) + 1e-12, step)
            grid[name] = [round(val, 10) for val in values.tolist()]
        elif dtype == "bool":
            grid[name] = [True, False]
        elif dtype in {"choice", "str", "string"}:
            values = spec.get("values") or spec.get("options") or spec.get("choices")
            if not values:
                raise ValueError(f"Choice parameter '{name}' requires a non-empty 'values' list for grid sampling.")
            grid[name] = list(values)
        else:
            raise ValueError(f"Unsupported parameter type for grid: {dtype}")
    return grid


def mutate_around(
    params: Dict[str, object],
    space: SpaceSpec,
    scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    rng = rng or np.random.default_rng()
    mutated = dict(params)
    for name, spec in space.items():
        if name not in params:
            continue
        dtype = spec["type"]
        value = params[name]
        if dtype == "int":
            width = max(int((spec["max"] - spec["min"]) * scale), int(spec.get("step", 1)))
            low = int(spec["min"])
            high = int(spec["max"])
            step = int(spec.get("step", 1)) or 1
            jitter = rng.integers(-width, width + 1)
            candidate = int(value) + jitter
            offset = candidate - low
            candidate = low + int(round(offset / step)) * step
            candidate = max(low, min(high, candidate))
            mutated[name] = candidate
        elif dtype == "float":
            span = float(spec["max"] - spec["min"]) * scale
            low = float(spec["min"])
            high = float(spec["max"])
            step = float(spec.get("step", 0.0))
            jitter = rng.normal(0.0, span or 1e-6)
            candidate = float(value) + jitter
            offset = candidate - low
            if step:
                candidate = low + round(offset / step) * step
            candidate = max(low, min(high, candidate))
            precision = int(spec.get("precision", 0)) if "precision" in spec else None
            if precision is not None and precision >= 0:
                candidate = round(candidate, precision)
                candidate = max(low, min(high, candidate))
            mutated[name] = float(candidate)
        elif dtype in {"bool", "choice", "str", "string"}:
            if rng.random() < 0.2:
                # Flip bool or pick another categorical option.
                if dtype == "bool":
                    mutated[name] = not bool(value)
                else:
                    values = list(
                        spec.get("values") or spec.get("options") or spec.get("choices") or []
                    )
                    if values:
                        choices = [option for option in values if option != value]
                        if choices:
                            mutated[name] = rng.choice(choices)
        else:
            continue
    return mutated


def get_search_spaces() -> List[Dict[str, object]]:
    """Optuna 탐색 공간 프리셋을 반환합니다."""

    spaces: List[Dict[str, object]] = [
        {
            "key": "sun_deluxe_core",
            "label": "쑨모멘텀 디럭스 – 핵심 파라미터",
            "space": build_space(
                {
                    "oscLen": {"type": "int", "min": 10, "max": 40, "step": 2},
                    "signalLen": {"type": "int", "min": 2, "max": 12, "step": 1},
                    "kcLen": {"type": "int", "min": 10, "max": 30, "step": 2},
                    "kcMult": {"type": "float", "min": 1.0, "max": 2.5, "step": 0.1},
                    "fluxLen": {"type": "int", "min": 10, "max": 40, "step": 2},
                    "fluxSmoothLen": {"type": "int", "min": 1, "max": 7, "step": 1},
                    "fluxSmoothType": {
                        "type": "choice",
                        "values": ["기본", "EMA", "HMA"],
                    },
                    "useFluxHeikin": {"type": "bool"},
                    "maType": {
                        "type": "choice",
                        "values": ["기본", "EMA", "HMA"],
                    },
                    "useDynamicThresh": {"type": "bool"},
                    "useSymThreshold": {
                        "type": "bool",
                        "requires": {"name": "useDynamicThresh", "equals": False},
                    },
                    "statThreshold": {
                        "type": "float",
                        "min": 20.0,
                        "max": 80.0,
                        "step": 2.0,
                        "requires": {"name": "useDynamicThresh", "equals": False},
                    },
                    "buyThreshold": {
                        "type": "float",
                        "min": 20.0,
                        "max": 60.0,
                        "step": 2.0,
                        "requires": {"name": "useDynamicThresh", "equals": False},
                    },
                    "sellThreshold": {
                        "type": "float",
                        "min": 20.0,
                        "max": 60.0,
                        "step": 2.0,
                        "requires": {"name": "useDynamicThresh", "equals": False},
                    },
                    "dynLen": {
                        "type": "int",
                        "min": 15,
                        "max": 60,
                        "step": 5,
                        "requires": {"name": "useDynamicThresh", "equals": True},
                    },
                    "dynMult": {
                        "type": "float",
                        "min": 0.8,
                        "max": 2.0,
                        "step": 0.1,
                        "requires": {"name": "useDynamicThresh", "equals": True},
                    },
                    "exitOpposite": {"type": "bool"},
                    "useMomFade": {"type": "bool"},
                    "momFadeRegLen": {
                        "type": "int",
                        "min": 10,
                        "max": 40,
                        "step": 2,
                        "requires": "useMomFade",
                    },
                    "momFadeBbLen": {
                        "type": "int",
                        "min": 10,
                        "max": 40,
                        "step": 2,
                        "requires": "useMomFade",
                    },
                    "momFadeKcLen": {
                        "type": "int",
                        "min": 10,
                        "max": 40,
                        "step": 2,
                        "requires": "useMomFade",
                    },
                    "momFadeBbMult": {
                        "type": "float",
                        "min": 1.5,
                        "max": 3.0,
                        "step": 0.25,
                        "requires": "useMomFade",
                    },
                    "momFadeKcMult": {
                        "type": "float",
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.25,
                        "requires": "useMomFade",
                    },
                }
            ),
        }
    ]

    return spaces
