"""Helpers for translating YAML search spaces to Optuna."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import optuna

SpaceSpec = Dict[str, Dict[str, object]]


@dataclass(frozen=True)
class CategoricalDistribution:
    name: str
    choices: Sequence[object]


@dataclass(frozen=True)
class IntDistribution:
    name: str
    low: int
    high: int
    step: int = 1


@dataclass(frozen=True)
class FloatDistribution:
    name: str
    low: float
    high: float
    step: float = 0.1


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


def get_search_spaces() -> List[object]:
    """Optuna 탐색에 사용할 기본 파라미터 공간을 반환합니다."""

    spaces = [
        # 1. Squeeze Momentum
        CategoricalDistribution(name="use_squeeze_momentum", choices=[True]),
        IntDistribution(name="len", low=5, high=40, step=1),
        IntDistribution(name="sig", low=1, high=20, step=1),
        IntDistribution(name="bbLen", low=10, high=100, step=1),
        FloatDistribution(name="bbMult", low=0.5, high=3.0, step=0.1),
        IntDistribution(name="kcLen", low=10, high=100, step=1),
        FloatDistribution(name="kcMult", low=0.5, high=3.0, step=0.1),

        # 2. Directional Flux
        CategoricalDistribution(name="use_directional_flux", choices=[True]),
        IntDistribution(name="dfl", low=5, high=50, step=1),
        IntDistribution(name="dfSmoothLen", low=1, high=20, step=1),
        CategoricalDistribution(name="dfh", choices=[True, False]),

        # 3. Exit Options
        CategoricalDistribution(name="exitOpposite", choices=[True, False]),
        CategoricalDistribution(name="useMomFade", choices=[True, False]),
        CategoricalDistribution(name="fadeMode", choices=["VN", "Legacy"]),
        CategoricalDistribution(name="useAtrStop", choices=[True, False]),
        FloatDistribution(name="atrStopMult", low=0.5, high=5.0, step=0.1),
        CategoricalDistribution(name="useFixedStop", choices=[True, False]),
        FloatDistribution(name="fixedStopPct", low=0.5, high=5.0, step=0.1),
        CategoricalDistribution(name="useStopLoss", choices=[True, False]),
        IntDistribution(name="stopLookback", low=2, high=50, step=1),
        CategoricalDistribution(name="usePivotStop", choices=[True, False]),
        IntDistribution(name="pivotLen", low=2, high=20, step=1),
        CategoricalDistribution(name="useAtrTrail", choices=[True, False]),
        IntDistribution(name="atrTrailLen", low=5, high=100, step=1),
        FloatDistribution(name="atrTrailMult", low=1.0, high=7.0, step=0.1),
        CategoricalDistribution(name="useBreakeven", choices=[True, False]),
        FloatDistribution(name="breakevenMult", low=0.5, high=5.0, step=0.1),
        CategoricalDistribution(name="useTimeStop", choices=[True, False]),
        IntDistribution(name="maxHoldBars", low=5, high=100, step=1),
    ]

    return spaces
