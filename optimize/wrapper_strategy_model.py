"""Utility wrapper for ``optimize.strategy_model.run_backtest``."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from optimize.strategy_model import run_backtest as _original_run_backtest


_DISABLE_KEYS = [
    "useHtfTrend",
    "useRangeFilter",
    "useRegimeFilter",
    "useHmaFilter",
    "useSlopeFilter",
    "useDistanceGuard",
    "useEquitySlopeFilter",
    "usePivotHtf",
    "useSqzGate",
    "useStructureGate",
]


def _patched_params(params: Optional[Dict[str, float | bool | str]]) -> Dict[str, float | bool | str]:
    patched: Dict[str, float | bool | str] = dict(params or {})
    for key in _DISABLE_KEYS:
        patched[key] = False
    return patched


def run_backtest_wrapped(
    df: pd.DataFrame,
    params: Dict[str, float | bool | str],
    fees: Dict[str, float],
    risk: Dict[str, float | bool],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    """Run the original backtest after forcing specific HTF options off."""

    return _original_run_backtest(
        df,
        _patched_params(params),
        fees,
        risk,
        htf_df=htf_df,
        min_trades=min_trades,
    )


def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, float | bool | str],
    fees: Dict[str, float],
    risk: Dict[str, float | bool],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    """Compatibility alias that calls :func:`run_backtest_wrapped`."""

    return run_backtest_wrapped(
        df,
        params,
        fees,
        risk,
        htf_df=htf_df,
        min_trades=min_trades,
    )
