"""
Wrapper around the original `run_backtest` function from the
``optimize.strategy_model`` module.

The purpose of this wrapper is to enforce that various high time
frame‑related options are disabled when running the backtest.  The
user indicated that they never use these filters in the Python
back‑tester and that enabling them only slows the calculation down.

By copying the incoming ``params`` dictionary and overriding the
relevant keys to ``False``, this wrapper ensures that the underlying
``run_backtest`` function behaves as if those options were never
requested.  All other parameters are forwarded unchanged.

To use this wrapper, import ``run_backtest`` from this module
instead of directly from ``optimize.strategy_model``.

Note: this module depends on the original ``optimize.strategy_model``
being importable from your environment.  No other functionality from
``strategy_model`` is modified.
"""

from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

# Import the original run_backtest function.  It will be invoked once
# we've patched the parameter dictionary.
from optimize.strategy_model import run_backtest as _original_run_backtest


def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, float | bool | str],
    fees: Dict[str, float],
    risk: Dict[str, float | bool],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    """Run a backtest with certain high time‑frame options forcibly disabled.

    Parameters
    ----------
    df : pd.DataFrame
        The OHLCV data set to backtest.  This must already have a
        DatetimeIndex and the columns ``open``, ``high``, ``low``,
        ``close`` and ``volume``.
    params : Dict[str, float | bool | str]
        The parameter dictionary.  This wrapper will create a copy of
        the dictionary and set several HTF‑related keys to ``False``.
    fees : Dict[str, float]
        Commission and slippage settings.
    risk : Dict[str, float | bool]
        Risk management parameters such as leverage and initial capital.
    htf_df : Optional[pd.DataFrame], default ``None``
        Optional high time frame data.  This is forwarded unchanged.
    min_trades : Optional[int], default ``None``
        Minimum trades requirement.  Forwarded unchanged.

    Returns
    -------
    Dict[str, float]
        A dictionary of aggregated performance metrics, as returned
        by the original ``run_backtest``.

    This wrapper does not modify the underlying implementation of
    ``run_backtest``.  It simply ensures that certain high
    time‑frame features (trend, range, regime, HMA filter, slope
    filter, distance guard, equity slope filter, pivot HTF, squeeze
    gate and structure gate) are disabled before delegating to the
    original function.
    """
    # Create a copy of the params dictionary to avoid mutating the
    # caller's object.  If ``params`` is ``None`` it would be
    # unexpected, but we guard against it by starting with an empty
    # dictionary.
    patched_params: Dict[str, float | bool | str] = dict(params or {})

    # Keys corresponding to high‑time‑frame or complex filters that
    # should be disabled for the Python back‑tester.  See
    # ``strategy_model.py`` for the full list of available options.
    disable_keys = [
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

    # Set each of the above keys to False.  If the key does not
    # already exist, it will simply be inserted with a False value.
    for key in disable_keys:
        patched_params[key] = False

    # Delegate to the original backtest function with the patched
    # parameter dictionary.  All other arguments are forwarded
    # unchanged.
    return _original_run_backtest(
        df,
        patched_params,
        fees,
        risk,
        htf_df=htf_df,
        min_trades=min_trades,
    )