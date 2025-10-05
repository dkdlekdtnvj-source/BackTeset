"""
Wrapper around the original ``optimize.metrics`` module to adjust
behaviour of the ``sortino_ratio`` function.  In the original
implementation the Sortino ratio returns zero if there are no
downside returns.  This can lead to misleading metrics when a
strategy only produces non‑negative returns but still has variance.

The wrapper re‑exports all public names from ``optimize.metrics`` and
redefines ``sortino_ratio`` to fall back to a Sharpe‑style
calculation when there are no downside returns.  If there are no
returns at all or the standard deviation is zero, the ratio will
still return zero.

To use the patched metrics, import from this module instead of
``optimize.metrics``.
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Sequence
import numpy as np
import pandas as pd

# Import everything from the original metrics module.  The wildcard
# import is safe here because we are essentially wrapping the module.
from optimize.metrics import *  # type: ignore


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Compute the Sortino ratio with a fallback when there are no downside returns.

    Parameters
    ----------
    returns : pd.Series
        A series of periodic returns (percentage or fractional).
    risk_free : float, default 0.0
        The risk‑free rate used as the target return.  Any returns
        below this level are considered downside risk.

    Returns
    -------
    float
        The Sortino ratio, defined as the expected return minus the
        risk free rate, divided by the standard deviation of
        downside returns.  When there are no returns below the
        ``risk_free`` threshold the function falls back to a Sharpe
        ratio calculation (using the overall standard deviation).
    """
    # Identify returns below the risk‑free rate.  Replace infinities and
    # drop NaNs to avoid propagating them into the standard deviation.
    downside = returns[returns < risk_free]
    downside = downside.replace([np.inf, -np.inf], np.nan).dropna()

    # When there are no downside returns, fall back to a Sharpe
    # calculation.  This avoids returning zero for strategies that never
    # produce losses but still have variability.
    if downside.empty:
        cleaned = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if cleaned.empty:
            return 0.0
        with np.errstate(invalid="ignore"):
            std = cleaned.std(ddof=0)
        if std == 0 or np.isnan(std):
            return 0.0
        return float((cleaned.mean() - risk_free) / std)

    # Otherwise compute the Sortino ratio normally.
    expected = returns.replace([np.inf, -np.inf], np.nan).dropna().mean() - risk_free
    with np.errstate(invalid="ignore"):
        downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(expected / downside_std)