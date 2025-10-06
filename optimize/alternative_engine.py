"""Alternate backtesting engines integration.

This module provides a wrapper interface to optional third-party backtesting
frameworks such as **PyBroker** and **vectorbt**.  It exposes a single
function, :func:`run_backtest_alternative`, which is designed to mirror the
signature of :func:`~optimize.strategy_model.run_backtest`.  When invoked,
the function attempts to delegate the backtest to the requested engine
(``engine`` argument).  If the specified engine is not installed or
integration has not yet been implemented, the function will raise a
``NotImplementedError``.  This design allows the core optimisation loop to
fallback to the native Python implementation without causing compilation
errors when the optional dependencies are missing.

Currently, this module contains only stubs for ``pybroker`` and
``vectorbt``.  Full integration would involve mapping the Pine strategy
logic to the respective framework's API.  The stubs are provided to
demonstrate how such an integration could be wired into the optimisation
pipeline without breaking the build.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

import logging

LOGGER = logging.getLogger(__name__)

try:
    # Optional import for PyBroker
    import pybroker  # type: ignore  # noqa: F401
    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False

try:
    # Optional import for vectorbt
    import vectorbt  # type: ignore  # noqa: F401
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False


def run_backtest_alternative(
    df: pd.DataFrame,
    params: Dict[str, object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
    *,
    engine: str = "vectorbt",
) -> Dict[str, float]:
    """
    Execute a backtest using an alternative engine.

    This function attempts to delegate the backtest to either **vectorbt** or
    **PyBroker** based on the value of the ``engine`` argument.  If the
    requested engine is not available or integration has not been implemented,
    a ``NotImplementedError`` will be raised.  Callers should catch this
    exception and fallback to the default Python implementation as needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Minute-level OHLCV data to be backtested.
    params : Dict[str, object]
        Strategy parameters, consistent with those accepted by
        :func:`~optimize.strategy_model.run_backtest`.
    fees : Dict[str, float]
        Trading fee configuration.
    risk : Dict[str, float]
        Risk management configuration.
    htf_df : Optional[pandas.DataFrame], optional
        Higher time frame data, by default ``None``.
    min_trades : Optional[int], optional
        Minimum number of trades required to avoid penalties.  Not yet
        incorporated into the alternative engines.
    engine : str, optional
        Name of the engine to use.  Supported values are ``"vectorbt"``
        and ``"pybroker"``.  Defaults to ``"vectorbt"``.

    Returns
    -------
    Dict[str, float]
        A dictionary of aggregated performance metrics.

    Raises
    ------
    NotImplementedError
        If the requested engine is unavailable or integration has not been
        implemented.
    ImportError
        If the requested engine is not installed in the runtime environment.
    """
    engine = str(engine or "").strip().lower()
    if engine in {"vectorbt", "vectorbtpro", "vbt"}:
        if not VECTORBT_AVAILABLE:
            raise ImportError("vectorbt is not installed in this environment")
        # Placeholder: integration with vectorbt should be implemented here
        # A full implementation would convert the Pine strategy logic into
        # vectorbt's portfolio API using indicator functions and entries/exits.
        raise NotImplementedError(
            "vectorbt engine integration is not yet implemented."
        )
    elif engine in {"pybroker", "pb"}:
        if not PYBROKER_AVAILABLE:
            raise ImportError("pybroker is not installed in this environment")
        # Placeholder: integration with PyBroker should be implemented here
        # A complete implementation would wrap the strategy logic into a
        # PyBroker Strategy, configure feeds and commissions, and run the backtest.
        raise NotImplementedError(
            "pybroker engine integration is not yet implemented."
        )
    else:
        raise NotImplementedError(f"Unknown alternative engine: {engine}")
