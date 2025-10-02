"""
Reusable components for backtesting and strategy development.

This module exposes a lightweight API on top of the internal `strategy-Backtest`
package. It provides:

* `normalize_timeframe` – convert generic timeframe strings (e.g. "60m") to
  Binance‑compatible tokens (e.g. "1h"). This helper prevents common
  ExchangeError: Invalid interval exceptions when using ccxt.
* `DataCache` – a caching wrapper for downloading and storing OHLCV data from
  Binance. The cache ensures subsequent runs re‑use locally saved CSV files
  instead of repeatedly hitting the exchange API.
* `fetch_data` – a convenience function that instantiates a `DataCache` on
  demand and returns a pandas `DataFrame` containing OHLCV candles for the
  requested symbol, timeframe, and window.

Other scripts or notebooks can `import module` and call these functions
directly instead of parsing YAML or invoking the command‑line interface.

Example:

    >>> from strategy_Backtest.module import fetch_data, normalize_timeframe
    >>> candles = fetch_data("BINANCE:BTCUSDT", "60m", "2024-01-01", "2025-01-01")
    >>> print(candles.tail())

The returned DataFrame contains `open`, `high`, `low`, `close`, and `volume`
columns indexed by UTC timestamps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .datafeed.binance_client import normalize_timeframe  # re-export for convenience
from .datafeed.cache import DataCache


def fetch_data(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    *,
    cache_dir: str | Path = "data",
    futures: bool = False,
    allow_partial: bool = False,
    max_retries: Optional[int] = None,
    retry_wait: Optional[float] = None,
    rate_limit: Optional[float] = None,
) -> pd.DataFrame:
    """Download or load OHLCV candles for a single symbol/timeframe window.

    This helper instantiates a `DataCache` pointing at `cache_dir` and uses
    it to fetch candles. The timeframe is normalised automatically so that
    aliases like "60m" map to valid Binance intervals. Additional client
    options (max_retries, retry_wait, rate_limit) can be provided to override
    the defaults used when constructing the underlying `BinanceClient`.

    Args:
        symbol: Market symbol, optionally prefixed with "BINANCE:". Must be a
            tradable USDT pair on Binance.
        timeframe: Candlestick interval (e.g. "1m", "3m", "1h", "60m").
        start: ISO date (YYYY‑MM‑DD) marking the beginning of the dataset.
        end: ISO date (YYYY‑MM‑DD) marking the end of the dataset.
        cache_dir: Directory where downloaded CSV files are stored. Defaults to
            ``"data"`` relative to the current working directory.
        futures: If ``True``, use the Binance USDT‑M futures API; otherwise
            access the spot market.
        allow_partial: When ``True``, keep rows containing NaN values. When
            ``False`` (default) drop any incomplete candles.
        max_retries: Override the default number of network retries for the
            underlying client. ``None`` uses the package default.
        retry_wait: Seconds between retries (backoff multiplier). ``None`` uses
            the package default.
        rate_limit: Minimum seconds between API calls. ``None`` uses the
            package default.

    Returns:
        A pandas DataFrame indexed by UTC timestamps with columns ``open``,
        ``high``, ``low``, ``close``, and ``volume``.
    """

    # Ensure the cache directory is a Path instance
    cache_path = Path(cache_dir)
    cache = DataCache(root=cache_path, futures=futures)
    # Propagate optional client overrides if provided
    if max_retries is not None:
        cache._client.max_retries = int(max_retries)
    if retry_wait is not None:
        cache._client.retry_wait = float(retry_wait)
    if rate_limit is not None:
        cache._client.rate_limit = float(rate_limit)
    # Use the cache to fetch data; `DataCache` normalises the timeframe for
    # logging and caching but delegates the actual download to `BinanceClient`.
    frame = cache.get(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        allow_partial=allow_partial,
    )
    return frame


__all__ = [
    "normalize_timeframe",
    "DataCache",
    "fetch_data",
]