"""Client utilities for fetching OHLCV data from Binance with retries."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency handled via requirements
    raise RuntimeError("ccxt is required for Binance data access") from exc

LOGGER = logging.getLogger(__name__)

_BINANCE_ALLOWED_TIMEFRAMES = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
)

_BINANCE_ALLOWED_TIMEFRAMES_LOWER = {
    value.lower(): value for value in _BINANCE_ALLOWED_TIMEFRAMES
}

_MINUTE_TO_BINANCE_TIMEFRAME = {
    1: "1m",
    3: "3m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    120: "2h",
    240: "4h",
    360: "6h",
    480: "8h",
    720: "12h",
    1440: "1d",
    4320: "3d",
    10080: "1w",
    43200: "1M",
}


def normalize_timeframe(timeframe: str) -> str:
    """Normalize timeframe strings to Binance-compatible values."""

    tf = str(timeframe).strip()
    if not tf:
        raise ValueError("Timeframe must be a non-empty string")

    if tf in _BINANCE_ALLOWED_TIMEFRAMES:
        return tf

    tf_lower = tf.lower()
    if tf_lower in _BINANCE_ALLOWED_TIMEFRAMES_LOWER:
        return _BINANCE_ALLOWED_TIMEFRAMES_LOWER[tf_lower]

    if tf_lower.endswith("m"):
        digits = tf_lower[:-1]
    else:
        digits = tf_lower

    if digits.isdigit():
        minutes = int(digits)
        if minutes in _MINUTE_TO_BINANCE_TIMEFRAME:
            return _MINUTE_TO_BINANCE_TIMEFRAME[minutes]

    raise ValueError(
        "Unsupported timeframe for Binance USDM: "
        f"{timeframe}. Allowed values: {sorted(_BINANCE_ALLOWED_TIMEFRAMES)}"
    )


def _to_milliseconds(dt: datetime | str | pd.Timestamp) -> int:
    ts = pd.Timestamp(dt, tz="UTC")
    return int(ts.timestamp() * 1000)


def _parse_symbol(symbol: str) -> str:
    if symbol.upper().startswith("BINANCE:"):
        return symbol.split(":", 1)[1]
    return symbol


@dataclass
class BinanceClient:
    """Thin wrapper around ccxt.binance with retry and rate-limit handling."""

    futures: bool = False
    max_retries: int = 5
    retry_wait: float = 2.0
    rate_limit: float = 0.2

    def __post_init__(self) -> None:
        exchange_class = ccxt.binanceusdm if self.futures else ccxt.binance
        self._client = exchange_class({"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | str,
        end: datetime | str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        normalized_timeframe = normalize_timeframe(timeframe)
        market_symbol = _parse_symbol(symbol)
        since = _to_milliseconds(start)
        end_ms = _to_milliseconds(end)
        all_rows: List[List[float]] = []

        while since < end_ms:
            for attempt in range(1, self.max_retries + 1):
                try:
                    batch = self._client.fetch_ohlcv(
                        market_symbol,
                        timeframe=normalized_timeframe,
                        since=since,
                        limit=limit,
                    )
                    break
                except ccxt.NetworkError as err:  # pragma: no cover - network failure path
                    LOGGER.warning("Network error from Binance: %s (attempt %d)", err, attempt)
                    time.sleep(self.retry_wait * attempt)
                except ccxt.ExchangeError as err:
                    LOGGER.error("Exchange error from Binance: %s", err)
                    raise
            else:  # no break
                raise RuntimeError("Exceeded maximum retries while fetching data")

            if not batch:
                break

            all_rows.extend(batch)
            since = batch[-1][0] + self._client.parse_timeframe(normalized_timeframe) * 1000
            time.sleep(self.rate_limit)

        if not all_rows:
            raise ValueError("No OHLCV data returned from Binance")

        frame = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("timestamp").sort_index()
        mask = (frame.index >= pd.Timestamp(start, tz="UTC")) & (frame.index <= pd.Timestamp(end, tz="UTC"))
        frame = frame.loc[mask]
        return frame


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: datetime | str,
    end: datetime | str,
    futures: bool = False,
    **kwargs,
) -> pd.DataFrame:
    client = BinanceClient(futures=futures, **kwargs)
    return client.fetch_ohlcv(symbol, timeframe, start, end)
