"""Caching helper for Binance OHLCV downloads."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .binance_client import BinanceClient

LOGGER = logging.getLogger(__name__)


@dataclass
class DataCache:
    root: Path
    futures: bool = False

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._client = BinanceClient(futures=self.futures)

    @staticmethod
    def _filename(symbol: str, timeframe: str, start: str, end: str) -> str:
        """Return a normalised cache filename for the given symbol/timeframe.

        Historically the cache key embedded the raw `timeframe` string (e.g. "60m").
        However Binance's API only accepts canonical interval identifiers such as
        "1h", "2h", etc. To avoid mismatches between the requested timeframe and
        the actual data fetched (e.g. `60m` being mapped internally to `1h`), we
        normalise the timeframe before constructing the filename. This ensures
        both the on‑disk cache and the API request use the same canonical value.

        Args:
            symbol: Market symbol (with optional prefix like "BINANCE:").
            timeframe: Desired candle interval (may be an alias like "60m").
            start: Start date (YYYY‑MM‑DD).
            end:   End date (YYYY‑MM‑DD).

        Returns:
            A filename incorporating the normalized timeframe.
        """
        # Resolve any exchange prefixes and slashes for file naming
        clean_symbol = symbol.replace(":", "_").replace("/", "")
        # Normalise the timeframe to a Binance‑compatible token (e.g. 60m -> 1h)
        from .binance_client import normalize_timeframe

        try:
            normalised = normalize_timeframe(timeframe)
        except Exception:
            # If normalisation fails, fall back to the raw input for the filename;
            # the underlying client will raise a clearer exception when fetching.
            normalised = timeframe
        return f"BINANCE_{clean_symbol}_{normalised}_{start.replace('-', '')}_{end.replace('-', '')}.csv"

    def _full_path(self, symbol: str, timeframe: str, start: str, end: str) -> Path:
        # Construct the full path using the normalised filename. See `_filename`
        # for the rationale behind normalising the timeframe.
        return self.root / self._filename(symbol, timeframe, start, end)

    def get(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        allow_partial: bool = False,
    ) -> pd.DataFrame:
        # Use the normalised filename for caching but also ensure the fetch uses a
        # canonical timeframe. The underlying `fetch_ohlcv` call will normalise
        # again, but normalising here lets us log the actual interval being
        # requested.
        path = self._full_path(symbol, timeframe, start, end)
        # Attempt to normalise the timeframe for logging. If it fails, we use
        # the raw string; the fetch will handle errors appropriately.
        from .binance_client import normalize_timeframe

        try:
            display_tf = normalize_timeframe(timeframe)
        except Exception:
            display_tf = timeframe

        if path.exists():
            LOGGER.info("Loading cached data from %s", path)
            frame = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        else:
            LOGGER.info("Downloading %s %s from Binance", symbol, display_tf)
            frame = self._client.fetch_ohlcv(symbol, timeframe, start, end)
            self._persist(path, frame)

        frame = frame.sort_index().loc[:, ["open", "high", "low", "close", "volume"]]
        # Deduplicate and optionally drop NaNs
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = frame.dropna(how="any") if not allow_partial else frame
        return frame

    def _persist(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        export = frame.copy()
        export.index.name = "timestamp"
        export.to_csv(path)
        LOGGER.info("Saved %s rows to %s", len(export), path)
