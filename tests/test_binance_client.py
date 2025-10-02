import sys
import types

import pytest

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.Timestamp = lambda *args, **kwargs: None
    pandas_stub.DataFrame = object
    pandas_stub.to_datetime = lambda *args, **kwargs: None
    sys.modules["pandas"] = pandas_stub

if "ccxt" not in sys.modules:
    ccxt_stub = types.ModuleType("ccxt")

    class _DummyExchange:  # pragma: no cover - helper for import stubbing
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, *args, **kwargs):  # type: ignore[override]
            return self

        def fetch_ohlcv(self, *args, **kwargs):  # pragma: no cover - not used in tests
            raise NotImplementedError

        def parse_timeframe(self, *args, **kwargs):  # pragma: no cover - not used in tests
            raise NotImplementedError

    ccxt_stub.binance = _DummyExchange
    ccxt_stub.binanceusdm = _DummyExchange
    ccxt_stub.NetworkError = Exception
    ccxt_stub.ExchangeError = Exception
    sys.modules["ccxt"] = ccxt_stub

from datafeed.binance_client import normalize_timeframe


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1m", "1m"),
        ("1M", "1M"),
        ("1H", "1h"),
        ("6H", "6h"),
        ("3d", "3d"),
        ("1W", "1w"),
        ("60m", "1h"),
        ("120m", "2h"),
        ("240", "4h"),
        ("43200", "1M"),
    ],
)
def test_normalize_timeframe_valid(value, expected):
    assert normalize_timeframe(value) == expected


def test_normalize_timeframe_invalid():
    with pytest.raises(ValueError):
        normalize_timeframe("7m")
