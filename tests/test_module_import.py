"""module.py 임포트 경로 폴백 동작을 검증합니다."""

from __future__ import annotations

import importlib
import sys
import types


def _ensure_ccxt_stub() -> None:
    """ccxt 의존성이 없는 환경에서도 모듈을 임포트할 수 있도록 스텁을 주입합니다."""

    if "ccxt" in sys.modules:
        return

    ccxt_stub = types.ModuleType("ccxt")

    class _DummyExchange:  # pragma: no cover - 테스트 헬퍼
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - 문서화 필요 없음
            pass

        def __call__(self, *args, **kwargs):  # type: ignore[override]
            return self

        def fetch_ohlcv(self, *args, **kwargs):  # pragma: no cover - 사용되지 않음
            raise NotImplementedError

        def parse_timeframe(self, *args, **kwargs):  # pragma: no cover - 사용되지 않음
            raise NotImplementedError

    ccxt_stub.binance = _DummyExchange
    ccxt_stub.binanceusdm = _DummyExchange
    ccxt_stub.NetworkError = Exception
    ccxt_stub.ExchangeError = Exception
    sys.modules["ccxt"] = ccxt_stub


def test_module_import_and_public_api():
    """루트 레벨에서 ``import module`` 이 성공하고 공개 API 가 노출되는지 확인합니다."""

    _ensure_ccxt_stub()
    sys.modules.pop("module", None)
    module = importlib.import_module("module")

    assert hasattr(module, "normalize_timeframe")
    assert callable(module.normalize_timeframe)
    assert hasattr(module, "DataCache")

