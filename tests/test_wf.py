import copy
from typing import Iterator, List

import pandas as pd

from optimize import wf


def _make_metrics_sequence() -> List[dict]:
    return [
        {"NetProfit": 1.0, "Trades": 15, "Valid": True},
        {"NetProfit": 0.5, "Trades": 3, "Valid": True},
        {"NetProfit": 1.1, "Trades": 14, "Valid": True},
        {"NetProfit": 0.8, "Trades": 7, "Valid": True},
        {"NetProfit": 1.2, "Trades": 16, "Valid": True},
        {"NetProfit": -0.2, "Trades": 4, "Valid": True},
    ]


def _stub_backtest(monkeypatch, sequence: List[dict]) -> None:
    iterator: Iterator[dict] = iter(sequence)

    def _fake_backtest(*args, **kwargs):
        try:
            metrics = next(iterator)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise AssertionError("run_backtest 호출 횟수가 예상보다 많습니다") from exc
        return copy.deepcopy(metrics)

    monkeypatch.setattr(wf, "run_backtest", _fake_backtest)


def test_run_walk_forward_min_trades_filters(monkeypatch):
    index = pd.date_range("2022-01-01", periods=10, freq="h")
    df = pd.DataFrame({"close": range(10)}, index=index)
    params = {}
    fees = {}
    risk = {}

    def _run_with_min_trades(value):
        metrics = _make_metrics_sequence()
        _stub_backtest(monkeypatch, metrics)
        return wf.run_walk_forward(
            df,
            params,
            fees,
            risk,
            train_bars=4,
            test_bars=2,
            step=2,
            min_trades=value,
        )

    default_summary = _run_with_min_trades(None)
    assert default_summary["count"] == 3
    assert all(bool(seg.test_metrics.get("Valid", True)) for seg in default_summary["segments"])

    filtered_summary = _run_with_min_trades(5)
    assert len(filtered_summary["segments"]) == 3
    assert filtered_summary["count"] == 1
    assert filtered_summary["oos_mean"] == filtered_summary["oos_median"] == 0.8
    flags = [bool(seg.test_metrics.get("Valid", True)) for seg in filtered_summary["segments"]]
    assert flags == [False, True, False]
    assert filtered_summary["segments"][0].test_metrics["Valid"] == 0.0
    assert filtered_summary["segments"][1].test_metrics.get("Valid", True)
