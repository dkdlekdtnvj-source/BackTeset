from __future__ import annotations

import pandas as pd
import pytest

from optimize.wf import run_walk_forward


def _build_frame() -> pd.DataFrame:
    index = pd.date_range("2024-05-01", periods=12, freq="1h", tz="UTC")
    base = pd.Series(range(len(index)), index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": base + 0.1,
            "high": base + 0.2,
            "low": base - 0.1,
            "close": base,
            "volume": 1.0,
        }
    )


def test_walk_forward_parallel_matches_serial(monkeypatch):
    df = _build_frame()

    def fake_run_backtest(data, params, fees, risk, htf_df=None, min_trades=None):
        return {"NetProfit": float(data["close"].sum()), "Valid": True}

    monkeypatch.setattr("optimize.wf.run_backtest", fake_run_backtest)

    params = {}
    fees = {}
    risk = {}

    serial = run_walk_forward(
        df,
        params,
        fees,
        risk,
        train_bars=4,
        test_bars=2,
        step=2,
        n_jobs=1,
    )

    parallel = run_walk_forward(
        df,
        params,
        fees,
        risk,
        train_bars=4,
        test_bars=2,
        step=2,
        n_jobs=2,
        executor="thread",
    )

    assert parallel["count"] == serial["count"]
    assert len(parallel["segments"]) == len(serial["segments"])
    assert parallel["oos_mean"] == pytest.approx(serial["oos_mean"])
    assert parallel["oos_median"] == pytest.approx(serial["oos_median"])

    for left, right in zip(serial["segments"], parallel["segments"]):
        assert left.train_start == right.train_start
        assert left.test_end == right.test_end
        assert left.train_metrics == right.train_metrics
        assert left.test_metrics == right.test_metrics
