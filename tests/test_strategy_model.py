import math

import numpy as np

import pandas as pd
import pytest

from optimize.strategy_model import (
    _bars_since_mask,
    _rolling_rma_last,
    _security_series,
    _true_range,
    run_backtest,
)


def _make_ohlcv(prices):
    index = pd.date_range("2025-07-01", periods=len(prices), freq="1min", tz="UTC")
    close = pd.Series(prices, index=index)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        }
    )
    return df


def _base_params(**overrides):
    params = {
        "oscLen": 3,
        "signalLen": 1,
        "fluxLen": 3,
        "useFluxHeikin": False,
        "useDynamicThresh": False,
        "useSymThreshold": True,
        "statThreshold": 0.0,
        "startDate": "2025-07-01T00:00:00",
        "allowLongEntry": True,
        "allowShortEntry": False,
        "debugForceLong": True,
    }
    params.update(overrides)
    return params


FEES = {"commission_pct": 0.0, "slippage_ticks": 0.0}
RISK = {"initial_capital": 1000.0, "min_trades": 0, "min_hold_bars": 0, "max_consecutive_losses": 10}


def test_debug_force_long_creates_trade():
    df = _make_ohlcv([100, 101, 102, 103, 104, 105])
    params = _base_params(useTimeStop=True, maxHoldBars=1)

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] >= 1


def test_daily_loss_guard_freezes_after_loss():
    prices = [100, 99, 98, 97, 96, 95, 94, 93]
    df = _make_ohlcv(prices)
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        useDailyLossGuard=True,
        dailyLossLimit=0.5,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 1
    assert metrics["GuardFrozen"] == 1.0


def test_min_trades_argument_marks_invalid_when_threshold_not_met():
    prices = [100, 99, 98, 97, 96, 95, 94, 93]
    df = _make_ohlcv(prices)
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        useDailyLossGuard=True,
        dailyLossLimit=0.5,
    )

    metrics = run_backtest(df, params, FEES, RISK, min_trades=2)

    assert metrics["Trades"] == pytest.approx(1.0)
    assert metrics["MinTrades"] == pytest.approx(2.0)
    assert not metrics["Valid"]


def test_squeeze_gate_blocks_without_release():
    df = _make_ohlcv([100] * 20)
    params = _base_params(
        useSqzGate=True,
        sqzReleaseBars=0,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 0


def test_stop_distance_guard_prevents_entry():
    prices = [100, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0]
    df = _make_ohlcv(prices)
    params = _base_params(
        useStopDistanceGuard=True,
        maxStopAtrMult=0.5,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 0


def test_timestamp_column_with_invalid_rows_is_cleaned():
    prices = [100, 101, 102, 103, 104, 105, 106]
    df = _make_ohlcv(prices)
    raw = df.reset_index().rename(columns={"index": "timestamp"})

    raw.loc[2, "timestamp"] = None  # invalid timestamp row -> should be dropped
    raw.loc[3, "close"] = "bad"  # non-numeric OHLC value -> should be coerced then dropped
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    raw.loc[len(raw) - 1, "timestamp"] = raw.loc[1, "timestamp"]  # duplicate timestamp

    params = _base_params(useTimeStop=True, maxHoldBars=1)

    metrics = run_backtest(raw, params, FEES, RISK)

    returns = metrics["Returns"]
    assert isinstance(returns, pd.Series)
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert returns.index.tz is not None
    assert 0 < len(returns) < len(raw)


def test_short_stop_handles_missing_candidates():
    prices = [110, 109, 108, 107, 106, 105, 104, 103]
    df = _make_ohlcv(prices)
    params = _base_params(
        allowLongEntry=False,
        allowShortEntry=True,
        debugForceLong=False,
        debugForceShort=True,
        useStructureGate=True,
        useBOS=True,
        useCHOCH=True,
        useStopLoss=True,
        stopLookback=3,
        usePivotStop=True,
        useTimeStop=True,
        maxHoldBars=1,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] >= 1


def test_security_series_resamples_monthly_timeframe():
    index = pd.date_range("2025-01-01", periods=65, freq="D", tz="UTC")
    close = pd.Series(range(len(index)), index=index)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
        }
    )

    captured = {}

    def _compute(resampled: pd.DataFrame) -> pd.Series:
        captured["index"] = resampled.index
        return resampled["close"]

    result = _security_series(df, "1M", _compute)

    assert "index" in captured
    assert captured["index"].freqstr in {"MS", "M"}

    period_index = result.index.tz_localize(None).to_period("M")
    unique_per_month = result.groupby(period_index).nunique()
    assert (unique_per_month == 1).all()


def test_mom_fade_bars_since_vectorised_matches_reference():
    hist = pd.Series(
        [0.5, -0.2, -0.1, 0.3, 0.2, -0.4, 0.1, 0.0, 0.6, -0.3],
        index=pd.date_range("2025-01-01", periods=10, freq="1min", tz="UTC"),
    )

    nonpos_mask = hist.le(0)
    nonneg_mask = hist.ge(0)

    vector_nonpos = _bars_since_mask(nonpos_mask)
    vector_nonneg = _bars_since_mask(nonneg_mask)

    def _reference(mask: pd.Series) -> pd.Series:
        results = []
        for idx, _ in enumerate(mask):
            count = 0
            found = False
            for lookback in range(idx, -1, -1):
                if mask.iloc[lookback]:
                    results.append(float(count))
                    found = True
                    break
                count += 1
            if not found:
                results.append(float("inf"))
        return pd.Series(results, index=mask.index, dtype=float)

    ref_nonpos = _reference(nonpos_mask)
    ref_nonneg = _reference(nonneg_mask)

    for left, right in zip(vector_nonpos, ref_nonpos):
        if math.isinf(right):
            assert math.isinf(left)
        else:
            assert left == pytest.approx(right)

    for left, right in zip(vector_nonneg, ref_nonneg):
        if math.isinf(right):
            assert math.isinf(left)
        else:
            assert left == pytest.approx(right)


def test_volatility_guard_atr_matches_reference():
    prices = [100, 101, 100.5, 102, 101.5, 103, 102.5, 104, 103.5, 105]
    df = _make_ohlcv(prices)
    window = 3

    tr_series = _true_range(df)
    atr_values = _rolling_rma_last(tr_series.to_numpy(dtype=float), window)

    computed = []
    for idx, close_value in enumerate(df["close"].to_numpy(dtype=float)):
        if idx >= window and not math.isnan(atr_values[idx]) and close_value != 0.0:
            computed.append(atr_values[idx] / close_value * 100.0)
        else:
            computed.append(0.0)

    expected = []
    for idx in range(len(df)):
        if idx >= window:
            window_tr = tr_series.iloc[idx - window + 1 : idx + 1].to_numpy(dtype=float)
            acc = window_tr[0]
            for value in window_tr[1:]:
                acc = (acc * (window - 1) + value) / window
            expected.append(acc / df["close"].iloc[idx] * 100.0)
        else:
            expected.append(0.0)

    assert computed == pytest.approx(expected)


def test_rolling_rma_last_matches_recursive_formula():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    result = _rolling_rma_last(values, 3)

    assert np.isnan(result[0])
    assert np.isnan(result[1])
    expected_tail = [2.0, 2.6666666667, 3.4444444444]
    assert result[2:] == pytest.approx(expected_tail)

    zero_length = _rolling_rma_last(values, 0)
    assert np.isnan(zero_length).all()
