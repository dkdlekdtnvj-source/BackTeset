import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna", reason="optimize.run 모듈은 optuna 의존성을 필요로 합니다.")
pytest.importorskip("ccxt", reason="데이터 캐시 초기화에는 ccxt 의존성이 필요합니다.")
pytest.importorskip("matplotlib", reason="리포트 모듈은 matplotlib 을 요구합니다.")
pytest.importorskip("seaborn", reason="리포트 모듈은 seaborn 을 요구합니다.")

from optimize.metrics import (
    ObjectiveSpec,
    Trade,
    aggregate_metrics,
    equity_curve_from_returns,
    evaluate_objective_values,
    max_drawdown,
    normalise_objectives,
    score_metrics,
)
from optimize.run import combine_metrics
from optimize.strategy_model import run_backtest


def _base_params(**overrides):
    params = {
        "oscLen": 12,
        "signalLen": 3,
        "bbLen": 20,
        "kcLen": 18,
        "bbMult": 1.4,
        "kcMult": 1.0,
        "fluxLen": 14,
        "fluxSmoothLen": 1,
        "useFluxHeikin": True,
        "useDynamicThresh": False,
        "useSymThreshold": True,
        "statThreshold": 38.0,
        "useHTF": False,
    }
    params.update(overrides)
    return params


def test_equity_curve_and_drawdown():
    returns = pd.Series([0.01, -0.02, 0.015, -0.01])
    equity = equity_curve_from_returns(returns, initial=1.0)
    dd = max_drawdown(equity)
    assert equity.iloc[-1] == pytest.approx(0.9946, rel=1e-3)
    assert dd <= 0


def test_aggregate_metrics_basic():
    trades = [
        Trade(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), "long", 1.0, 100, 101, 0.01, 0.01, 0.02, -0.01, 5),
        Trade(pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"), "short", 1.0, 105, 104, 0.01, 0.01, 0.03, -0.02, 4),
    ]
    returns = pd.Series([0.01, 0.0, -0.005])
    metrics = aggregate_metrics(trades, returns)
    assert metrics["NetProfit"] != 0
    assert metrics["Trades"] == pytest.approx(2.0)
    assert metrics["WinRate"] == 1.0
    assert metrics["ProfitFactor"] > 0
    assert "WeeklyNetProfit" in metrics
    assert "Expectancy" in metrics


def test_aggregate_metrics_no_losses_is_finite():
    trades = [
        Trade(
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-02"),
            "long",
            1.0,
            100,
            102,
            0.02,
            0.02,
            0.03,
            -0.01,
            3,
        ),
        Trade(
            pd.Timestamp("2023-01-03"),
            pd.Timestamp("2023-01-04"),
            "short",
            1.0,
            105,
            103,
            0.02,
            0.02,
            0.02,
            -0.01,
            4,
        ),
    ]
    returns = pd.Series([0.01, 0.015, 0.02], index=pd.date_range("2023-01-01", periods=3, freq="D"))
    metrics = aggregate_metrics(trades, returns)

    assert math.isfinite(metrics["ProfitFactor"])
    assert math.isfinite(metrics["Sortino"])
    assert math.isfinite(metrics["Sharpe"])


def test_run_backtest_deterministic():
    data = pd.read_csv("tests/tests_data/sample_ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    params = _base_params(oscLen=10, signalLen=3)
    fees = {"commission_pct": 0.0005, "slippage_ticks": 1}
    risk = {
        "leverage": 2,
        "qty_pct": 10,
        "min_trades": 1,
        "min_hold_bars": 0,
        "max_consecutive_losses": 10,
        "penalty_trade": 0.0,
        "penalty_hold": 0.0,
        "penalty_consecutive_loss": 0.0,
    }

    first = run_backtest(data, params, fees, risk)
    second = run_backtest(data, params, fees, risk)
    assert first["Trades"] == second["Trades"]
    assert first["NetProfit"] == second["NetProfit"]
    assert first["Valid"] == second["Valid"]


def test_run_backtest_injects_penalty_settings():
    data = pd.read_csv("tests/tests_data/sample_ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    params = _base_params(
        useDynamicThresh=False,
        useSymThreshold=True,
        holdPenalty="4.5",
        consecutiveLossPenalty=6.0,
    )
    fees = {"commission_pct": 0.0, "slippage_ticks": 0}
    risk = {
        "leverage": 1,
        "qty_pct": 5,
        "min_trades": "7",
        "min_hold_bars": "3",
        "max_consecutive_losses": "4",
        "penalty_trade": "2.5",
    }

    metrics = run_backtest(data, params, fees, risk)

    assert metrics["TradePenalty"] == pytest.approx(2.5)
    assert metrics["HoldPenalty"] == pytest.approx(4.5)
    assert metrics["ConsecutiveLossPenalty"] == pytest.approx(6.0)
    assert metrics["MinTrades"] == pytest.approx(7.0)
    assert metrics["MinHoldBars"] == pytest.approx(3.0)
    assert metrics["MaxConsecutiveLossLimit"] == pytest.approx(4.0)


def test_run_backtest_clamps_negative_penalties():
    data = pd.read_csv("tests/tests_data/sample_ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    params = _base_params(
        holdPenalty=-3.0,
        consecutiveLossPenalty="-8.0",
    )
    fees = {"commission_pct": 0.0, "slippage_ticks": 0}
    risk = {
        "leverage": 1,
        "qty_pct": 5,
        "min_trades": 1,
        "min_hold_bars": 0,
        "max_consecutive_losses": 5,
        "penalty_trade": -2.0,
    }

    metrics = run_backtest(data, params, fees, risk)

    assert metrics["TradePenalty"] >= 0
    assert metrics["HoldPenalty"] >= 0
    assert metrics["ConsecutiveLossPenalty"] >= 0


def test_run_backtest_handles_nan_risk_values():
    data = pd.read_csv("tests/tests_data/sample_ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    params = _base_params(
        useDynamicThresh=False,
        useSymThreshold=True,
        minTrades="4",
        minHoldBars="2",
        maxConsecutiveLosses="3",
        penaltyTrade="3.0",
        holdPenalty="5.5",
        consecutiveLossPenalty="7.0",
    )
    fees = {"commission_pct": 0.0, "slippage_ticks": 0}
    risk = {
        "leverage": 1,
        "qty_pct": "nan",
        "liq_buffer_pct": "nan",
        "min_trades": "nan",
        "min_hold_bars": "nan",
        "max_consecutive_losses": "nan",
        "penalty_trade": "nan",
    }

    metrics = run_backtest(data, params, fees, risk)

    assert metrics["MinTrades"] == pytest.approx(4.0)
    assert metrics["MinHoldBars"] == pytest.approx(2.0)
    assert metrics["MaxConsecutiveLossLimit"] == pytest.approx(3.0)
    assert metrics["TradePenalty"] == pytest.approx(3.0)
    assert metrics["HoldPenalty"] == pytest.approx(5.5)
    assert metrics["ConsecutiveLossPenalty"] == pytest.approx(7.0)
    assert metrics["Trades"] >= 0
    assert np.isfinite(metrics["TradePenalty"])


def test_run_backtest_defaults_missing_penalties_to_zero():
    data = pd.read_csv("tests/tests_data/sample_ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    params = _base_params(useDynamicThresh=False, useSymThreshold=True)
    fees = {"commission_pct": 0.0, "slippage_ticks": 0}
    risk = {
        "leverage": 1,
        "qty_pct": 5,
        "min_trades": 2,
        "min_hold_bars": 1,
        "max_consecutive_losses": 3,
    }

    metrics = run_backtest(data, params, fees, risk)

    assert metrics["TradePenalty"] == pytest.approx(0.0)
    assert metrics["HoldPenalty"] == pytest.approx(0.0)
    assert metrics["ConsecutiveLossPenalty"] == pytest.approx(0.0)


def test_event_filter_blocks_trades():
    data = pd.read_csv("tests/tests_data/sample_ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    params = _base_params(
        oscLen=4,
        signalLen=2,
        useEventFilter=True,
        eventWindows="2023-01-01T00:00:00Z/2023-01-01T01:00:00Z",
    )
    fees = {"commission_pct": 0.0, "slippage_ticks": 0}
    risk = {"leverage": 1, "qty_pct": 10, "min_trades": 0, "min_hold_bars": 0, "max_consecutive_losses": 10}

    results = run_backtest(data, params, fees, risk)
    assert results["Trades"] == 0


def test_time_stop_exit_reason():
    data = pd.read_csv("tests/tests_data/sample_ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    step = data.index[1] - data.index[0]
    last_row = data.iloc[-1]
    extra_index = pd.date_range(
        data.index[-1] + step,
        periods=3,
        freq=step,
        tz=data.index.tz,
    )
    extra = pd.DataFrame({col: last_row[col] for col in data.columns}, index=extra_index)
    data = pd.concat([data, extra])
    params = _base_params(
        oscLen=4,
        signalLen=2,
        useTimeStop=True,
        maxHoldBars=2,
        useDynamicThresh=False,
        useSymThreshold=False,
        statThreshold=0.0,
        buyThreshold=0.0,
        sellThreshold=0.0,
        bbLen=4,
        kcLen=4,
        bbMult=1.0,
        kcMult=1.0,
        fluxLen=2,
        useFluxHeikin=False,
        requireMomentumCross=False,
        debugForceLong=True,
        startDate=data.index[0].isoformat(),
    )
    fees = {"commission_pct": 0.0, "slippage_ticks": 0}
    risk = {
        "leverage": 1,
        "qty_pct": 10,
        "min_trades": 0,
        "min_hold_bars": 0,
        "max_consecutive_losses": 10,
    }

    results = run_backtest(data, params, fees, risk)
    assert results["Trades"] >= 1
    reasons = [trade.reason for trade in results["TradesList"]]
    assert any(reason == "time_stop" for reason in reasons)


def test_score_metrics_handles_objectives():
    metrics = {
        "NetProfit": 0.5,
        "MaxDD": -0.1,
        "Sortino": 1.2,
        "WinRate": 0.6,
        "Trades": 50,
        "MinTrades": 20,
        "AvgHoldBars": 5,
        "MinHoldBars": 1,
        "MaxConsecutiveLosses": 2,
        "MaxConsecutiveLossLimit": 5,
    }
    score = score_metrics(metrics, ["NetProfit", "MaxDD", "Sortino", "WinRate"])
    assert score > 0


def test_score_metrics_applies_penalties():
    metrics = {
        "NetProfit": 0.5,
        "Trades": 5,
        "MinTrades": 10,
        "TradePenalty": 1.0,
        "AvgHoldBars": 0.5,
        "MinHoldBars": 1.0,
        "HoldPenalty": 0.5,
        "MaxConsecutiveLosses": 4,
        "MaxConsecutiveLossLimit": 2,
        "ConsecutiveLossPenalty": 1.0,
    }
    score = score_metrics(metrics, ["NetProfit"])
    assert score < 0.5


def test_normalise_objectives_handles_direction():
    specs = normalise_objectives([{"name": "MaxDD", "goal": "minimize", "weight": 2.0}])
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "MaxDD"
    assert spec.direction == "minimize"
    assert spec.weight == pytest.approx(2.0)


def test_score_metrics_minimize_goal():
    metrics = {"AvgHoldBars": 10.0}
    score = score_metrics(metrics, [{"name": "AvgHoldBars", "goal": "minimize"}])
    assert score == pytest.approx(-10.0)


def test_evaluate_objective_values_handles_penalties_and_weights():
    specs = [
        ObjectiveSpec(name="NetProfit", goal="maximize", weight=1.0),
        ObjectiveSpec(name="MaxDD", goal="minimize", weight=2.0),
        ObjectiveSpec(name="Sortino", goal="maximize", weight=0.0),
    ]
    metrics = {"NetProfit": float("nan"), "MaxDD": float("nan"), "Sortino": float("nan")}

    values = evaluate_objective_values(metrics, specs, non_finite_penalty=-1e6)

    assert values[0] < 0  # maximize 항목은 큰 음수 패널티로 처리되어야 함
    assert values[1] > 0  # 최소화 항목은 큰 양수 패널티를 받아야 함
    assert values[2] == 0  # weight가 0이면 패널티도 0이어야 함


def test_combine_metrics_respects_series_length():
    def make_metrics(profits, start):
        index = pd.date_range(start=start, periods=len(profits), freq="H")
        returns = pd.Series(profits, index=index)
        trades = [
            Trade(
                entry_time=ts,
                exit_time=ts + pd.Timedelta(hours=1),
                direction="long" if profit >= 0 else "short",
                size=1.0,
                entry_price=1.0,
                exit_price=1.0 + profit,
                profit=profit,
                return_pct=profit,
                mfe=profit,
                mae=-abs(profit),
                bars_held=1,
                reason="test",
            )
            for ts, profit in zip(index, profits)
        ]
        metrics = aggregate_metrics(trades, returns)
        metrics["Returns"] = returns
        metrics["TradesList"] = trades
        metrics.update(
            {
                "MinTrades": 0.0,
                "MinHoldBars": 0.0,
                "MaxConsecutiveLossLimit": 10.0,
                "TradePenalty": 0.0,
                "HoldPenalty": 0.0,
                "ConsecutiveLossPenalty": 0.0,
                "Valid": True,
            }
        )
        return metrics

    metrics_a = make_metrics([0.02, -0.01, 0.03], pd.Timestamp("2024-01-01"))
    metrics_b = make_metrics([-0.05, 0.04], pd.Timestamp("2024-02-01"))

    combined = combine_metrics([metrics_a, metrics_b])
    expected_returns = pd.concat([metrics_a["Returns"], metrics_b["Returns"]]).sort_index()
    expected_trades = sorted(metrics_a["TradesList"] + metrics_b["TradesList"], key=lambda t: t.entry_time)
    expected = aggregate_metrics(expected_trades, expected_returns)

    assert combined["Trades"] == expected["Trades"]
    assert combined["NetProfit"] == pytest.approx(expected["NetProfit"])
    assert combined["Sortino"] == pytest.approx(expected["Sortino"])


def test_combine_metrics_preserves_penalty_metadata():
    base = aggregate_metrics([], pd.Series(dtype=float))
    base["Returns"] = pd.Series(dtype=float)
    base["TradesList"] = []
    base["Valid"] = True

    secondary = aggregate_metrics([], pd.Series(dtype=float))
    secondary["Returns"] = pd.Series(dtype=float)
    secondary["TradesList"] = []
    secondary.update(
        {
            "MinTrades": 4.0,
            "MinHoldBars": 2.0,
            "MaxConsecutiveLossLimit": 3.0,
            "TradePenalty": 2.0,
            "HoldPenalty": 1.5,
            "ConsecutiveLossPenalty": 0.75,
            "Valid": True,
        }
    )

    combined = combine_metrics([base, secondary])

    assert combined["MinTrades"] == pytest.approx(4.0)
    assert combined["MinHoldBars"] == pytest.approx(2.0)
    assert combined["MaxConsecutiveLossLimit"] == pytest.approx(3.0)
    assert combined["TradePenalty"] == pytest.approx(2.0)
    assert combined["HoldPenalty"] == pytest.approx(1.5)
    assert combined["ConsecutiveLossPenalty"] == pytest.approx(0.75)


def test_combine_metrics_defaults_missing_penalties_to_zero():
    base = aggregate_metrics([], pd.Series(dtype=float))
    base["Returns"] = pd.Series(dtype=float)
    base["TradesList"] = []
    base["Valid"] = True

    combined = combine_metrics([base])

    assert combined["TradePenalty"] == pytest.approx(0.0)
    assert combined["HoldPenalty"] == pytest.approx(0.0)
    assert combined["ConsecutiveLossPenalty"] == pytest.approx(0.0)


def test_combine_metrics_simple_override_forces_basic_profile():
    idx = pd.date_range("2024-03-01", periods=5, freq="1H")
    returns = pd.Series([0.01, -0.005, 0.007, 0.0, 0.012], index=idx)
    metrics = {
        "Returns": returns,
        "GrossProfit": 180.0,
        "GrossLoss": -90.0,
        "Trades": 12,
        "Wins": 7,
        "Losses": 5,
        "AvgHoldBars": 3.0,
        "NetProfit": 0.08,
        "Valid": True,
    }

    combined = combine_metrics([metrics], simple_override=True)

    assert combined["SimpleMetricsOnly"] is True
    assert combined["ProfitFactor"] == pytest.approx(2.0)
    assert combined["Trades"] == 12
    assert combined["WinRate"] == pytest.approx(7 / 12)
    assert combined["Valid"] is True
