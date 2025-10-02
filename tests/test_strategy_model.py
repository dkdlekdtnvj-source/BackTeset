import importlib
from datetime import datetime, timedelta, timezone

import pandas as pd

from optimize.strategy_model import run_backtest


def _make_ohlcv(prices):
    pd_mod = importlib.import_module("pandas")
    if not hasattr(pd_mod, "DatetimeIndex"):
        pd_mod = importlib.reload(pd_mod)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = pd_mod.DatetimeIndex([start + timedelta(minutes=i) for i in range(len(prices))])
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
        "utKey": 1.2,
        "utAtrLen": 3,
        "useHeikin": False,
        "rsiLen": 6,
        "stochLen": 6,
        "kLen": 1,
        "dLen": 1,
        "obLevel": 80,
        "osLevel": 20,
        "stMode": "Bounce",
        "atrLen": 4,
        "initStopMult": 1.2,
        "trailAtrMult": 1.5,
        "trailStartPct": 0.8,
        "trailGapPct": 0.3,
        "usePercentStop": True,
        "stopPct": 1.0,
        "takePct": 2.0,
        "breakevenPct": 0.0,
        "maxHoldBars": 0,
        "useFlipExit": True,
        "cooldownBars": 0,
    }
    params.update(overrides)
    return params


FEES = {"commission_pct": 0.0, "slippage_ticks": 0.0}
RISK = {"initial_capital": 1000.0, "qty_pct": 100.0, "min_trades": 0}


def test_basic_long_trade_executes():
    prices = [100, 99, 98, 99, 100, 101, 102, 103]
    df = _make_ohlcv(prices)
    params = _base_params(debugForceLong=True, stMode="Cross", osLevel=60, obLevel=40, utKey=0.8, initStopMult=0.9)

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] >= 1
    assert metrics["NetProfitAbs"] != 0


def test_cooldown_after_loss_blocks_next_entry():
    # 짧은 시퀀스로 손절 후 추가 진입이 발생하지 않도록 구성한다.
    prices = [100, 99, 98, 97]
    df = _make_ohlcv(prices)
    params = _base_params(
        debugForceLong=True,
        cooldownBars=3,
        takePct=0.5,
        stopPct=0.3,
        utKey=0.7,
        stMode="Cross",
        breakevenPct=1.0,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 1


def test_max_hold_forces_exit_reason():
    prices = [100, 101, 102, 103, 104, 105]
    df = _make_ohlcv(prices)
    params = _base_params(debugForceLong=True, maxHoldBars=1, takePct=50.0, stopPct=50.0, trailStartPct=50.0, stMode="Cross", utKey=0.8)

    metrics = run_backtest(df, params, FEES, RISK)
    trades = metrics["TradesList"]

    assert trades
    assert any(trade.reason.startswith("time_") for trade in trades)


def test_dataframe_cleanup_handles_invalid_rows():
    base_prices = [100, 101, 102, 103, 104, 105]
    raw = _make_ohlcv(base_prices).reset_index().rename(columns={"index": "timestamp"})
    raw.loc[2, "timestamp"] = None
    raw.loc[3, "close"] = "bad"
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    raw.loc[len(raw) - 1, "timestamp"] = raw.loc[1, "timestamp"]

    params = _base_params()
    metrics = run_backtest(raw, params, FEES, RISK)

    returns = metrics["Returns"]
    assert isinstance(returns, pd.Series)
    pd_mod = importlib.import_module("pandas")
    assert isinstance(returns.index, pd_mod.DatetimeIndex)
    assert returns.index.tz is not None
    assert len(returns) < len(raw)


def test_flip_exit_closes_position():
    prices = [
        100.0,
        101.78,
        101.77,
        99.79,
        99.31,
        101.22,
        102.67,
        100.92,
        100.68,
        102.6,
        102.1,
        102.68,
        103.81,
        105.26,
        103.93,
    ]
    df = _make_ohlcv(prices)
    params = _base_params(
        debugForceLong=True,
        useFlipExit=True,
        utKey=1.2,
        utAtrLen=5,
        stMode="Cross",
        atrLen=7,
        initStopMult=3.0,
        trailAtrMult=5.0,
        trailStartPct=5.0,
        trailGapPct=2.0,
        usePercentStop=False,
    )

    metrics = run_backtest(df, params, FEES, RISK)
    trades = metrics["TradesList"]

    assert trades
    assert any("flip" in trade.reason for trade in trades)
