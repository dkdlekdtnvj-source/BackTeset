from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from optimize import alternative_engine as alt
from optimize.run import combine_metrics


class DummyRecords:
    def __init__(self, records: List[Dict[str, object]]):
        self.records_readable = pd.DataFrame(records)


class DummyTrades:
    def __init__(self, records: List[Dict[str, object]]):
        self.records_readable = pd.DataFrame(records)


class DummyPortfolio:
    def __init__(self, close_series: pd.Series, returns: np.ndarray, records: List[Dict[str, object]]):
        self._close = close_series
        self._returns = returns
        self.trades = DummyTrades(records)

    @property
    def close(self) -> pd.Series:
        return self._close

    @property
    def returns(self) -> np.ndarray:
        return self._returns

    @staticmethod
    def from_signals(
        close: pd.Series,
        entries: np.ndarray,
        exits: np.ndarray,
        short_entries: np.ndarray,
        short_exits: np.ndarray,
        fees: float,
        size: float,
        size_type: str,
        execute_on_close: bool,
        upon_opposite_entry: str,
    ) -> "DummyPortfolio":
        returns = np.array([0.02, -0.01, 0.03], dtype=float)
        records = [
            {
                "Entry Timestamp": close.index[0],
                "Exit Timestamp": close.index[1],
                "Direction": "Long",
                "Size": size,
                "Avg Entry Price": float(close.iloc[0]),
                "Avg Exit Price": float(close.iloc[1]),
                "PnL": 10.0,
                "Return": 0.1,
            }
        ]
        return DummyPortfolio(close, returns, records)


def test_vectorbt_backtest_returns_trades_and_returns(monkeypatch):
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1.0, 1.0, 1.0],
        },
        index=index,
    )

    parsed = alt._ParsedInputs(
        df=df,
        htf_df=None,
        start_ts=index[0],
        commission_pct=0.0005,
        slippage_ticks=0.0,
        leverage=1.0,
        initial_capital=1000.0,
        capital_pct=1.0,
        allow_long=True,
        allow_short=False,
        require_cross=False,
        exit_opposite=False,
        min_trades=0,
        min_hold_bars=0,
        max_consecutive_losses=10,
    )

    def fake_validate(params: Dict[str, object]) -> None:
        return None

    def fake_signals(df: pd.DataFrame, params: Dict[str, object], parsed_inputs: alt._ParsedInputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        entries = np.array([True, False, False])
        exits = np.array([False, True, False])
        shorts = np.zeros_like(entries, dtype=bool)
        return entries, exits, shorts, shorts

    def fake_apply(parsed_inputs, params, long_entries, long_exits, short_entries, short_exits):
        return long_entries, long_exits, short_entries, short_exits

    monkeypatch.setattr(alt, "_validate_feature_flags", fake_validate)
    monkeypatch.setattr(alt, "_build_signals", fake_signals)
    monkeypatch.setattr(alt, "_apply_exit_overrides", fake_apply)
    monkeypatch.setattr(alt, "VECTORBT_AVAILABLE", True)
    monkeypatch.setattr(alt, "_VBT_MODULE", type("DummyModule", (), {"Portfolio": DummyPortfolio}))

    metrics = alt._vectorbt_backtest(parsed, params={})

    assert "TradesList" in metrics
    assert "Returns" in metrics
    assert isinstance(metrics["TradesList"], list)
    assert isinstance(metrics["Returns"], pd.Series)
    assert len(metrics["Returns"]) == len(df)

    combined = combine_metrics([metrics])
    assert combined["Trades"] == len(metrics["TradesList"])
    assert combined["NetProfit"] != 0.0


@pytest.mark.parametrize("use_flux_heikin", [False, True])
def test_compute_indicators_with_mod_flux_matches_manual_calculation(use_flux_heikin):
    index = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    base = np.linspace(100.0, 110.0, num=len(index))
    df = pd.DataFrame(
        {
            "open": base + np.random.default_rng(42).normal(0, 0.5, size=len(index)),
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + np.random.default_rng(24).normal(0, 0.3, size=len(index)),
            "volume": np.linspace(1.0, 2.0, num=len(index)),
        },
        index=index,
    )

    params = {
        "oscLen": 10,
        "signalLen": 3,
        "fluxLen": 8,
        "fluxSmoothLen": 3,
        "useFluxHeikin": use_flux_heikin,
        "useModFlux": True,
        "kcLen": 10,
        "kcMult": 1.0,
        "bbLen": 12,
        "bbMult": 1.2,
        "maType": "SMA",
    }

    *_unused, flux_hist = alt._compute_indicators(df, params)

    flux_df = alt._heikin_ashi(df) if use_flux_heikin else df
    plus_di, minus_di, _ = alt._dmi(flux_df, params["fluxLen"])
    flux_denom = (plus_di + minus_di).replace(0.0, np.nan)
    mod_flux_ratio = (plus_di - minus_di).divide(flux_denom)
    mod_flux_ratio = mod_flux_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    flux_half = max(int(np.round(params["fluxLen"] / 2.0)), 1)
    mod_flux_core = alt._rma(mod_flux_ratio, flux_half) * 100.0
    expected = mod_flux_core.rolling(
        params["fluxSmoothLen"], min_periods=params["fluxSmoothLen"]
    ).mean()

    pdt.assert_series_equal(flux_hist, expected, check_names=False, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("use_flux_heikin", [False, True])
def test_compute_indicators_with_directional_flux_applies_additional_smoothing(use_flux_heikin):
    index = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    base = np.linspace(120.0, 135.0, num=len(index))
    df = pd.DataFrame(
        {
            "open": base + np.random.default_rng(11).normal(0, 0.4, size=len(index)),
            "high": base + 1.2,
            "low": base - 0.8,
            "close": base + np.random.default_rng(7).normal(0, 0.25, size=len(index)),
            "volume": np.linspace(1.0, 3.0, num=len(index)),
        },
        index=index,
    )

    params = {
        "oscLen": 12,
        "signalLen": 4,
        "fluxLen": 9,
        "fluxSmoothLen": 4,
        "useFluxHeikin": use_flux_heikin,
        "useModFlux": False,
        "kcLen": 15,
        "kcMult": 1.1,
        "bbLen": 18,
        "bbMult": 1.3,
        "maType": "EMA",
    }

    *_unused, flux_hist = alt._compute_indicators(df, params)

    flux_df = alt._heikin_ashi(df) if use_flux_heikin else df
    flux_raw = alt._directional_flux(flux_df, params["fluxLen"], params["fluxSmoothLen"])
    expected = flux_raw.rolling(params["fluxSmoothLen"], min_periods=params["fluxSmoothLen"]).mean()

    pdt.assert_series_equal(flux_hist, expected, check_names=False, check_exact=False, rtol=1e-12, atol=1e-12)
