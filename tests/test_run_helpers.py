from typing import Optional

import pandas as pd
import pytest

from optimize.run import (
    DatasetSpec,
    _filter_basic_factor_params,
    _group_datasets,
    _normalise_periods,
    _resolve_symbol_entry,
    _restrict_to_basic_factors,
    _select_datasets_for_params,
    parse_args,
)


def test_normalise_periods_returns_configured_entries():
    periods_cfg = [{"from": "2023-01-01", "to": "2023-06-30"}]
    base = {"from": "2022-01-01", "to": "2022-12-31"}

    result = _normalise_periods(periods_cfg, base)

    assert result == [{"from": "2023-01-01", "to": "2023-06-30"}]


def test_normalise_periods_falls_back_to_base_when_empty():
    base = {"from": "2022-01-01", "to": "2022-12-31"}

    result = _normalise_periods([], base)

    assert result == [{"from": "2022-01-01", "to": "2022-12-31"}]


@pytest.mark.parametrize(
    "payload",
    [
        [{"from": "2023-01-01"}],
        [{"to": "2023-06-30"}],
        ["2023-01-01/2023-06-30"],
    ],
)
def test_normalise_periods_raises_for_invalid_entries(payload):
    base = {}

    with pytest.raises(ValueError):
        _normalise_periods(payload, base)


def test_resolve_symbol_entry_uses_alias_map():
    alias, resolved = _resolve_symbol_entry(
        "BINANCE:XPLUSDT", {"BINANCE:XPLUSDT": "BINANCE:XPLAUSDT"}
    )

    assert alias == "BINANCE:XPLUSDT"
    assert resolved == "BINANCE:XPLAUSDT"


def test_resolve_symbol_entry_accepts_mapping_definition():
    entry = {"alias": "BINANCE:ASTERUSDT", "symbol": "BINANCE:ASTRUSDT"}
    alias, resolved = _resolve_symbol_entry(entry, {})

    assert alias == "BINANCE:ASTERUSDT"
    assert resolved == "BINANCE:ASTRUSDT"


def test_parse_args_accepts_trial_override():
    args = parse_args(["--n-trials", "25"])

    assert args.n_trials == 25


def test_parse_args_exposes_space_flags():
    args = parse_args(["--full-space"])

    assert args.full_space is True
    assert args.basic_factors_only is False

    args = parse_args(["--basic-factors-only"])

    assert args.full_space is False
    assert args.basic_factors_only is True


def test_basic_factor_filter_toggle():
    space = {"oscLen": {"type": "int"}, "custom": {"type": "int"}}

    restricted = _restrict_to_basic_factors(space)
    assert restricted == {"oscLen": {"type": "int"}}

    restored = _restrict_to_basic_factors(space, enabled=False)
    assert restored == {"oscLen": {"type": "int"}, "custom": {"type": "int"}}


def test_basic_factor_param_filter_toggle():
    params = {"oscLen": 12, "custom": 7}

    filtered = _filter_basic_factor_params(params)
    assert filtered == {"oscLen": 12}

    unfiltered = _filter_basic_factor_params(params, enabled=False)
    assert unfiltered == params


def _make_dataset(timeframe: str, htf: Optional[str]) -> DatasetSpec:
    idx = pd.date_range("2024-01-01", periods=3, freq="1min")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [10, 11, 12],
        },
        index=idx,
    )
    return DatasetSpec(
        symbol="BINANCE:TESTUSDT",
        timeframe=timeframe,
        start="2024-01-01",
        end="2024-01-02",
        df=df,
        htf=None,
        htf_timeframe=htf,
        source_symbol="BINANCE:TESTUSDT",
    )


def test_select_datasets_respects_timeframe_and_htf_choice():
    datasets = [_make_dataset("1m", "15m"), _make_dataset("3m", "1h")]
    groups, timeframe_groups, default_key = _group_datasets(datasets)
    params_cfg = {"timeframe": "1m", "htf_timeframes": ["15m", "1h"]}
    params = {"timeframe": "3m", "htf": "1h"}

    key, selection = _select_datasets_for_params(params_cfg, groups, timeframe_groups, default_key, params)

    assert key == ("3m", "1h")
    assert selection == [datasets[1]]


def test_select_datasets_falls_back_when_htf_disabled():
    datasets = [_make_dataset("1m", "15m"), _make_dataset("1m", None)]
    groups, timeframe_groups, default_key = _group_datasets(datasets)
    params_cfg = {"timeframe": "1m", "htf_timeframes": ["15m"]}
    params = {"timeframe": "1m", "htf": "none"}

    key, selection = _select_datasets_for_params(params_cfg, groups, timeframe_groups, default_key, params)

    assert key[0] == "1m"
    assert all(dataset.timeframe == "1m" for dataset in selection)
