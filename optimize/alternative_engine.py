"""고속 외부 백테스트 엔진 호환 레이어.

이 모듈은 :mod:`optimize.strategy_model` 의 ``run_backtest`` 함수를
대체할 수 있는 호환 레이어를 제공한다. 현재는 **vectorbt** 기반 실행
경로를 우선 지원하며, 동일한 규칙(모멘텀 교차, 동적 임계값, `exitOpposite`
등)을 그대로 적용해 결과를 집계한다. 외부 엔진이 준비되지 않았거나
필수 기능을 아직 매핑하지 못했다면 :class:`NotImplementedError` 를
발생시켜 기본 파이썬 구현으로 안전하게 폴백하도록 설계했다.

PyBroker 통합은 향후 추가 예정이므로, 해당 엔진이 선택되면 현재는
명시적으로 ``NotImplementedError`` 를 발생시킨다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimize.metrics import Trade, aggregate_metrics
from optimize.strategy_model import (  # 재사용 가능한 보조 함수들
    _atr,
    _bars_since_mask,
    _directional_flux,
    _heikin_ashi,
    _linreg,
    _rma,
    _sma,
    _std,
)

LOGGER = logging.getLogger(__name__)


# vectorbt/pybroker 는 선택적 의존성이므로 "필요할 때" 불러온다.
try:  # pragma: no cover - 런타임 환경에 따라 달라짐
    import vectorbt  # type: ignore  # noqa: F401

    _VBT_MODULE = vectorbt
    VECTORBT_AVAILABLE = True
except Exception:  # pragma: no cover - 미설치 시 자동 폴백
    _VBT_MODULE = None
    VECTORBT_AVAILABLE = False

try:  # pragma: no cover - 선택적 모듈
    import pybroker  # type: ignore  # noqa: F401

    PYBROKER_AVAILABLE = True
except Exception:  # pragma: no cover - 미설치 시 False
    PYBROKER_AVAILABLE = False


_SUPPORTED_ENGINES = {"vectorbt", "vectorbtpro", "vbt"}
_PYBROKER_ENGINES = {"pybroker", "pb"}


@dataclass
class _ParsedInputs:
    """전처리된 입력 및 파생 설정 값 컨테이너."""

    df: pd.DataFrame
    htf_df: Optional[pd.DataFrame]
    start_ts: pd.Timestamp
    commission_pct: float
    slippage_ticks: float
    leverage: float
    initial_capital: float
    capital_pct: float
    allow_long: bool
    allow_short: bool
    require_cross: bool
    exit_opposite: bool
    min_trades: int
    min_hold_bars: int
    max_consecutive_losses: int


@dataclass
class _MomFadeContext:
    """모멘텀 페이드 청산 계산에 필요한 선행 시계열과 설정 값."""

    hist: pd.Series
    abs_hist: pd.Series
    abs_prev: pd.Series
    abs_prev2: pd.Series
    since_nonpos: pd.Series
    since_nonneg: pd.Series
    gate_valid: pd.Series
    bars_required: int
    zero_delay: int
    min_abs: float
    release_only: bool
    require_two: bool


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan"}:
            return default
        if text in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "f", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_float(value: object, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _ensure_datetime_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """OHLCV 프레임이 UTC 기반 DatetimeIndex 를 갖도록 강제한다."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        if "timestamp" in frame.columns:
            ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            mask = ts.notna()
            if not mask.any():
                raise TypeError(
                    f"{label} 데이터프레임에 유효한 timestamp 컬럼이 없습니다."
                )
            frame = frame.loc[mask].copy()
            frame.index = ts[mask]
            frame = frame.drop(columns=["timestamp"])
        else:
            raise TypeError(
                f"{label} 데이터프레임은 DatetimeIndex 혹은 timestamp 컬럼이 필요합니다."
            )

    if frame.index.tz is None:
        frame = frame.tz_localize("UTC")
    else:
        frame = frame.tz_convert("UTC")
    return frame


def _normalise_ohlcv(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        dup = int(frame.index.duplicated(keep="last").sum())
        if dup:
            LOGGER.warning("%s 데이터에서 중복 인덱스 %d개를 제거합니다.", label, dup)
        frame = frame[~frame.index.duplicated(keep="last")]

    required = ["open", "high", "low", "close", "volume"]
    for column in required:
        if column not in frame.columns:
            continue
        coerced = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = coerced

    before = len(frame)
    frame = frame.dropna(subset=[col for col in required if col in frame.columns])
    dropped = before - len(frame)
    if dropped:
        LOGGER.warning("%s 데이터에서 결측 OHLCV 행 %d개를 제거했습니다.", label, dropped)
    if len(frame) < 2:
        raise ValueError(f"{label} 데이터가 부족하여 백테스트를 수행할 수 없습니다.")
    return frame


def _parse_core_settings(
    df: pd.DataFrame,
    params: Dict[str, object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    *,
    min_trades: Optional[int],
    htf_df: Optional[pd.DataFrame],
) -> _ParsedInputs:
    df = _ensure_datetime_index(df, "가격")
    df = _normalise_ohlcv(df, "가격")
    if htf_df is not None:
        htf_df = _normalise_ohlcv(_ensure_datetime_index(htf_df, "HTF"), "HTF")

    start_raw = params.get("startDate")
    try:
        start_ts = pd.to_datetime(start_raw, utc=True) if start_raw else df.index[0]
    except Exception:
        start_ts = df.index[0]
    if start_ts < df.index[0]:
        start_ts = df.index[0]

    commission_pct = _coerce_float(
        fees.get("commission_pct", params.get("commission_value", 0.0005)), 0.0005
    )
    slippage_ticks = _coerce_float(fees.get("slippage_ticks", params.get("slipTicks")), 0.0)
    leverage = _coerce_float(risk.get("leverage", params.get("leverage")), 10.0)
    initial_capital = _coerce_float(
        risk.get("initial_capital", params.get("initial_capital")), 500.0
    )
    base_qty_pct = _coerce_float(params.get("baseQtyPercent"), 30.0) / 100.0

    allow_long = _coerce_bool(params.get("allowLongEntry"), True)
    allow_short = _coerce_bool(params.get("allowShortEntry"), True)
    require_cross = True
    exit_opposite = _coerce_bool(params.get("exitOpposite"), True)

    if min_trades is not None:
        min_trades_value = max(int(min_trades), 0)
    else:
        min_trades_value = max(
            _coerce_int(
                params.get("minTrades", risk.get("min_trades", 0)),
                0,
            ),
            0,
        )

    min_hold_bars = max(_coerce_int(params.get("minHoldBars"), 0), 0)
    max_consecutive_losses = max(
        _coerce_int(params.get("maxConsecutiveLosses"), 3),
        0,
    )

    return _ParsedInputs(
        df=df,
        htf_df=htf_df,
        start_ts=start_ts,
        commission_pct=float(abs(commission_pct)),
        slippage_ticks=float(abs(slippage_ticks)),
        leverage=float(abs(leverage)) if leverage else 1.0,
        initial_capital=float(abs(initial_capital)) if initial_capital else 1.0,
        capital_pct=float(max(base_qty_pct, 0.0)),
        allow_long=allow_long,
        allow_short=allow_short,
        require_cross=require_cross,
        exit_opposite=exit_opposite,
        min_trades=min_trades_value,
        min_hold_bars=min_hold_bars,
        max_consecutive_losses=max_consecutive_losses,
    )


def _validate_feature_flags(params: Dict[str, object]) -> None:
    unsupported = [
        "useBreakevenStop",
        "usePivotStop",
        "useAtrProfit",
        "useDynVol",
        "useStopDistanceGuard",
        "useTimeStop",
        "useKASA",
        "useBETiers",
        "useShock",
        "useReversal",
        "useSqzGate",
        "useStructureGate",
        "useSizingOverride",
        "useDrawdownScaling",
        "usePerfAdaptiveRisk",
        "useDailyStopLoss",
        "useSessionFilter",
        "useDayFilter",
        "useAdx",
        "useEma",
        "useBbFilter",
        "useStochRsi",
        "useObv",
        "useAtrDiff",
        "useHtfTrend",
        "useHmaFilter",
        "useRangeFilter",
        "useEventFilter",
        "useSlopeFilter",
        "useDistanceGuard",
    ]
    enabled = [name for name in unsupported if _coerce_bool(params.get(name), False)]
    if enabled:
        raise NotImplementedError(
            "다음 옵션은 vectorbt 호환 레이어에서 아직 지원되지 않습니다: "
            + ", ".join(sorted(enabled))
        )


def _compute_indicators(
    df: pd.DataFrame,
    params: Dict[str, object],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    osc_len = max(_coerce_int(params.get("oscLen"), 20), 1)
    sig_len = max(_coerce_int(params.get("signalLen"), 3), 1)
    use_same_len = _coerce_bool(params.get("useSameLen"), False)
    bb_len = osc_len if use_same_len else max(_coerce_int(params.get("bbLen"), 20), 1)
    kc_len = osc_len if use_same_len else max(_coerce_int(params.get("kcLen"), 18), 1)
    bb_mult = _coerce_float(params.get("bbMult"), 1.4)
    kc_mult = _coerce_float(params.get("kcMult"), 1.0)

    flux_len = max(_coerce_int(params.get("fluxLen"), 14), 1)
    flux_smooth_len = max(_coerce_int(params.get("fluxSmoothLen"), 1), 1)
    flux_use_ha = _coerce_bool(params.get("useFluxHeikin"), True)

    hl2 = (df["high"] + df["low"]) / 2.0
    atr_kc = _atr(df, kc_len).replace(0.0, np.nan)
    kc_basis = _sma(hl2, kc_len)
    kc_range = atr_kc * kc_mult
    kc_upper = kc_basis + kc_range
    kc_lower = kc_basis - kc_range
    kc_average = (kc_upper + kc_lower) / 2.0
    midline = (hl2 + kc_average) / 2.0
    norm = (df["close"] - midline) / atr_kc * 100.0
    momentum = _linreg(norm, osc_len)
    mom_signal = _sma(momentum, sig_len)

    prev_mom = momentum.shift(1).fillna(momentum)
    prev_sig = mom_signal.shift(1).fillna(mom_signal)
    cross_up = (prev_mom <= prev_sig) & (momentum > mom_signal)
    cross_down = (prev_mom >= prev_sig) & (momentum < mom_signal)

    flux_df = _heikin_ashi(df) if flux_use_ha else df
    flux_raw = _directional_flux(flux_df, flux_len)
    if flux_smooth_len > 1:
        flux_hist = flux_raw.rolling(flux_smooth_len, min_periods=flux_smooth_len).mean()
    else:
        flux_hist = flux_raw

    return momentum, mom_signal, cross_up.astype(bool), cross_down.astype(bool), flux_hist


def _resolve_thresholds(
    momentum: pd.Series,
    params: Dict[str, object],
) -> Tuple[pd.Series, pd.Series]:
    use_dynamic = _coerce_bool(params.get("useDynamicThresh"), True)
    use_sym = _coerce_bool(params.get("useSymThreshold"), False)
    stat_threshold = _coerce_float(params.get("statThreshold"), 38.0)
    buy_threshold = _coerce_float(params.get("buyThreshold"), 36.0)
    sell_threshold = _coerce_float(params.get("sellThreshold"), 36.0)
    if use_dynamic:
        dyn_len = max(_coerce_int(params.get("dynLen"), 21), 1)
        dyn_mult = _coerce_float(params.get("dynMult"), 1.1)
        dyn_series = momentum.rolling(dyn_len, min_periods=dyn_len).std() * dyn_mult
        fallback = abs(stat_threshold) if stat_threshold else dyn_series.dropna().mean()
        if not np.isfinite(fallback) or fallback == 0:
            fallback = 1.0
        dyn_series = dyn_series.abs().fillna(abs(fallback))
        buy = -dyn_series
        sell = dyn_series
    else:
        if use_sym:
            buy_val = -abs(stat_threshold)
            sell_val = abs(stat_threshold)
        else:
            buy_val = -abs(buy_threshold)
            sell_val = abs(sell_threshold)
        index = momentum.index
        buy = pd.Series(buy_val, index=index)
        sell = pd.Series(sell_val, index=index)
    return buy, sell


def _build_signals(
    df: pd.DataFrame,
    params: Dict[str, object],
    parsed: _ParsedInputs,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    momentum, mom_signal, cross_up, cross_down, flux_hist = _compute_indicators(df, params)
    buy_thresh, sell_thresh = _resolve_thresholds(momentum, params)

    base_long = (momentum < buy_thresh) & (flux_hist > 0)
    base_short = (momentum > sell_thresh) & (flux_hist < 0)

    if parsed.require_cross:
        base_long &= cross_up
        base_short &= cross_down

    if not parsed.allow_long:
        base_long = pd.Series(False, index=df.index)
    if not parsed.allow_short:
        base_short = pd.Series(False, index=df.index)

    long_exits = cross_down.copy()
    short_exits = cross_up.copy()
    if parsed.exit_opposite:
        long_exits |= base_short
        short_exits |= base_long

    base_long &= df.index >= parsed.start_ts
    base_short &= df.index >= parsed.start_ts
    long_exits &= df.index >= parsed.start_ts
    short_exits &= df.index >= parsed.start_ts

    return (
        base_long.astype(bool),
        long_exits.astype(bool),
        base_short.astype(bool),
        short_exits.astype(bool),
    )


def _prepare_mom_fade_context(
    df: pd.DataFrame,
    params: Dict[str, object],
) -> Optional[_MomFadeContext]:
    """모멘텀 페이드 청산에 필요한 시계열과 파라미터를 계산한다."""

    use_mom_fade = _coerce_bool(params.get("useMomFade"), False)
    if not use_mom_fade:
        return None

    mom_fade_bars = max(_coerce_int(params.get("momFadeBars"), 1), 0)
    mom_fade_reg_len = max(_coerce_int(params.get("momFadeRegLen"), 20), 1)
    mom_fade_bb_len = max(_coerce_int(params.get("momFadeBbLen"), 20), 1)
    mom_fade_kc_len = max(_coerce_int(params.get("momFadeKcLen"), 20), 1)
    mom_fade_bb_mult = _coerce_float(params.get("momFadeBbMult"), 2.0)
    mom_fade_kc_mult = _coerce_float(params.get("momFadeKcMult"), 1.5)
    mom_fade_use_true_range = _coerce_bool(params.get("momFadeUseTrueRange"), True)
    mom_fade_zero_delay = max(_coerce_int(params.get("momFadeZeroDelay"), 0), 0)
    mom_fade_min_abs = max(0.0, _coerce_float(params.get("momFadeMinAbs"), 0.0))
    mom_fade_release_only = _coerce_bool(params.get("momFadeReleaseOnly"), True)
    mom_fade_require_two = _coerce_bool(params.get("momFadeRequireTwoBars"), False)
    sqz_release_bars = max(_coerce_int(params.get("sqzReleaseBars"), 5), 0)

    mom_fade_source = (df["high"] + df["low"] + df["close"]) / 3.0
    mom_fade_basis = _sma(mom_fade_source, mom_fade_bb_len)
    mom_fade_dev = _std(mom_fade_source, mom_fade_bb_len) * mom_fade_bb_mult
    prev_close = df["close"].shift().fillna(df["close"])
    if mom_fade_use_true_range:
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        mom_range = _rma(tr, mom_fade_kc_len)
    else:
        mom_range = _sma((df["high"] - df["low"]).abs(), mom_fade_kc_len)
    mom_range = mom_range * mom_fade_kc_mult
    mom_mid = mom_fade_basis
    mom_fade_hist = _linreg(mom_fade_source - mom_mid, mom_fade_reg_len)
    mom_fade_abs = mom_fade_hist.abs()
    mom_fade_abs_prev = mom_fade_abs.shift(1).fillna(mom_fade_abs)
    mom_fade_abs_prev2 = mom_fade_abs.shift(2).fillna(mom_fade_abs_prev)
    mom_fade_nonpos = mom_fade_hist.le(0)
    mom_fade_nonneg = mom_fade_hist.ge(0)
    mom_fade_since_nonpos = _bars_since_mask(mom_fade_nonpos)
    mom_fade_since_nonneg = _bars_since_mask(mom_fade_nonneg)

    gate_dev = mom_fade_dev
    gate_atr = mom_range
    gate_sq_on = (gate_dev < gate_atr).fillna(False).astype(bool)
    gate_sq_prev = gate_sq_on.shift(fill_value=False)
    gate_sq_rel = gate_sq_prev & np.logical_not(gate_sq_on)
    gate_rel_idx = gate_sq_rel.cumsum()
    gate_rel_idx = gate_rel_idx.where(gate_sq_rel, np.nan).ffill()
    if gate_rel_idx.notna().any():
        gate_bars_since_release = (
            df.index.to_series()
            .groupby(gate_rel_idx)
            .cumcount()
            .reindex(df.index, fill_value=np.nan)
        )
    else:
        gate_bars_since_release = pd.Series(np.nan, index=df.index)
    gate_bars_since_release = gate_bars_since_release.fillna(np.inf)
    gate_release_seen = gate_bars_since_release != np.inf
    gate_valid = (gate_release_seen & (gate_bars_since_release <= sqz_release_bars) & (~gate_sq_on)).astype(bool)

    return _MomFadeContext(
        hist=mom_fade_hist,
        abs_hist=mom_fade_abs,
        abs_prev=mom_fade_abs_prev,
        abs_prev2=mom_fade_abs_prev2,
        since_nonpos=mom_fade_since_nonpos,
        since_nonneg=mom_fade_since_nonneg,
        gate_valid=gate_valid,
        bars_required=mom_fade_bars,
        zero_delay=mom_fade_zero_delay,
        min_abs=mom_fade_min_abs,
        release_only=mom_fade_release_only,
        require_two=mom_fade_require_two,
    )


def _apply_exit_overrides(
    parsed: _ParsedInputs,
    params: Dict[str, object],
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """스탑 및 모멘텀 페이드 조건을 반영해 청산 신호를 조정한다."""

    use_stop_loss = _coerce_bool(params.get("useStopLoss"), False)
    use_atr_trail = _coerce_bool(params.get("useAtrTrail"), False)
    use_mom_fade = _coerce_bool(params.get("useMomFade"), False)

    if not any((use_stop_loss, use_atr_trail, use_mom_fade)):
        return long_entries, long_exits, short_entries, short_exits

    df = parsed.df
    index = df.index
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    atr_values = None
    atr_trail_mult = 0.0
    if use_atr_trail:
        atr_len = max(_coerce_int(params.get("atrTrailLen"), 7), 1)
        atr_trail_mult = _coerce_float(params.get("atrTrailMult"), 2.5)
        atr_values = _atr(df, atr_len).to_numpy(dtype=float)

    swing_low = None
    swing_high = None
    if use_stop_loss:
        lookback = max(_coerce_int(params.get("stopLookback"), 5), 1)
        swing_low = df["low"].rolling(lookback, min_periods=lookback).min().to_numpy(dtype=float)
        swing_high = df["high"].rolling(lookback, min_periods=lookback).max().to_numpy(dtype=float)

    mom_ctx = _prepare_mom_fade_context(df, params) if use_mom_fade else None
    if use_mom_fade and mom_ctx is None:
        use_mom_fade = False

    hist_arr = mom_ctx.hist.to_numpy(dtype=float) if mom_ctx else None
    abs_arr = mom_ctx.abs_hist.to_numpy(dtype=float) if mom_ctx else None
    abs_prev_arr = mom_ctx.abs_prev.to_numpy(dtype=float) if mom_ctx else None
    abs_prev2_arr = mom_ctx.abs_prev2.to_numpy(dtype=float) if mom_ctx else None
    since_nonpos_arr = mom_ctx.since_nonpos.to_numpy(dtype=float) if mom_ctx else None
    since_nonneg_arr = mom_ctx.since_nonneg.to_numpy(dtype=float) if mom_ctx else None
    gate_valid_arr = mom_ctx.gate_valid.to_numpy(dtype=bool) if mom_ctx else None

    long_entries_arr = long_entries.to_numpy(dtype=bool)
    short_entries_arr = short_entries.to_numpy(dtype=bool)
    long_exits_arr = long_exits.to_numpy(dtype=bool)
    short_exits_arr = short_exits.to_numpy(dtype=bool)

    position_dir = 0
    bars_held = 0
    long_stop_price = np.nan
    short_stop_price = np.nan

    for i in range(n):
        skip_entry = False

        if position_dir != 0:
            long_entries_arr[i] = False
            short_entries_arr[i] = False

        if position_dir > 0:
            bars_held += 1
            exit_long = bool(long_exits_arr[i])

            # ATR 트레일링 스탑
            if use_atr_trail and atr_values is not None:
                atr_val = atr_values[i]
                if np.isfinite(atr_val):
                    candidate = close[i] - atr_val * atr_trail_mult
                    if np.isfinite(candidate):
                        if np.isnan(long_stop_price):
                            long_stop_price = candidate
                        else:
                            long_stop_price = max(long_stop_price, candidate)

            # 스윙 로우 기반 스탑
            if use_stop_loss and swing_low is not None:
                ref = swing_low[i]
                if np.isfinite(ref):
                    if np.isnan(long_stop_price):
                        long_stop_price = ref
                    else:
                        long_stop_price = max(long_stop_price, ref)

            if np.isfinite(long_stop_price) and low[i] <= long_stop_price:
                exit_long = True

            if use_mom_fade and mom_ctx is not None:
                hist_val = hist_arr[i]
                if np.isfinite(hist_val):
                    fade_abs = abs_arr[i]
                    prev_abs = abs_prev_arr[i]
                    prev2_abs = abs_prev2_arr[i]
                    fade_abs_falling = (
                        fade_abs < prev_abs if mom_ctx.bars_required <= 1 else fade_abs <= prev_abs
                    )
                    fade_abs_two = (not mom_ctx.require_two) or (
                        fade_abs <= prev_abs and prev_abs <= prev2_abs
                    )
                    delay_ok = (
                        mom_ctx.zero_delay <= 0
                        or since_nonpos_arr[i] > float(mom_ctx.zero_delay)
                    )
                    min_abs_ok = mom_ctx.min_abs <= 0 or fade_abs >= mom_ctx.min_abs
                    release_ok = (not mom_ctx.release_only) or gate_valid_arr[i]
                    if (
                        hist_val > 0
                        and fade_abs_falling
                        and fade_abs_two
                        and delay_ok
                        and min_abs_ok
                        and release_ok
                        and bars_held >= mom_ctx.bars_required
                    ):
                        exit_long = True

            long_exits_arr[i] = exit_long
            if exit_long:
                position_dir = 0
                bars_held = 0
                long_stop_price = np.nan
                skip_entry = True
                long_entries_arr[i] = False
                short_entries_arr[i] = False

        elif position_dir < 0:
            bars_held += 1
            exit_short = bool(short_exits_arr[i])

            if use_atr_trail and atr_values is not None:
                atr_val = atr_values[i]
                if np.isfinite(atr_val):
                    candidate = close[i] + atr_val * atr_trail_mult
                    if np.isfinite(candidate):
                        if np.isnan(short_stop_price):
                            short_stop_price = candidate
                        else:
                            short_stop_price = min(short_stop_price, candidate)

            if use_stop_loss and swing_high is not None:
                ref = swing_high[i]
                if np.isfinite(ref):
                    if np.isnan(short_stop_price):
                        short_stop_price = ref
                    else:
                        short_stop_price = min(short_stop_price, ref)

            if np.isfinite(short_stop_price) and high[i] >= short_stop_price:
                exit_short = True

            if use_mom_fade and mom_ctx is not None:
                hist_val = hist_arr[i]
                if np.isfinite(hist_val):
                    fade_abs = abs_arr[i]
                    prev_abs = abs_prev_arr[i]
                    prev2_abs = abs_prev2_arr[i]
                    fade_abs_falling = (
                        fade_abs < prev_abs if mom_ctx.bars_required <= 1 else fade_abs <= prev_abs
                    )
                    fade_abs_two = (not mom_ctx.require_two) or (
                        fade_abs <= prev_abs and prev_abs <= prev2_abs
                    )
                    delay_ok = (
                        mom_ctx.zero_delay <= 0
                        or since_nonneg_arr[i] > float(mom_ctx.zero_delay)
                    )
                    min_abs_ok = mom_ctx.min_abs <= 0 or fade_abs >= mom_ctx.min_abs
                    release_ok = (not mom_ctx.release_only) or gate_valid_arr[i]
                    if (
                        hist_val < 0
                        and fade_abs_falling
                        and fade_abs_two
                        and delay_ok
                        and min_abs_ok
                        and release_ok
                        and bars_held >= mom_ctx.bars_required
                    ):
                        exit_short = True

            short_exits_arr[i] = exit_short
            if exit_short:
                position_dir = 0
                bars_held = 0
                short_stop_price = np.nan
                skip_entry = True
                long_entries_arr[i] = False
                short_entries_arr[i] = False

        else:
            bars_held = 0
            long_stop_price = np.nan
            short_stop_price = np.nan

        if position_dir == 0 and not skip_entry:
            long_entry = long_entries_arr[i]
            short_entry = short_entries_arr[i]
            if long_entry and not short_entry:
                position_dir = 1
                bars_held = 0
                long_stop_price = np.nan
                if use_atr_trail and atr_values is not None:
                    atr_val = atr_values[i]
                    if np.isfinite(atr_val):
                        candidate = close[i] - atr_val * atr_trail_mult
                        if np.isfinite(candidate):
                            long_stop_price = candidate
                if use_stop_loss and swing_low is not None:
                    ref = swing_low[i]
                    if np.isfinite(ref):
                        long_stop_price = ref if np.isnan(long_stop_price) else max(long_stop_price, ref)
            elif short_entry and not long_entry:
                position_dir = -1
                bars_held = 0
                short_stop_price = np.nan
                if use_atr_trail and atr_values is not None:
                    atr_val = atr_values[i]
                    if np.isfinite(atr_val):
                        candidate = close[i] + atr_val * atr_trail_mult
                        if np.isfinite(candidate):
                            short_stop_price = candidate
                if use_stop_loss and swing_high is not None:
                    ref = swing_high[i]
                    if np.isfinite(ref):
                        short_stop_price = ref if np.isnan(short_stop_price) else min(short_stop_price, ref)

    return (
        pd.Series(long_entries_arr, index=index, dtype=bool),
        pd.Series(long_exits_arr, index=index, dtype=bool),
        pd.Series(short_entries_arr, index=index, dtype=bool),
        pd.Series(short_exits_arr, index=index, dtype=bool),
    )


def _bars_between(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> int:
    try:
        start_loc = index.get_loc(start)
    except KeyError:
        start_loc = index.get_indexer([start])[0]
    try:
        end_loc = index.get_loc(end)
    except KeyError:
        end_loc = index.get_indexer([end])[0]
    return max(int(end_loc - start_loc), 1)


def _vectorbt_backtest(
    parsed: _ParsedInputs,
    params: Dict[str, object],
) -> Dict[str, float]:
    if not VECTORBT_AVAILABLE or _VBT_MODULE is None:  # pragma: no cover - 환경 의존
        raise ImportError("vectorbt 가 설치되어 있지 않습니다.")

    _validate_feature_flags(params)

    long_entries, long_exits, short_entries, short_exits = _build_signals(
        parsed.df, params, parsed
    )

    long_entries, long_exits, short_entries, short_exits = _apply_exit_overrides(
        parsed,
        params,
        long_entries,
        long_exits,
        short_entries,
        short_exits,
    )

    trade_value = parsed.initial_capital * parsed.capital_pct * parsed.leverage
    if trade_value <= 0:
        trade_value = parsed.initial_capital * parsed.leverage

    pf = _VBT_MODULE.Portfolio.from_signals(
        parsed.df["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=parsed.commission_pct,
        size=trade_value,
        size_type="value",
        direction="both",
        upon_opposite_entry="close",
    )
    pf = pf.close()

    raw_returns = pf.returns()
    returns = pd.Series(np.asarray(raw_returns), index=parsed.df.index)

    records = pf.trades.records_readable
    trades: List[Trade] = []
    if not records.empty:
        for row in records.to_dict("records"):
            entry_ts = pd.Timestamp(row.get("Entry Timestamp"))
            exit_ts = pd.Timestamp(row.get("Exit Timestamp"))
            if pd.isna(exit_ts):
                continue
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            else:
                entry_ts = entry_ts.tz_convert("UTC")
            if exit_ts.tz is None:
                exit_ts = exit_ts.tz_localize("UTC")
            else:
                exit_ts = exit_ts.tz_convert("UTC")
            direction = str(row.get("Direction", "long")).strip().lower()
            trade = Trade(
                entry_time=entry_ts,
                exit_time=exit_ts,
                direction="long" if direction.startswith("long") else "short",
                size=float(row.get("Size", 0.0) or 0.0),
                entry_price=float(row.get("Avg Entry Price", 0.0) or 0.0),
                exit_price=float(row.get("Avg Exit Price", 0.0) or 0.0),
                profit=float(row.get("PnL", 0.0) or 0.0),
                return_pct=float(row.get("Return", 0.0) or 0.0),
                mfe=float("nan"),
                mae=float("nan"),
                bars_held=_bars_between(parsed.df.index, entry_ts, exit_ts),
                reason="vectorbt",
            )
            trades.append(trade)

    metrics = aggregate_metrics(trades, returns, simple=False)
    metrics.setdefault("InitialCapital", parsed.initial_capital)
    metrics.setdefault("Leverage", parsed.leverage)
    metrics.setdefault("Commission", parsed.commission_pct)
    metrics.setdefault("SlippageTicks", parsed.slippage_ticks)
    metrics["Engine"] = "vectorbt"
    metrics["MinTrades"] = float(parsed.min_trades)
    metrics["MinHoldBars"] = float(parsed.min_hold_bars)
    metrics["MaxConsecutiveLossLimit"] = float(parsed.max_consecutive_losses)

    valid = (
        metrics.get("Trades", 0.0) >= parsed.min_trades
        and metrics.get("AvgHoldBars", 0.0) >= parsed.min_hold_bars
        and metrics.get("MaxConsecutiveLosses", 0.0) <= parsed.max_consecutive_losses
        and not metrics.get("LosslessProfitFactor", False)
    )
    metrics["Valid"] = bool(valid)
    return metrics


def run_backtest_alternative(
    df: pd.DataFrame,
    params: Dict[str, object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
    *,
    engine: str = "vectorbt",
) -> Dict[str, float]:
    """외부 엔진으로 백테스트를 실행한다."""

    engine_name = str(engine or "").strip().lower()
    parsed = _parse_core_settings(
        df,
        params,
        fees,
        risk,
        min_trades=min_trades,
        htf_df=htf_df,
    )

    if engine_name in _SUPPORTED_ENGINES:
        return _vectorbt_backtest(parsed, params)
    if engine_name in _PYBROKER_ENGINES:
        if not PYBROKER_AVAILABLE:  # pragma: no cover - 설치 여부에 따라 다름
            raise ImportError("pybroker 가 설치되어 있지 않습니다.")
        raise NotImplementedError(
            "pybroker 엔진 호환 레이어는 아직 구현되지 않았습니다."
        )
    raise NotImplementedError(f"알 수 없는 대체 엔진: {engine}")

