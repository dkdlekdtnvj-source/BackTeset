"""Python 백테스트 엔진 – TradingView `매직1분VN` 최종본을 재현합니다."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import Trade, aggregate_metrics


LOGGER = logging.getLogger(__name__)


# =====================================================================================
# === 보조 계산 함수들 ===============================================================
# =====================================================================================


def _ensure_series(values: Iterable[float], index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index, dtype=float)


def _ema(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.ewm(span=length, adjust=False).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.rolling(length, min_periods=length).mean()


def _std(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.rolling(length, min_periods=length).std(ddof=0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    return _rma(_true_range(df), length)


def _linreg(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    if length == 1:
        return series.copy()

    idx = np.arange(length, dtype=float)

    def _calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        slope, intercept = np.polyfit(idx, values, 1)
        return slope * (length - 1) + intercept

    return series.rolling(length, min_periods=length).apply(_calc, raw=True)


def _timeframe_to_offset(timeframe: str) -> Optional[str]:
    tf = str(timeframe).strip()
    if not tf:
        return None
    if tf.endswith("m"):
        return f"{int(tf[:-1])}min"
    if tf.endswith("h"):
        return f"{int(tf[:-1])}H"
    if tf.endswith("D"):
        return f"{int(tf[:-1])}D"
    if tf.endswith("W"):
        return f"{int(tf[:-1])}W"
    if tf.isdigit():
        return f"{int(tf)}min"
    return None


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    offset = _timeframe_to_offset(timeframe)
    if offset is None:
        return df
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df.resample(offset, label="right", closed="right").agg(agg)
    return resampled.dropna()


def _security_series(
    df: pd.DataFrame, timeframe: str, compute: callable, default: float = np.nan
) -> pd.Series:
    if timeframe in {"", "0", None}:
        out = compute(df)
        return out if isinstance(out, pd.Series) else _ensure_series(out, df.index)
    resampled = _resample_ohlcv(df, timeframe)
    if resampled.empty:
        out = compute(df)
        return out if isinstance(out, pd.Series) else _ensure_series(out, df.index)
    result = compute(resampled)
    if not isinstance(result, pd.Series):
        result = _ensure_series(result, resampled.index)
    result = result.reindex(df.index, method="ffill")
    return result.fillna(default)


def _max_ignore_nan(*values: float) -> float:
    """NaN 을 무시하면서 최대값을 계산합니다.

    전달된 값이 모두 NaN 이거나 ``None`` 이면 ``np.nan`` 을 돌려 빈 시퀀스에 대한 ``max`` 호출을
    회피합니다. ``float`` 로 강제 변환 가능한 항목만 고려해 예외 발생 가능성을 낮춥니다.
    """

    cleaned: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(numeric):
            continue
        cleaned.append(numeric)

    if not cleaned:
        return np.nan
    return max(cleaned)


def _min_ignore_nan(*values: float) -> float:
    """NaN 을 무시하면서 최소값을 계산합니다.

    ``values`` 가 모두 비어 있거나 유효하지 않은 경우 ``np.nan`` 을 반환해 ``min`` 의 빈 시퀀스
    예외를 방지합니다.
    """

    cleaned: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(numeric):
            continue
        cleaned.append(numeric)

    if not cleaned:
        return np.nan
    return min(cleaned)


def _pivot_series(series: pd.Series, left: int, right: int, is_high: bool) -> pd.Series:
    left = max(int(left), 1)
    right = max(int(right), 1)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.to_numpy()
    for idx in range(left, len(series) - right):
        window = values[idx - left : idx + right + 1]
        center = window[left]
        if is_high and center == window.max():
            result.iloc[idx + right] = center
        if not is_high and center == window.min():
            result.iloc[idx + right] = center
    return result.ffill()


def _dmi(df: pd.DataFrame, length: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    length = max(int(length), 1)
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = _rma(pd.Series(plus_dm, index=df.index), length)
    minus_dm = _rma(pd.Series(minus_dm, index=df.index), length)
    tr = _atr(df, length).replace(0.0, np.nan)
    plus_di = 100.0 * (plus_dm / tr)
    minus_di = 100.0 * (minus_dm / tr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
    adx = _rma(dx.fillna(0.0), length)
    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx.fillna(0.0)


def _rsi(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    diff = series.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    avg_gain = _rma(up, length)
    avg_loss = _rma(down, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)


def _stoch_rsi(series: pd.Series, length: int) -> pd.Series:
    rsi = _rsi(series, length)
    lowest = rsi.rolling(length, min_periods=length).min()
    highest = rsi.rolling(length, min_periods=length).max()
    denom = (highest - lowest).replace(0, np.nan)
    return ((rsi - lowest) / denom * 100.0).fillna(50.0)


def _obv_slope(close: pd.Series, volume: pd.Series, smooth: int) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    obv = (direction * volume.fillna(0.0)).cumsum()
    return _ema(obv.diff().fillna(0.0), max(int(smooth), 1))


def _estimate_tick(series: pd.Series) -> float:
    diffs = series.diff().abs()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return float(series.iloc[-1]) * 1e-6 if len(series) else 0.01
    return float(diffs.min())


def _cross_over(prev_a: float, prev_b: float, a: float, b: float) -> bool:
    return prev_a <= prev_b and a > b


def _cross_under(prev_a: float, prev_b: float, a: float, b: float) -> bool:
    return prev_a >= prev_b and a < b


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = ha_close.copy()
    if len(df) > 0:
        ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = pd.concat([ha_open, ha_close, df["high"]], axis=1).max(axis=1)
    ha_low = pd.concat([ha_open, ha_close, df["low"]], axis=1).min(axis=1)
    ha["open"] = ha_open
    ha["close"] = ha_close
    ha["high"] = ha_high
    ha["low"] = ha_low
    return ha


def _directional_flux(df: pd.DataFrame, length: int) -> pd.Series:
    length = max(int(length), 1)
    high = df["high"]
    low = df["low"]
    prev_high = high.shift()
    prev_low = low.shift()
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = _rma(pd.Series(plus_dm, index=df.index), length)
    minus_dm = _rma(pd.Series(minus_dm, index=df.index), length)
    atr = _atr(df, length).replace(0, np.nan)
    plus_di = 100 * (plus_dm / atr)
    minus_di = 100 * (minus_dm / atr)
    return plus_di - minus_di


@dataclass
class Position:
    direction: int = 0
    qty: float = 0.0
    avg_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    bars_held: int = 0
    highest: float = np.nan
    lowest: float = np.nan


@dataclass
class EquityState:
    initial_capital: float
    equity: float
    net_profit: float = 0.0
    withdrawable: float = 0.0
    tradable_capital: float = 0.0
    peak_equity: float = 0.0
    daily_start_capital: float = 0.0
    daily_peak_capital: float = 0.0
    week_start_equity: float = 0.0
    week_peak_equity: float = 0.0


def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, float | bool | str],
    fees: Dict[str, float],
    risk: Dict[str, float | bool],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    """TradingView `매직1분VN` 최종본과 동등한 파이썬 백테스트."""

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError("DataFrame must contain OHLCV columns")

    def _ensure_datetime_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if isinstance(frame.index, pd.DatetimeIndex):
            idx = frame.index
            if idx.tz is None:
                frame = frame.copy()
                frame.index = idx.tz_localize("UTC")
            return frame

        frame = frame.copy()
        if "timestamp" in frame.columns:
            converted = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            valid_mask = converted.notna()
            invalid = (~valid_mask).sum()
            if invalid:
                LOGGER.warning(
                    "%s 데이터프레임에서 timestamp 컬럼의 %d개 행을 제거했습니다.",
                    label,
                    int(invalid),
                )
            if valid_mask.any():
                frame = frame.loc[valid_mask].copy()
                frame.index = converted[valid_mask]
                frame.drop(columns=["timestamp"], inplace=True)
                return frame
        raise TypeError(
            f"{label} 데이터프레임은 DatetimeIndex 를 가져야 합니다. "
            "timestamp 컬럼이 있다면 UTC 로 변환한 뒤 다시 실행해주세요."
        )

    df = _ensure_datetime_index(df, "가격")
    if htf_df is not None:
        htf_df = _ensure_datetime_index(htf_df, "HTF")

    def _normalise_ohlcv(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        frame = frame.copy()
        frame.sort_index(inplace=True)

        if frame.index.has_duplicates:
            dup_count = int(frame.index.duplicated(keep="last").sum())
            if dup_count:
                LOGGER.warning("%s 데이터프레임에서 중복 인덱스 %d개를 제거합니다.", label, dup_count)
            frame = frame[~frame.index.duplicated(keep="last")]

        for column in required_cols:
            if column not in frame.columns:
                continue
            coerced = pd.to_numeric(frame[column], errors="coerce")
            invalid_count = int((coerced.isna() & frame[column].notna()).sum())
            if invalid_count:
                LOGGER.warning(
                    "%s 데이터프레임의 %s 열에서 비수치 값 %d개를 NaN 으로 치환했습니다.",
                    label,
                    column,
                    invalid_count,
                )
            frame[column] = coerced

        before = len(frame)
        frame = frame.dropna(subset=list(required_cols))
        dropped = before - len(frame)
        if dropped:
            LOGGER.warning("%s 데이터프레임에서 결측 OHLCV 행 %d개를 제거했습니다.", label, int(dropped))

        if len(frame) < 2:
            raise ValueError(f"{label} 데이터가 부족하여 백테스트를 진행할 수 없습니다.")

        return frame

    df = _normalise_ohlcv(df, "가격")
    if htf_df is not None:
        htf_df = _normalise_ohlcv(htf_df, "HTF")

    def _coerce_bool(value: object, default: bool) -> bool:
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

    def bool_param(name: str, default: bool, *, enabled: bool = True) -> bool:
        if not enabled:
            return default
        return _coerce_bool(params.get(name, default), default)

    def int_param(name: str, default: int, *, enabled: bool = True) -> int:
        if not enabled:
            return int(default)
        value = params.get(name, default)
        if isinstance(value, bool):
            return int(value)
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    def float_param(name: str, default: float, *, enabled: bool = True) -> float:
        if not enabled:
            return float(default)
        value = params.get(name, default)
        if isinstance(value, bool):
            return float(int(value))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _coerce_float_value(value: object, default: float) -> float:
        """임의의 값을 ``float`` 로 강제 변환합니다."""

        if value is None:
            return float(default)
        if isinstance(value, bool):
            return float(int(value))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _resolve_penalty_value(*names: str, default: float) -> float:
        fallback = float(default)
        if not np.isfinite(fallback) or fallback < 0:
            fallback = 1.0
        for source in (risk, params):
            if not isinstance(source, dict):
                continue
            for name in names:
                if name not in source or source[name] is None:
                    continue
                candidate = _coerce_float_value(source[name], fallback)
                if not np.isfinite(candidate):
                    continue
                return float(abs(candidate))
        return abs(fallback)

    def _resolve_requirement_value(*names: str, default: float) -> float:
        fallback = float(default)
        if not np.isfinite(fallback):
            fallback = 0.0
        for source in (risk, params):
            if not isinstance(source, dict):
                continue
            for name in names:
                if name not in source or source[name] is None:
                    continue
                candidate = _coerce_float_value(source[name], fallback)
                if np.isfinite(candidate):
                    return float(candidate)
        return fallback

    def _apply_penalty_settings(
        target: Dict[str, float],
        *,
        min_trades_value: float,
        min_hold_value: float,
        max_loss_streak: float,
    ) -> None:
        target["MinTrades"] = float(max(0.0, min_trades_value))
        target["MinHoldBars"] = float(max(0.0, min_hold_value))
        target["MaxConsecutiveLossLimit"] = float(max(0.0, max_loss_streak))

        penalty_specs = {
            "TradePenalty": (("penalty_trade", "penaltyTrade", "tradePenalty"), 0.0),
            "HoldPenalty": (("penalty_hold", "penaltyHold", "holdPenalty"), 0.0),
            "ConsecutiveLossPenalty": (
                (
                    "penalty_consecutive_loss",
                    "penaltyConsecutiveLoss",
                    "consecutiveLossPenalty",
                ),
                0.0,
            ),
        }

        for key, (names, default_value) in penalty_specs.items():
            resolved = _resolve_penalty_value(*names, default=default_value)
            if not np.isfinite(resolved):
                resolved = default_value
            target[key] = float(max(0.0, resolved))

    def str_param(name: str, default: str, *, enabled: bool = True) -> str:
        if not enabled:
            return str(default)
        value = params.get(name, default)
        return str(value) if value is not None else str(default)

    # Pine 입력 매핑 -----------------------------------------------------------------
    osc_len = int_param("oscLen", 12)
    sig_len = int_param("signalLen", 3)
    use_same_len = bool_param("useSameLen", False)
    bb_len = osc_len if use_same_len else int_param("bbLen", 20)
    kc_len = osc_len if use_same_len else int_param("kcLen", 18)
    bb_mult = float_param("bbMult", 1.4)
    kc_mult = float_param("kcMult", 1.0)

    flux_len = int_param("fluxLen", 14)
    flux_smooth_len = int_param("fluxSmoothLen", 1)
    flux_use_ha = bool_param("useFluxHeikin", True)

    use_dynamic_thresh = bool_param("useDynamicThresh", True)
    use_sym_threshold = bool_param("useSymThreshold", False)
    stat_threshold = float_param("statThreshold", 38.0)
    buy_threshold = float_param("buyThreshold", 36.0)
    sell_threshold = float_param("sellThreshold", 36.0)
    dyn_len = int_param("dynLen", 21, enabled=use_dynamic_thresh)
    dyn_mult = float_param("dynMult", 1.1, enabled=use_dynamic_thresh)
    require_momentum_cross = bool_param("requireMomentumCross", True)

    start_ts = pd.to_datetime(params.get("startDate", "2025-07-01T00:00:00"), utc=True)

    leverage = float(risk.get("leverage", params.get("leverage", 10.0)))
    commission_pct = float(fees.get("commission_pct", params.get("commission_value", 0.0005)))
    slippage_ticks = float(fees.get("slippage_ticks", params.get("slipTicks", 1)))
    initial_capital = float(risk.get("initial_capital", params.get("initial_capital", 500.0)))

    allow_long_entry = bool_param("allowLongEntry", True)
    allow_short_entry = bool_param("allowShortEntry", True)
    debug_force_long = bool_param("debugForceLong", False)
    debug_force_short = bool_param("debugForceShort", False)
    reentry_bars = int_param("reentryBars", 0)

    if bool_param("useSessionFilter", False):
        LOGGER.warning("세션 필터 기능은 안정성 문제로 인해 현재 비활성화됩니다.")
    if bool_param("useDayFilter", False):
        LOGGER.warning("요일 필터 기능은 안정성 문제로 인해 현재 비활성화됩니다.")
    if bool_param("useEventFilter", False):
        LOGGER.warning("이벤트 필터 기능은 안정성 문제로 인해 현재 비활성화됩니다.")

    # 리스크/지갑 --------------------------------------------------------------------
    base_qty_percent = float_param("baseQtyPercent", 30.0)
    qty_override = risk.get("qty_pct")
    if qty_override is None:
        qty_override = risk.get("qtyPercent")
    if qty_override is not None:
        resolved_qty = _coerce_float_value(qty_override, base_qty_percent)
        if np.isfinite(resolved_qty):
            base_qty_percent = resolved_qty
    use_sizing_override = bool_param("useSizingOverride", False)
    sizing_mode = str_param("sizingMode", "자본 비율")
    advanced_percent = float_param("advancedPercent", 25.0, enabled=use_sizing_override)
    fixed_usd_amount = float_param("fixedUsdAmount", 100.0, enabled=use_sizing_override)
    fixed_contract_size = float_param("fixedContractSize", 1.0, enabled=use_sizing_override)
    risk_sizing_type = str_param("riskSizingType", "손절 기반 %", enabled=use_sizing_override)
    base_risk_pct = float_param("baseRiskPct", 0.6)
    risk_contract_size = float_param("riskContractSize", 1.0, enabled=use_sizing_override)
    use_wallet = bool_param("useWallet", False)
    profit_reserve_pct = (
        float_param("profitReservePct", 20.0, enabled=use_wallet) / 100.0 if use_wallet else 0.0
    )
    apply_reserve_to_sizing = bool_param("applyReserveToSizing", True, enabled=use_wallet)
    min_tradable_capital = float_param("minTradableCapital", 250.0)
    use_drawdown_scaling = bool_param("useDrawdownScaling", False)
    drawdown_trigger_pct = float_param("drawdownTriggerPct", 7.0, enabled=use_drawdown_scaling)
    drawdown_risk_scale = float_param("drawdownRiskScale", 0.5, enabled=use_drawdown_scaling)

    use_perf_adaptive_risk = bool_param("usePerfAdaptiveRisk", False)
    par_lookback = int_param("parLookback", 6, enabled=use_perf_adaptive_risk)
    par_min_trades = int_param("parMinTrades", 3, enabled=use_perf_adaptive_risk)
    par_hot_win_rate = float_param("parHotWinRate", 65.0, enabled=use_perf_adaptive_risk)
    par_cold_win_rate = float_param("parColdWinRate", 35.0, enabled=use_perf_adaptive_risk)
    par_hot_mult = float_param("parHotRiskMult", 1.25, enabled=use_perf_adaptive_risk)
    par_cold_mult = float_param("parColdRiskMult", 0.35, enabled=use_perf_adaptive_risk)
    par_pause_on_cold = bool_param("parPauseOnCold", True, enabled=use_perf_adaptive_risk)

    min_trades_default = (
        _coerce_float_value(min_trades, 0.0)
        if min_trades is not None
        else _coerce_float_value(params.get("minTrades"), 0.0)
    )
    if not np.isfinite(min_trades_default):
        min_trades_default = 0.0
    min_trades_value = _resolve_requirement_value(
        "min_trades",
        "minTrades",
        "minTradesReq",
        default=min_trades_default,
    )
    try:
        min_trades_req = max(0, int(float(min_trades_value)))
    except (TypeError, ValueError, OverflowError):
        min_trades_req = max(0, int(min_trades_default))

    use_daily_loss_guard = bool_param("useDailyLossGuard", False)
    daily_loss_limit = float_param("dailyLossLimit", 80.0)
    use_daily_profit_lock = bool_param("useDailyProfitLock", False)
    daily_profit_target = float_param("dailyProfitTarget", 120.0)
    use_weekly_profit_lock = bool_param("useWeeklyProfitLock", False)
    weekly_profit_target = float_param("weeklyProfitTarget", 250.0)
    use_loss_streak_guard = bool_param("useLossStreakGuard", False)
    max_consecutive_loss_default = int_param("maxConsecutiveLosses", 3)
    max_consecutive_value = _resolve_requirement_value(
        "max_consecutive_losses",
        "maxConsecutiveLosses",
        "maxLossStreak",
        default=float(max_consecutive_loss_default),
    )
    try:
        max_consecutive_losses = max(0, int(float(max_consecutive_value)))
    except (TypeError, ValueError, OverflowError):
        max_consecutive_losses = max(0, int(max_consecutive_loss_default))
    use_capital_guard = bool_param("useCapitalGuard", False)
    capital_guard_pct = float_param("capitalGuardPct", 20.0)
    max_daily_losses = int_param("maxDailyLosses", 0)
    max_weekly_dd = float_param("maxWeeklyDD", 0.0)
    max_guard_fires = int_param("maxGuardFires", 0)
    use_guard_exit = bool_param("useGuardExit", False)
    maintenance_margin_pct = float_param("maintenanceMarginPct", 0.5)
    preempt_ticks = int_param("preemptTicks", 8)
    liq_buffer_raw = _coerce_float_value(risk.get("liq_buffer_pct"), 0.0)
    if not np.isfinite(liq_buffer_raw):
        liq_buffer_raw = 0.0
    liq_buffer_pct = max(liq_buffer_raw, 0.0)

    simple_metrics_only = (
        bool_param("simpleMetricsOnly", False)
        or bool_param("simpleProfitOnly", False)
        or _coerce_bool(risk.get("simpleMetricsOnly"), False)
        or _coerce_bool(risk.get("simpleProfitOnly"), False)
    )

    use_volatility_guard = bool_param("useVolatilityGuard", False)
    volatility_lookback = int_param("volatilityLookback", 50, enabled=use_volatility_guard)
    volatility_lower_pct = float_param("volatilityLowerPct", 0.15, enabled=use_volatility_guard)
    volatility_upper_pct = float_param("volatilityUpperPct", 2.5, enabled=use_volatility_guard)

    # 필터 옵션 ---------------------------------------------------------------------
    use_adx = bool_param("useAdx", False)
    use_atr_diff = bool_param("useAtrDiff", False)
    adx_len = int_param("adxLen", 10, enabled=use_adx or use_atr_diff)
    adx_thresh = float_param("adxThresh", 15.0, enabled=use_adx)
    use_ema = bool_param("useEma", False)
    ema_fast_len = int_param("emaFastLen", 8, enabled=use_ema)
    ema_slow_len = int_param("emaSlowLen", 20, enabled=use_ema)
    ema_mode = str_param("emaMode", "Trend", enabled=use_ema)
    use_bb_filter = bool_param("useBb", False)
    bb_filter_len = int_param("bbLenFilter", 20, enabled=use_bb_filter)
    bb_filter_mult = float_param("bbMultFilter", 2.0, enabled=use_bb_filter)
    use_stoch_rsi = bool_param("useStochRsi", False)
    stoch_len = int_param("stochLen", 14, enabled=use_stoch_rsi)
    stoch_ob = float_param("stochOB", 80.0, enabled=use_stoch_rsi)
    stoch_os = float_param("stochOS", 20.0, enabled=use_stoch_rsi)
    use_obv = bool_param("useObv", False)
    obv_smooth_len = int_param("obvSmoothLen", 3, enabled=use_obv)
    adx_atr_tf = str_param("adxAtrTf", "5", enabled=use_adx or use_atr_diff)
    use_htf_trend = bool_param("useHtfTrend", False)
    htf_trend_tf = str_param("htfTrendTf", "240", enabled=use_htf_trend)
    htf_ma_len = int_param("htfMaLen", 20, enabled=use_htf_trend)
    use_hma_filter = bool_param("useHmaFilter", False)
    hma_len = int_param("hmaLen", 20, enabled=use_hma_filter)
    use_range_filter = bool_param("useRangeFilter", False)
    range_tf = str_param("rangeTf", "5", enabled=use_range_filter)
    range_bars = int_param("rangeBars", 20, enabled=use_range_filter)
    range_percent = float_param("rangePercent", 1.0, enabled=use_range_filter)
    use_event_filter = False  # 이벤트 필터는 안정성 문제로 비활성화

    use_regime_filter = bool_param("useRegimeFilter", False)
    ctx_htf_tf = str_param("ctxHtfTf", "240", enabled=use_regime_filter)
    ctx_htf_ema_len = int_param("ctxHtfEmaLen", 120, enabled=use_regime_filter)
    ctx_htf_adx_len = int_param("ctxHtfAdxLen", 14, enabled=use_regime_filter)
    ctx_htf_adx_th = float_param("ctxHtfAdxTh", 22.0, enabled=use_regime_filter)
    use_slope_filter = bool_param("useSlopeFilter", False)
    slope_lookback = int_param("slopeLookback", 8, enabled=use_slope_filter)
    slope_min_pct = float_param("slopeMinPct", 0.06, enabled=use_slope_filter)
    use_distance_guard = bool_param("useDistanceGuard", False)
    distance_atr_len = int_param("distanceAtrLen", 21, enabled=use_distance_guard)
    distance_trend_len = int_param("distanceTrendLen", 55, enabled=use_distance_guard)
    distance_max_atr = float_param("distanceMaxAtr", 2.4, enabled=use_distance_guard)
    use_equity_slope_filter = bool_param("useEquitySlopeFilter", False)
    eq_slope_len = int_param("eqSlopeLen", 120, enabled=use_equity_slope_filter)

    use_sqz_gate = bool_param("useSqzGate", False)
    sqz_release_bars = int_param("sqzReleaseBars", 5, enabled=use_sqz_gate)
    use_structure_gate = bool_param("useStructureGate", False)
    structure_gate_mode = str_param("structureGateMode", "어느 하나 충족", enabled=use_structure_gate)
    use_bos = bool_param("useBOS", False, enabled=use_structure_gate)
    use_choch = bool_param("useCHOCH", False, enabled=use_structure_gate)
    bos_state_bars = int_param("bos_stateBars", 5, enabled=use_bos)
    choch_state_bars = int_param("choch_stateBars", 5, enabled=use_choch)
    bos_tf = str_param("bosTf", "15", enabled=use_bos or use_choch)
    bos_lookback = int_param("bosLookback", 50, enabled=use_bos)
    pivot_left = int_param("pivotLeft_vn", 5, enabled=use_bos or use_choch)
    pivot_right = int_param("pivotRight_vn", 5, enabled=use_bos or use_choch)

    use_reversal = bool_param("useReversal", False)
    reversal_delay_sec = float_param("reversalDelaySec", 0.0, enabled=use_reversal)

    # 출구 옵션 ---------------------------------------------------------------------
    exit_opposite = bool_param("exitOpposite", True)
    use_mom_fade = bool_param("useMomFade", False)
    mom_params_enabled = use_mom_fade or use_sqz_gate
    mom_fade_bars = int_param("momFadeBars", 1, enabled=use_mom_fade)
    mom_fade_reg_len = int_param("momFadeRegLen", 20, enabled=mom_params_enabled)
    mom_fade_bb_len = int_param("momFadeBbLen", 20, enabled=mom_params_enabled)
    mom_fade_kc_len = int_param("momFadeKcLen", 20, enabled=mom_params_enabled)
    mom_fade_bb_mult = float_param("momFadeBbMult", 2.0, enabled=mom_params_enabled)
    mom_fade_kc_mult = float_param("momFadeKcMult", 1.5, enabled=mom_params_enabled)
    mom_fade_use_true_range = bool_param("momFadeUseTrueRange", True, enabled=mom_params_enabled)
    mom_fade_zero_delay = int_param("momFadeZeroDelay", 0, enabled=use_mom_fade)
    mom_fade_min_abs = float_param("momFadeMinAbs", 0.0, enabled=use_mom_fade)
    mom_fade_release_only = bool_param("momFadeReleaseOnly", True, enabled=use_mom_fade)
    mom_fade_min_bars_after_rel = int_param("momFadeMinBarsAfterRel", 1, enabled=use_mom_fade)
    mom_fade_window_bars = int_param("momFadeWindowBars", 6, enabled=use_mom_fade)
    mom_fade_require_two = bool_param("momFadeRequireTwoBars", False, enabled=use_mom_fade)

    use_stop_loss = bool_param("useStopLoss", False)
    stop_lookback = int_param("stopLookback", 5, enabled=use_stop_loss)
    use_atr_trail = bool_param("useAtrTrail", False)
    atr_trail_len = int_param("atrTrailLen", 7, enabled=use_atr_trail)
    atr_trail_mult = float_param("atrTrailMult", 2.5, enabled=use_atr_trail)
    use_breakeven_stop = bool_param("useBreakevenStop", False)
    breakeven_mult = float_param("breakevenMult", 1.0, enabled=use_breakeven_stop)
    use_pivot_stop = bool_param("usePivotStop", False)
    pivot_len = int_param("pivotLen", 5, enabled=use_pivot_stop)
    use_pivot_htf = bool_param("usePivotHtf", False, enabled=use_pivot_stop)
    pivot_tf = str_param("pivotTf", "5", enabled=use_pivot_stop)
    use_atr_profit = bool_param("useAtrProfit", False)
    atr_profit_mult = float_param("atrProfitMult", 2.0, enabled=use_atr_profit)
    use_dyn_vol = bool_param("useDynVol", False)
    use_stop_distance_guard = bool_param("useStopDistanceGuard", False)
    max_stop_atr_mult = float_param("maxStopAtrMult", 2.8, enabled=use_stop_distance_guard)
    use_time_stop = bool_param("useTimeStop", False)
    max_hold_bars = int_param("maxHoldBars", 45, enabled=use_time_stop)
    min_hold_default = _coerce_float_value(params.get("minHoldBars"), 0.0)
    if not np.isfinite(min_hold_default):
        min_hold_default = 0.0
    min_hold_value = _resolve_requirement_value(
        "min_hold_bars",
        "minHoldBars",
        "minHold",
        default=min_hold_default,
    )
    try:
        min_hold_bars_param = max(0, int(float(min_hold_value)))
    except (TypeError, ValueError, OverflowError):
        min_hold_bars_param = max(0, int(min_hold_default))
    use_kasa = bool_param("useKASA", False)
    kasa_rsi_len = int_param("kasa_rsiLen", 14, enabled=use_kasa)
    kasa_rsi_ob = float_param("kasa_rsiOB", 72.0, enabled=use_kasa)
    kasa_rsi_os = float_param("kasa_rsiOS", 28.0, enabled=use_kasa)
    use_be_tiers = bool_param("useBETiers", False)

    use_shock = bool_param("useShock", False)
    atr_fast_len = int_param("atrFastLen", 5, enabled=use_shock)
    atr_slow_len = int_param("atrSlowLen", 20, enabled=use_shock)
    shock_mult = float_param("shockMult", 2.5, enabled=use_shock)
    shock_action = str_param("shockAction", "손절 타이트닝", enabled=use_shock)

    # =================================================================================
    # === 인디케이터 선계산 ===========================================================
    # =================================================================================

    tick_size = _estimate_tick(df["close"])
    slip_value = tick_size * slippage_ticks

    hl2 = (df["high"] + df["low"]) / 2.0
    bb_len_eff = osc_len if use_same_len else bb_len
    kc_len_eff = osc_len if use_same_len else kc_len

    bb_basis = _sma(hl2, bb_len_eff)
    highest = df["high"].rolling(osc_len, min_periods=osc_len).max()
    lowest = df["low"].rolling(osc_len, min_periods=osc_len).min()
    channel_mid = (highest + lowest) / 2.0
    avg_line = (bb_basis + channel_mid) / 2.0
    atr_primary = _atr(df, osc_len).replace(0.0, np.nan)
    norm = (df["close"] - avg_line) / atr_primary * 100.0
    momentum = _linreg(norm, osc_len)
    mom_signal = _sma(momentum, sig_len)

    flux_df = _heikin_ashi(df) if flux_use_ha else df
    flux_raw = _directional_flux(flux_df, flux_len)
    if flux_smooth_len > 1:
        flux_hist = flux_raw.rolling(flux_smooth_len, min_periods=flux_smooth_len).mean()
    else:
        flux_hist = flux_raw

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
    mom_fade_abs_prev = mom_fade_abs.shift().fillna(mom_fade_abs)
    mom_fade_abs_prev2 = mom_fade_abs.shift(2).fillna(mom_fade_abs_prev)

    gate_dev = mom_fade_dev
    gate_atr = mom_range
    gate_sq_on = (gate_dev < gate_atr).fillna(False).astype(bool)
    gate_sq_prev = gate_sq_on.shift(fill_value=False)
    gate_sq_rel = gate_sq_prev & np.logical_not(gate_sq_on)
    gate_rel_idx = gate_sq_rel.cumsum()
    gate_rel_idx = gate_rel_idx.where(gate_sq_rel, np.nan).ffill()
    gate_bars_since_release = (df.index.to_series().groupby(gate_rel_idx).cumcount()).fillna(np.inf)

    if use_dynamic_thresh:
        dyn_window = max(dyn_len, 1)
        dyn_series = momentum.rolling(dyn_window, min_periods=dyn_window).std() * dyn_mult
        fallback = abs(stat_threshold) if stat_threshold else dyn_series.dropna().mean()
        if not np.isfinite(fallback) or fallback == 0:
            fallback = 1.0
        dyn_series = dyn_series.fillna(fallback).abs()
        buy_thresh_series = -dyn_series
        sell_thresh_series = dyn_series
    else:
        if use_sym_threshold:
            buy_val = -abs(stat_threshold)
            sell_val = abs(stat_threshold)
        else:
            buy_val = -abs(buy_threshold)
            sell_val = abs(sell_threshold)
        buy_thresh_series = pd.Series(buy_val, index=df.index)
        sell_thresh_series = pd.Series(sell_val, index=df.index)

    atr_len_series = _atr(df, osc_len)

    # 보조 필터 선계산 --------------------------------------------------------------
    if use_adx or use_atr_diff:
        adx_df = _resample_ohlcv(df, adx_atr_tf) if adx_atr_tf not in {"", "0"} else df
        _, _, adx_raw = _dmi(adx_df, adx_len)
        adx_series = adx_raw.reindex(df.index, method="ffill").fillna(0.0)
        atr_htf = _atr(adx_df, adx_len)
        atr_diff = (atr_htf - _sma(atr_htf, adx_len)).reindex(df.index, method="ffill").fillna(0.0)
    else:
        adx_series = pd.Series(0.0, index=df.index)
        atr_diff = pd.Series(0.0, index=df.index)

    if use_ema:
        ema_fast = _ema(df["close"], ema_fast_len)
        ema_slow = _ema(df["close"], ema_slow_len)
    else:
        ema_fast = df["close"]
        ema_slow = df["close"]

    if use_bb_filter:
        bb_filter_basis = _sma(df["close"], bb_filter_len)
        bb_filter_dev = _std(df["close"], bb_filter_len)
        bb_filter_upper = bb_filter_basis + bb_filter_dev * bb_filter_mult
        bb_filter_lower = bb_filter_basis - bb_filter_dev * bb_filter_mult
    else:
        bb_filter_basis = df["close"]
        bb_filter_dev = pd.Series(0.0, index=df.index)
        bb_filter_upper = df["close"]
        bb_filter_lower = df["close"]

    stoch_rsi_val = _stoch_rsi(df["close"], stoch_len) if use_stoch_rsi else pd.Series(50.0, index=df.index)
    obv_slope = _obv_slope(df["close"], df["volume"], obv_smooth_len) if use_obv else pd.Series(0.0, index=df.index)

    if use_htf_trend:
        htf_ma = _security_series(df, htf_trend_tf, lambda data: _ema(data["close"], htf_ma_len))
        htf_trend_up = df["close"] > htf_ma
        htf_trend_down = df["close"] < htf_ma
    else:
        htf_ma = df["close"]
        htf_trend_up = pd.Series(True, index=df.index)
        htf_trend_down = pd.Series(True, index=df.index)

    hma_value = _ema(df["close"], hma_len) if use_hma_filter else df["close"]

    if use_range_filter:
        range_high = _security_series(df, range_tf, lambda data: data["high"].rolling(range_bars).max())
        range_low = _security_series(df, range_tf, lambda data: data["low"].rolling(range_bars).min())
        range_perc = (range_high - range_low) / range_low.replace(0.0, np.nan) * 100.0
        in_range_box = range_perc <= range_percent
    else:
        range_high = df["high"]
        range_low = df["low"]
        in_range_box = pd.Series(False, index=df.index)

    event_mask = pd.Series(False, index=df.index)
    if use_event_filter:
        windows_raw = str_param("eventWindows", "")
        for segment in windows_raw.split(","):
            if "/" not in segment:
                continue
            start_str, end_str = segment.split("/", 1)
            try:
                start = pd.to_datetime(start_str.strip(), utc=True)
                end = pd.to_datetime(end_str.strip(), utc=True)
            except ValueError:
                continue
            if pd.isna(start) or pd.isna(end):
                continue
            if end < start:
                start, end = end, start
            event_mask |= (df.index >= start) & (df.index <= end)

    if use_slope_filter:
        slope_basis = _ema(df["close"], slope_lookback)
        slope_prev = slope_basis.shift(slope_lookback).fillna(slope_basis)
        slope_pct = np.where(slope_basis != 0, (slope_basis - slope_prev) / slope_basis * 100.0, 0.0)
        slope_pct = pd.Series(slope_pct, index=df.index)
        slope_ok_long = slope_pct >= slope_min_pct
        slope_ok_short = slope_pct <= -slope_min_pct
    else:
        slope_ok_long = pd.Series(True, index=df.index)
        slope_ok_short = pd.Series(True, index=df.index)

    if use_distance_guard:
        distance_atr = _atr(df, distance_atr_len)
        vwap = df["close"].rolling(distance_atr_len, min_periods=1).mean()
        vw_distance = (df["close"] - vwap).abs() / distance_atr.replace(0.0, np.nan)
        trend_ma = _ema(df["close"], distance_trend_len)
        trend_distance = (df["close"] - trend_ma).abs() / distance_atr.replace(0.0, np.nan)
        distance_ok = (vw_distance <= distance_max_atr) & (trend_distance <= distance_max_atr)
    else:
        distance_ok = pd.Series(True, index=df.index)

    kasa_rsi = _rsi(df["close"], kasa_rsi_len) if use_kasa else pd.Series(50.0, index=df.index)

    if use_regime_filter:
        ctx_close = _security_series(df, ctx_htf_tf, lambda data: data["close"])
        ctx_ema = _security_series(df, ctx_htf_tf, lambda data: _ema(data["close"], ctx_htf_ema_len))
        ctx_adx = _security_series(df, ctx_htf_tf, lambda data: _dmi(data, ctx_htf_adx_len)[2])
        regime_long_ok = (ctx_close > ctx_ema) & (ctx_adx > ctx_htf_adx_th)
        regime_short_ok = (ctx_close < ctx_ema) & (ctx_adx > ctx_htf_adx_th)
    else:
        regime_long_ok = pd.Series(True, index=df.index)
        regime_short_ok = pd.Series(True, index=df.index)

    # 구조 게이트 ------------------------------------------------------------
    if use_structure_gate:
        bos_high = _security_series(
            df,
            bos_tf,
            lambda data: data["high"].rolling(bos_lookback, min_periods=bos_lookback).max(),
        )
        bos_low = _security_series(
            df,
            bos_tf,
            lambda data: data["low"].rolling(bos_lookback, min_periods=bos_lookback).min(),
        )
        bos_high_ref = bos_high.shift()
        bos_low_ref = bos_low.shift()
        if use_bos:
            bos_long_event = (df["close"] > bos_high_ref).where(~bos_high_ref.isna(), True)
            bos_short_event = (df["close"] < bos_low_ref).where(~bos_low_ref.isna(), True)
            bos_long_state = (
                bos_long_event.rolling(bos_state_bars, min_periods=1).max().fillna(True).astype(bool)
            )
            bos_short_state = (
                bos_short_event.rolling(bos_state_bars, min_periods=1).max().fillna(True).astype(bool)
            )
        else:
            bos_long_state = pd.Series(True, index=df.index)
            bos_short_state = pd.Series(True, index=df.index)

        pivot_high_ctx = _security_series(
            df,
            bos_tf,
            lambda data: _pivot_series(data["high"], pivot_left, pivot_right, True),
        )
        pivot_low_ctx = _security_series(
            df,
            bos_tf,
            lambda data: _pivot_series(data["low"], pivot_left, pivot_right, False),
        )
        if use_choch:
            choch_long_event = (df["close"] > pivot_high_ctx).where(~pivot_high_ctx.isna(), True)
            choch_short_event = (df["close"] < pivot_low_ctx).where(~pivot_low_ctx.isna(), True)
            choch_long_state = (
                choch_long_event.rolling(choch_state_bars, min_periods=1).max().fillna(True).astype(bool)
            )
            choch_short_state = (
                choch_short_event.rolling(choch_state_bars, min_periods=1).max().fillna(True).astype(bool)
            )
        else:
            choch_long_state = pd.Series(True, index=df.index)
            choch_short_state = pd.Series(True, index=df.index)
    else:
        bos_long_state = pd.Series(True, index=df.index)
        bos_short_state = pd.Series(True, index=df.index)
        choch_long_state = pd.Series(True, index=df.index)
        choch_short_state = pd.Series(True, index=df.index)

    # 스탑 계산용 시리즈 -------------------------------------------------------
    atr_trail_series = (
        _atr(df, atr_trail_len)
        if (use_atr_trail or use_breakeven_stop or use_be_tiers or use_dyn_vol or use_atr_profit)
        else pd.Series(np.nan, index=df.index)
    )
    pivot_low_series = (
        _pivot_series(df["low"], pivot_len, pivot_len, False) if use_stop_loss else pd.Series(np.nan, index=df.index)
    )
    pivot_high_series = (
        _pivot_series(df["high"], pivot_len, pivot_len, True) if use_stop_loss else pd.Series(np.nan, index=df.index)
    )
    swing_low_series = (
        df["low"].rolling(stop_lookback).min() if use_stop_loss else pd.Series(np.nan, index=df.index)
    )
    swing_high_series = (
        df["high"].rolling(stop_lookback).max() if use_stop_loss else pd.Series(np.nan, index=df.index)
    )
    if use_pivot_stop and use_pivot_htf:
        pivot_low_htf = _security_series(
            df, pivot_tf, lambda data: _pivot_series(data["low"], pivot_len, pivot_len, False)
        )
        pivot_high_htf = _security_series(
            df, pivot_tf, lambda data: _pivot_series(data["high"], pivot_len, pivot_len, True)
        )
    else:
        pivot_low_htf = pd.Series(np.nan, index=df.index)
        pivot_high_htf = pd.Series(np.nan, index=df.index)

    if use_shock:
        atr_fast = _atr(df, atr_fast_len)
        atr_slow = _sma(atr_fast, atr_slow_len)
        shock_series = atr_fast > atr_slow * shock_mult
    else:
        shock_series = pd.Series(False, index=df.index)

    if use_dyn_vol:
        atr_ratio = atr_trail_series / df["close"]
        bb_dev20 = _std(df["close"], 20) * 2.0
        bb_width = (bb_dev20 * 2.0) / df["close"]
        ma50 = _sma(df["close"], 50)
        ma_dist = (df["close"] - ma50).abs() / df["close"]
        dyn_metric = (atr_ratio.fillna(0.0) + bb_width.fillna(0.0) + ma_dist.fillna(0.0)) / 3.0
        dyn_factor_series = dyn_metric + 1.0
        dyn_factor_series = dyn_factor_series.clip(lower=0.5, upper=3.0)
    else:
        dyn_factor_series = pd.Series(1.0, index=df.index)

    # =================================================================================
    # === 상태 초기화 =================================================================
    # =================================================================================

    state = EquityState(
        initial_capital=initial_capital,
        equity=initial_capital,
        tradable_capital=initial_capital,
        peak_equity=initial_capital,
        daily_start_capital=initial_capital,
        daily_peak_capital=initial_capital,
        week_start_equity=initial_capital,
        week_peak_equity=initial_capital,
    )
    position = Position()
    trades: List[Trade] = []

    recent_trade_results: List[float] = []
    guard_frozen = False
    guard_fired_total = 0
    loss_streak = 0
    daily_losses = 0
    reentry_countdown = 0
    reversal_countdown = 0
    last_position_dir = 0
    highest_since_entry = np.nan
    lowest_since_entry = np.nan
    pos_bars = 0

    equity_trace: List[float] = [state.equity]
    returns_series = pd.Series(0.0, index=df.index)

    def record_trade_profit(pnl: float) -> None:
        nonlocal loss_streak, daily_losses
        prev_equity = state.equity
        state.equity += pnl
        state.net_profit += pnl
        state.peak_equity = max(state.peak_equity, state.equity)
        equity_trace.append(state.equity)
        if pnl < 0:
            loss_streak += 1
            daily_losses += 1
        elif pnl > 0:
            loss_streak = 0

    def calc_order_size(close_price: float, stop_distance: float, risk_mult: float) -> float:
        if close_price <= 0:
            return 0.0
        effective_scale = base_risk_pct
        if use_drawdown_scaling and state.peak_equity > 0:
            dd = (state.peak_equity - state.equity) / state.peak_equity * 100.0
            if dd > drawdown_trigger_pct:
                effective_scale *= drawdown_risk_scale
        if use_perf_adaptive_risk and recent_trade_results:
            wins = sum(1 for x in recent_trade_results if x > 0)
            win_rate = wins / len(recent_trade_results) * 100.0
            if len(recent_trade_results) >= par_min_trades:
                if win_rate >= par_hot_win_rate:
                    effective_scale *= par_hot_mult
                elif win_rate <= par_cold_win_rate:
                    effective_scale *= par_cold_mult
        mult = max(risk_mult, 0.0)
        if not use_sizing_override:
            pct_to_use = max(base_qty_percent * mult * (effective_scale / base_risk_pct if base_risk_pct > 0 else 1.0), 0.0)
            capital_portion = state.tradable_capital * pct_to_use / 100.0
            return (capital_portion * leverage) / close_price
        if sizing_mode == "자본 비율":
            pct_to_use = max(advanced_percent * mult, 0.0)
            capital_portion = state.tradable_capital * pct_to_use / 100.0
            return (capital_portion * leverage) / close_price
        if sizing_mode == "고정 금액 (USD)":
            usd_to_use = max(fixed_usd_amount * mult, 0.0)
            return (usd_to_use * leverage) / close_price
        if sizing_mode == "고정 계약":
            return max(fixed_contract_size * mult, 0.0)
        if sizing_mode == "리스크 기반":
            if risk_sizing_type == "고정 계약":
                return max(risk_contract_size * mult, 0.0)
            if stop_distance <= 0 or np.isnan(stop_distance):
                return 0.0
            risk_pct = max(effective_scale * mult, 0.0)
            risk_capital = state.tradable_capital * risk_pct / 100.0
            return risk_capital / (stop_distance + slip_value) if risk_capital > 0 else 0.0
        return 0.0

    def close_position(ts: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal position, highest_since_entry, lowest_since_entry, pos_bars, guard_frozen, guard_fired_total, last_position_dir
        if position.direction == 0 or position.entry_time is None:
            return
        qty = position.qty
        direction = position.direction
        exit_price = price - slip_value if direction > 0 else price + slip_value
        pnl = (exit_price - position.avg_price) * direction * qty
        fees_paid = (position.avg_price + exit_price) * qty * commission_pct
        pnl -= fees_paid
        record_trade_profit(pnl)
        returns_series.loc[ts] += pnl / state.initial_capital if state.initial_capital else 0.0
        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=ts,
                direction="long" if direction > 0 else "short",
                size=qty,
                entry_price=position.avg_price,
                exit_price=exit_price,
                profit=pnl,
                return_pct=pnl / state.initial_capital if state.initial_capital else 0.0,
                mfe=np.nan,
                mae=np.nan,
                bars_held=position.bars_held,
                reason=reason,
            )
        )
        last_position_dir = direction
        position = Position()
        highest_since_entry = np.nan
        lowest_since_entry = np.nan
        pos_bars = 0
        if use_perf_adaptive_risk:
            recent_trade_results.append(pnl)
            if len(recent_trade_results) > par_lookback:
                recent_trade_results.pop(0)
        reentry_countdown = reentry_bars

    def bars_since(series: pd.Series, idx: int, condition: callable) -> int:
        count = 0
        for lookback in range(idx, -1, -1):
            if condition(series.iloc[lookback]):
                return count
            count += 1
        return int(1e9)

    prev_guard_state = guard_frozen

    for idx, ts in enumerate(df.index):
        if ts < start_ts:
            continue

        row = df.iloc[idx]

        if idx > 0:
            prev_day = df.index[idx - 1].date()
            if ts.date() != prev_day:
                state.daily_start_capital = state.tradable_capital
                state.daily_peak_capital = state.tradable_capital
                daily_losses = 0
                guard_frozen = False
            prev_week = df.index[idx - 1].isocalendar()[1]
            if ts.isocalendar()[1] != prev_week:
                state.week_start_equity = state.equity
                state.week_peak_equity = state.equity

        if use_wallet and state.net_profit > 0:
            state.withdrawable += state.net_profit * profit_reserve_pct
        effective_equity = state.equity - state.withdrawable if (use_wallet and apply_reserve_to_sizing) else state.equity
        state.tradable_capital = max(effective_equity, state.initial_capital * 0.01)
        state.peak_equity = max(state.peak_equity, state.equity)
        state.daily_peak_capital = max(state.daily_peak_capital, state.tradable_capital)
        state.week_peak_equity = max(state.week_peak_equity, state.equity)

        daily_pnl = state.tradable_capital - state.daily_start_capital
        weekly_pnl = state.equity - state.week_start_equity
        weekly_dd = (
            (state.week_peak_equity - state.equity) / state.week_peak_equity * 100.0
            if state.week_peak_equity > 0
            else 0.0
        )

        daily_loss_breached = use_daily_loss_guard and daily_pnl <= -abs(daily_loss_limit)
        daily_profit_reached = use_daily_profit_lock and daily_pnl >= abs(daily_profit_target)
        weekly_profit_reached = use_weekly_profit_lock and weekly_pnl >= abs(weekly_profit_target)
        loss_streak_breached = use_loss_streak_guard and loss_streak >= max_consecutive_losses
        capital_breached = use_capital_guard and state.equity <= state.initial_capital * (1 - capital_guard_pct / 100.0)
        weekly_dd_breached = max_weekly_dd > 0 and weekly_dd >= max_weekly_dd
        loss_count_breached = max_daily_losses > 0 and daily_losses >= max_daily_losses
        guard_fire_limit = max_guard_fires > 0 and guard_fired_total >= max_guard_fires

        atr_pct_val = 0.0
        if use_volatility_guard:
            atr_window = max(volatility_lookback, 1)
            if idx >= atr_window:
                atr_pct_val = _atr(df.iloc[idx - atr_window + 1 : idx + 1], atr_window).iloc[-1] / row["close"] * 100.0
            else:
                atr_pct_val = 0.0
        is_vol_ok = (not use_volatility_guard) or (
            volatility_lower_pct <= atr_pct_val <= volatility_upper_pct
        )

        performance_pause = False
        if use_perf_adaptive_risk and recent_trade_results:
            wins = sum(1 for x in recent_trade_results if x > 0)
            win_rate = wins / len(recent_trade_results) * 100.0
            if len(recent_trade_results) >= par_min_trades and win_rate <= par_cold_win_rate and par_pause_on_cold:
                performance_pause = True

        should_freeze = (
            daily_loss_breached
            or daily_profit_reached
            or weekly_profit_reached
            or loss_streak_breached
            or capital_breached
            or weekly_dd_breached
            or loss_count_breached
            or guard_fire_limit
            or performance_pause
            or state.tradable_capital < min_tradable_capital
        )
        if should_freeze:
            guard_frozen = True

        guard_activated = guard_frozen and not prev_guard_state
        prev_guard_state = guard_frozen

        if guard_activated and position.direction != 0:
            close_position(ts, row["close"], "Guard Halt")
            guard_fired_total += 1

        if use_guard_exit and position.direction != 0 and not guard_activated:
            qty = abs(position.qty)
            if qty > 0:
                entry_price = position.avg_price
                initial_margin = (qty * entry_price) / leverage
                maint_margin = (qty * entry_price) * (maintenance_margin_pct / 100.0)
                offset = (initial_margin - maint_margin) / qty if qty > 0 else 0.0
                liq_price = entry_price - offset if position.direction > 0 else entry_price + offset
                if liq_buffer_pct > 0:
                    buffer = entry_price * (liq_buffer_pct / 100.0)
                    if position.direction > 0:
                        liq_price -= buffer
                    else:
                        liq_price += buffer
                preempt_price = liq_price + preempt_ticks * tick_size if position.direction > 0 else liq_price - preempt_ticks * tick_size
                hit_guard = row["low"] <= preempt_price if position.direction > 0 else row["high"] >= preempt_price
                if hit_guard:
                    close_position(ts, row["close"], "Guard Exit")
                    guard_frozen = True
                    guard_fired_total += 1

        can_trade = (not guard_frozen) and is_vol_ok

        if reentry_countdown > 0 and position.direction == 0:
            reentry_countdown -= 1
        if reversal_countdown > 0 and position.direction == 0:
            reversal_countdown -= 1

        if position.direction != 0:
            pos_bars += 1
            if position.direction > 0:
                highest_since_entry = row["high"] if np.isnan(highest_since_entry) else max(highest_since_entry, row["high"])
                lowest_since_entry = row["low"] if np.isnan(lowest_since_entry) else min(lowest_since_entry, row["low"])
            else:
                lowest_since_entry = row["low"] if np.isnan(lowest_since_entry) else min(lowest_since_entry, row["low"])
                highest_since_entry = row["high"] if np.isnan(highest_since_entry) else max(highest_since_entry, row["high"])
            position.bars_held += 1

        prev_idx = max(idx - 1, 0)
        prev_momentum = momentum.iloc[prev_idx]
        prev_signal = mom_signal.iloc[prev_idx]
        mom_val = momentum.iloc[idx]
        sig_val = mom_signal.iloc[idx]
        flux_val = flux_hist.iloc[idx]
        buy_thresh_val = buy_thresh_series.iloc[idx]
        sell_thresh_val = sell_thresh_series.iloc[idx]

        cross_up = _cross_over(prev_momentum, prev_signal, mom_val, sig_val)
        cross_down = _cross_under(prev_momentum, prev_signal, mom_val, sig_val)

        long_cross_ok = cross_up or not require_momentum_cross
        short_cross_ok = cross_down or not require_momentum_cross
        base_long_trigger = long_cross_ok and mom_val < buy_thresh_val and flux_val > 0
        base_short_trigger = short_cross_ok and mom_val > sell_thresh_val and flux_val < 0
        base_long_signal = debug_force_long or base_long_trigger
        base_short_signal = debug_force_short or base_short_trigger

        long_ok = True
        short_ok = True

        if use_adx:
            long_ok &= adx_series.iloc[idx] > adx_thresh
            short_ok &= adx_series.iloc[idx] > adx_thresh
        if use_ema:
            if ema_mode == "Crossover":
                long_ok &= ema_fast.iloc[idx] > ema_slow.iloc[idx]
                short_ok &= ema_fast.iloc[idx] < ema_slow.iloc[idx]
            else:
                long_ok &= row["close"] > ema_slow.iloc[idx]
                short_ok &= row["close"] < ema_slow.iloc[idx]
        if use_bb_filter:
            long_ok &= (row["close"] <= bb_filter_basis.iloc[idx]) or (row["close"] < bb_filter_lower.iloc[idx])
            short_ok &= (row["close"] >= bb_filter_basis.iloc[idx]) or (row["close"] > bb_filter_upper.iloc[idx])
        if use_stoch_rsi:
            long_ok &= stoch_rsi_val.iloc[idx] <= stoch_os
            short_ok &= stoch_rsi_val.iloc[idx] >= stoch_ob
        if use_obv:
            long_ok &= obv_slope.iloc[idx] > 0
            short_ok &= obv_slope.iloc[idx] < 0
        if use_atr_diff:
            long_ok &= atr_diff.iloc[idx] > 0
            short_ok &= atr_diff.iloc[idx] > 0
        if use_htf_trend:
            long_ok &= bool(htf_trend_up.iloc[idx])
            short_ok &= bool(htf_trend_down.iloc[idx])
        if use_hma_filter:
            long_ok &= row["close"] > hma_value.iloc[idx]
            short_ok &= row["close"] < hma_value.iloc[idx]
        if use_range_filter:
            long_ok &= not bool(in_range_box.iloc[idx])
            short_ok &= not bool(in_range_box.iloc[idx])
        if use_event_filter:
            long_ok &= not bool(event_mask.iloc[idx])
            short_ok &= not bool(event_mask.iloc[idx])
        if use_slope_filter:
            long_ok &= bool(slope_ok_long.iloc[idx])
            short_ok &= bool(slope_ok_short.iloc[idx])
        if use_distance_guard:
            long_ok &= bool(distance_ok.iloc[idx])
            short_ok &= bool(distance_ok.iloc[idx])
        if use_equity_slope_filter and len(equity_trace) >= eq_slope_len:
            equity_window = pd.Series(equity_trace[-eq_slope_len:])
            eq_slope = _linreg(equity_window, min(eq_slope_len, len(equity_window))).iloc[-1]
            long_ok &= eq_slope >= 0
            short_ok &= eq_slope <= 0
        long_ok &= bool(regime_long_ok.iloc[idx])
        short_ok &= bool(regime_short_ok.iloc[idx])

        structure_require_all = structure_gate_mode == "모두 충족"
        structure_long_pass = True
        structure_short_pass = True
        if use_structure_gate:
            if structure_require_all:
                structure_long_pass = (not use_bos or bool(bos_long_state.iloc[idx])) and (
                    not use_choch or bool(choch_long_state.iloc[idx])
                )
                structure_short_pass = (not use_bos or bool(bos_short_state.iloc[idx])) and (
                    not use_choch or bool(choch_short_state.iloc[idx])
                )
            else:
                structure_long_pass = (
                    (use_bos and bool(bos_long_state.iloc[idx]))
                    or (use_choch and bool(choch_long_state.iloc[idx]))
                    or (not use_bos and not use_choch)
                )
                structure_short_pass = (
                    (use_bos and bool(bos_short_state.iloc[idx]))
                    or (use_choch and bool(choch_short_state.iloc[idx]))
                    or (not use_bos and not use_choch)
                )
            long_ok &= structure_long_pass
            short_ok &= structure_short_pass

        gate_release_seen = gate_bars_since_release.iloc[idx] != np.inf
        gate_sq_valid = gate_release_seen and gate_bars_since_release.iloc[idx] <= sqz_release_bars and not gate_sq_on.iloc[idx]
        if use_sqz_gate:
            long_ok &= gate_sq_valid
            short_ok &= gate_sq_valid

        if use_structure_gate:
            base_long_signal = base_long_signal and structure_long_pass
            base_short_signal = base_short_signal and structure_short_pass

        enter_long = (
            allow_long_entry
            and can_trade
            and base_long_signal
            and long_ok
            and position.direction == 0
            and reentry_countdown == 0
        )
        enter_short = (
            allow_short_entry
            and can_trade
            and base_short_signal
            and short_ok
            and position.direction == 0
            and reentry_countdown == 0
        )

        if use_reversal and reversal_countdown == 0 and position.direction == 0 and last_position_dir != 0 and can_trade:
            if last_position_dir == 1:
                enter_short = True
            elif last_position_dir == -1:
                enter_long = True
            last_position_dir = 0

        exit_long = False
        exit_short = False
        exit_long_reason: Optional[str] = None
        exit_short_reason: Optional[str] = None

        if position.direction > 0:
            if exit_opposite and base_short_signal and position.bars_held >= min_hold_bars_param:
                exit_long = True
                exit_long_reason = exit_long_reason or "opposite_signal"
            fade_abs_falling = mom_fade_abs.iloc[idx] < mom_fade_abs_prev.iloc[idx] if mom_fade_bars <= 1 else mom_fade_abs.iloc[idx] <= mom_fade_abs_prev.iloc[idx]
            fade_abs_two = (not mom_fade_require_two) or (
                mom_fade_abs.iloc[idx] <= mom_fade_abs_prev.iloc[idx]
                and mom_fade_abs_prev.iloc[idx] <= mom_fade_abs_prev2.iloc[idx]
            )
            fade_delay_long = mom_fade_zero_delay <= 0 or bars_since(mom_fade_hist, idx, lambda v: v <= 0) > mom_fade_zero_delay
            fade_min_abs_ok = mom_fade_min_abs <= 0 or mom_fade_abs.iloc[idx] >= mom_fade_min_abs
            fade_release_ok = (not mom_fade_release_only) or gate_sq_valid
            if (
                use_mom_fade
                and mom_fade_hist.iloc[idx] > 0
                and fade_abs_falling
                and fade_abs_two
                and fade_delay_long
                and fade_min_abs_ok
                and fade_release_ok
                and position.bars_held >= mom_fade_bars
            ):
                exit_long = True
                exit_long_reason = exit_long_reason or "mom_fade"
            if use_time_stop and max_hold_bars > 0 and position.bars_held >= max_hold_bars:
                exit_long = True
                exit_long_reason = exit_long_reason or "time_stop"
            if use_kasa and kasa_rsi.iloc[idx] < kasa_rsi_ob and kasa_rsi.iloc[max(idx - 1, 0)] >= kasa_rsi_ob:
                exit_long = True
                exit_long_reason = exit_long_reason or "kasa_exit"
        elif position.direction < 0:
            if exit_opposite and base_long_signal and position.bars_held >= min_hold_bars_param:
                exit_short = True
                exit_short_reason = exit_short_reason or "opposite_signal"
            fade_abs_falling = mom_fade_abs.iloc[idx] < mom_fade_abs_prev.iloc[idx] if mom_fade_bars <= 1 else mom_fade_abs.iloc[idx] <= mom_fade_abs_prev.iloc[idx]
            fade_abs_two = (not mom_fade_require_two) or (
                mom_fade_abs.iloc[idx] <= mom_fade_abs_prev.iloc[idx]
                and mom_fade_abs_prev.iloc[idx] <= mom_fade_abs_prev2.iloc[idx]
            )
            fade_delay_short = mom_fade_zero_delay <= 0 or bars_since(mom_fade_hist, idx, lambda v: v >= 0) > mom_fade_zero_delay
            fade_min_abs_ok = mom_fade_min_abs <= 0 or mom_fade_abs.iloc[idx] >= mom_fade_min_abs
            fade_release_ok = (not mom_fade_release_only) or gate_sq_valid
            if (
                use_mom_fade
                and mom_fade_hist.iloc[idx] < 0
                and fade_abs_falling
                and fade_abs_two
                and fade_delay_short
                and fade_min_abs_ok
                and fade_release_ok
                and position.bars_held >= mom_fade_bars
            ):
                exit_short = True
                exit_short_reason = exit_short_reason or "mom_fade"
            if use_time_stop and max_hold_bars > 0 and position.bars_held >= max_hold_bars:
                exit_short = True
                exit_short_reason = exit_short_reason or "time_stop"
            if use_kasa and kasa_rsi.iloc[idx] > kasa_rsi_os and kasa_rsi.iloc[max(idx - 1, 0)] <= kasa_rsi_os:
                exit_short = True
                exit_short_reason = exit_short_reason or "kasa_exit"

        is_shock = use_shock and bool(shock_series.iloc[idx])
        if position.direction > 0 and is_shock and shock_action == "즉시 청산":
            close_position(ts, row["close"], "Volatility Shock")
            continue
        if position.direction < 0 and is_shock and shock_action == "즉시 청산":
            close_position(ts, row["close"], "Volatility Shock")
            continue

        if position.direction > 0 and (exit_long or (is_shock and shock_action == "손절 타이트닝")):
            if exit_long:
                close_position(ts, row["close"], exit_long_reason or "Exit Long")
                continue
        if position.direction < 0 and (exit_short or (is_shock and shock_action == "손절 타이트닝")):
            if exit_short:
                close_position(ts, row["close"], exit_short_reason or "Exit Short")
                continue

        if position.direction > 0:
            stop_long = np.nan
            if use_atr_trail and not np.isnan(atr_trail_series.iloc[idx]):
                stop_long = row["close"] - atr_trail_series.iloc[idx] * atr_trail_mult * dyn_factor_series.iloc[idx]
            if use_stop_loss:
                swing_low = swing_low_series.iloc[idx]
                stop_long = _max_ignore_nan(stop_long, swing_low)
                if use_pivot_stop:
                    pivot_ref = pivot_low_htf.iloc[idx] if use_pivot_htf else pivot_low_series.iloc[idx]
                    stop_long = _max_ignore_nan(stop_long, pivot_ref)
            if use_breakeven_stop and not np.isnan(highest_since_entry) and not np.isnan(atr_trail_series.iloc[idx]):
                move = highest_since_entry - position.avg_price
                trigger = atr_trail_series.iloc[idx] * breakeven_mult * dyn_factor_series.iloc[idx]
                if move >= trigger:
                    stop_long = _max_ignore_nan(stop_long, position.avg_price)
            if use_be_tiers and not np.isnan(highest_since_entry):
                atr_seed = atr_len_series.iloc[idx]
                if atr_seed > 0 and (highest_since_entry - position.avg_price) >= atr_seed:
                    stop_long = _max_ignore_nan(stop_long, position.avg_price)
            if not np.isnan(stop_long) and row["low"] <= stop_long:
                close_position(ts, stop_long, "Stop Long")
                continue
            if use_atr_profit and not np.isnan(atr_trail_series.iloc[idx]):
                target = position.avg_price + atr_trail_series.iloc[idx] * atr_profit_mult * dyn_factor_series.iloc[idx]
                if row["high"] >= target:
                    close_position(ts, target, "ATR Profit Long")
                    continue
        elif position.direction < 0:
            stop_short = np.nan
            if use_atr_trail and not np.isnan(atr_trail_series.iloc[idx]):
                stop_short = row["close"] + atr_trail_series.iloc[idx] * atr_trail_mult * dyn_factor_series.iloc[idx]
            if use_stop_loss:
                swing_high = swing_high_series.iloc[idx]
                stop_short = _min_ignore_nan(stop_short, swing_high)
                if use_pivot_stop:
                    pivot_ref = pivot_high_htf.iloc[idx] if use_pivot_htf else pivot_high_series.iloc[idx]
                    stop_short = _min_ignore_nan(stop_short, pivot_ref)
            if use_breakeven_stop and not np.isnan(lowest_since_entry) and not np.isnan(atr_trail_series.iloc[idx]):
                move = position.avg_price - lowest_since_entry
                trigger = atr_trail_series.iloc[idx] * breakeven_mult * dyn_factor_series.iloc[idx]
                if move >= trigger:
                    stop_short = _min_ignore_nan(stop_short, position.avg_price)
            if use_be_tiers and not np.isnan(lowest_since_entry):
                atr_seed = atr_len_series.iloc[idx]
                if atr_seed > 0 and (position.avg_price - lowest_since_entry) >= atr_seed:
                    stop_short = _min_ignore_nan(stop_short, position.avg_price)
            if not np.isnan(stop_short) and row["high"] >= stop_short:
                close_position(ts, stop_short, "Stop Short")
                continue
            if use_atr_profit and not np.isnan(atr_trail_series.iloc[idx]):
                target = position.avg_price - atr_trail_series.iloc[idx] * atr_profit_mult * dyn_factor_series.iloc[idx]
                if row["low"] <= target:
                    close_position(ts, target, "ATR Profit Short")
                    continue

        if position.direction == 0:
            if enter_long:
                stop_hint = atr_len_series.iloc[idx]
                if use_stop_loss:
                    swing_low = swing_low_series.iloc[idx]
                    if not np.isnan(swing_low):
                        stop_hint = max(stop_hint, row["close"] - swing_low) if not np.isnan(stop_hint) else row["close"] - swing_low
                    if use_pivot_stop:
                        pivot_ref = pivot_low_htf.iloc[idx] if use_pivot_htf else pivot_low_series.iloc[idx]
                        if not np.isnan(pivot_ref):
                            dist_pivot = row["close"] - pivot_ref
                            stop_hint = max(stop_hint, dist_pivot) if not np.isnan(stop_hint) else dist_pivot
                if use_atr_trail and not np.isnan(atr_trail_series.iloc[idx]):
                    atr_dist = atr_trail_series.iloc[idx] * atr_trail_mult
                    stop_hint = max(stop_hint, atr_dist) if not np.isnan(stop_hint) else atr_dist
                if np.isnan(stop_hint) or stop_hint <= 0:
                    stop_hint = tick_size
                stop_for_size = max(stop_hint, tick_size)
                guard_ok = (
                    (not use_stop_distance_guard)
                    or np.isnan(atr_len_series.iloc[idx])
                    or stop_for_size <= atr_len_series.iloc[idx] * max_stop_atr_mult
                )
                qty = calc_order_size(row["close"], stop_for_size, 1.0)
                if guard_ok and qty > 0:
                    position = Position(direction=1, qty=qty, avg_price=row["close"], entry_time=ts)
                    highest_since_entry = row["high"]
                    lowest_since_entry = row["low"]
                    pos_bars = 0
                    reversal_countdown = int(reversal_delay_sec // 60) if reversal_delay_sec > 0 else 0
            elif enter_short:
                stop_hint = atr_len_series.iloc[idx]
                if use_stop_loss:
                    swing_high = swing_high_series.iloc[idx]
                    if not np.isnan(swing_high):
                        stop_hint = max(stop_hint, swing_high - row["close"]) if not np.isnan(stop_hint) else swing_high - row["close"]
                    if use_pivot_stop:
                        pivot_ref = pivot_high_htf.iloc[idx] if use_pivot_htf else pivot_high_series.iloc[idx]
                        if not np.isnan(pivot_ref):
                            dist_pivot = pivot_ref - row["close"]
                            stop_hint = max(stop_hint, dist_pivot) if not np.isnan(stop_hint) else dist_pivot
                if use_atr_trail and not np.isnan(atr_trail_series.iloc[idx]):
                    atr_dist = atr_trail_series.iloc[idx] * atr_trail_mult
                    stop_hint = max(stop_hint, atr_dist) if not np.isnan(stop_hint) else atr_dist
                if np.isnan(stop_hint) or stop_hint <= 0:
                    stop_hint = tick_size
                stop_for_size = max(stop_hint, tick_size)
                guard_ok = (
                    (not use_stop_distance_guard)
                    or np.isnan(atr_len_series.iloc[idx])
                    or stop_for_size <= atr_len_series.iloc[idx] * max_stop_atr_mult
                )
                qty = calc_order_size(row["close"], stop_for_size, 1.0)
                if guard_ok and qty > 0:
                    position = Position(direction=-1, qty=qty, avg_price=row["close"], entry_time=ts)
                    highest_since_entry = row["high"]
                    lowest_since_entry = row["low"]
                    pos_bars = 0
                    reversal_countdown = int(reversal_delay_sec // 60) if reversal_delay_sec > 0 else 0

    if position.direction != 0 and position.entry_time is not None:
        close_position(df.index[-1], df.iloc[-1]["close"], "EndOfData")

    metrics = aggregate_metrics(trades, returns_series, simple=simple_metrics_only)
    if simple_metrics_only:
        metrics["SimpleMetricsOnly"] = True
    metrics["FinalEquity"] = state.equity
    metrics["NetProfitAbs"] = state.net_profit
    metrics["GuardFrozen"] = float(guard_frozen)
    metrics["TradesList"] = trades
    metrics["Returns"] = returns_series
    metrics["Withdrawable"] = state.withdrawable
    _apply_penalty_settings(
        metrics,
        min_trades_value=min_trades_req,
        min_hold_value=min_hold_bars_param,
        max_loss_streak=max_consecutive_losses,
    )
    metrics["Valid"] = (
        metrics.get("Trades", 0.0) >= min_trades_req
        and metrics.get("AvgHoldBars", 0.0) >= min_hold_bars_param
        and metrics.get("MaxConsecutiveLosses", 0.0) <= max_consecutive_losses
    )
    return metrics







