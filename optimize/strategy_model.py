"""PPP Vishva Algo 경량 프로파일용 파이썬 백테스트 엔진."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .metrics import Trade, aggregate_metrics

LOGGER = logging.getLogger(__name__)


# =====================================================================================
# === 보조 계산 함수 =================================================================
# =====================================================================================


def _ensure_series(values: Iterable[float], index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index, dtype=float)


def _ema(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.rolling(length, min_periods=length).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _rma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    return _rma(_true_range(df), length)


def _rsi(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    diff = series.diff()
    up = diff.clip(lower=0.0)
    down = -diff.clip(upper=0.0)
    avg_gain = _rma(up, length)
    avg_loss = _rma(down, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)


def _stoch(values: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    lowest = values.rolling(length, min_periods=length).min()
    highest = values.rolling(length, min_periods=length).max()
    denom = (highest - lowest).replace(0.0, np.nan)
    return ((values - lowest) / denom * 100.0).fillna(50.0)


def _estimate_tick(series: pd.Series) -> float:
    diffs = series.diff().abs()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        if len(series):
            return float(series.iloc[-1]) * 1e-6
        return 0.01
    return float(diffs.min())


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


def _normalise_ohlcv(df: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = df.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "timestamp" not in frame.columns:
            raise TypeError(f"{label} 데이터프레임은 DatetimeIndex 가 필요합니다.")
        converted = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        valid_mask = converted.notna()
        dropped = (~valid_mask).sum()
        if dropped:
            LOGGER.warning("%s 데이터에서 잘못된 timestamp %d개를 제거했습니다.", label, int(dropped))
        frame = frame.loc[valid_mask].copy()
        frame.index = converted[valid_mask]
        frame.drop(columns=["timestamp"], inplace=True)
    if frame.index.tz is None:
        frame = frame.copy()
        frame.index = frame.index.tz_localize("UTC")

    frame.sort_index(inplace=True)
    if frame.index.has_duplicates:
        dup = int(frame.index.duplicated(keep="last").sum())
        if dup:
            LOGGER.warning("%s 데이터에서 중복 인덱스 %d개를 제거했습니다.", label, dup)
        frame = frame[~frame.index.duplicated(keep="last")]

    required = ["open", "high", "low", "close", "volume"]
    for column in required:
        if column not in frame.columns:
            raise ValueError(f"{label} 데이터에 '{column}' 열이 없습니다.")
        coerced = pd.to_numeric(frame[column], errors="coerce")
        bad = int((coerced.isna() & frame[column].notna()).sum())
        if bad:
            LOGGER.warning("%s 데이터의 %s 열에서 비수치 값 %d개를 NaN 으로 치환했습니다.", label, column, bad)
        frame[column] = coerced
    before = len(frame)
    frame = frame.dropna(subset=required)
    removed = before - len(frame)
    if removed:
        LOGGER.warning("%s 데이터에서 결측 OHLCV 행 %d개를 제거했습니다.", label, int(removed))
    if len(frame) < 2:
        raise ValueError(f"{label} 데이터가 부족하여 백테스트를 진행할 수 없습니다.")
    return frame


@dataclass
class Position:
    direction: int = 0
    qty: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    bars_held: int = 0
    high_watermark: float = np.nan
    low_watermark: float = np.nan
    max_favourable: float = 0.0
    max_adverse: float = 0.0


# =====================================================================================
# === 메인 백테스트 루틴 =============================================================
# =====================================================================================


def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, float | bool | str],
    fees: Dict[str, float],
    risk: Dict[str, float | bool],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    """PPP Vishva Algo 경량 버전과 동등한 파이썬 백테스트."""

    del htf_df  # 상위 타임프레임 필터는 경량 버전에서 사용하지 않습니다.

    price_df = _normalise_ohlcv(df, "가격")

    def bool_param(name: str, default: bool) -> bool:
        value = params.get(name, default)
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
        return default

    def int_param(name: str, default: int) -> int:
        value = params.get(name, default)
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    def float_param(name: str, default: float) -> float:
        value = params.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def str_param(name: str, default: str) -> str:
        value = params.get(name, default)
        return str(value) if value is not None else str(default)

    # === 파라미터 해석 ===
    ut_key = float_param("utKey", 4.0)
    ut_atr_len = int_param("utAtrLen", 10)
    use_heikin = bool_param("useHeikin", False)

    rsi_len = int_param("rsiLen", 14)
    stoch_len = int_param("stochLen", 14)
    k_len = int_param("kLen", 3)
    d_len = int_param("dLen", 3)
    ob_level = float_param("obLevel", 80.0)
    os_level = float_param("osLevel", 20.0)
    st_mode = str_param("stMode", "Bounce").lower()
    debug_force_long = bool_param("debugForceLong", False)
    debug_force_short = bool_param("debugForceShort", False)

    atr_len = int_param("atrLen", 14)
    init_stop_mult = float_param("initStopMult", 1.8)
    trail_atr_mult = float_param("trailAtrMult", 2.5)
    trail_start_pct = float_param("trailStartPct", 1.0) / 100.0
    trail_gap_pct = float_param("trailGapPct", 0.5) / 100.0
    use_percent_stop = bool_param("usePercentStop", True)
    stop_pct = float_param("stopPct", 1.5) / 100.0
    take_pct = float_param("takePct", 2.5) / 100.0
    breakeven_pct = float_param("breakevenPct", 0.0) / 100.0
    max_hold_bars = int_param("maxHoldBars", 0)
    use_flip_exit = bool_param("useFlipExit", True)
    cooldown_bars = int_param("cooldownBars", 0)

    initial_capital = float(risk.get("initial_capital", params.get("initial_capital", 1000.0)))
    leverage = float(risk.get("leverage", params.get("leverage", 10.0)))
    qty_pct = float(risk.get("qty_pct", params.get("qty_pct", 100.0)))
    commission_pct = float(
        fees.get(
            "commission_pct",
            fees.get("fee_pct", risk.get("fee_pct", params.get("fee_pct", 0.0005))),
        )
    )
    slippage_ticks = float(
        fees.get(
            "slippage_ticks",
            risk.get("slippage_ticks", params.get("slippage_ticks", 0.0)),
        )
    )

    simple_metrics_only = bool(risk.get("simpleMetricsOnly") or risk.get("simpleProfitOnly"))
    min_trades_req = int(min_trades if min_trades is not None else risk.get("min_trades", 0))
    min_hold_bars_req = int(risk.get("min_hold_bars", 0))
    max_loss_streak = int(risk.get("max_consecutive_losses", 999999))

    # === 지표 사전 계산 ===
    ha_df = _heikin_ashi(price_df) if use_heikin else price_df
    ut_src = ha_df["close"]
    atr_ut = _atr(price_df, max(ut_atr_len, 1)).bfill().ffill()
    ut_trail = np.full(len(price_df), np.nan, dtype=float)

    for i in range(len(price_df)):
        loss = ut_key * atr_ut.iloc[i] if np.isfinite(atr_ut.iloc[i]) else np.nan
        price = ut_src.iloc[i]
        if not np.isfinite(loss):
            ut_trail[i] = ut_trail[i - 1] if i > 0 else price
            continue
        if i == 0 or not np.isfinite(ut_trail[i - 1]):
            ut_trail[i] = price - loss
            continue
        prev_trail = ut_trail[i - 1]
        prev_price = ut_src.iloc[i - 1]
        if price > prev_trail and prev_price > prev_trail:
            ut_trail[i] = max(prev_trail, price - loss)
        elif price < prev_trail and prev_price < prev_trail:
            ut_trail[i] = min(prev_trail, price + loss)
        else:
            ut_trail[i] = price - loss if price > prev_trail else price + loss

    ema1 = _ema(ut_src, 1)
    rsi_val = _rsi(price_df["close"], rsi_len)
    stoch_val = _stoch(rsi_val, stoch_len)
    k = _sma(stoch_val, max(k_len, 1)).bfill()
    d = _sma(k, max(d_len, 1)).bfill()
    atr_series = _atr(price_df, max(atr_len, 1)).bfill()

    tick_size = _estimate_tick(price_df["close"])
    slip_value = tick_size * slippage_ticks

    position = Position()
    trades: List[Trade] = []
    equity = float(initial_capital)
    returns = pd.Series(0.0, index=price_df.index, dtype=float)
    cooldown = 0

    def _close_position(ts: pd.Timestamp, exit_price: float, reason: str) -> None:
        nonlocal position, equity, cooldown
        if position.direction == 0 or position.entry_time is None:
            return
        qty = position.qty
        direction = position.direction
        fill_price = exit_price - slip_value if direction > 0 else exit_price + slip_value
        pnl = (fill_price - position.entry_price) * direction * qty
        fees_paid = (position.entry_price + fill_price) * qty * commission_pct
        pnl -= fees_paid
        equity += pnl
        returns.loc[ts] += pnl / initial_capital if initial_capital else 0.0

        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=ts,
                direction="long" if direction > 0 else "short",
                size=qty,
                entry_price=position.entry_price,
                exit_price=fill_price,
                profit=pnl,
                return_pct=pnl / initial_capital if initial_capital else 0.0,
                mfe=position.max_favourable,
                mae=position.max_adverse,
                bars_held=position.bars_held,
                reason=reason,
            )
        )
        if pnl < 0 and cooldown_bars > 0:
            cooldown = cooldown_bars
        position = Position()

    for i, (ts, row) in enumerate(price_df.iterrows()):
        if cooldown > 0:
            cooldown -= 1

        price = row["close"]
        prev_ema = ema1.iloc[i - 1] if i > 0 else ema1.iloc[i]
        prev_trail = ut_trail[i - 1] if i > 0 else ut_trail[i]
        ut_buy = bool(price > ut_trail[i] and prev_ema <= prev_trail and ema1.iloc[i] > ut_trail[i])
        ut_sell = bool(price < ut_trail[i] and prev_trail <= prev_ema and ut_trail[i] > ema1.iloc[i])

        if st_mode == "bounce":
            st_long = bool(k.iloc[i] < os_level and k.iloc[i - 1] <= d.iloc[i - 1] and k.iloc[i] > d.iloc[i]) if i > 0 else False
            st_short = bool(k.iloc[i] > ob_level and k.iloc[i - 1] >= d.iloc[i - 1] and k.iloc[i] < d.iloc[i]) if i > 0 else False
        else:
            st_long = bool(k.iloc[i - 1] <= d.iloc[i - 1] and k.iloc[i] > d.iloc[i] and k.iloc[i] < 50) if i > 0 else False
            st_short = bool(k.iloc[i - 1] >= d.iloc[i - 1] and k.iloc[i] < d.iloc[i] and k.iloc[i] > 50) if i > 0 else False

        long_signal = ut_buy and st_long
        short_signal = ut_sell and st_short

        if position.direction != 0:
            position.bars_held += 1
            qty = position.qty
            if position.direction > 0:
                move_high = (row["high"] - position.entry_price) * qty
                move_low = (row["low"] - position.entry_price) * qty
                position.max_favourable = max(position.max_favourable, move_high)
                position.max_adverse = min(position.max_adverse, move_low)
                position.high_watermark = row["high"] if np.isnan(position.high_watermark) else max(position.high_watermark, row["high"])
            else:
                move_high = (position.entry_price - row["low"]) * qty
                move_low = (position.entry_price - row["high"]) * qty
                position.max_favourable = max(position.max_favourable, move_high)
                position.max_adverse = min(position.max_adverse, move_low)
                position.low_watermark = row["low"] if np.isnan(position.low_watermark) else min(position.low_watermark, row["low"])

        atr_val = atr_series.iloc[i] if np.isfinite(atr_series.iloc[i]) else np.nan
        exit_triggered = False

        if position.direction > 0:
            stop_candidates: List[float] = []
            if np.isfinite(atr_val):
                stop_candidates.append(position.entry_price - atr_val * init_stop_mult)
                stop_candidates.append(price - atr_val * trail_atr_mult)
            if use_percent_stop:
                stop_candidates.append(position.entry_price * (1 - stop_pct))
            if breakeven_pct >= 0 and not np.isnan(position.high_watermark):
                if position.high_watermark >= position.entry_price * (1 + breakeven_pct):
                    stop_candidates.append(position.entry_price * (1 + breakeven_pct))
            if trail_start_pct > 0 and not np.isnan(position.high_watermark):
                if position.high_watermark >= position.entry_price * (1 + trail_start_pct):
                    stop_candidates.append(price * (1 - trail_gap_pct))

            stop_price = max([s for s in stop_candidates if np.isfinite(s)], default=np.nan)
            take_price = (
                position.entry_price * (1 + take_pct)
                if use_percent_stop and take_pct > 0
                else np.nan
            )

            if np.isfinite(stop_price) and row["low"] <= stop_price:
                _close_position(ts, stop_price, "stop_long")
                exit_triggered = True
            elif np.isfinite(take_price) and row["high"] >= take_price:
                _close_position(ts, take_price, "take_long")
                exit_triggered = True
            elif max_hold_bars > 0 and position.bars_held >= max_hold_bars:
                _close_position(ts, price, "time_long")
                exit_triggered = True
            elif use_flip_exit and ut_sell:
                _close_position(ts, price, "flip_long")
                exit_triggered = True

        elif position.direction < 0:
            stop_candidates = []
            if np.isfinite(atr_val):
                stop_candidates.append(position.entry_price + atr_val * init_stop_mult)
                stop_candidates.append(price + atr_val * trail_atr_mult)
            if use_percent_stop:
                stop_candidates.append(position.entry_price * (1 + stop_pct))
            if breakeven_pct >= 0 and not np.isnan(position.low_watermark):
                if position.low_watermark <= position.entry_price * (1 - breakeven_pct):
                    stop_candidates.append(position.entry_price * (1 - breakeven_pct))
            if trail_start_pct > 0 and not np.isnan(position.low_watermark):
                if position.low_watermark <= position.entry_price * (1 - trail_start_pct):
                    stop_candidates.append(price * (1 + trail_gap_pct))

            stop_price = min([s for s in stop_candidates if np.isfinite(s)], default=np.nan)
            take_price = (
                position.entry_price * (1 - take_pct)
                if use_percent_stop and take_pct > 0
                else np.nan
            )

            if np.isfinite(stop_price) and row["high"] >= stop_price:
                _close_position(ts, stop_price, "stop_short")
                exit_triggered = True
            elif np.isfinite(take_price) and row["low"] <= take_price:
                _close_position(ts, take_price, "take_short")
                exit_triggered = True
            elif max_hold_bars > 0 and position.bars_held >= max_hold_bars:
                _close_position(ts, price, "time_short")
                exit_triggered = True
            elif use_flip_exit and ut_buy:
                _close_position(ts, price, "flip_short")
                exit_triggered = True

        if exit_triggered:
            continue

        force_long = debug_force_long and position.direction == 0
        force_short = debug_force_short and position.direction == 0

        if position.direction == 0 and cooldown == 0:
            if force_long or long_signal:
                capital = equity * max(qty_pct, 0.0) / 100.0
                qty = (capital * max(leverage, 0.0)) / price if price > 0 else 0.0
                if qty > 0:
                    position = Position(
                        direction=1,
                        qty=qty,
                        entry_price=price,
                        entry_time=ts,
                        high_watermark=row["high"],
                        low_watermark=row["low"],
                    )
            elif force_short or short_signal:
                capital = equity * max(qty_pct, 0.0) / 100.0
                qty = (capital * max(leverage, 0.0)) / price if price > 0 else 0.0
                if qty > 0:
                    position = Position(
                        direction=-1,
                        qty=qty,
                        entry_price=price,
                        entry_time=ts,
                        high_watermark=row["high"],
                        low_watermark=row["low"],
                    )

    if position.direction != 0:
        _close_position(price_df.index[-1], price_df.iloc[-1]["close"], "eod")

    metrics = aggregate_metrics(trades, returns, simple=simple_metrics_only)
    if simple_metrics_only:
        metrics["SimpleMetricsOnly"] = True
    metrics["FinalEquity"] = equity
    metrics["NetProfitAbs"] = equity - initial_capital
    metrics["TradesList"] = trades
    metrics["Returns"] = returns
    metrics["Withdrawable"] = 0.0
    metrics["GuardFrozen"] = 0.0
    metrics["MinTrades"] = float(max(0, min_trades_req))
    metrics["MinHoldBars"] = float(max(0, min_hold_bars_req))
    metrics["MaxConsecutiveLossLimit"] = float(max(0, max_loss_streak))
    metrics["Valid"] = (
        metrics.get("Trades", 0.0) >= min_trades_req
        and metrics.get("AvgHoldBars", 0.0) >= min_hold_bars_req
        and metrics.get("MaxConsecutiveLosses", 0.0) <= max_loss_streak
    )
    return metrics
