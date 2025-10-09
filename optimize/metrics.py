"""Performance metric calculations for optimisation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12
LOSSLESS_GROSS_LOSS_PCT = 1e-3
LOSSLESS_ANOMALY_FLAG = "lossless_profit_factor"
MICRO_LOSS_ANOMALY_FLAG = "micro_loss_profit_factor"
_INITIAL_BALANCE_KEYS = (
    "InitialCapital",
    "InitialEquity",
    "InitialBalance",
    "StartingBalance",
)


LOGGER = logging.getLogger(__name__)


def _resolve_initial_balance(target: Dict[str, float], default: float = 1.0) -> float:
    for key in _INITIAL_BALANCE_KEYS:
        value = target.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric) and numeric != 0:
            return abs(numeric)
    return abs(default)


def lossless_gross_loss_threshold(target: Dict[str, float], *, default_initial: float = 1.0) -> float:
    override = target.get("LosslessGrossLossThreshold")
    if override is not None:
        try:
            numeric = abs(float(override))
        except (TypeError, ValueError):
            numeric = float("nan")
        else:
            if np.isfinite(numeric) and numeric > 0:
                return max(numeric, EPS)

    base = _resolve_initial_balance(target, default=default_initial)
    threshold = abs(base) * LOSSLESS_GROSS_LOSS_PCT
    return max(threshold, EPS)


def detect_lossless_profit_factor(
    *,
    trades: float,
    wins: float,
    losses: float,
    gross_loss: float,
    threshold: float,
) -> str | None:
    try:
        trades = float(trades)
        wins = float(wins)
        losses = float(losses)
        gross_loss = float(gross_loss)
        threshold = max(float(threshold), EPS)
    except (TypeError, ValueError):
        return None

    if trades <= 0 or wins <= 0:
        return None

    abs_loss = abs(gross_loss)
    if abs_loss <= EPS and losses <= 0:
        return LOSSLESS_ANOMALY_FLAG
    if abs_loss <= threshold:
        return MICRO_LOSS_ANOMALY_FLAG
    return None


def apply_lossless_anomaly(
    target: Dict[str, float],
    *,
    trades: float | None = None,
    wins: float | None = None,
    losses: float | None = None,
    gross_loss: float | None = None,
    threshold: float | None = None,
) -> Optional[Tuple[str, float, float, float, float]]:
    """
    Detect and record lossless or micro-loss anomalies on the provided metric
    dictionary ``target``.  Lossless 케이스에서는 ``ProfitFactor`` 열을 문자열
    ``"overfactor"`` 로 교체해 사용자에게 비정상 상태임을 알리고, 원본 수치는
    ``LosslessProfitFactorValue`` 와 ``DisplayedProfitFactor`` 에 각각 보존한다.
    ``DisplayedProfitFactor`` 는 학습·정량 비교용으로 항상 부동소수 값(기본 0)
    을 유지하며, ``ProfitFactor`` 는 사용자 뷰에서 원래 계산값 또는 표시용
    문자열을 노출한다.

    Parameters
    ----------
    target : Dict[str, float]
        The metric dictionary to annotate.  It must contain keys such as
        ``Trades``, ``Wins``, ``Losses``, and ``GrossLoss``.  This
        function will update ``AnomalyFlags``, ``LosslessGrossLossThreshold``,
        ``LosslessProfitFactorValue``, ``DisplayedProfitFactor`` 및
        ``ProfitFactor`` in place if an anomaly is detected.
    trades, wins, losses, gross_loss : optional
        Overrides for the corresponding values found in ``target``.  If
        ``None``, values will be extracted from ``target`` instead.
    threshold : float, optional
        A custom threshold for micro-loss detection.  If not provided,
        ``lossless_gross_loss_threshold`` will be used.

    Returns
    -------
    Optional[Tuple[str, float, float, float, float]]
        A tuple of `(flag, trades_val, wins_val, abs_loss, threshold_val)`
        when an anomaly is detected, otherwise ``None``.
    """

    def _coerce(value: object, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    trades_val = _coerce(trades, _coerce(target.get("Trades")))
    wins_val = _coerce(wins, _coerce(target.get("Wins")))
    losses_val = _coerce(losses, _coerce(target.get("Losses")))
    gross_loss_val = _coerce(gross_loss, _coerce(target.get("GrossLoss")))

    if threshold is None:
        threshold_val = lossless_gross_loss_threshold(target)
    else:
        try:
            threshold_val = max(float(threshold), EPS)
        except (TypeError, ValueError):
            threshold_val = lossless_gross_loss_threshold(target)

    target["LosslessGrossLossThreshold"] = threshold_val

    flag = detect_lossless_profit_factor(
        trades=trades_val,
        wins=wins_val,
        losses=losses_val,
        gross_loss=gross_loss_val,
        threshold=threshold_val,
    )

    if not flag:
        return None

    # Record the anomaly flag
    existing = target.get("AnomalyFlags")
    if isinstance(existing, str):
        flags = [token.strip() for token in existing.split(",") if token.strip()]
    elif isinstance(existing, (list, tuple)):
        flags = [str(token) for token in existing if str(token)]
    else:
        flags = []
    if flag not in flags:
        flags.append(flag)
    target["AnomalyFlags"] = flags

    # Profit factor anomalies: we treat complete lossless and micro-loss cases
    # differently.  In the lossless case (no losing trades), the true ratio is
    # mathematically undefined, so we keep the original value but expose a
    # sentinel "overfactor" string to the user.  In the micro-loss case we
    # preserve the original profit factor instead of forcing it to zero.  This
    # avoids confusing logs such as ``profit_factor: 0.000`` when a nearly
    # lossless result is achieved.  The ``LosslessProfitFactorValue`` always
    # stores the raw value for reproducibility.
    original_pf = target.get("ProfitFactor")
    if original_pf is None:
        original_pf = float("nan")
    # Record the original value
    target["LosslessProfitFactorValue"] = original_pf
    target["LosslessProfitFactor"] = True
    # Micro-loss: keep the original PF but zero out the displayed metric so
    # weighted aggregates are not distorted by near-lossless ratios.
    if flag == MICRO_LOSS_ANOMALY_FLAG:
        target["DisplayedProfitFactor"] = 0.0
        target["ProfitFactor"] = original_pf
    elif flag == LOSSLESS_ANOMALY_FLAG:
        # True lossless case: set ProfitFactor to a sentinel string while
        # preserving the numeric value separately.  DisplayedProfitFactor
        # is set to 0 to avoid skewing weighted averages.
        target["DisplayedProfitFactor"] = 0.0
        target["ProfitFactor"] = "overfactor"
    else:
        # Unexpected flag type: fall back to original logic
        target["DisplayedProfitFactor"] = original_pf
        target["ProfitFactor"] = original_pf

    return flag, trades_val, wins_val, abs(gross_loss_val), threshold_val


@dataclass
class Trade:
    """Container describing the outcome of a single trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    size: float
    entry_price: float
    exit_price: float
    profit: float
    return_pct: float
    mfe: float
    mae: float
    bars_held: int
    reason: str = ""


@dataclass(frozen=True)
class ObjectiveSpec:
    """Normalised representation of an optimisation objective."""

    name: str
    weight: float = 1.0
    goal: str = "maximize"

    @property
    def direction(self) -> str:
        goal = str(self.goal).lower()
        if goal in {"minimise", "minimize", "min", "lower"}:
            return "minimize"
        return "maximize"

    @property
    def is_minimize(self) -> bool:
        return self.direction == "minimize"


def equity_curve_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Create an equity curve from percentage returns."""

    equity = (1 + returns.fillna(0)).cumprod() * initial
    return equity


def max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown as a negative percentage."""

    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    downside = returns[returns < risk_free]
    if not downside.empty:
        downside = downside.replace([np.inf, -np.inf], np.nan).dropna()
    if downside.empty:
        return 0.0
    expected = returns.replace([np.inf, -np.inf], np.nan).dropna().mean() - risk_free
    with np.errstate(invalid="ignore"):
        downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(expected / downside_std)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    cleaned = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return 0.0
    with np.errstate(invalid="ignore"):
        std = cleaned.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((cleaned.mean() - risk_free) / std)


def profit_factor(trades: Iterable[Trade]) -> float:
    """
    Compute the profit factor (gross profit divided by absolute gross loss).

    In general the profit factor is defined as

        PF = gross_profit / abs(gross_loss)

    where gross_profit is the sum of all positive trade profits and
    gross_loss is the sum of all negative trade profits.  If there are no
    losing trades (i.e. gross_loss == 0), the ratio is mathematically
    undefined and tends to infinity.  Rather than artificially capping the
    value or coercing to 0 or 1, this function returns ``np.inf`` in the
    lossless case.  The caller is responsible for deciding how to handle
    extremely large or infinite profit factors (e.g. flagging as
    "overfactor" or recording an anomaly).
    """
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in trades:
        profit = float(trade.profit)
        if profit > 0:
            gross_profit += profit
        else:
            gross_loss += profit

    if gross_loss == 0.0:
        # No losses at all; return infinity to denote undefined ratio.
        return float('inf') if gross_profit != 0.0 else 0.0
    denom = abs(gross_loss)
    if denom == 0.0:
        return 0.0
    return float(gross_profit / denom)


def win_rate(trades: Sequence[Trade]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for trade in trades if trade.profit > 0)
    return wins / len(trades)


def average_rr(trades: Sequence[Trade]) -> float:
    rs = [trade.mfe / abs(trade.mae) for trade in trades if trade.mae < 0]
    return float(np.mean(rs)) if rs else 0.0


def average_hold_time(trades: Sequence[Trade]) -> float:
    holds = [trade.bars_held for trade in trades]
    return float(np.mean(holds)) if holds else 0.0


def _consecutive_losses(trades: Sequence[Trade]) -> int:
    streak = 0
    worst = 0
    for trade in trades:
        if trade.profit < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def _weekly_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    weekly = returns.resample("W").sum()
    return weekly.dropna()


def aggregate_metrics(
    trades: List[Trade], returns: pd.Series, *, simple: bool = False
) -> Dict[str, float]:
    """Aggregate trade-level information into rich performance metrics."""

    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = equity_curve_from_returns(returns, initial=1.0)
    net_profit = float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]) if len(equity) > 1 else 0.0

    gross_profit = float(sum(max(trade.profit, 0.0) for trade in trades))
    gross_loss = float(sum(min(trade.profit, 0.0) for trade in trades))
    wins = sum(1 for trade in trades if trade.profit > 0)
    losses = sum(1 for trade in trades if trade.profit < 0)

    fallback_keys = ("Sortino", "Sharpe")

    # Nested helper declared below within aggregate_metrics.

    if simple:
        metrics: Dict[str, float] = {
            "NetProfit": net_profit,
            "TotalReturn": net_profit,
            "ProfitFactor": float(profit_factor(trades)),
            "Trades": float(len(trades)),
            "Wins": float(wins),
            "Losses": float(losses),
            "GrossProfit": gross_profit,
            "GrossLoss": gross_loss,
            "AvgHoldBars": float(average_hold_time(trades)),
            "MaxConsecutiveLosses": float(_consecutive_losses(trades)),
            "WinRate": float(win_rate(trades)),
        }
        for key in fallback_keys:
            if key in metrics and not np.isfinite(metrics[key]):
                metrics[key] = 0.0
        result = apply_lossless_anomaly(
            metrics,
            trades=len(trades),
            wins=wins,
            losses=losses,
            gross_loss=gross_loss,
        )
        if result:
            flag, trades_val, wins_val, abs_loss, threshold = result
            if flag == LOSSLESS_ANOMALY_FLAG:
                LOGGER.info(
                    "손실이 없는 결과(trades=%d, wins=%d)로 ProfitFactor='overfactor' 및 DisplayedProfitFactor=0으로 표기합니다.",
                    int(trades_val),
                    int(wins_val),
                )
            else:
                LOGGER.warning(
                    "미세 손실 %.6g (임계값 %.6g 이하)로 DisplayedProfitFactor=0으로 고정합니다. trades=%d, wins=%d",
                    abs_loss,
                    threshold,
                    int(trades_val),
                    int(wins_val),
                )
        return metrics

    weekly = _weekly_returns(returns)
    weekly_mean = float(weekly.mean()) if not weekly.empty else 0.0
    weekly_std = float(weekly.std(ddof=0)) if len(weekly) > 1 else 0.0

    metrics: Dict[str, float] = {
        "NetProfit": net_profit,
        "TotalReturn": net_profit,
        "MaxDD": float(max_drawdown(equity)),
        "WinRate": float(win_rate(trades)),
        "ProfitFactor": float(profit_factor(trades)),
        "Sortino": float(sortino_ratio(returns)),
        "Sharpe": float(sharpe_ratio(returns)),
        "AvgRR": float(average_rr(trades)),
        "AvgHoldBars": float(average_hold_time(trades)),
        "Trades": float(len(trades)),
        "Wins": float(wins),
        "Losses": float(losses),
        "GrossProfit": gross_profit,
        "GrossLoss": gross_loss,
        "Expectancy": float((gross_profit + gross_loss) / len(trades)) if trades else 0.0,
        "WeeklyNetProfit": weekly_mean,
        "WeeklyReturnStd": weekly_std,
        "MaxConsecutiveLosses": float(_consecutive_losses(trades)),
    }

    mfe = [trade.mfe for trade in trades]
    mae = [trade.mae for trade in trades]
    metrics["AvgMFE"] = float(np.mean(mfe)) if mfe else 0.0
    metrics["AvgMAE"] = float(np.mean(mae)) if mae else 0.0
    for key in fallback_keys:
        if key in metrics and not np.isfinite(metrics[key]):
            metrics[key] = 0.0
    result = apply_lossless_anomaly(
        metrics,
        trades=len(trades),
        wins=wins,
        losses=losses,
        gross_loss=gross_loss,
    )
    if result:
        flag, trades_val, wins_val, abs_loss, threshold = result
        if flag == LOSSLESS_ANOMALY_FLAG:
            LOGGER.info(
                "손실이 없는 결과(trades=%d, wins=%d)로 ProfitFactor='overfactor' 및 DisplayedProfitFactor=0으로 표기합니다.",
                int(trades_val),
                int(wins_val),
            )
        else:
            LOGGER.warning(
                "미세 손실 %.6g (임계값 %.6g 이하)로 DisplayedProfitFactor=0으로 고정합니다. trades=%d, wins=%d",
                abs_loss,
                threshold,
                int(trades_val),
                int(wins_val),
            )
    return metrics


def normalise_objectives(objectives: Iterable[object]) -> List[ObjectiveSpec]:
    """Coerce raw objective declarations into :class:`ObjectiveSpec` entries."""

    specs: List[ObjectiveSpec] = []
    for obj in objectives:
        if isinstance(obj, ObjectiveSpec):
            specs.append(obj)
            continue
        if isinstance(obj, str):
            specs.append(ObjectiveSpec(name=obj))
            continue
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("metric")
            if not name:
                continue
            weight = float(obj.get("weight", 1.0))
            if "minimize" in obj:
                goal = "minimize" if bool(obj.get("minimize")) else "maximize"
            elif "maximize" in obj:
                goal = "maximize" if bool(obj.get("maximize")) else "minimize"
            else:
                goal_raw = obj.get("goal") or obj.get("direction") or obj.get("target")
                if goal_raw is None:
                    goal = "maximize"
                else:
                    goal_text = str(goal_raw).lower()
                    if goal_text in {"min", "minimise", "minimize", "lower"}:
                        goal = "minimize"
                    elif goal_text in {"max", "maximise", "maximize", "higher"}:
                        goal = "maximize"
                    else:
                        goal = "maximize"
            specs.append(ObjectiveSpec(name=str(name), weight=weight, goal=goal))
    return specs


def _objective_iterator(objectives: Iterable[object]) -> Iterable[ObjectiveSpec]:
    for spec in normalise_objectives(objectives):
        yield spec


def evaluate_objective_values(
    metrics: Dict[str, float],
    objectives: Sequence[ObjectiveSpec],
    non_finite_penalty: float,
) -> Tuple[float, ...]:
    """Transform metric dict into ordered objective values respecting directions."""

    penalty = abs(float(non_finite_penalty))
    values: List[float] = []
    for spec in objectives:
        raw = metrics.get(spec.name)
        # Convert the raw metric to a numeric value.  Certain sentinel
        # strings (e.g. "overfactor" or the ProfitFactor check label) are
        # intentionally ignored during objective evaluation and treated as
        # neutral values.  If the value cannot be coerced to float and is
        # not one of these sentinel values, ``nan`` is used to trigger the
        # non-finite penalty.
        try:
            # If raw is a string sentinel, handle separately
            if isinstance(raw, str):
                raw_str = raw.strip().lower()
                if raw_str in {"overfactor", "체크 필요"}:
                    numeric = 0.0
                else:
                    numeric = float(raw)
            else:
                numeric = float(raw)
        except Exception:
            numeric = float("nan")

        name_lower = spec.name.lower()
        if name_lower in {"maxdd", "maxdrawdown"}:
            numeric = abs(numeric) if spec.is_minimize else -abs(numeric)

        if not np.isfinite(numeric):
            weight = abs(float(spec.weight))
            if weight == 0:
                numeric = 0.0
            else:
                base = penalty if spec.is_minimize else -penalty
                numeric = base * weight
        else:
            numeric *= float(spec.weight)

        values.append(numeric)

    return tuple(values)


def score_metrics(metrics: Dict[str, float], objectives: Iterable[object]) -> float:
    """Score a metric dictionary according to weighted objectives and penalties."""

    score = 0.0
    for spec in _objective_iterator(objectives):
        value = metrics.get(spec.name)
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        name_lower = spec.name.lower()
        if name_lower in {"maxdd", "maxdrawdown"}:
            contribution = -abs(numeric)
        elif spec.is_minimize:
            contribution = -numeric
        else:
            contribution = numeric
        score += float(spec.weight) * contribution

    trades = float(metrics.get("Trades", 0))
    min_trades = metrics.get("MinTrades")
    if min_trades is not None and trades < float(min_trades):
        penalty = float(metrics.get("TradePenalty", 1.0))
        score -= (float(min_trades) - trades) * penalty

    avg_hold = float(metrics.get("AvgHoldBars", 0.0))
    min_hold = metrics.get("MinHoldBars")
    if min_hold is not None and avg_hold < float(min_hold):
        penalty = float(metrics.get("HoldPenalty", 1.0))
        score -= (float(min_hold) - avg_hold) * penalty

    losses = float(metrics.get("MaxConsecutiveLosses", 0.0))
    loss_cap = metrics.get("MaxConsecutiveLossLimit")
    if loss_cap is not None and losses > float(loss_cap):
        penalty = float(metrics.get("ConsecutiveLossPenalty", 1.0))
        score -= (losses - float(loss_cap)) * penalty

    return float(score)


__all__ = [
    "Trade",
    "ObjectiveSpec",
    "evaluate_objective_values",
    "aggregate_metrics",
    "equity_curve_from_returns",
    "max_drawdown",
    "normalise_objectives",
    "score_metrics",
    "LOSSLESS_GROSS_LOSS_PCT",
    "LOSSLESS_ANOMALY_FLAG",
    "MICRO_LOSS_ANOMALY_FLAG",
    "lossless_gross_loss_threshold",
    "detect_lossless_profit_factor",
    "apply_lossless_anomaly",
]
