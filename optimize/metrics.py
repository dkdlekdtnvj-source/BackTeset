"""핵심 백테스트 지표 계산 유틸리티."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

EPS: float = 1e-9
LOSSLESS_GROSS_LOSS_PCT: float = 1e-3
LOSSLESS_ANOMALY_FLAG: str = "lossless_profit_factor"
MICRO_LOSS_ANOMALY_FLAG: str = "micro_loss_profit_factor"


@dataclass(frozen=True)
class Trade:
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
    bars_held: float
    reason: str = ""

    @property
    def is_win(self) -> bool:
        return float(self.profit) > 0.0

    @property
    def is_loss(self) -> bool:
        return float(self.profit) < 0.0


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    goal: str = "maximize"
    weight: float = 1.0

    @property
    def direction(self) -> str:
        goal = (self.goal or "").strip().lower()
        if goal in {"min", "minimise", "minimize", "down"}:
            return "minimize"
        return "maximize"


MetricMapping = MutableMapping[str, object]
TradeLike = Union[Trade, MutableMapping[str, object]]


def _extract_profit(trade: TradeLike) -> float:
    if isinstance(trade, Trade):
        return float(trade.profit)
    if isinstance(trade, MutableMapping):
        try:
            return float(trade.get("profit", 0.0))
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(getattr(trade, "profit"))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def profit_factor(trades: Iterable[TradeLike]) -> float:
    gross_profit = 0.0
    gross_loss = 0.0
    for trade in trades:
        profit = _extract_profit(trade)
        if not np.isfinite(profit) or profit == 0.0:
            continue
        if profit > 0:
            gross_profit += profit
        else:
            gross_loss += profit
    if abs(gross_loss) < EPS:
        return float("inf") if gross_profit > EPS else 0.0
    return float(gross_profit / abs(gross_loss))


def equity_curve_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    if returns is None:
        return pd.Series([], dtype=float)
    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan)
    series = series.fillna(0.0)
    equity = (1.0 + series).cumprod() * float(initial)
    equity.index = returns.index
    return equity


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    series = pd.Series(equity, dtype=float).replace([np.inf, -np.inf], np.nan)
    series = series.ffill().fillna(0.0)
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdowns = series / running_max - 1.0
    return float(drawdowns.min() if not drawdowns.empty else 0.0)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    if returns is None or len(returns) == 0:
        return 0.0
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    excess = clean - float(risk_free)
    std = float(excess.std(ddof=0))
    if std < EPS or not np.isfinite(std):
        return 0.0
    mean = float(excess.mean())
    return float(mean / std)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    if returns is None or len(returns) == 0:
        return 0.0
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    downside = clean[clean < risk_free]
    downside = downside.replace([np.inf, -np.inf], np.nan).dropna()
    excess_mean = float(clean.mean() - risk_free)
    if downside.empty:
        std = float(clean.std(ddof=0))
        if std < EPS or not np.isfinite(std):
            return 0.0
        return float(excess_mean / std)
    downside_std = float(downside.std(ddof=0))
    if downside_std < EPS or not np.isfinite(downside_std):
        return 0.0
    return float(excess_mean / downside_std)


def _ensure_list(iterable: Iterable[TradeLike]) -> List[TradeLike]:
    return list(iterable) if not isinstance(iterable, list) else iterable


def _win_loss_counts(profits: np.ndarray) -> Tuple[int, int, int]:
    wins = int(np.sum(profits > EPS))
    losses = int(np.sum(profits < -EPS))
    ties = int(len(profits) - wins - losses)
    return wins, losses, ties


def _max_consecutive_losses(profits: Sequence[float]) -> int:
    worst = 0
    current = 0
    for profit in profits:
        if profit < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def _avg_hold(trades: Sequence[TradeLike]) -> float:
    if not trades:
        return 0.0
    holds: List[float] = []
    for trade in trades:
        try:
            value = float(getattr(trade, "bars_held"))
        except (AttributeError, TypeError, ValueError):
            if isinstance(trade, MutableMapping):
                try:
                    value = float(trade.get("bars_held", 0.0))
                except (TypeError, ValueError):
                    value = 0.0
            else:
                value = 0.0
        if np.isfinite(value):
            holds.append(value)
    if not holds:
        return 0.0
    return float(np.mean(holds))


def _weekly_net_profit(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    if isinstance(returns.index, pd.DatetimeIndex):
        weekly = returns.resample("W").sum(min_count=1)
        weekly = weekly.replace([np.inf, -np.inf], np.nan).dropna()
        if weekly.empty:
            return 0.0
        return float(weekly.mean())
    return float(returns.sum())


def _prepare_anomaly_flags(flags: Union[None, str, Sequence[str]]) -> List[str]:
    if flags is None:
        return []
    if isinstance(flags, str):
        tokens = [token.strip() for token in flags.split(",") if token.strip()]
        return list(dict.fromkeys(tokens))
    cleaned = [str(token).strip() for token in flags if str(token).strip()]
    return list(dict.fromkeys(cleaned))


def apply_lossless_anomaly(
    metrics: MetricMapping, *, threshold: Optional[float] = None
) -> Optional[Tuple[str, float, float, float, float]]:
    gross_loss = float(metrics.get("GrossLoss", 0.0) or 0.0)
    abs_loss = abs(gross_loss)
    if threshold is None:
        base = metrics.get("InitialCapital") or metrics.get("InitialEquity") or 1.0
        try:
            base_value = abs(float(base))
        except (TypeError, ValueError):
            base_value = 1.0
        threshold = base_value * LOSSLESS_GROSS_LOSS_PCT
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        threshold_value = 0.0
    if threshold_value <= 0:
        return None
    metrics["LosslessGrossLossThreshold"] = float(threshold_value)
    if abs_loss <= threshold_value + EPS:
        trades_val = float(metrics.get("Trades", 0.0) or 0.0)
        wins_val = float(metrics.get("Wins", 0.0) or 0.0)
        metrics["LosslessProfitFactor"] = True
        metrics["ProfitFactor"] = 0.0
        flags = _prepare_anomaly_flags(metrics.get("AnomalyFlags"))
        if abs_loss <= EPS:
            flag = LOSSLESS_ANOMALY_FLAG
            flags.append(flag)
        elif abs_loss <= threshold_value * 0.75:
            flag = MICRO_LOSS_ANOMALY_FLAG
            flags.extend([LOSSLESS_ANOMALY_FLAG, MICRO_LOSS_ANOMALY_FLAG])
        else:
            flag = LOSSLESS_ANOMALY_FLAG
            flags.append(flag)
        metrics["AnomalyFlags"] = list(dict.fromkeys(flags))
        return (
            flag,
            float(trades_val),
            float(wins_val),
            float(abs_loss),
            float(threshold_value),
        )
    return None


def aggregate_metrics(
    trades: Iterable[TradeLike],
    returns: Optional[pd.Series],
    *,
    simple: bool = False,
) -> Dict[str, object]:
    trade_list = _ensure_list(list(trades))
    profits = np.array([_extract_profit(trade) for trade in trade_list], dtype=float) if trade_list else np.array([], dtype=float)

    gross_profit = float(profits[profits > 0].sum()) if profits.size else 0.0
    gross_loss = float(profits[profits < 0].sum()) if profits.size else 0.0
    net_profit = gross_profit + gross_loss

    wins, losses, ties = _win_loss_counts(profits)
    trades_count = len(trade_list)

    metrics: Dict[str, object] = {
        "Trades": float(trades_count),
        "Wins": float(wins),
        "Losses": float(losses),
        "GrossProfit": float(gross_profit),
        "GrossLoss": float(gross_loss),
        "NetProfit": float(net_profit),
        "WinRate": float(wins / trades_count) if trades_count else 0.0,
        "Expectancy": float(net_profit / trades_count) if trades_count else 0.0,
        "AvgHoldBars": _avg_hold(trade_list),
        "MaxConsecutiveLosses": float(_max_consecutive_losses(profits)),
        "TradesList": trade_list,
        "LosslessProfitFactor": False,
        "AnomalyFlags": [],
    }

    if returns is not None:
        returns_series = pd.Series(returns, dtype=float)
    else:
        returns_series = pd.Series([], dtype=float)
    metrics["Returns"] = returns_series

    total_return = float(returns_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum()) if not returns_series.empty else float(net_profit)
    metrics["TotalReturn"] = total_return

    metrics["ProfitFactor"] = profit_factor(trade_list)

    equity = equity_curve_from_returns(returns_series, initial=1.0) if not returns_series.empty else pd.Series([], dtype=float)
    metrics["EquityCurve"] = equity
    metrics["MaxDD"] = max_drawdown(equity) if not equity.empty else 0.0
    metrics["Sharpe"] = sharpe_ratio(returns_series)
    metrics["Sortino"] = sortino_ratio(returns_series)
    metrics["WeeklyNetProfit"] = _weekly_net_profit(returns_series)

    if ties and not metrics["LosslessProfitFactor"]:
        # Ensure expectancies remain finite when only breakeven trades exist.
        metrics["Expectancy"] = 0.0

    apply_lossless_anomaly(metrics)

    metrics["SimpleMetricsOnly"] = bool(simple)

    metrics.setdefault("Valid", True)
    return metrics


def normalise_objectives(objectives: Sequence[Union[str, Dict[str, object], ObjectiveSpec]]) -> List[ObjectiveSpec]:
    specs: List[ObjectiveSpec] = []
    for obj in objectives:
        if isinstance(obj, ObjectiveSpec):
            specs.append(obj)
            continue
        if isinstance(obj, str):
            specs.append(ObjectiveSpec(name=obj))
            continue
        if isinstance(obj, Dict):
            name = str(obj.get("name"))
            if not name:
                raise ValueError("Objective 명칭은 필수입니다.")
            goal = str(obj.get("goal") or obj.get("direction") or "maximize")
            weight = obj.get("weight", 1.0)
            try:
                weight_value = float(weight)
            except (TypeError, ValueError):
                weight_value = 1.0
            specs.append(ObjectiveSpec(name=name, goal=goal, weight=weight_value))
            continue
        raise TypeError(f"지원되지 않는 objective 정의: {obj!r}")
    return specs


def evaluate_objective_values(
    metrics: MetricMapping,
    objectives: Sequence[Union[str, Dict[str, object], ObjectiveSpec]],
    non_finite_penalty: float = -1e9,
) -> List[float]:
    specs = normalise_objectives(objectives)
    values: List[float] = []
    for spec in specs:
        weight = float(spec.weight)
        if weight == 0:
            values.append(0.0)
            continue
        raw = metrics.get(spec.name)
        numeric: Optional[float]
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            numeric = None
        if numeric is None or not np.isfinite(numeric):
            penalty = float(non_finite_penalty) * weight
            if spec.direction == "minimize":
                penalty = -penalty
            values.append(penalty)
            continue
        value = numeric * weight
        if spec.direction == "minimize":
            value = -value
        values.append(float(value))
    return values


def score_metrics(
    metrics: MetricMapping,
    objectives: Sequence[Union[str, Dict[str, object], ObjectiveSpec]],
    *,
    non_finite_penalty: float = -1e9,
) -> float:
    objective_scores = evaluate_objective_values(metrics, objectives, non_finite_penalty)
    score = float(np.sum(objective_scores)) if objective_scores else 0.0

    def _coerce(value: object) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return float(numeric)

    trades = _coerce(metrics.get("Trades"))
    min_trades = _coerce(metrics.get("MinTrades"))
    if trades < min_trades:
        penalty = _coerce(metrics.get("TradePenalty")) or (min_trades - trades)
        score -= abs(penalty)

    avg_hold = _coerce(metrics.get("AvgHoldBars"))
    min_hold = _coerce(metrics.get("MinHoldBars"))
    if avg_hold < min_hold:
        penalty = _coerce(metrics.get("HoldPenalty")) or (min_hold - avg_hold)
        score -= abs(penalty)

    max_losses = _coerce(metrics.get("MaxConsecutiveLosses"))
    max_loss_limit = _coerce(metrics.get("MaxConsecutiveLossLimit"))
    if max_loss_limit and max_losses > max_loss_limit:
        penalty = _coerce(metrics.get("ConsecutiveLossPenalty")) or (max_losses - max_loss_limit)
        score -= abs(penalty)

    return float(score)

