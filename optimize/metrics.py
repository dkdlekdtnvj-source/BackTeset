"""Performance metric calculations for optimisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


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
    gross_profit = sum(max(trade.profit, 0.0) for trade in trades)
    gross_loss = sum(min(trade.profit, 0.0) for trade in trades)
    denominator = max(abs(gross_loss), EPS)
    return float(gross_profit / denominator)


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

    fallback_keys = ("ProfitFactor", "Sortino", "Sharpe")

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
        try:
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
]
