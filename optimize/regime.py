"""Regime detection helpers for warm-start selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class RegimeSummary:
    label: str
    volatility: float
    slope: float
    session: str


def _categorise(value: float, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    if value <= low:
        return low_label
    if value >= high:
        return high_label
    return mid_label


def detect_regime_label(df: pd.DataFrame) -> RegimeSummary:
    if df.empty or "close" not in df:
        return RegimeSummary(label="unknown", volatility=float("nan"), slope=float("nan"), session="unknown")

    closes = df["close"].astype(float)
    returns = closes.pct_change().dropna()
    window = min(len(returns), 240)
    if window == 0:
        return RegimeSummary(label="unknown", volatility=float("nan"), slope=float("nan"), session="unknown")

    recent_returns = returns.iloc[-window:]
    vol = float(recent_returns.std(ddof=0))
    lookback = min(len(closes) - 1, 240)
    slope = float("nan")
    if lookback > 1:
        past_price = closes.iloc[-lookback]
        if past_price != 0:
            slope = float((closes.iloc[-1] - past_price) / past_price)
    else:
        slope = 0.0

    vol_label = _categorise(vol, 0.005, 0.02, "lowvol", "midvol", "highvol")
    slope_label = _categorise(slope, -0.01, 0.01, "bear", "flat", "bull")

    last_ts = df.index[-1]
    if isinstance(last_ts, pd.Timestamp):
        hour = last_ts.hour
    else:
        hour = 0
    if 7 <= hour < 12:
        session = "asia"
    elif 12 <= hour < 18:
        session = "eu"
    else:
        session = "us"

    label = "-".join([vol_label, slope_label, session])
    return RegimeSummary(label=label, volatility=vol, slope=slope, session=session)


def summarise_regime_performance(
    bank_entry: Dict[str, object],
    summary: RegimeSummary,
    metric: str = "oos_mean",
) -> Dict[str, object]:
    score = float(bank_entry.get(metric, 0.0) or 0.0)
    payload = {
        "label": summary.label,
        "score": score,
        "volatility": summary.volatility,
        "slope": summary.slope,
        "session": summary.session,
    }
    return payload
