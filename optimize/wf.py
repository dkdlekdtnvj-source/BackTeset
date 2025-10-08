"""Walk-forward analysis utilities."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .strategy_model import run_backtest


def _clean_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool)):
            clean[key] = float(value)
    return clean


def _run_walk_forward_segment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_htf: Optional[pd.DataFrame],
    test_htf: Optional[pd.DataFrame],
    params: Dict[str, float | bool],
    fees: Dict[str, float],
    risk: Dict[str, float],
    min_trades: Optional[int],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    train_metrics = run_backtest(
        train_df,
        params,
        fees,
        risk,
        htf_df=train_htf,
        min_trades=min_trades,
    )
    test_metrics = run_backtest(
        test_df,
        params,
        fees,
        risk,
        htf_df=test_htf,
        min_trades=min_trades,
    )
    return _clean_metrics(train_metrics), _clean_metrics(test_metrics)


@dataclass
class SegmentResult:
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def run_walk_forward(
    df: pd.DataFrame,
    params: Dict[str, float | bool],
    fees: Dict[str, float],
    risk: Dict[str, float],
    train_bars: int,
    test_bars: int,
    step: int,
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
    *,
    n_jobs: int = 1,
    executor: str = "process",
) -> Dict[str, object]:
    segments: List[SegmentResult] = []
    total = len(df)

    if total == 0:
        return {"segments": [], "oos_mean": 0.0, "oos_median": 0.0, "count": 0}

    train_bars = int(train_bars)
    test_bars = int(test_bars)
    step = max(int(step), 1)

    if train_bars <= 0 or train_bars >= total:
        train_bars = total
    if test_bars <= 0 or train_bars + test_bars > total:
        metrics = run_backtest(
            df,
            params,
            fees,
            risk,
            htf_df=htf_df,
            min_trades=min_trades,
        )
        clean = _clean_metrics(metrics)
        return {
            "segments": segments,
            "oos_mean": float(clean.get("NetProfit", 0.0)),
            "oos_median": float(clean.get("NetProfit", 0.0)),
            "count": 0,
            "full_run": clean,
        }

    start = 0
    segment_inputs: List[
        Tuple[
            pd.DataFrame,
            pd.DataFrame,
            Optional[pd.DataFrame],
            Optional[pd.DataFrame],
        ]
    ] = []
    boundaries: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    while start + train_bars + test_bars <= total:
        train_slice = slice(start, start + train_bars)
        test_slice = slice(start + train_bars, start + train_bars + test_bars)
        train_df = df.iloc[train_slice]
        test_df = df.iloc[test_slice]

        if train_df.empty or test_df.empty:
            break

        train_htf = htf_df.loc[: train_df.index[-1]] if htf_df is not None else None
        test_htf = htf_df.loc[: test_df.index[-1]] if htf_df is not None else None

        segment_inputs.append((train_df, test_df, train_htf, test_htf))
        boundaries.append(
            (train_df.index[0], train_df.index[-1], test_df.index[0], test_df.index[-1])
        )
        start += step

    if not segment_inputs:
        summary = {
            "segments": segments,
            "oos_mean": 0.0,
            "oos_median": 0.0,
            "count": 0,
        }
        return summary

    jobs = max(1, int(n_jobs))
    executor_choice = executor.lower().strip()
    parallel_enabled = jobs > 1 and len(segment_inputs) > 1

    results: List[Optional[Tuple[Dict[str, float], Dict[str, float]]]] = [
        None
        for _ in segment_inputs
    ]

    if parallel_enabled:
        if executor_choice == "thread":
            pool_cls = ThreadPoolExecutor
        else:
            pool_cls = ProcessPoolExecutor
        with pool_cls(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    _run_walk_forward_segment,
                    segment_inputs[idx][0],
                    segment_inputs[idx][1],
                    segment_inputs[idx][2],
                    segment_inputs[idx][3],
                    params,
                    fees,
                    risk,
                    min_trades,
                ): idx
                for idx in range(len(segment_inputs))
            }
            for future in as_completed(futures):
                idx = futures[future]
                train_metrics, test_metrics = future.result()
                results[idx] = (train_metrics, test_metrics)
    else:
        for idx, (train_df_seg, test_df_seg, train_htf_seg, test_htf_seg) in enumerate(
            segment_inputs
        ):
            train_metrics, test_metrics = _run_walk_forward_segment(
                train_df_seg,
                test_df_seg,
                train_htf_seg,
                test_htf_seg,
                params,
                fees,
                risk,
                min_trades,
            )
            results[idx] = (train_metrics, test_metrics)

    for idx, boundary in enumerate(boundaries):
        result_pair = results[idx]
        if result_pair is None:
            continue
        train_metrics, test_metrics = result_pair
        segments.append(
            SegmentResult(
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                train_start=boundary[0],
                train_end=boundary[1],
                test_start=boundary[2],
                test_end=boundary[3],
            )
        )

    oos_returns = [seg.test_metrics.get("NetProfit", 0.0) for seg in segments if seg.test_metrics.get("Valid", True)]
    oos_series = pd.Series(oos_returns) if oos_returns else pd.Series(dtype=float)
    summary = {
        "segments": segments,
        "oos_mean": float(oos_series.mean()) if not oos_series.empty else 0.0,
        "oos_median": float(oos_series.median()) if not oos_series.empty else 0.0,
        "count": len(segments),
    }
    return summary


def run_purged_kfold(
    df: pd.DataFrame,
    params: Dict[str, float | bool],
    fees: Dict[str, float],
    risk: Dict[str, float],
    *,
    k: int = 5,
    embargo: float = 0.01,
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, object]:
    k = max(int(k), 2)
    total = len(df)
    if total == 0 or k <= 1:
        return {"folds": [], "mean": 0.0, "median": 0.0, "count": 0}

    fold_size = total // k
    embargo_bars = int(total * float(max(embargo, 0.0)))
    folds: List[Dict[str, object]] = []

    for fold in range(k):
        test_start = fold * fold_size
        test_end = total if fold == k - 1 else (fold + 1) * fold_size
        if test_start >= test_end:
            continue
        test_df = df.iloc[test_start:test_end]
        if test_df.empty:
            continue

        train_mask = np.ones(total, dtype=bool)
        start_embargo = max(0, test_start - embargo_bars)
        end_embargo = min(total, test_end + embargo_bars)
        train_mask[start_embargo:end_embargo] = False
        train_df = df.iloc[train_mask]
        if train_df.empty:
            continue

        train_htf = htf_df if htf_df is None else htf_df.loc[train_df.index]
        test_htf = htf_df if htf_df is None else htf_df.loc[test_df.index]

        train_metrics = run_backtest(
            train_df,
            params,
            fees,
            risk,
            htf_df=train_htf,
            min_trades=min_trades,
        )
        test_metrics = run_backtest(
            test_df,
            params,
            fees,
            risk,
            htf_df=test_htf,
            min_trades=min_trades,
        )

        folds.append(
            {
                "fold": fold,
                "train_range": (train_df.index[0], train_df.index[-1]),
                "test_range": (test_df.index[0], test_df.index[-1]),
                "train_metrics": _clean_metrics(train_metrics),
                "test_metrics": _clean_metrics(test_metrics),
            }
        )

    scores = [fold["test_metrics"].get("NetProfit", 0.0) for fold in folds]
    series = pd.Series(scores) if scores else pd.Series(dtype=float)
    return {
        "folds": folds,
        "mean": float(series.mean()) if not series.empty else 0.0,
        "median": float(series.median()) if not series.empty else 0.0,
        "count": len(folds),
    }
