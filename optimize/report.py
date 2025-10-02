"""Report generation utilities for optimisation runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


from optimize.metrics import normalise_objectives


def _objective_iterator(objectives: Iterable[object]) -> Iterable[Tuple[str, float]]:
    for spec in normalise_objectives(objectives):
        yield spec.name, float(spec.weight)


def _flatten_results(results: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aggregated_rows: List[Dict[str, object]] = []
    dataset_rows: List[Dict[str, object]] = []

    for record in results:
        base_row: Dict[str, object] = {
            "trial": record.get("trial"),
            "score": record.get("score"),
            "valid": record.get("valid", True),
        }
        base_row.update(record.get("params", {}))
        for key, value in record.get("metrics", {}).items():
            if isinstance(value, (int, float, bool)):
                base_row[key] = value
        aggregated_rows.append(base_row)

        for dataset in record.get("datasets", []):
            ds_row: Dict[str, object] = {
                "trial": record.get("trial"),
                "score": record.get("score"),
                "valid": dataset.get("metrics", {}).get("Valid", True),
                "dataset": dataset.get("name"),
            }
            ds_row.update(dataset.get("meta", {}))
            ds_row.update(record.get("params", {}))
            for key, value in dataset.get("metrics", {}).items():
                if isinstance(value, (int, float, bool)):
                    ds_row[key] = value
            dataset_rows.append(ds_row)

    return pd.DataFrame(aggregated_rows), pd.DataFrame(dataset_rows)


def _annotate_objectives(df: pd.DataFrame, objectives: Iterable[object]) -> pd.DataFrame:
    if df.empty:
        return df

    composite = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for name, weight in _objective_iterator(objectives):
        if name not in df.columns:
            continue
        series = df[name].astype(float)
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            z = pd.Series(0.0, index=df.index)
        else:
            z = (series - series.mean()) / std
        df[f"{name}_z"] = z
        composite += weight * z
        total_weight += abs(weight)

    if total_weight:
        df["CompositeScore"] = composite / total_weight
    else:
        df["CompositeScore"] = composite
    return df


def export_results(results: List[Dict[str, object]], objectives: Iterable[object], output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dir(output_dir)
    agg_df, dataset_df = _flatten_results(results)
    agg_df = _annotate_objectives(agg_df, objectives)
    agg_df.to_csv(output_dir / "results.csv", index=False)
    if not dataset_df.empty:
        dataset_df.to_csv(output_dir / "results_datasets.csv", index=False)
    return agg_df, dataset_df


def export_best(best: Dict[str, object], wf_summary: Dict[str, object], output_dir: Path) -> None:
    segments_payload = []
    for seg in wf_summary.get("segments", []):
        segments_payload.append(
            {
                "train": [seg.train_start.isoformat(), seg.train_end.isoformat()],
                "test": [seg.test_start.isoformat(), seg.test_end.isoformat()],
                "train_metrics": seg.train_metrics,
                "test_metrics": seg.test_metrics,
            }
        )

    payload = {
        "params": best.get("params"),
        "metrics": best.get("metrics"),
        "score": best.get("score"),
        "datasets": best.get("datasets", []),
        "walk_forward": {
            "oos_mean": wf_summary.get("oos_mean"),
            "oos_median": wf_summary.get("oos_median"),
            "count": wf_summary.get("count"),
            "segments": segments_payload,
            "candidates": wf_summary.get("candidates", []),
        },
    }
    (output_dir / "best.json").write_text(json.dumps(payload, indent=2))


def export_heatmap(metrics_df: pd.DataFrame, params: List[str], metric: str, plots_dir: Path) -> None:
    if len(params) < 2 or metrics_df.empty or metric not in metrics_df.columns:
        return
    x_param, y_param = params[:2]
    if x_param not in metrics_df or y_param not in metrics_df:
        return
    pivot = metrics_df.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc="mean")
    if pivot.empty:
        return
    _ensure_dir(plots_dir)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=False, cmap="viridis")
    plt.title(f"{metric} heatmap ({y_param} vs {x_param})")
    plt.tight_layout()
    plt.savefig(plots_dir / "heatmap.png")
    plt.close()


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(part) for part in col if str(part))
            for col in df.columns.values
        ]
    return df


def export_timeframe_summary(dataset_df: pd.DataFrame, output_dir: Path) -> None:
    if dataset_df.empty:
        return
    if "timeframe" not in dataset_df.columns or "htf_timeframe" not in dataset_df.columns:
        return

    df = dataset_df.copy()
    df["timeframe"] = df["timeframe"].astype(str)
    df["htf_timeframe"] = df["htf_timeframe"].replace({"": "None"}).astype(str)

    metrics = [
        "NetProfit",
        "Sortino",
        "ProfitFactor",
        "MaxDD",
        "WinRate",
        "WeeklyNetProfit",
        "Trades",
    ]
    present = [metric for metric in metrics if metric in df.columns]
    if not present:
        return

    summary = (
        df.groupby(["timeframe", "htf_timeframe"], dropna=False)[present]
        .agg(["mean", "median", "max"])
        .sort_index()
    )
    if summary.empty:
        return

    summary = summary.round(6).reset_index()
    summary = _flatten_multiindex_columns(summary)
    summary.to_csv(output_dir / "results_timeframe_summary.csv", index=False)

    sort_candidates = [
        "Sortino_mean",
        "Sortino_median",
        "ProfitFactor_mean",
        "NetProfit_mean",
    ]
    sort_metric: Optional[str] = next((name for name in sort_candidates if name in summary.columns), None)
    rankings = summary.sort_values(sort_metric, ascending=False) if sort_metric else summary
    rankings.to_csv(output_dir / "results_timeframe_rankings.csv", index=False)


def generate_reports(
    results: List[Dict[str, object]],
    best: Dict[str, object],
    wf_summary: Dict[str, object],
    objectives: Iterable[object],
    output_dir: Path,
) -> None:
    agg_df, dataset_df = export_results(results, objectives, output_dir)
    export_best(best, wf_summary, output_dir)
    export_timeframe_summary(dataset_df, output_dir)

    params = list(best.get("params", {}).keys())
    metric_name = next((name for name, _ in _objective_iterator(objectives)), "NetProfit")
    plots_dir = output_dir / "plots"
    export_heatmap(agg_df, params, metric_name, plots_dir)


def write_trials_dataframe(study: optuna.study.Study, output_dir: Path) -> None:
    _ensure_dir(output_dir)
    try:
        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    except Exception:
        return
    if trials_df.empty:
        return
    trials_df.to_csv(output_dir / "trials.csv", index=False)


def write_bank_file(output_dir: Path, payload: Dict[str, object]) -> None:
    (output_dir / "bank.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
