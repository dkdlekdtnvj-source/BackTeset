from pathlib import Path

import matplotlib
import pandas as pd

from optimize.report import generate_reports

matplotlib.use("Agg")


def _make_dataset(symbol: str, timeframe: str, htf: str, metrics: dict) -> dict:
    meta = {
        "symbol": symbol,
        "source_symbol": symbol,
        "timeframe": timeframe,
        "from": "2024-01-01",
        "to": "2025-09-25",
        "htf_timeframe": htf,
    }
    payload = {"name": f"{symbol}_{timeframe}_{htf}", "meta": meta, "metrics": metrics}
    return payload


def test_generate_reports_emits_timeframe_summary(tmp_path: Path) -> None:
    dataset_metrics_a = {
        "Valid": True,
        "NetProfit": 0.25,
        "Sortino": 1.8,
        "ProfitFactor": 1.6,
        "MaxDD": -0.12,
        "WinRate": 0.58,
        "WeeklyNetProfit": 0.015,
        "Trades": 140,
    }
    dataset_metrics_b = {
        "Valid": True,
        "NetProfit": 0.42,
        "Sortino": 2.1,
        "ProfitFactor": 1.9,
        "MaxDD": -0.09,
        "WinRate": 0.61,
        "WeeklyNetProfit": 0.019,
        "Trades": 128,
    }
    dataset_metrics_c = {
        "Valid": True,
        "NetProfit": 0.3,
        "Sortino": 1.6,
        "ProfitFactor": 1.4,
        "MaxDD": -0.15,
        "WinRate": 0.52,
        "WeeklyNetProfit": 0.012,
        "Trades": 150,
    }

    results = [
        {
            "trial": 0,
            "score": 1.0,
            "params": {"oscLen": 12, "statThreshold": 38.0},
            "metrics": {"NetProfit": 0.25, "Sortino": 1.8},
            "datasets": [
                _make_dataset("BINANCE:ENAUSDT", "1m", "15m", dataset_metrics_a),
                _make_dataset("BINANCE:ENAUSDT", "3m", "1h", dataset_metrics_b),
            ],
        },
        {
            "trial": 1,
            "score": 1.2,
            "params": {"oscLen": 14, "statThreshold": 42.0},
            "metrics": {"NetProfit": 0.3, "Sortino": 1.7},
            "datasets": [
                _make_dataset("BINANCE:ENAUSDT", "1m", "15m", dataset_metrics_c),
            ],
        },
    ]

    best = {"params": {"oscLen": 12, "statThreshold": 38.0}, "metrics": {"NetProfit": 0.25}, "score": 1.0}
    wf_summary = {}

    generate_reports(results, best, wf_summary, ["NetProfit"], tmp_path)

    summary_path = tmp_path / "results_timeframe_summary.csv"
    ranking_path = tmp_path / "results_timeframe_rankings.csv"

    assert summary_path.exists()
    assert ranking_path.exists()

    summary_df = pd.read_csv(summary_path)
    ranking_df = pd.read_csv(ranking_path)

    assert {"timeframe", "htf_timeframe"}.issubset(summary_df.columns)
    assert (summary_df["timeframe"] == "1m").any()
    assert "Sortino_mean" in ranking_df.columns
