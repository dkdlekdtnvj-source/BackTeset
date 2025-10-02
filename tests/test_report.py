from pathlib import Path
from typing import Optional

import importlib
import matplotlib
import pandas as pd

# 일부 테스트 환경에서 pandas 가 경량 스텁으로 로드되는 경우가 있어 DataFrame 생성이 실패할 수 있다.
# 실제 pandas 구현을 확실히 사용하도록 필요 시 재로딩한다.
if not hasattr(pd, "DataFrame"):
    pd = importlib.reload(importlib.import_module("pandas"))

from optimize.report import generate_reports

matplotlib.use("Agg")


def _make_dataset(symbol: str, timeframe: str, htf: Optional[str], metrics: dict) -> dict:
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
    dataset_metrics_d = {
        "Valid": True,
        "NetProfit": 0.18,
        "Sortino": 1.2,
        "ProfitFactor": 1.1,
        "MaxDD": -0.2,
        "WinRate": 0.48,
        "WeeklyNetProfit": 0.01,
        "Trades": 90,
    }

    results = [
        {
            "trial": 0,
            "score": 1.0,
            "params": {"utKey": 3.8, "stMode": "Bounce"},
            "metrics": {"NetProfit": 0.25, "Sortino": 1.8},
            "datasets": [
                _make_dataset("BINANCE:ENAUSDT", "1m", "15m", dataset_metrics_a),
                _make_dataset("BINANCE:ENAUSDT", "3m", "1h", dataset_metrics_b),
            ],
        },
        {
            "trial": 1,
            "score": 1.2,
            "params": {"utKey": 4.2, "stMode": "Cross"},
            "metrics": {"NetProfit": 0.3, "Sortino": 1.7},
            "datasets": [
                _make_dataset("BINANCE:ENAUSDT", "1m", "15m", dataset_metrics_c),
                _make_dataset("BINANCE:ENAUSDT", "5m", None, dataset_metrics_d),
            ],
        },
    ]

    best = {"params": {"utKey": 3.8, "stMode": "Bounce"}, "metrics": {"NetProfit": 0.25}, "score": 1.0}
    wf_summary = {}

    generate_reports(results, best, wf_summary, ["NetProfit"], tmp_path)

    summary_path = tmp_path / "results_timeframe_summary.csv"
    ranking_path = tmp_path / "results_timeframe_rankings.csv"

    assert summary_path.exists()
    assert ranking_path.exists()

    summary_df = pd.read_csv(summary_path, keep_default_na=False)
    ranking_df = pd.read_csv(ranking_path, keep_default_na=False)

    assert {"timeframe", "htf_timeframe"}.issubset(summary_df.columns)
    assert (summary_df["timeframe"] == "1m").any()
    assert "Sortino_mean" in ranking_df.columns
    assert (summary_df["htf_timeframe"] == "None").any()
    assert (ranking_df["htf_timeframe"] == "None").any()
