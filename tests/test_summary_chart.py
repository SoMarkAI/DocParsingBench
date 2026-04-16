import pytest

from docparsingbench.visualize.data_aggregator import AggregatedResult, ModelScores
from docparsingbench.visualize.summary_chart import generate_summary_chart_from_agg


def _make_models(n: int) -> list:
    models = []
    for i in range(n):
        models.append(ModelScores(
            model_name=f"model_{i}",
            display_name=f"Model {i}",
            dpb=0.9 - i * 0.05,
            text=0.85 - i * 0.04,
            formula=0.80 - i * 0.03,
            table=0.75 - i * 0.02,
        ))
    return models


def _make_agg(n: int) -> AggregatedResult:
    models = _make_models(n)
    return AggregatedResult(
        all=models,
        by_industry={},
        industries=[],
        best_per_col={"dpb": models[0].display_name, "text": models[0].display_name,
                      "formula": models[0].display_name, "table": models[0].display_name},
    )


def test_chart_generates_png(tmp_path):
    agg = _make_agg(5)
    out = tmp_path / "chart.png"
    result = generate_summary_chart_from_agg(agg, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_chart_single_model(tmp_path):
    agg = _make_agg(1)
    out = tmp_path / "single.png"
    result = generate_summary_chart_from_agg(agg, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_chart_output_path_returned(tmp_path):
    agg = _make_agg(3)
    out = tmp_path / "sub" / "chart.png"
    result = generate_summary_chart_from_agg(agg, out)
    assert result == out
