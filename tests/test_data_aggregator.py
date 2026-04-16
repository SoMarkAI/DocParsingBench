import json

from docparsingbench.visualize.data_aggregator import aggregate


def _write_result(path, score: float) -> None:
    payload = {
        "summary": {
            "avg_dpb": score,
            "avg_text": score,
            "avg_formula": score,
            "avg_table": score,
        },
        "reports": [
            {
                "file": "finance_banking_00001.md",
                "summary": {
                    "dpb_score": score,
                    "text_score": score,
                    "display_formula_score": score,
                    "table_score": score,
                    "weights": {"alpha": 1, "beta": 1, "gamma": 1},
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_supports_prefix_exclusion(tmp_path):
    labels = {
        "data": [
            {"md": "finance_banking_00001.md", "industry": "finance"},
        ]
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_result(results_dir / "paddle_1_5.result.json", 0.99)
    _write_result(results_dir / "dots_ocr_1_5.result.json", 0.75)
    _write_result(results_dir / "deepseek_ocr_2.result.json", 0.66)

    agg_all = aggregate(labels_path, results_dir)
    assert [m.model_name for m in agg_all.all] == ["paddle_1_5", "dots_ocr_1_5", "deepseek_ocr_2"]

    agg_filtered = aggregate(labels_path, results_dir, exclude_model_prefixes=["paddle_1_5"])
    assert [m.model_name for m in agg_filtered.all] == ["dots_ocr_1_5", "deepseek_ocr_2"]
    assert [m.model_name for m in agg_filtered.by_industry["finance"]] == ["dots_ocr_1_5", "deepseek_ocr_2"]
