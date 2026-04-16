import argparse
import json
from pathlib import Path

import pytest

import docparsingbench.cli as cli
from docparsingbench.cli import UserInputError


def _make_dataset(tmp_path):
    dataset_dir = tmp_path / "DocParsingBench"
    gt_dir = dataset_dir / "markdowns"
    results_dir = tmp_path / "results"
    gt_dir.mkdir(parents=True)
    results_dir.mkdir()
    (gt_dir / "education_chemistry_00001.md").write_text("hello", encoding="utf-8")
    return dataset_dir, gt_dir, results_dir


def test_cmd_summary_chart_generates_labels_when_omitted(tmp_path, monkeypatch):
    dataset_dir, gt_dir, results_dir = _make_dataset(tmp_path)
    captured = {}

    def fake_generate(labels_path, results_dir, output_path, dpi=150, figsize=(16, 10), y_limits=(30.0, 100.0)):
        captured["labels_path"] = Path(labels_path)
        captured["results_dir"] = Path(results_dir)
        captured["output_path"] = Path(output_path)
        captured["dpi"] = dpi
        captured["y_limits"] = y_limits
        return Path(output_path)

    monkeypatch.setattr(cli, "summary_chart_generate", fake_generate)

    args = argparse.Namespace(
        labels=None,
        gt=str(gt_dir),
        results=str(results_dir),
        output=str(tmp_path / "chart.png"),
        dpi=200,
        y_min=35.0,
        y_max=98.0,
    )

    cli.cmd_summary_chart(args)

    labels_path = captured["labels_path"]
    assert labels_path == dataset_dir / "labels.json"
    assert captured["results_dir"] == results_dir
    assert captured["output_path"] == tmp_path / "chart.png"
    assert captured["dpi"] == 200
    assert captured["y_limits"] == (35.0, 98.0)

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    assert payload["label_schema"] == [
        {
            "industry": "education",
            "sub_industries": [{"sub-industry": "chemistry"}],
        }
    ]
    assert payload["data"] == [
        {
            "img": "",
            "md": "education_chemistry_00001.md",
            "industry": "education",
            "sub-industry": "chemistry",
        }
    ]


def test_cmd_summary_chart_errors_for_missing_explicit_labels(tmp_path):
    _, gt_dir, results_dir = _make_dataset(tmp_path)
    args = argparse.Namespace(
        labels=str(tmp_path / "missing-labels.json"),
        gt=str(gt_dir),
        results=str(results_dir),
        output=str(tmp_path / "chart.png"),
        dpi=150,
        y_min=30.0,
        y_max=100.0,
    )

    with pytest.raises(UserInputError) as exc_info:
        cli.cmd_summary_chart(args)

    assert "Explicit labels path does not exist" in str(exc_info.value)


def test_cmd_summary_chart_errors_for_invalid_y_limits(tmp_path):
    _, gt_dir, results_dir = _make_dataset(tmp_path)
    args = argparse.Namespace(
        labels=None,
        gt=str(gt_dir),
        results=str(results_dir),
        output=str(tmp_path / "chart.png"),
        dpi=150,
        y_min=100.0,
        y_max=30.0,
    )

    with pytest.raises(UserInputError) as exc_info:
        cli.cmd_summary_chart(args)

    assert "y_min (100.0) must be smaller than y_max (30.0)" in str(exc_info.value)
