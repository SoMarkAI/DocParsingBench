import argparse
import json
from pathlib import Path

import pytest

import docparsingbench.cli as cli
from docparsingbench.cli import cmd_eval, UserInputError


def _write_config(path):
    path.write_text(
        "\n".join(
            [
                "skip_chemical: false",
                "drop_img: true",
                "paragraph:",
                "  match_metric: NED",
                "formula:",
                "  metric: NED",
                "  match_metric: NED",
                "table:",
                "  match_metric: NED",
            ]
        ),
        encoding="utf-8",
    )


def _write_config_skip_chemical_true(path):
    path.write_text(
        "\n".join(
            [
                "skip_chemical: true",
                "drop_img: true",
                "paragraph:",
                "  match_metric: NED",
                "formula:",
                "  metric: NED",
                "  match_metric: NED",
                "table:",
                "  match_metric: NED",
            ]
        ),
        encoding="utf-8",
    )


def test_cmd_eval_raises_when_pred_only_has_nested_md(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    nested_dir = pred_dir / "nested"
    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "result.json"

    gt_dir.mkdir()
    pred_dir.mkdir()
    nested_dir.mkdir()
    (gt_dir / "a.md").write_text("hello", encoding="utf-8")
    (nested_dir / "a.md").write_text("hello", encoding="utf-8")
    _write_config(cfg_path)

    args = argparse.Namespace(gt=str(gt_dir), pred=str(pred_dir), config=str(cfg_path), out=str(out_path))
    with pytest.raises(UserInputError) as exc_info:
        cmd_eval(args)
    msg = str(exc_info.value)
    assert "found nested .md files" in msg
    assert "Fix:" in msg
    assert "move .md files to pred_dir root" in msg


def test_cmd_eval_raises_when_pred_has_no_md(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "result.json"

    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "a.md").write_text("hello", encoding="utf-8")
    _write_config(cfg_path)

    args = argparse.Namespace(gt=str(gt_dir), pred=str(pred_dir), config=str(cfg_path), out=str(out_path))
    with pytest.raises(UserInputError) as exc_info:
        cmd_eval(args)
    msg = str(exc_info.value)
    assert "Pred dir has no top-level .md files" in msg
    assert "Fix:" in msg
    assert "prediction markdown files under pred_dir root" in msg


def test_cmd_eval_raises_when_gt_pred_have_no_common_md(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "result.json"

    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "a.md").write_text("hello", encoding="utf-8")
    (pred_dir / "b.md").write_text("hello", encoding="utf-8")
    _write_config(cfg_path)

    args = argparse.Namespace(gt=str(gt_dir), pred=str(pred_dir), config=str(cfg_path), out=str(out_path))
    with pytest.raises(UserInputError) as exc_info:
        cmd_eval(args)
    msg = str(exc_info.value)
    assert "No matched .md filenames" in msg
    assert "Fix:" in msg
    assert "same basename on both sides" in msg


def test_cmd_eval_prints_summary_after_directory_eval(tmp_path, capsys, monkeypatch):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred_model_a"
    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "result.json"
    generated = {}

    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "education_chemistry_00001.md").write_text("hello", encoding="utf-8")
    (pred_dir / "education_chemistry_00001.md").write_text("hello", encoding="utf-8")
    _write_config(cfg_path)

    def fake_generate(labels_path, results_dir, output_path, dpi=150, figsize=(16, 10), y_limits=(30.0, 100.0)):
        generated["labels_path"] = Path(labels_path)
        generated["results_dir"] = Path(results_dir)
        generated["output_path"] = Path(output_path)
        generated["y_limits"] = y_limits
        return Path(output_path)

    monkeypatch.setattr(cli, "summary_chart_generate", fake_generate)

    args = argparse.Namespace(gt=str(gt_dir), pred=str(pred_dir), config=str(cfg_path), out=str(out_path), labels=None)
    cmd_eval(args)
    stdout = capsys.readouterr().out

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    summary_line = next((line for line in lines if line.startswith("SUMMARY|")), "")
    assert summary_line
    assert summary_line.startswith("SUMMARY|")
    assert "Model=pred_model_a" in summary_line
    assert "Files=1" in summary_line
    assert "DPB=1.0000" in summary_line
    assert "Text=1.0000" in summary_line
    assert "Formula=0.0000" in summary_line
    assert "Table=0.0000" in summary_line
    assert "FormulaRenderFailures=0" in summary_line
    assert "ElapsedSec=" in summary_line
    assert f"Output={out_path}" in summary_line
    assert f"NOTE|SummaryChart=generated|Path={out_path.parent / 'summary_chart.png'}" in lines
    assert generated["results_dir"] == out_path.parent
    assert generated["output_path"] == out_path.parent / "summary_chart.png"
    assert generated["y_limits"] == (30.0, 100.0)
    assert generated["labels_path"] == tmp_path / "labels.json"
    labels_payload = json.loads(generated["labels_path"].read_text(encoding="utf-8"))
    assert labels_payload["data"][0]["md"] == "education_chemistry_00001.md"
    assert labels_payload["data"][0]["industry"] == "education"
    assert labels_payload["data"][0]["sub-industry"] == "chemistry"

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert payload["summary"]["files"] == 1


def test_cmd_eval_creates_output_parent_directory(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred_model_b"
    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "new" / "nested" / "result.json"

    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "a.md").write_text("hello", encoding="utf-8")
    (pred_dir / "a.md").write_text("hello", encoding="utf-8")
    _write_config(cfg_path)

    args = argparse.Namespace(gt=str(gt_dir), pred=str(pred_dir), config=str(cfg_path), out=str(out_path))
    cmd_eval(args)

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary"]["files"] == 1


def test_cmd_eval_reports_skip_reason_when_skip_chemical_enabled(tmp_path, capsys):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred_model_c"
    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "result.json"

    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "a.md").write_text(r"text \smiles{C}", encoding="utf-8")
    (pred_dir / "a.md").write_text("text", encoding="utf-8")
    _write_config_skip_chemical_true(cfg_path)

    args = argparse.Namespace(gt=str(gt_dir), pred=str(pred_dir), config=str(cfg_path), out=str(out_path))
    cmd_eval(args)
    stdout = capsys.readouterr().out

    assert "Skipped a.md: contains chemical formula" in stdout
    assert "(skip_chemical=true)" not in stdout
    assert "NOTE|SkippedBreakdown=" not in stdout
