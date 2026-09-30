from types import SimpleNamespace

import pytest

from docparsingbench.cli import cmd_validate
from docparsingbench.validation import validate_markdown, validate_prediction_directory


def test_validate_markdown_flags_noncanonical_structure():
    result = validate_markdown(
        '<h3>Discussion</h3>\n\n<math display="block">x + y</math>\n\n'
        "<table><tr><td>A</td></tr></table>"
    )

    assert result["unsupported_html_math_count"] == 1
    assert result["noncanonical_html_heading_count"] == 1
    assert result["embedded_display_math_count"] == 0
    assert result["segment_counts"]["display_formula"] == 0
    assert result["segment_counts"]["table"] == 1
    assert result["warnings"] == [
        "unsupported_html_math",
        "noncanonical_html_heading",
    ]


def test_validate_markdown_ignores_html_examples_and_table_cells():
    result = validate_markdown(
        '```html\n<h3>Example</h3>\n<math>x</math>\n```\n\n'
        "<table><tr><td><math>x</math></td></tr></table>"
    )

    assert result["unsupported_html_math_count"] == 0
    assert result["noncanonical_html_heading_count"] == 0
    assert result["embedded_display_math_count"] == 0


def test_validate_markdown_flags_complete_display_math_embedded_in_text_block():
    result = validate_markdown("Introduction\n$$\nx + y\n$$\n(6)\n")

    assert result["segment_counts"]["display_formula"] == 0
    assert result["embedded_display_math_count"] == 1
    assert result["warnings"] == ["embedded_display_math"]


def test_validate_markdown_accepts_isolated_multiline_display_math():
    result = validate_markdown("Introduction\n\n$$\nx + y\n$$\n\n(6)\n")

    assert result["segment_counts"]["display_formula"] == 1
    assert result["embedded_display_math_count"] == 0
    assert result["warnings"] == []


def test_validate_markdown_does_not_pair_stray_delimiters_across_paragraphs():
    result = validate_markdown("first fragment $$\n\nsecond fragment $$\n")

    assert result["embedded_display_math_count"] == 0
    assert result["warnings"] == []


def test_validate_prediction_directory_checks_coverage_and_empty_files(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "a.md").write_text("alpha", encoding="utf-8")
    (gt_dir / "b.md").write_text("beta", encoding="utf-8")
    (pred_dir / "a.md").write_text("", encoding="utf-8")
    (pred_dir / "extra.md").write_text("## Extra\n\n$$x$$", encoding="utf-8")

    result = validate_prediction_directory(pred_dir, gt_dir=gt_dir)

    assert result["summary"]["status"] == "error"
    assert result["missing_files"] == ["b.md"]
    assert result["extra_files"] == ["extra.md"]
    assert result["empty_files"] == ["a.md"]


def test_validate_prediction_directory_treats_extra_files_as_error(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "a.md").write_text("alpha", encoding="utf-8")
    (pred_dir / "a.md").write_text("alpha", encoding="utf-8")
    (pred_dir / "extra.md").write_text("extra", encoding="utf-8")

    result = validate_prediction_directory(pred_dir, gt_dir=gt_dir)

    assert result["summary"]["status"] == "error"
    assert result["extra_files"] == ["extra.md"]


def test_validate_prediction_directory_summarizes_embedded_display_math(tmp_path):
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    (pred_dir / "a.md").write_text("$$\nx + y\n$$\n(6)\n", encoding="utf-8")

    result = validate_prediction_directory(pred_dir)

    assert result["summary"]["status"] == "warning"
    assert result["summary"]["embedded_display_math_files"] == 1
    assert result["summary"]["embedded_display_math_blocks"] == 1
    assert result["embedded_display_math_files"] == ["a.md"]


def test_cmd_validate_returns_nonzero_for_coverage_errors(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    out_path = tmp_path / "validation.json"
    gt_dir.mkdir()
    pred_dir.mkdir()
    (gt_dir / "a.md").write_text("alpha", encoding="utf-8")

    with pytest.raises(SystemExit, match="1"):
        cmd_validate(
            SimpleNamespace(pred=str(pred_dir), gt=str(gt_dir), out=str(out_path))
        )

    assert out_path.is_file()
