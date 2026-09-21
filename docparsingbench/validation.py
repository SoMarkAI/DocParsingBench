import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from docparsingbench.markdown_segmenter import split_markdown


HTML_MATH_PATTERN = re.compile(r"<math\b[\s\S]*?</math>", re.IGNORECASE)
HTML_HEADING_PATTERN = re.compile(r"<h[1-6]\b[\s\S]*?</h[1-6]>", re.IGNORECASE)
FENCED_CODE_PATTERN = re.compile(r"(?:```|~~~)[\s\S]*?(?:```|~~~)")
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]*`")
HTML_TABLE_PATTERN = re.compile(r"<table\b[\s\S]*?</table>", re.IGNORECASE)


def _diagnostic_text(markdown: str) -> str:
    """Mask regions where HTML-like text is content rather than DPB structure."""
    text = FENCED_CODE_PATTERN.sub(" ", markdown)
    text = INLINE_CODE_PATTERN.sub(" ", text)
    return HTML_TABLE_PATTERN.sub(" ", text)


def validate_markdown(markdown: str) -> Dict[str, Any]:
    """Check one prediction against DPB's canonical Markdown contract."""
    segments = split_markdown(markdown)
    segment_counts = {
        kind: sum(segment.type == kind for segment in segments)
        for kind in ("text", "display_formula", "table", "image")
    }
    diagnostic_text = _diagnostic_text(markdown)
    unsupported_html_math = len(HTML_MATH_PATTERN.findall(diagnostic_text))
    noncanonical_html_headings = len(HTML_HEADING_PATTERN.findall(diagnostic_text))
    warnings: List[str] = []
    if unsupported_html_math:
        warnings.append("unsupported_html_math")
    if noncanonical_html_headings:
        warnings.append("noncanonical_html_heading")

    return {
        "empty": not markdown.strip(),
        "segment_counts": segment_counts,
        "unsupported_html_math_count": unsupported_html_math,
        "noncanonical_html_heading_count": noncanonical_html_headings,
        "warnings": warnings,
    }


def validate_prediction_directory(
    pred_dir: Path,
    *,
    gt_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if not pred_dir.is_dir():
        raise NotADirectoryError(f"Prediction directory does not exist: {pred_dir}")
    if gt_dir is not None and not gt_dir.is_dir():
        raise NotADirectoryError(f"Ground-truth directory does not exist: {gt_dir}")

    pred_index = {path.name: path for path in pred_dir.glob("*.md")}
    expected_names = (
        sorted(path.name for path in gt_dir.glob("*.md"))
        if gt_dir is not None
        else sorted(pred_index)
    )
    expected_set = set(expected_names)
    missing_files = sorted(expected_set - set(pred_index))
    extra_files = sorted(set(pred_index) - expected_set) if gt_dir is not None else []

    reports: List[Dict[str, Any]] = []
    for name in sorted(pred_index):
        report = validate_markdown(pred_index[name].read_text(encoding="utf-8"))
        report["file"] = name
        reports.append(report)

    empty_files = sorted(report["file"] for report in reports if report["empty"])
    unsupported_html_math_files = sorted(
        report["file"] for report in reports if report["unsupported_html_math_count"]
    )
    noncanonical_html_heading_files = sorted(
        report["file"] for report in reports if report["noncanonical_html_heading_count"]
    )
    error_count = len(missing_files) + len(extra_files) + len(empty_files)
    warning_count = len(unsupported_html_math_files) + len(noncanonical_html_heading_files)
    status = "error" if error_count else "warning" if warning_count else "ok"

    return {
        "summary": {
            "status": status,
            "prediction_files": len(pred_index),
            "expected_files": len(expected_names),
            "missing_files": len(missing_files),
            "extra_files": len(extra_files),
            "empty_files": len(empty_files),
            "unsupported_html_math_files": len(unsupported_html_math_files),
            "unsupported_html_math_blocks": sum(
                report["unsupported_html_math_count"] for report in reports
            ),
            "noncanonical_html_heading_files": len(noncanonical_html_heading_files),
            "noncanonical_html_heading_blocks": sum(
                report["noncanonical_html_heading_count"] for report in reports
            ),
        },
        "missing_files": missing_files,
        "extra_files": extra_files,
        "empty_files": empty_files,
        "unsupported_html_math_files": unsupported_html_math_files,
        "noncanonical_html_heading_files": noncanonical_html_heading_files,
        "reports": reports,
    }
