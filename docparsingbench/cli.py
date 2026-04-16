import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from docparsingbench.config.schema import load_config
from docparsingbench.core import evaluate_pair, evaluate_single
from docparsingbench.labels import LabelsError, resolve_labels_path
from docparsingbench.markdown_segmenter import split_markdown, normalize_markdown

def viz_launch(*args, **kwargs):
    # Keep lazy import here to avoid importing visualization stack for non-visualize commands.
    from docparsingbench.visualize.vis_result import launch as _launch
    _launch(*args, **kwargs)

def leaderboard_html_generate(*args, **kwargs):
    # Lazy import to avoid pulling the data aggregator unless needed.
    from docparsingbench.visualize.leaderboard_html import (
        generate_leaderboard_html as _gen,
    )
    return _gen(*args, **kwargs)


def summary_chart_generate(*args, **kwargs):
    # Keep lazy import to avoid loading matplotlib for non-chart commands.
    from docparsingbench.visualize.summary_chart import generate_summary_chart as _gen
    return _gen(*args, **kwargs)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class UserInputError(ValueError):
    """Raised when CLI inputs are invalid and need user action."""


def _resolve_labels_arg(labels_arg: Any, gt_arg: Any) -> Path:
    try:
        return resolve_labels_path(labels_arg, gt_arg)
    except LabelsError as exc:
        raise UserInputError(str(exc)) from exc


def _nested_md_files(path: Path) -> List[Path]:
    return sorted(p for p in path.rglob("*.md") if p.parent != path)


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _ensure_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _print_eval_summary(
    *,
    model: str,
    files: Any,
    dpb: Any,
    text: Any,
    formula: Any,
    table: Any,
    formula_render_failures: Any,
    elapsed_sec: float,
    output: Path,
):
    parts = [
        f"Model={model}",
        f"Files={files}",
        f"DPB={_format_metric(dpb)}",
        f"Text={_format_metric(text)}",
        f"Formula={_format_metric(formula)}",
        f"Table={_format_metric(table)}",
        f"FormulaRenderFailures={formula_render_failures}",
        f"ElapsedSec={elapsed_sec:.3f}",
        f"Output={output}",
    ]
    print("SUMMARY|" + "|".join(parts))


def _prepare_eval_file_index(gt_path: Path, pred_path: Path) -> Tuple[List[Path], Dict[str, Path]]:
    gt_files = sorted(gt_path.glob("*.md"))
    pred_files = sorted(pred_path.glob("*.md"))

    if not gt_files:
        raise UserInputError(
            "\n".join(
                [
                    f"GT dir has no top-level .md files: {gt_path}",
                    "Fix: put GT markdown files directly under gt_dir (e.g. gt_dir/a.md).",
                ]
            )
        )

    if not pred_files:
        nested = _nested_md_files(pred_path)
        if nested:
            examples = ", ".join(str(p.relative_to(pred_path)) for p in nested[:3])
            raise UserInputError(
                "\n".join(
                    [
                        f"Pred dir is invalid: found nested .md files ({examples}).",
                        "Fix: move .md files to pred_dir root (e.g. pred_dir/a.md), then rerun eval.",
                    ]
                )
            )
        raise UserInputError(
            "\n".join(
                [
                    f"Pred dir has no top-level .md files: {pred_path}",
                    "Fix: put prediction markdown files under pred_dir root (e.g. pred_dir/a.md), then rerun eval.",
                ]
            )
        )

    pred_index = {p.name: p for p in pred_files}
    if not any(g.name in pred_index for g in gt_files):
        raise UserInputError(
            "\n".join(
                [
                    "No matched .md filenames between gt_dir and pred_dir.",
                    "Fix: ensure same basename on both sides (e.g. gt_dir/a.md and pred_dir/a.md).",
                ]
            )
        )

    return gt_files, pred_index


def cmd_visualize(args):
    labels_path = _resolve_labels_arg(getattr(args, "labels", None), getattr(args, "gt", None))
    viz_launch(str(labels_path), args.img, args.gt, args.pred, args.result)

def cmd_leaderboard_html(args):
    labels_path = _resolve_labels_arg(getattr(args, "labels", None), getattr(args, "gt", None))
    exclude_model_prefixes = list(getattr(args, "exclude_model_prefix", []) or [])
    out_path = leaderboard_html_generate(
        labels_path=labels_path,
        results_dir=Path(args.results),
        output_path=Path(args.output),
        exclude_model_prefixes=exclude_model_prefixes or None,
    )
    print(f"Wrote standalone leaderboard HTML → {out_path}")


def cmd_summary_chart(args):
    labels_path = _resolve_labels_arg(getattr(args, "labels", None), getattr(args, "gt", None))
    exclude_model_prefixes = list(getattr(args, "exclude_model_prefix", []) or [])
    y_min = float(getattr(args, "y_min", 30.0))
    y_max = float(getattr(args, "y_max", 100.0))
    if y_min >= y_max:
        raise UserInputError(f"Invalid summary chart y-limits: y_min ({y_min}) must be smaller than y_max ({y_max}).")
    chart_kwargs = {}
    if exclude_model_prefixes:
        chart_kwargs["exclude_model_prefixes"] = exclude_model_prefixes
    chart_kwargs["y_limits"] = (y_min, y_max)
    summary_chart_generate(
        labels_path=labels_path,
        results_dir=Path(args.results),
        output_path=Path(args.output),
        dpi=args.dpi,
        **chart_kwargs,
    )


def cmd_eval(args):
    eval_started = time.perf_counter()
    cfg = load_config(args.config)
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    out_path = _ensure_output_path(Path(args.out))
    summary_kwargs = None
    reports = []
    resolved_labels_path = None
    if gt_path.is_dir() and getattr(args, "labels", None):
        resolved_labels_path = _resolve_labels_arg(getattr(args, "labels", None), gt_path)
    if gt_path.is_dir() and pred_path.is_dir():
        placeholder = cfg.paragraph.formula_placeholder
        vis_dir = str(out_path.parent / "cdm_vis") if cfg.formula.metric == "CDM" and cfg.visualize else None
        gt_component_index = {}
        gt_files, pred_index = _prepare_eval_file_index(gt_path, pred_path)
        for g in gt_files:
            p = pred_index.get(g.name)
            if not p:
                continue
            gt_md = _read_text(g)
            pred_md = _read_text(p)
            gt_segments = split_markdown(gt_md, placeholder=placeholder, drop_img=cfg.drop_img)
            gt_components = {
                "text": any(s.type == "text" for s in gt_segments),
                "display_formula": any(s.type == "display_formula" for s in gt_segments),
                "table": any(s.type == "table" for s in gt_segments),
            }
            gt_component_index[g.name] = gt_components
            report = evaluate_pair(gt_md, pred_md, cfg, vis_dir=vis_dir, vis_prefix=g.stem)
            report["file"] = g.name
            if report.get("skipped"):
                print(f"Skipped {g.name}: {report.get('reason')}")
            reports.append(report)
        
        valid_reports = [r for r in reports if not r.get("skipped")]
        text_reports = [r for r in valid_reports if gt_component_index.get(r.get("file", ""), {}).get("text")]
        formula_reports = [r for r in valid_reports if gt_component_index.get(r.get("file", ""), {}).get("display_formula")]
        table_reports = [r for r in valid_reports if gt_component_index.get(r.get("file", ""), {}).get("table")]
        summary = {
            "files": len(valid_reports),
            "avg_dpb": (sum(r["summary"]["dpb_score"] for r in valid_reports) / len(valid_reports)) if valid_reports else 0.0,
            "avg_text": (sum(r["summary"]["text_score"] for r in text_reports) / len(text_reports)) if text_reports else 0.0,
            "avg_formula": (sum(r["summary"]["display_formula_score"] for r in formula_reports) / len(formula_reports)) if formula_reports else 0.0,
            "avg_table": (sum(r["summary"]["table_score"] for r in table_reports) / len(table_reports)) if table_reports else 0.0,
            "formula_render_failures": sum(r.get("formula_render_failures", 0) for r in valid_reports),
        }
        out = {"summary": summary, "reports": reports}
        summary_kwargs = dict(
            model=pred_path.name,
            files=summary["files"],
            dpb=summary["avg_dpb"],
            text=summary["avg_text"],
            formula=summary["avg_formula"],
            table=summary["avg_table"],
            formula_render_failures=summary["formula_render_failures"],
            output=out_path,
        )
    else:
        report = evaluate_pair(_read_text(gt_path), _read_text(pred_path), cfg)
        out = report
        if report.get("skipped"):
            summary_kwargs = dict(
                model=pred_path.stem if pred_path.suffix else pred_path.name,
                files=0,
                dpb=None,
                text=None,
                formula=None,
                table=None,
                formula_render_failures=0,
                output=out_path,
            )
        else:
            summary = report.get("summary", {})
            summary_kwargs = dict(
                model=pred_path.stem if pred_path.suffix else pred_path.name,
                files=1,
                dpb=summary.get("dpb_score"),
                text=summary.get("text_score"),
                formula=summary.get("display_formula_score"),
                table=summary.get("table_score"),
                formula_render_failures=report.get("formula_render_failures", 0),
                output=out_path,
            )
    elapsed = time.perf_counter() - eval_started
    summary_kwargs["elapsed_sec"] = elapsed
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_eval_summary(**summary_kwargs)
    if gt_path.is_dir():
        if not cfg.summary_chart.enable:
            print("NOTE|SummaryChart=skipped|Reason=disabled_in_config")
        else:
            if resolved_labels_path is None:
                try:
                    resolved_labels_path = resolve_labels_path(None, gt_path)
                except LabelsError as exc:
                    error_text = " ".join(str(exc).splitlines())
                    print(f"NOTE|SummaryChart=skipped|Reason=labels_auto_generate_failed|Error={error_text}")
                    return
            chart_path = out_path.parent / cfg.summary_chart.output_path
            try:
                y_limits = (cfg.summary_chart.y_min, cfg.summary_chart.y_max)
                if y_limits[0] >= y_limits[1]:
                    raise UserInputError(
                        f"Invalid summary_chart y-limits in config: y_min ({y_limits[0]}) must be smaller than y_max ({y_limits[1]})."
                    )
                summary_chart_generate(
                    labels_path=resolved_labels_path,
                    results_dir=out_path.parent,
                    output_path=chart_path,
                    y_limits=y_limits,
                )
                print(f"NOTE|SummaryChart=generated|Path={chart_path}")
            except Exception as exc:
                # Chart generation should not fail the eval pipeline.
                print(f"NOTE|SummaryChart=failed|Error={exc}")


def cmd_segment(args):
    md = Path(args.infile).read_text(encoding="utf-8")
    segs = split_markdown(md)
    payload = [{"type": s.type, "raw": s.raw, "text_no_formula": s.text_no_formula, "inline_formulas": s.inline_formulas} for s in segs]
    out_path = _ensure_output_path(Path(args.out))
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _segment_payload(md: str, drop_img: bool) -> List[Dict[str, Any]]:
    segs = split_markdown(md, drop_img=drop_img)
    return [
        {
            "type": s.type,
            "raw": s.raw,
            "text_no_formula": s.text_no_formula,
            "inline_formulas": s.inline_formulas,
        }
        for s in segs
    ]


def _render_segments_md(segs: List[Dict[str, Any]]) -> str:
    if not segs:
        return "_No segments_"
    lines = []
    for idx, seg in enumerate(segs):
        lines.append(f"{idx}. `{seg['type']}`: `{seg['raw']}`")
    return "\n".join(lines)


def cmd_segment_report(args):
    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    out_path = _ensure_output_path(Path(args.out))
    if args.files:
        files = args.files
    else:
        gt_names = {p.name for p in gt_dir.glob("*.md")}
        pred_names = {p.name for p in pred_dir.glob("*.md")}
        files = sorted(gt_names & pred_names)

    sections = ["# Bad Case Segmented Results", ""]
    for name in files:
        gt_path = gt_dir / name
        pred_path = pred_dir / name
        if not gt_path.exists() or not pred_path.exists():
            continue

        gt_raw = _read_text(gt_path)
        pred_raw = _read_text(pred_path)
        gt_normalized = normalize_markdown(gt_raw)
        pred_normalized = normalize_markdown(pred_raw)
        gt_segments = _segment_payload(gt_raw, drop_img=True)
        pred_segments = _segment_payload(pred_raw, drop_img=True)

        sections.extend(
            [
                f"## `{name}`",
                "",
                "### GT Raw Excerpt",
                "```md",
                gt_raw.rstrip(),
                "```",
                "",
                "### Pred Raw Excerpt",
                "```md",
                pred_raw.rstrip(),
                "```",
                "",
                "### GT After Normalization",
                "```md",
                gt_normalized.rstrip(),
                "```",
                "",
                "### Pred After Normalization",
                "```md",
                pred_normalized.rstrip(),
                "```",
                "",
                "### GT Final Segments (drop_img=True)",
                _render_segments_md(gt_segments),
                "",
                "### Pred Final Segments (drop_img=True)",
                _render_segments_md(pred_segments),
                "",
            ]
        )

    out_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def _cmd_single(args, kind: str):
    cfg = load_config(args.config)
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    out_path = _ensure_output_path(Path(args.out))
    if gt_path.is_dir() and pred_path.is_dir():
        reports = []
        pred_index = {p.name: p for p in pred_path.glob("*.md")}
        for g in gt_path.glob("*.md"):
            p = pred_index.get(g.name)
            if not p:
                continue
            r = evaluate_single(_read_text(g), _read_text(p), cfg, kind)
            r["file"] = g.name
            reports.append(r)
        summary = {
            "files": len(reports),
            "avg_score": (sum(rr.get("score", 0.0) for rr in reports) / len(reports)) if reports else 0.0,
            "type": kind,
        }
        out = {"summary": summary, "reports": reports}
    else:
        report = evaluate_single(_read_text(gt_path), _read_text(pred_path), cfg, kind)
        out = report
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(prog="dpb", description="DocParsingBench CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="Evaluate Markdown parsing quality")
    p_eval.add_argument("--gt", required=True)
    p_eval.add_argument("--pred", required=True)
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--out", required=True)
    p_eval.add_argument(
        "--labels",
        default=None,
        help="Optional path to labels.json. When omitted, labels are auto-generated next to the GT directory for batch eval.",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_seg = sub.add_parser("segment", help="Segment markdown and extract inline formulas")
    p_seg.add_argument("--in", dest="infile", required=True)
    p_seg.add_argument("--out", required=True)
    p_seg.set_defaults(func=cmd_segment)

    p_seg_report = sub.add_parser("segment-report", help="Generate a markdown report of normalized and final segmentation results")
    p_seg_report.add_argument("--gt", required=True, help="Path to ground truth markdown directory")
    p_seg_report.add_argument("--pred", required=True, help="Path to predicted markdown directory")
    p_seg_report.add_argument("--out", required=True, help="Path to output markdown report")
    p_seg_report.add_argument("files", nargs="*", help="Markdown filenames to include")
    p_seg_report.set_defaults(func=cmd_segment_report)

    p_viz = sub.add_parser("visualize", help="Visualize evaluation results")
    p_viz.add_argument("--labels", default=None, help="Optional path to labels.json")
    p_viz.add_argument("--img", required=True, help="Path to image directory")
    p_viz.add_argument("--gt", required=True, help="Path to ground truth markdown directory")
    p_viz.add_argument("--pred", required=True, help="Path to predicted markdown directory")
    p_viz.add_argument("--result", required=True, help="Path to result.json from eval command")
    p_viz.set_defaults(func=cmd_visualize)

    p_text = sub.add_parser("text", help="Evaluate text paragraphs only")
    p_text.add_argument("--gt", required=True)
    p_text.add_argument("--pred", required=True)
    p_text.add_argument("--config", required=True)
    p_text.add_argument("--out", required=True)
    p_text.set_defaults(func=lambda args: _cmd_single(args, "text"))

    p_formula = sub.add_parser("formula", help="Evaluate formulas only")
    p_formula.add_argument("--gt", required=True)
    p_formula.add_argument("--pred", required=True)
    p_formula.add_argument("--config", required=True)
    p_formula.add_argument("--out", required=True)
    p_formula.set_defaults(func=lambda args: _cmd_single(args, "formula"))

    p_table = sub.add_parser("table", help="Evaluate tables only")
    p_table.add_argument("--gt", required=True)
    p_table.add_argument("--pred", required=True)
    p_table.add_argument("--config", required=True)
    p_table.add_argument("--out", required=True)
    p_table.set_defaults(func=lambda args: _cmd_single(args, "table"))

    p_lbh = sub.add_parser(
        "leaderboard-html",
        help="Generate a standalone, shareable leaderboard HTML file",
    )
    p_lbh.add_argument("--labels", default=None, help="Optional path to labels.json")
    p_lbh.add_argument(
        "--gt",
        default=None,
        help="Path to ground truth markdown directory or dataset root. Required when --labels is omitted.",
    )
    p_lbh.add_argument("--results", required=True, help="Path to results directory containing *.result.json")
    p_lbh.add_argument("--output", required=True, help="Output HTML path")
    p_lbh.add_argument(
        "--exclude-model-prefix",
        action="append",
        default=[],
        help="Exclude models whose result filename starts with this prefix. Repeatable.",
    )
    p_lbh.set_defaults(func=cmd_leaderboard_html)

    p_chart = sub.add_parser("summary-chart", help="Generate summary bar chart from evaluation results")
    p_chart.add_argument("--labels", default=None, help="Optional path to labels.json")
    p_chart.add_argument(
        "--gt",
        default=None,
        help="Path to ground truth markdown directory or dataset root. Required when --labels is omitted.",
    )
    p_chart.add_argument("--results", required=True, help="Path to results directory containing *.result.json")
    p_chart.add_argument("--output", required=True, help="Output PNG path")
    p_chart.add_argument("--dpi", type=int, default=150, help="Output DPI (default: 150)")
    p_chart.add_argument("--y-min", type=float, default=30.0, help="Summary chart y-axis minimum (default: 30)")
    p_chart.add_argument("--y-max", type=float, default=100.0, help="Summary chart y-axis maximum (default: 100)")
    p_chart.add_argument(
        "--exclude-model-prefix",
        action="append",
        default=[],
        help="Exclude models whose result filename starts with this prefix. Repeatable.",
    )
    p_chart.set_defaults(func=cmd_summary_chart)

    args = parser.parse_args()
    try:
        args.func(args)
    except UserInputError as exc:
        parser.exit(2, f"\n[DPB Input Error]\n{exc}\n")


if __name__ == "__main__":
    main()
