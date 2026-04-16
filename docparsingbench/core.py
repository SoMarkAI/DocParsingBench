import json
import time
from typing import Dict, Any, List

from docparsingbench.config.schema import Config
from docparsingbench.markdown_segmenter import split_markdown, Segment, extract_inline_formulas
from docparsingbench.evaluation.segment_compare import match_and_score, FormulaSimilarityEngine
from docparsingbench.evaluation.dpb import dpb_aggregate
from docparsingbench.metrics.text_distance import ned
from docparsingbench.metrics.teds import calculate_teds, calculate_teds_structure_only


def evaluate_pair(gt_md: str, pred_md: str, cfg: Config, vis_dir: str = None, vis_prefix: str = None) -> Dict[str, Any]:
    if cfg.skip_chemical and r"\smiles{" in gt_md:
        return {"skipped": True, "reason": "contains chemical formula"}

    perf = {}
    t0 = time.time()
    placeholder = cfg.paragraph.formula_placeholder
    gt_segments = split_markdown(gt_md, placeholder=placeholder, drop_img=cfg.drop_img)
    pred_segments = split_markdown(pred_md, placeholder=placeholder, drop_img=cfg.drop_img)
    perf["segment_ms"] = (time.time() - t0) * 1000

    def _filter(segments: List[Segment], kind: str) -> List[Segment]:
        return [s for s in segments if s.type == kind]

    text_gt = _filter(gt_segments, "text")
    text_pred = _filter(pred_segments, "text")
    formula_gt = _filter(gt_segments, "display_formula")
    formula_pred = _filter(pred_segments, "display_formula")
    table_gt = _filter(gt_segments, "table")
    table_pred = _filter(pred_segments, "table")

    t1 = time.time()
    text_res = match_and_score(
        text_gt,
        text_pred,
        kind="text",
        alpha=cfg.DPB.alpha,
        paragraph_match_metric=cfg.paragraph.match_metric,
        formula_match_metric=cfg.formula.match_metric,
        chromedriver_path=cfg.chromedriver_path,
        progress_enable=cfg.progress.enable,
    )
    formula_res = match_and_score(
        formula_gt,
        formula_pred,
        kind="display_formula",
        formula_match_metric=cfg.formula.match_metric,
        formula_score_metric=cfg.formula.metric,
        chromedriver_path=cfg.chromedriver_path,
        progress_enable=cfg.progress.enable,
        vis_dir=vis_dir,
        vis_prefix=vis_prefix,
    )
    table_res = match_and_score(
        table_gt,
        table_pred,
        kind="table",
        table_match_metric=cfg.table.match_metric,
        chromedriver_path=cfg.chromedriver_path,
        progress_enable=cfg.progress.enable,
    )
    perf["match_ms"] = (time.time() - t1) * 1000

    a = cfg.DPB.alpha
    b = cfg.DPB.beta
    g = cfg.DPB.gamma
    text_present = (len(text_gt) > 0) or (len(text_pred) > 0)
    formula_present = (len(formula_gt) > 0) or (len(formula_pred) > 0)
    table_present = (len(table_gt) > 0) or (len(table_pred) > 0)
    denom = (a if text_present else 0.0) + (b if formula_present else 0.0) + (g if table_present else 0.0)
    if denom > 0.0:
        sa = (a if text_present else 0.0) / denom
        sb = (b if formula_present else 0.0) / denom
        sg = (g if table_present else 0.0) / denom
    else:
        sa, sb, sg = a, b, g
    result = dpb_aggregate(text_res.scores, formula_res.scores, table_res.scores, sa, sb, sg)
    out = {
        "summary": result,
        "matches": {
            "text": {"pairs": text_res.pairs, "scores": text_res.scores, "detail": text_res.detail},
            "display_formula": {"pairs": formula_res.pairs, "scores": formula_res.scores, "detail": formula_res.detail},
            "table": {"pairs": table_res.pairs, "scores": table_res.scores, "detail": table_res.detail},
        },
        "counts": {"gt": len(gt_segments), "pred": len(pred_segments)},
        "formula_render_failures": formula_res.render_failures,
    }
    if cfg.perf.enable:
        out["perf"] = perf
    return out


def evaluate_single(gt: str, pred: str, cfg: Config, type: str) -> Dict[str, Any]:
    if type == "text":
        perf = {}
        t0 = time.time()
        placeholder = cfg.paragraph.formula_placeholder
        gt_segments = split_markdown(gt or "", placeholder=placeholder)
        pred_segments = split_markdown(pred or "", placeholder=placeholder)
        text_gt = [s for s in gt_segments if s.type == "text"]
        text_pred = [s for s in pred_segments if s.type == "text"]
        perf["segment_ms"] = (time.time() - t0) * 1000
        t1 = time.time()
        res = match_and_score(
            text_gt,
            text_pred,
            kind="text",
            alpha=cfg.DPB.alpha,
            paragraph_match_metric=cfg.paragraph.match_metric,
            formula_match_metric=cfg.formula.metric,
            chromedriver_path=cfg.chromedriver_path,
            progress_enable=cfg.progress.enable,
        )
        perf["match_ms"] = (time.time() - t1) * 1000
        avg = (sum(res.scores) / len(res.scores)) if res.scores else 0.0
        out = {
            "type": "text",
            "score": avg,
            "matches": {"pairs": res.pairs, "scores": res.scores, "detail": res.detail},
            "counts": {"gt": len(text_gt), "pred": len(text_pred)},
        }
        if cfg.perf.enable:
            out["perf"] = perf
        return out
    elif type == "formula":
        fm = cfg.formula.metric
        if fm == "CDM":
            engine = FormulaSimilarityEngine("CDM", chromedriver_path=cfg.chromedriver_path)
            result = engine.compute(gt or "", pred or "", visualize=False)
            if result.metric_used == "CDM":
                return {
                    "type": "formula",
                    "score": float(result.sim),
                    "detail": {
                        "metric": "CDM",
                        "metric_used": "CDM",
                        "f1": float(result.sim),
                        "recall": float(result.recall or 0.0),
                        "precision": float(result.precision or 0.0),
                    },
                }
            return {
                "type": "formula",
                "score": float(result.sim),
                "detail": {
                    "metric": "NED",
                    "metric_used": result.metric_used,
                    "sim": float(result.sim),
                    "fallback_from": result.fallback_from,
                    "fallback_reason": result.fallback_reason,
                    "render_failure_delta": result.render_failure_delta,
                },
            }
        else:
            sim = 1.0 - ned(gt or "", pred or "")
            return {
                "type": "formula",
                "score": sim,
                "detail": {
                    "metric": "NED",
                    "metric_used": "NED",
                    "sim": sim,
                },
            }
    elif type == "table":
        tm = cfg.table.match_metric
        if tm == "TEDS":
            sim = calculate_teds(gt or "", pred or "")
            return {
                "type": "table",
                "score": sim,
                "detail": {
                    "metric": "TEDS",
                    "sim": sim,
                    "structure_only": False,
                },
            }
        elif tm == "TEDS-S":
            sim = calculate_teds_structure_only(gt or "", pred or "")
            return {
                "type": "table",
                "score": sim,
                "detail": {
                    "metric": "TEDS-S",
                    "sim": sim,
                    "structure_only": True,
                },
            }
        else:
            sim = 1.0 - ned(gt or "", pred or "")
            return {
                "type": "table",
                "score": sim,
                "detail": {
                    "metric": "NED",
                    "sim": sim,
                    "structure_only": False,
                },
            }
    else:
        return {"type": type, "score": 0.0, "detail": {}}
