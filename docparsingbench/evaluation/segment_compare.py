import os
import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from docparsingbench.markdown_segmenter import Segment
from docparsingbench.metrics.text_distance import ned
from docparsingbench.metrics.cdm import CDM
from docparsingbench.metrics.teds import calculate_teds, calculate_teds_structure_only
from docparsingbench.hungarian import linear_sum_assignment
from tqdm import tqdm


@dataclass
class CompareResult:
    """Matching and scoring result.
    - `pairs`: list of index tuples (gt_index, pred_index)
    - `scores`: similarity scores corresponding to each pair ([0,1])
    - `detail`: detail dict for each pair, including metric used and intermediate scores
    - `render_failures`: number of CDM formula render failures (including KaTeX red errors)
    """
    pairs: List[Tuple[int, int]]
    scores: List[float]
    detail: List[Dict[str, Any]]
    render_failures: int = 0


@dataclass
class FormulaScoreResult:
    sim: float
    metric_used: str
    recall: Optional[float] = None
    precision: Optional[float] = None
    vis_img: Any = None
    fallback_from: Optional[str] = None
    fallback_reason: Optional[str] = None
    render_failure_delta: int = 0


class CDMInitializationError(RuntimeError):
    """Raised when CDM backend cannot be initialized."""


class FormulaSimilarityEngine:
    def __init__(self, formula_metric: str, chromedriver_path: Optional[str] = None):
        assert formula_metric in {"CDM", "NED"}, f"Unsupported formula.metric: {formula_metric}"
        self.formula_metric = formula_metric
        self.chromedriver_path = chromedriver_path
        self._cdm: Optional[CDM] = None
        self._cdm_unavailable: bool = False
        self._cdm_init_err: Optional[str] = None
        self.render_failures: int = 0

    @staticmethod
    def _warn_fallback(reason: str, render_failure_delta: int = 0):
        extra = f" render_failure_delta={render_failure_delta}." if render_failure_delta > 0 else ""
        warnings.warn(
            f"CDM fallback to NED for current formula pair (reason={reason}).{extra}",
            RuntimeWarning,
            stacklevel=3,
        )

    def _raise_init_error(self):
        detail = f" Original error: {self._cdm_init_err}" if self._cdm_init_err else ""
        raise CDMInitializationError(
            "CDM initialization failed. Please set formula.match_metric/formula.metric to NED "
            "in config (or provide a valid chromedriver_path)." + detail
        )

    def _fallback_ned(self, gt: str, pred: str, reason: str, render_failure_delta: int = 0) -> FormulaScoreResult:
        self._warn_fallback(reason=reason, render_failure_delta=render_failure_delta)
        if render_failure_delta > 0:
            self.render_failures += int(render_failure_delta)
        return FormulaScoreResult(
            sim=1.0 - ned(gt or "", pred or ""),
            metric_used="NED-fallback",
            fallback_from="CDM",
            fallback_reason=reason,
            render_failure_delta=int(render_failure_delta),
        )

    def _ensure_cdm(self) -> bool:
        if self.formula_metric != "CDM":
            return False
        if self._cdm is not None:
            return True
        if self._cdm_unavailable:
            self._raise_init_error()
        try:
            self._cdm = CDM(chromedriver_path=self.chromedriver_path)
            return True
        except Exception as exc:
            self._cdm_unavailable = True
            self._cdm_init_err = f"{type(exc).__name__}: {exc}"
            self._raise_init_error()

    @staticmethod
    def _is_valid_metric(value: Any) -> bool:
        try:
            f = float(value)
        except Exception:
            return False
        return math.isfinite(f) and 0.0 <= f <= 1.0

    def compute(self, gt: str, pred: str, visualize: bool = False) -> FormulaScoreResult:
        if self.formula_metric == "NED":
            return FormulaScoreResult(sim=1.0 - ned(gt or "", pred or ""), metric_used="NED")

        self._ensure_cdm()

        assert self._cdm is not None
        before_failures = int(getattr(self._cdm, "render_failure_count", 0))
        try:
            # Suppress noisy low-level warning from skimage RANSAC inside CDM.
            # We rely on CDM output and render_failure_count for fallback decisions.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="No inliers found. Model not fitted",
                    category=UserWarning,
                )
                f1, recall, precision, vis_img = self._cdm.compute(gt or "", pred or "", visualize=visualize)
        except Exception:
            return self._fallback_ned(gt, pred, reason="compute_exception")

        after_failures = int(getattr(self._cdm, "render_failure_count", 0))
        delta = max(after_failures - before_failures, 0)

        if not (self._is_valid_metric(f1) and self._is_valid_metric(recall) and self._is_valid_metric(precision)):
            return self._fallback_ned(gt, pred, reason="invalid_output", render_failure_delta=delta)

        if delta > 0:
            return self._fallback_ned(gt, pred, reason="render_failure", render_failure_delta=delta)

        return FormulaScoreResult(
            sim=float(f1),
            recall=float(recall),
            precision=float(precision),
            vis_img=vis_img,
            metric_used="CDM",
            render_failure_delta=delta,
        )


def _stage(enabled: bool, text: str):
    if enabled:
        msg = f" {text} "
        pad = 4
        width = max(len(msg) + pad * 2, 60)
        border = "─" * (width - 2)
        print(f"\n┌{border}┐")
        print(f"│{msg.center(width - 2)}│")
        print(f"└{border}┘")


def build_cost_matrix_text(
    gt_segments: List[Segment],
    pred_segments: List[Segment],
    alpha: float,
    paragraph_match_metric: str,
    formula_match_metric: str,
    chromedriver_path: Optional[str] = None,
    progress_enable: bool = False,
    desc: Optional[str] = None,
    formula_engine: Optional[FormulaSimilarityEngine] = None,
) -> List[List[float]]:
    matrix: List[List[float]] = []
    assert paragraph_match_metric in {"NED"}, f"Unsupported paragraph.match_metric: {paragraph_match_metric}"
    assert formula_match_metric in {"CDM", "NED"}, f"Unsupported formula.match_metric: {formula_match_metric}"
    engine = formula_engine or FormulaSimilarityEngine(formula_match_metric, chromedriver_path=chromedriver_path)
    total = len(gt_segments) * len(pred_segments)
    pbar = tqdm(total=total, desc=desc or "Text cost matrix", disable=not progress_enable)
    for g in gt_segments:
        row: List[float] = []
        for p in pred_segments:
            if paragraph_match_metric == "NED":
                text_sim = 1.0 - ned(g.text_no_formula or "", p.text_no_formula or "")
            gf = g.inline_formulas or []
            pf = p.inline_formulas or []
            if len(gf) == 0 and len(pf) == 0:
                formula_sim_avg = 1.0
            elif len(gf) == 0 or len(pf) == 0:
                formula_sim_avg = 0.0
            else:
                sim_matrix: List[List[float]] = []
                for gx in gf:
                    row_sim: List[float] = []
                    for px in pf:
                        f_res = engine.compute(gx, px, visualize=False)
                        row_sim.append(float(f_res.sim))
                    sim_matrix.append(row_sim)
                cost = [[1.0 - s for s in r] for r in sim_matrix]
                r_ind, c_ind = linear_sum_assignment(cost)
                if len(r_ind) == 0:
                    formula_sim_avg = 0.0
                else:
                    sims = [sim_matrix[i][j] for i, j in zip(r_ind, c_ind)]
                    formula_sim_avg = sum(sims) / len(sims)
            sim = alpha * text_sim + (1.0 - alpha) * formula_sim_avg
            row.append(1.0 - sim)
            pbar.update(1)
        matrix.append(row)
    pbar.close()
    return matrix


def build_cost_matrix_formula(
    gt_segments: List[Segment],
    pred_segments: List[Segment],
    formula_match_metric: str,
    chromedriver_path: Optional[str] = None,
    progress_enable: bool = False,
    desc: Optional[str] = None,
    formula_engine: Optional[FormulaSimilarityEngine] = None,
) -> List[List[float]]:
    assert formula_match_metric in {"CDM", "NED"}, f"Unsupported formula.match_metric: {formula_match_metric}"
    engine = formula_engine or FormulaSimilarityEngine(formula_match_metric, chromedriver_path=chromedriver_path)
    matrix: List[List[float]] = []
    total = len(gt_segments) * len(pred_segments)
    pbar = tqdm(total=total, desc=desc or "Display formula cost matrix", disable=not progress_enable)
    for g in gt_segments:
        row: List[float] = []
        for p in pred_segments:
            f_res = engine.compute(g.raw, p.raw, visualize=False)
            row.append(1.0 - float(f_res.sim))
            pbar.update(1)
        matrix.append(row)
    pbar.close()
    return matrix


def build_cost_matrix_table(gt_segments: List[Segment], pred_segments: List[Segment], table_match_metric: str, progress_enable: bool = False, desc: Optional[str] = None) -> List[List[float]]:
    assert table_match_metric in {"TEDS", "TEDS-S", "NED"}, f"Unsupported table.match_metric: {table_match_metric}"
    matrix: List[List[float]] = []
    total = len(gt_segments) * len(pred_segments)
    pbar = tqdm(total=total, desc=desc or "Table cost matrix", disable=not progress_enable)
    for g in gt_segments:
        row: List[float] = []
        for p in pred_segments:
            if table_match_metric == "TEDS":
                sim = calculate_teds(g.raw, p.raw)
            elif table_match_metric == "TEDS-S":
                sim = calculate_teds_structure_only(g.raw, p.raw)
            else:
                sim = 1.0 - ned(g.raw or "", p.raw or "")
            row.append(1.0 - sim)
            pbar.update(1)
        matrix.append(row)
    pbar.close()
    return matrix


def match_and_score(gt_segments: List[Segment], pred_segments: List[Segment], kind: str, vis_dir: Optional[str] = None, vis_prefix: Optional[str] = None, **kwargs) -> CompareResult:
    chromedriver_path: Optional[str] = kwargs.get("chromedriver_path", None)
    progress_enable: bool = bool(kwargs.get("progress_enable", False))
    if len(gt_segments) == 0 or len(pred_segments) == 0:
        _stage(progress_enable, f"No segments, skipping ({kind})")
        return CompareResult(pairs=[], scores=[], detail=[])
    if kind == "text":
        alpha = float(kwargs.get("alpha", 0.5))
        paragraph_match_metric = str(kwargs.get("paragraph_match_metric", "NED"))
        formula_match_metric = str(kwargs.get("formula_match_metric", "CDM"))
        formula_engine = FormulaSimilarityEngine(formula_match_metric, chromedriver_path=chromedriver_path)
        _stage(progress_enable, "Building cost matrix (text)")
        cost = build_cost_matrix_text(
            gt_segments,
            pred_segments,
            alpha,
            paragraph_match_metric,
            formula_match_metric,
            chromedriver_path,
            progress_enable,
            "Text cost matrix",
            formula_engine=formula_engine,
        )
        _stage(progress_enable, "Hungarian matching (text)")
        r, c = linear_sum_assignment(cost)
        pairs = list(zip(r, c))
        _stage(progress_enable, "Scoring matched pairs (text)")
        scores, detail = [], []
        pbar = tqdm(total=len(pairs), desc="Text pair scoring", disable=not progress_enable)
        for i, j in pairs:
            g = gt_segments[i]
            p = pred_segments[j]
            formula_fallback_count = 0
            formula_fallback_reasons: List[str] = []
            if paragraph_match_metric == "NED":
                text_sim = 1.0 - ned(g.text_no_formula or "", p.text_no_formula or "")
            gf = g.inline_formulas or []
            pf = p.inline_formulas or []
            pair_alpha = alpha
            if len(gf) == 0 and len(pf) == 0:
                formula_sim_avg = 1.0
                pair_alpha = 1.0  # No inline formulas, use text similarity only
            elif len(gf) == 0 or len(pf) == 0:
                formula_sim_avg = 0.0
            else:
                sim_matrix: List[List[float]] = []
                res_matrix: List[List[FormulaScoreResult]] = []
                for gx in gf:
                    row_sim: List[float] = []
                    row_res: List[FormulaScoreResult] = []
                    for px in pf:
                        f_res = formula_engine.compute(gx, px, visualize=False)
                        row_sim.append(float(f_res.sim))
                        row_res.append(f_res)
                    sim_matrix.append(row_sim)
                    res_matrix.append(row_res)
                cost_formula = [[1.0 - s for s in r] for r in sim_matrix]
                rf, cf = linear_sum_assignment(cost_formula)
                sims = [sim_matrix[ii][jj] for ii, jj in zip(rf, cf)] if len(rf) > 0 else []
                formula_sim_avg = (sum(sims) / len(sims)) if sims else 0.0
                matched_results = [res_matrix[ii][jj] for ii, jj in zip(rf, cf)] if len(rf) > 0 else []
                fallback_reasons = [
                    rlt.fallback_reason for rlt in matched_results if rlt.fallback_reason
                ]
                formula_fallback_count = len(fallback_reasons)
                formula_fallback_reasons = sorted(set(fallback_reasons))
            sim = pair_alpha * text_sim + (1.0 - pair_alpha) * formula_sim_avg
            entry = {
                "type": "text",
                "text_sim": text_sim,
                "formula_sim_avg": formula_sim_avg,
                "alpha": pair_alpha,
                "paragraph_match_metric": paragraph_match_metric,
                "formula_match_metric": formula_match_metric,
                "gt": g.raw,
                "pred": p.raw,
            }
            if formula_match_metric == "CDM":
                entry["formula_metric_used"] = "NED-fallback" if formula_fallback_count > 0 else "CDM"
                entry["formula_fallback_count"] = formula_fallback_count
                if formula_fallback_reasons:
                    entry["formula_fallback_reasons"] = formula_fallback_reasons
            scores.append(sim)
            detail.append(entry)
            pbar.update(1)
        pbar.close()
        _stage(progress_enable, "Done (text)")
        return CompareResult(
            pairs=pairs,
            scores=scores,
            detail=detail,
            render_failures=formula_engine.render_failures,
        )

    elif kind == "display_formula":
        formula_match_metric = str(kwargs.get("formula_match_metric", "CDM"))
        formula_score_metric = str(kwargs.get("formula_score_metric", formula_match_metric))
        match_engine = FormulaSimilarityEngine(formula_match_metric, chromedriver_path=chromedriver_path)
        score_engine = (
            match_engine
            if formula_score_metric == formula_match_metric
            else FormulaSimilarityEngine(formula_score_metric, chromedriver_path=chromedriver_path)
        )
        _stage(progress_enable, "Building cost matrix (display formula)")
        cost = build_cost_matrix_formula(
            gt_segments,
            pred_segments,
            formula_match_metric,
            chromedriver_path,
            progress_enable,
            "Display formula cost matrix",
            formula_engine=match_engine,
        )
        _stage(progress_enable, "Hungarian matching (display formula)")
        r, c = linear_sum_assignment(cost)
        pairs = list(zip(r, c))
        _stage(progress_enable, "Scoring matched pairs (display formula)")
        scores, detail = [], []
        if formula_score_metric == "CDM":
            do_vis = bool(vis_dir and vis_prefix)
            if do_vis:
                import cv2
                os.makedirs(vis_dir, exist_ok=True)
            pbar = tqdm(total=len(pairs), desc="Display formula pair scoring", disable=not progress_enable)
            for pair_idx, (i, j) in enumerate(pairs):
                f_res = score_engine.compute(gt_segments[i].raw, pred_segments[j].raw, visualize=do_vis)
                vis_fname = None
                if do_vis and f_res.metric_used == "CDM" and f_res.vis_img is not None and f_res.vis_img.size > 0:
                    vis_fname = f"{vis_prefix}_{pair_idx:03d}.png"
                    cv2.imwrite(os.path.join(vis_dir, vis_fname), f_res.vis_img)

                scores.append(float(f_res.sim))
                if f_res.metric_used == "CDM":
                    entry = {
                        "type": "display_formula",
                        "metric": "CDM",
                        "metric_used": "CDM",
                        "f1": float(f_res.sim),
                        "recall": float(f_res.recall or 0.0),
                        "precision": float(f_res.precision or 0.0),
                        "gt": gt_segments[i].raw,
                        "pred": pred_segments[j].raw,
                    }
                    if vis_fname:
                        entry["vis_path"] = vis_fname
                else:
                    entry = {
                        "type": "display_formula",
                        "metric": "NED",
                        "metric_used": f_res.metric_used,
                        "sim": float(f_res.sim),
                        "gt": gt_segments[i].raw,
                        "pred": pred_segments[j].raw,
                        "fallback_from": f_res.fallback_from,
                        "fallback_reason": f_res.fallback_reason,
                        "render_failure_delta": f_res.render_failure_delta,
                    }
                detail.append(entry)
                pbar.update(1)
            pbar.close()
        else:
            pbar = tqdm(total=len(pairs), desc="Display formula pair scoring", disable=not progress_enable)
            for i, j in pairs:
                sim = 1.0 - ned(gt_segments[i].raw or "", pred_segments[j].raw or "")
                scores.append(sim)
                detail.append({
                    "type": "display_formula",
                    "metric": "NED",
                    "metric_used": "NED",
                    "sim": sim,
                    "gt": gt_segments[i].raw,
                    "pred": pred_segments[j].raw,
                })
                pbar.update(1)
            pbar.close()
        render_failures = score_engine.render_failures if formula_score_metric == "CDM" else 0
        _stage(progress_enable, "Done (display formula)")
        return CompareResult(pairs=pairs, scores=scores, detail=detail, render_failures=render_failures)

    elif kind == "table":
        table_match_metric = str(kwargs.get("table_match_metric", "TEDS"))
        _stage(progress_enable, "Building cost matrix (table)")
        cost = build_cost_matrix_table(gt_segments, pred_segments, table_match_metric, progress_enable, "Table cost matrix")
        _stage(progress_enable, "Hungarian matching (table)")
        r, c = linear_sum_assignment(cost)
        pairs = list(zip(r, c))
        _stage(progress_enable, "Scoring matched pairs (table)")
        scores, detail = [], []
        pbar = tqdm(total=len(pairs), desc="Table pair scoring", disable=not progress_enable)
        for i, j in pairs:
            if table_match_metric == "TEDS":
                sim = calculate_teds(gt_segments[i].raw, pred_segments[j].raw)
                scores.append(sim)
                detail.append({
                    "type": "table",
                    "metric": "TEDS",
                    "sim": sim,
                    "structure_only": False,
                    "gt": gt_segments[i].raw,
                    "pred": pred_segments[j].raw,
                })
            elif table_match_metric == "TEDS-S":
                sim = calculate_teds_structure_only(gt_segments[i].raw, pred_segments[j].raw)
                scores.append(sim)
                detail.append({
                    "type": "table",
                    "metric": "TEDS-S",
                    "sim": sim,
                    "structure_only": True,
                    "gt": gt_segments[i].raw,
                    "pred": pred_segments[j].raw,
                })
            else:
                sim = 1.0 - ned(gt_segments[i].raw or "", pred_segments[j].raw or "")
                scores.append(sim)
                detail.append({
                    "type": "table",
                    "metric": "NED",
                    "sim": sim,
                    "structure_only": False,
                    "gt": gt_segments[i].raw,
                    "pred": pred_segments[j].raw,
                })
            pbar.update(1)
        pbar.close()
        _stage(progress_enable, "Done (table)")
        return CompareResult(pairs=pairs, scores=scores, detail=detail)

    else:
        return CompareResult(pairs=[], scores=[], detail=[])
