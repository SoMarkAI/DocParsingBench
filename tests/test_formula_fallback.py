import warnings

import pytest

from docparsingbench.config.schema import Config
from docparsingbench.core import evaluate_single
import docparsingbench.evaluation.segment_compare as sc
from docparsingbench.markdown_segmenter import Segment


def _formula_segments(*raws: str):
    return [Segment(type="display_formula", raw=r) for r in raws]


class _InitFailCDM:
    def __init__(self, chromedriver_path=None):
        raise RuntimeError("init failed")


class _ComputeFailCDM:
    def __init__(self, chromedriver_path=None):
        self.render_failure_count = 0

    def compute(self, gt, pred, visualize=False):
        raise RuntimeError("compute failed")


class _RenderFailureCDM:
    def __init__(self, chromedriver_path=None):
        self.render_failure_count = 0

    def compute(self, gt, pred, visualize=False):
        self.render_failure_count += 1
        return 0.9, 0.9, 0.9, None


class _InvalidOutputCDM:
    def __init__(self, chromedriver_path=None):
        self.render_failure_count = 0

    def compute(self, gt, pred, visualize=False):
        return float("nan"), 0.8, 0.8, None


class _LowScoreCDM:
    def __init__(self, chromedriver_path=None):
        self.render_failure_count = 0

    def compute(self, gt, pred, visualize=False):
        return 0.01, 0.01, 0.01, None


class _SkimageWarnCDM:
    def __init__(self, chromedriver_path=None):
        self.render_failure_count = 0

    def compute(self, gt, pred, visualize=False):
        warnings.warn("No inliers found. Model not fitted", UserWarning)
        return 0.8, 0.8, 0.8, None


def test_formula_init_failure_raises_with_ned_guidance(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _InitFailCDM)
    with pytest.raises(sc.CDMInitializationError, match="set formula.match_metric/formula.metric to NED"):
        sc.match_and_score(
            _formula_segments("a", "b"),
            _formula_segments("a", "b"),
            kind="display_formula",
            formula_match_metric="CDM",
            formula_score_metric="CDM",
        )


def test_formula_compute_exception_falls_back_to_ned_with_warning(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _ComputeFailCDM)
    with pytest.warns(RuntimeWarning, match="reason=compute_exception"):
        res = sc.match_and_score(
            _formula_segments("x"),
            _formula_segments("y"),
            kind="display_formula",
            formula_match_metric="CDM",
            formula_score_metric="CDM",
        )
    assert res.detail[0]["metric_used"] == "NED-fallback"
    assert res.detail[0]["fallback_reason"] == "compute_exception"


def test_formula_render_failure_count_increase_falls_back_with_warning(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _RenderFailureCDM)
    with pytest.warns(RuntimeWarning, match="reason=render_failure"):
        res = sc.match_and_score(
            _formula_segments("x"),
            _formula_segments("y"),
            kind="display_formula",
            formula_match_metric="CDM",
            formula_score_metric="CDM",
        )
    assert res.detail[0]["metric_used"] == "NED-fallback"
    assert res.detail[0]["fallback_reason"] == "render_failure"
    assert res.render_failures >= 1


def test_formula_invalid_output_falls_back_with_warning(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _InvalidOutputCDM)
    with pytest.warns(RuntimeWarning, match="reason=invalid_output"):
        res = sc.match_and_score(
            _formula_segments("x"),
            _formula_segments("y"),
            kind="display_formula",
            formula_match_metric="CDM",
            formula_score_metric="CDM",
        )
    assert res.detail[0]["metric_used"] == "NED-fallback"
    assert res.detail[0]["fallback_reason"] == "invalid_output"


def test_formula_low_score_keeps_cdm_without_warning(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _LowScoreCDM)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sc.match_and_score(
            _formula_segments("x"),
            _formula_segments("y"),
            kind="display_formula",
            formula_match_metric="CDM",
            formula_score_metric="CDM",
        )
    assert res.detail[0]["metric"] == "CDM"
    assert res.detail[0]["metric_used"] == "CDM"
    assert abs(res.scores[0] - 0.01) < 1e-12
    assert len(rec) == 0


def test_formula_suppresses_skimage_no_inliers_warning(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _SkimageWarnCDM)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sc.match_and_score(
            _formula_segments("x"),
            _formula_segments("y"),
            kind="display_formula",
            formula_match_metric="CDM",
            formula_score_metric="CDM",
        )
    assert res.detail[0]["metric_used"] == "CDM"
    assert len(rec) == 0


def test_evaluate_single_formula_init_failure_raises_with_ned_guidance(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _InitFailCDM)
    cfg = Config()
    with pytest.raises(sc.CDMInitializationError, match="set formula.match_metric/formula.metric to NED"):
        evaluate_single("x", "y", cfg, "formula")


def test_evaluate_single_formula_ned_detail_is_minimal():
    cfg = Config()
    cfg.formula.metric = "NED"
    out = evaluate_single("x", "y", cfg, "formula")
    detail = out["detail"]
    assert set(detail.keys()) == {"metric", "metric_used", "sim"}
    assert detail["metric"] == "NED"
    assert detail["metric_used"] == "NED"
    assert abs(out["score"] - detail["sim"]) < 1e-12


def test_evaluate_single_formula_cdm_detail_uses_cdm_fields(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _LowScoreCDM)
    cfg = Config()
    cfg.formula.metric = "CDM"
    out = evaluate_single("x", "y", cfg, "formula")
    detail = out["detail"]
    assert set(detail.keys()) == {"metric", "metric_used", "f1", "recall", "precision"}
    assert detail["metric"] == "CDM"
    assert detail["metric_used"] == "CDM"
    assert abs(out["score"] - detail["f1"]) < 1e-12


def test_evaluate_single_formula_cdm_fallback_detail_is_ned_shaped(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _ComputeFailCDM)
    cfg = Config()
    cfg.formula.metric = "CDM"
    with pytest.warns(RuntimeWarning, match="reason=compute_exception"):
        out = evaluate_single("x", "y", cfg, "formula")
    detail = out["detail"]
    assert set(detail.keys()) == {
        "metric",
        "metric_used",
        "sim",
        "fallback_from",
        "fallback_reason",
        "render_failure_delta",
    }
    assert detail["metric"] == "NED"
    assert detail["metric_used"] == "NED-fallback"
    assert detail["fallback_from"] == "CDM"
    assert detail["fallback_reason"] == "compute_exception"
    assert abs(out["score"] - detail["sim"]) < 1e-12


def test_text_inline_formula_init_failure_raises(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _InitFailCDM)
    gt = [
        Segment(
            type="text",
            raw="A [FORMULA]",
            text_no_formula="A [FORMULA]",
            inline_formulas=["x^2"],
        )
    ]
    pred = [
        Segment(
            type="text",
            raw="B [FORMULA]",
            text_no_formula="B [FORMULA]",
            inline_formulas=["y^2"],
        )
    ]
    with pytest.raises(sc.CDMInitializationError, match="set formula.match_metric/formula.metric to NED"):
        sc.match_and_score(
            gt,
            pred,
            kind="text",
            alpha=0.5,
            paragraph_match_metric="NED",
            formula_match_metric="CDM",
        )


def test_text_inline_formula_compute_failure_warns_and_falls_back(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _ComputeFailCDM)
    gt = [
        Segment(
            type="text",
            raw="A [FORMULA]",
            text_no_formula="A [FORMULA]",
            inline_formulas=["x^2"],
        )
    ]
    pred = [
        Segment(
            type="text",
            raw="B [FORMULA]",
            text_no_formula="B [FORMULA]",
            inline_formulas=["y^2"],
        )
    ]
    with pytest.warns(RuntimeWarning, match="reason=compute_exception"):
        res = sc.match_and_score(
            gt,
            pred,
            kind="text",
            alpha=0.5,
            paragraph_match_metric="NED",
            formula_match_metric="CDM",
        )
    assert len(res.detail) == 1
    assert res.detail[0]["formula_metric_used"] == "NED-fallback"
    assert res.detail[0]["formula_fallback_count"] >= 1
    assert res.render_failures == 0


def test_text_inline_formula_render_failure_propagates_render_failures(monkeypatch):
    monkeypatch.setattr(sc, "CDM", _RenderFailureCDM)
    gt = [
        Segment(
            type="text",
            raw="A [FORMULA]",
            text_no_formula="A [FORMULA]",
            inline_formulas=["x^2"],
        )
    ]
    pred = [
        Segment(
            type="text",
            raw="B [FORMULA]",
            text_no_formula="B [FORMULA]",
            inline_formulas=["y^2"],
        )
    ]
    with pytest.warns(RuntimeWarning, match="reason=render_failure"):
        res = sc.match_and_score(
            gt,
            pred,
            kind="text",
            alpha=0.5,
            paragraph_match_metric="NED",
            formula_match_metric="CDM",
        )
    assert res.render_failures >= 1


def test_text_alpha_not_leaked_between_pairs():
    gt = [
        Segment(type="text", raw="line0", text_no_formula="line0", inline_formulas=[]),
        Segment(type="text", raw="line1 [FORMULA]", text_no_formula="line1 [FORMULA]", inline_formulas=["x"]),
    ]
    pred = [
        Segment(type="text", raw="line0", text_no_formula="line0", inline_formulas=[]),
        Segment(type="text", raw="line1 [FORMULA]", text_no_formula="line1 [FORMULA]", inline_formulas=["y"]),
    ]
    res = sc.match_and_score(
        gt,
        pred,
        kind="text",
        alpha=0.2,
        paragraph_match_metric="NED",
        formula_match_metric="NED",
    )
    assert len(res.pairs) == 2
    pair_to_score = {pair: score for pair, score in zip(res.pairs, res.scores)}
    assert abs(pair_to_score[(0, 0)] - 1.0) < 1e-12
    assert abs(pair_to_score[(1, 1)] - 0.2) < 1e-12
