import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from docparsingbench.visualize.model_naming import humanize_model_name


@dataclass
class ModelScores:
    model_name: str
    display_name: str
    dpb: float
    text: float
    formula: float
    table: float


@dataclass
class AggregatedResult:
    all: List[ModelScores]
    by_industry: Dict[str, List[ModelScores]]
    industries: List[str]
    best_per_col: Dict[str, str]


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_prefixes(prefixes: Optional[List[str]]) -> List[str]:
    if not prefixes:
        return []
    return [prefix.strip().lower() for prefix in prefixes if prefix and prefix.strip()]


def aggregate(
    labels_path: Path,
    results_dir: Path,
    exclude_model_prefixes: Optional[List[str]] = None,
) -> AggregatedResult:
    """Aggregate evaluation results across models and industries.

    Parameters
    ----------
    labels_path : Path
        Path to ``labels.json`` containing industry mappings.
    results_dir : Path
        Directory containing ``<model>.result.json`` files.
    """
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    normalized_excluded_prefixes = _normalize_prefixes(exclude_model_prefixes)
    file_industry: Dict[str, str] = {}
    for entry in labels_data.get("data", []):
        file_industry[entry["md"]] = entry["industry"]

    industries_set: set = set()
    for entry in labels_data.get("data", []):
        industries_set.add(entry["industry"])
    industries = sorted(industries_set)

    result_files = sorted(results_dir.glob("*.result.json"))

    all_models: List[ModelScores] = []
    by_industry: Dict[str, List[ModelScores]] = {ind: [] for ind in industries}

    for rf in result_files:
        model_name = rf.name.replace(".result.json", "")
        display_name = humanize_model_name(model_name)
        normalized_model_name = model_name.lower()
        if any(normalized_model_name.startswith(prefix) for prefix in normalized_excluded_prefixes):
            continue
        data = json.loads(rf.read_text(encoding="utf-8"))
        summary = data.get("summary", {})

        # "All" view: use top-level summary directly
        ms = ModelScores(
            model_name=model_name,
            display_name=display_name,
            dpb=summary.get("avg_dpb", 0.0),
            text=summary.get("avg_text", 0.0),
            formula=summary.get("avg_formula", 0.0),
            table=summary.get("avg_table", 0.0),
        )
        all_models.append(ms)

        # Industry views: aggregate per-industry from individual reports
        reports = data.get("reports", [])
        # Collect per-industry scores
        ind_dpb: Dict[str, List[float]] = {ind: [] for ind in industries}
        ind_text: Dict[str, List[float]] = {ind: [] for ind in industries}
        ind_formula: Dict[str, List[float]] = {ind: [] for ind in industries}
        ind_table: Dict[str, List[float]] = {ind: [] for ind in industries}

        for report in reports:
            if report.get("skipped"):
                continue
            fname = report.get("file", "")
            ind = file_industry.get(fname)
            if ind is None:
                continue
            rsummary = report.get("summary", {})
            weights = rsummary.get("weights", {})

            ind_dpb[ind].append(rsummary.get("dpb_score", 0.0))
            if weights.get("alpha", 0) > 0:
                ind_text[ind].append(rsummary.get("text_score", 0.0))
            if weights.get("beta", 0) > 0:
                ind_formula[ind].append(rsummary.get("display_formula_score", 0.0))
            if weights.get("gamma", 0) > 0:
                ind_table[ind].append(rsummary.get("table_score", 0.0))

        for ind in industries:
            ind_ms = ModelScores(
                model_name=model_name,
                display_name=display_name,
                dpb=_safe_mean(ind_dpb[ind]),
                text=_safe_mean(ind_text[ind]),
                formula=_safe_mean(ind_formula[ind]),
                table=_safe_mean(ind_table[ind]),
            )
            by_industry[ind].append(ind_ms)

    # Sort all by DPB descending
    all_models.sort(key=lambda m: m.dpb, reverse=True)
    for ind in industries:
        by_industry[ind].sort(key=lambda m: m.dpb, reverse=True)

    # Best per column
    best_per_col = _compute_best(all_models)

    return AggregatedResult(
        all=all_models,
        by_industry=by_industry,
        industries=industries,
        best_per_col=best_per_col,
    )


def _compute_best(models: List[ModelScores]) -> Dict[str, str]:
    if not models:
        return {"dpb": "", "text": "", "formula": "", "table": ""}
    best = {}
    for col in ("dpb", "text", "formula", "table"):
        top = max(models, key=lambda m: getattr(m, col))
        best[col] = top.display_name
    return best
