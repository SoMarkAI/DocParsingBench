from typing import Dict, Any, List


def dpb_aggregate(text_scores: List[float], display_formula_scores: List[float], table_scores: List[float], alpha: float, beta: float, gamma: float) -> Dict[str, Any]:
    """DPB aggregation:
    - `text_scores`, `display_formula_scores`, `table_scores`: score lists for matched pairs of each category
    - `alpha`, `beta`, `gamma`: DPB weights (preset defaults, overridable by explicit values)
    Returns a dict containing sub-scores and overall score.
    """
    def avg(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    text_score = avg(text_scores)
    display_formula_score = avg(display_formula_scores)
    table_score = avg(table_scores)
    dpb = alpha * text_score + beta * display_formula_score + gamma * table_score
    return {
        "text_score": text_score,
        "display_formula_score": display_formula_score,
        "table_score": table_score,
        "dpb_score": dpb,
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
    }
