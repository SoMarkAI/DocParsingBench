import re

try:
    import Levenshtein  # type: ignore
except Exception as e:
    raise RuntimeError("Levenshtein is not installed or unavailable; install via: pip install levenshtein") from e


def _edit_distance(a: str, b: str) -> int:
    return int(Levenshtein.distance(a, b))


def clean_string(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"`+", "", s)
    s = re.sub(r"[*_~]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ned(gt: str, pred: str) -> float:
    g = clean_string(gt)
    p = clean_string(pred)
    dist = _edit_distance(g, p)
    norm_length = max(len(g), len(p))
    return (dist / norm_length) if norm_length > 0 else 0.0


def cer(gt: str, pred: str) -> float:
    g = clean_string(gt)
    p = clean_string(pred)
    dist = _edit_distance(g, p)
    gt_len = len(g)
    return (dist / gt_len) if gt_len > 0 else 0.0
