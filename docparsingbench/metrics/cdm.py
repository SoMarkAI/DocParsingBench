import sys
import os
import re
import importlib
from typing import Tuple

# If FASTCDM_SRC env var is set, load fastcdm source from that path (for local development)
# Example: export FASTCDM_SRC=/path/to/fastcdm
_fastcdm_src = os.environ.get("FASTCDM_SRC", "")
if _fastcdm_src and os.path.isdir(_fastcdm_src) and _fastcdm_src not in sys.path:
    sys.path.insert(0, _fastcdm_src)

try:
    from fastcdm.core import FastCDM, preprocess
    from fastcdm import core as fastcdm_core
    from fastcdm import colorize as fastcdm_colorize
    from fastcdm import latex_processor as fastcdm_latex_processor
    fastcdm_tokenize = importlib.import_module("fastcdm.tokenize")
except Exception as e:
    raise RuntimeError("fastcdm is not installed or unavailable; install fastcdm or set FASTCDM_SRC to point to the source directory") from e


_ORIGINAL_TOKEN_ADD_COLOR_RGB = fastcdm_latex_processor.token_add_color_RGB
_ORIGINAL_TOKENIZE = fastcdm_tokenize.tokenize
_SAFE_COLORIZER_INSTALLED = False
_SAFE_TOKENIZER_INSTALLED = False
_DIMENSION_VALUE_RE = re.compile(
    r"^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:pt|px|em|ex|mu|cm|mm|in|pc|bp|dd|cc|sp)?$"
)
_DIMENSION_UNITS = ("pt", "px", "em", "ex", "mu", "cm", "mm", "in", "pc", "bp", "dd", "cc", "sp")
_PROTECTED_TOKENIZER_COMMANDS = ("ddots", "cdots", "limits", "int")


def _skip_balanced_group(tokens, start, left="{", right="}"):
    if start >= len(tokens) or tokens[start] != left:
        return start
    depth = 0
    for idx in range(start, len(tokens)):
        if tokens[idx] == left:
            depth += 1
        elif tokens[idx] == right:
            depth -= 1
            if depth == 0:
                return idx + 1
    return start


def _is_dimension(value):
    return _DIMENSION_VALUE_RE.match(value) and value.endswith(_DIMENSION_UNITS)


def _safe_token_add_color_rgb(tokens, idx, token_list, brace_color=False):
    """Keep syntax-sensitive LaTeX commands and arguments out of token coloring."""
    token = tokens[idx]
    if token == r"\hline":
        tokens[idx] = ""
        return tokens, idx + 1, token_list
    if token in {r"\substack", r"\textcircled", r"\fbox", r"\xrightleftharpoons"}:
        return tokens, idx + 1, token_list
    if token in {r"\tag", r"\tag*"}:
        next_idx = idx + 1
        if next_idx < len(tokens) and tokens[next_idx] == "*":
            next_idx += 1
        if next_idx < len(tokens) and tokens[next_idx] == "{":
            next_idx = _skip_balanced_group(tokens, next_idx)
        return tokens, next_idx, token_list
    if token in {r"\hskip", r"\hspace", r"\hspace*"}:
        next_idx = idx + 1
        if next_idx < len(tokens) and tokens[next_idx] == "*":
            tokens[idx] = token + "*"
            del tokens[next_idx]
        if next_idx < len(tokens) and tokens[next_idx] == "{":
            group_end = _skip_balanced_group(tokens, next_idx)
            dimension = "".join(tokens[next_idx + 1:group_end - 1])
            if _is_dimension(dimension):
                tokens[next_idx + 1:group_end - 1] = [dimension]
                group_end = next_idx + 3
            next_idx = group_end
        else:
            dimension = ""
            end_idx = next_idx
            while end_idx < len(tokens) and end_idx - next_idx < 32:
                candidate = dimension + tokens[end_idx]
                if not re.match(r"^[+\-0-9.a-zA-Z]+$", candidate):
                    break
                dimension = candidate
                end_idx += 1
                if _is_dimension(dimension):
                    tokens[next_idx:end_idx] = [dimension]
                    next_idx += 1
                    break
        return tokens, next_idx, token_list
    return _ORIGINAL_TOKEN_ADD_COLOR_RGB(
        tokens, idx, token_list, brace_color=brace_color
    )


def _install_safe_colorizer():
    global _SAFE_COLORIZER_INSTALLED
    if _SAFE_COLORIZER_INSTALLED:
        return
    fastcdm_latex_processor.token_add_color_RGB = _safe_token_add_color_rgb
    fastcdm_colorize.token_add_color_RGB = _safe_token_add_color_rgb
    _SAFE_COLORIZER_INSTALLED = True


def _install_safe_tokenizer():
    global _SAFE_TOKENIZER_INSTALLED
    if _SAFE_TOKENIZER_INSTALLED:
        return
    def safe_tokenize(latex_code):
        protected_latex = latex_code
        placeholders = {}
        for index, command in enumerate(_PROTECTED_TOKENIZER_COMMANDS):
            marker = rf"\alpha_{{918273{index:02d}}}"
            normalized_marker = (
                rf"\alpha _ {{ 9 1 8 2 7 3 {index // 10} {index % 10} }}"
            )
            pattern = re.compile(rf"\\{re.escape(command)}(?![a-zA-Z])")
            protected_latex, count = pattern.subn(lambda _match: marker, protected_latex)
            if count:
                placeholders[normalized_marker] = rf"\{command}"
        success, normalized = _ORIGINAL_TOKENIZE(protected_latex)
        if success:
            for marker, command in placeholders.items():
                normalized = normalized.replace(marker, command)
        return success, normalized

    fastcdm_tokenize.tokenize = safe_tokenize
    fastcdm_core.tokenize = safe_tokenize
    _SAFE_TOKENIZER_INSTALLED = True


class CDM:
    """
    CDM formula comparison wrapper.
    Input: two LaTeX formula strings `gt`, `pred`
    Output: tuple `(f1, recall, precision)`
    Note: internally uses `fastcdm.FastCDM.compute(gt, pred)`; if `chromedriver_path` is not configured, fastcdm uses its own default.
    """

    def __init__(self, chromedriver_path: str = None):
        _install_safe_colorizer()
        _install_safe_tokenizer()
        self._impl = FastCDM(chromedriver=chromedriver_path)

    @property
    def render_failure_count(self) -> int:
        return getattr(self._impl, "render_failure_count", 0)

    def compute(self, gt: str, pred: str, visualize: bool = False) -> Tuple:
        if visualize:
            f1, recall, precision, vis_img = self._impl.compute(gt or "", pred or "", visualize=True)
        else:
            f1, recall, precision = self._impl.compute(gt or "", pred or "")
            vis_img = None
        return float(f1), float(recall), float(precision), vis_img
