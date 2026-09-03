import sys
import os
from typing import Tuple

# If FASTCDM_SRC env var is set, load fastcdm source from that path (for local development)
# Example: export FASTCDM_SRC=/path/to/fastcdm
_fastcdm_src = os.environ.get("FASTCDM_SRC", "")
if _fastcdm_src and os.path.isdir(_fastcdm_src) and _fastcdm_src not in sys.path:
    sys.path.insert(0, _fastcdm_src)

try:
    from fastcdm.core import FastCDM
    from fastcdm import colorize as fastcdm_colorize
    from fastcdm import latex_processor as fastcdm_latex_processor
except Exception as e:
    raise RuntimeError("fastcdm is not installed or unavailable; install fastcdm or set FASTCDM_SRC to point to the source directory") from e


_ORIGINAL_TOKEN_ADD_COLOR_RGB = fastcdm_latex_processor.token_add_color_RGB
_SAFE_DELIMITER_COLORIZER_INSTALLED = False


def _token_has_unsafe_delimiters(token: str) -> bool:
    """Return whether coloring this token could create invalid LaTeX structure."""
    balance = 0
    for char in token:
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
            if balance < 0:
                return True
    return balance != 0 or r"\\" in token


def _delimiter_safe_token_add_color_rgb(
    tokens, idx, token_list, brace_color=False
):
    token = tokens[idx]
    if token and _token_has_unsafe_delimiters(token):
        return tokens, idx + 1, token_list
    return _ORIGINAL_TOKEN_ADD_COLOR_RGB(
        tokens,
        idx,
        token_list,
        brace_color=brace_color,
    )


def _install_delimiter_safe_colorizer() -> None:
    global _SAFE_DELIMITER_COLORIZER_INSTALLED
    if _SAFE_DELIMITER_COLORIZER_INSTALLED:
        return
    fastcdm_latex_processor.token_add_color_RGB = (
        _delimiter_safe_token_add_color_rgb
    )
    fastcdm_colorize.token_add_color_RGB = (
        _delimiter_safe_token_add_color_rgb
    )
    _SAFE_DELIMITER_COLORIZER_INSTALLED = True


class CDM:
    """
    CDM formula comparison wrapper.
    Input: two LaTeX formula strings `gt`, `pred`
    Output: tuple `(f1, recall, precision)`
    Note: internally uses `fastcdm.FastCDM.compute(gt, pred)`; if `chromedriver_path` is not configured, fastcdm uses its own default.
    """

    def __init__(self, chromedriver_path: str = None):
        _install_delimiter_safe_colorizer()
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
