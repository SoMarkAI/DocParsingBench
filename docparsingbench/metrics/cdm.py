import sys
import os
import math
from typing import Tuple

# If FASTCDM_SRC env var is set, load fastcdm source from that path (for local development)
# Example: export FASTCDM_SRC=/path/to/fastcdm
_fastcdm_src = os.environ.get("FASTCDM_SRC", "")
if _fastcdm_src and os.path.isdir(_fastcdm_src) and _fastcdm_src not in sys.path:
    sys.path.insert(0, _fastcdm_src)

try:
    from fastcdm.core import FastCDM, preprocess
except Exception as e:
    raise RuntimeError("fastcdm is not installed or unavailable; install fastcdm or set FASTCDM_SRC to point to the source directory") from e


class CDM:
    """
    CDM formula comparison wrapper.
    Input: two LaTeX formula strings `gt`, `pred`
    Output: tuple `(f1, recall, precision)`
    Note: internally uses `fastcdm.FastCDM.compute(gt, pred)`; if `chromedriver_path` is not configured, fastcdm uses its own default.
    """

    def __init__(self, chromedriver_path: str = None):
        self._impl = FastCDM(chromedriver=chromedriver_path)

    @property
    def render_failure_count(self) -> int:
        return getattr(self._impl, "render_failure_count", 0)

    def compute(self, gt: str, pred: str, visualize: bool = False) -> Tuple:
        self._prepare_render_width(gt or "", pred or "")
        if visualize:
            f1, recall, precision, vis_img = self._impl.compute(gt or "", pred or "", visualize=True)
        else:
            f1, recall, precision = self._impl.compute(gt or "", pred or "")
            vis_img = None
        return float(f1), float(recall), float(precision), vis_img

    def _prepare_render_width(self, gt: str, pred: str) -> None:
        worker = self._impl.render_worker
        if worker is None:
            return
        gt_latex, _ = preprocess(gt)
        pred_latex, _ = preprocess(pred)
        contents = [
            f"$$${value}$$" if not value.startswith("$$") else value
            for value in (gt_latex, pred_latex)
        ]
        initial_width = 2000
        driver = worker.driver
        driver.set_window_size(initial_width, worker.window_init_height)
        driver.execute_script("render(arguments[0], false)", contents)
        required_width = driver.execute_script(
            "return Math.max(0, ...[...document.querySelectorAll('.screenshot')]"
            ".map(e => Math.ceil(Math.max(e.scrollWidth, e.getBoundingClientRect().width))))"
        )
        target_width = min(max(initial_width, int(math.ceil(required_width)) + 40), 16000)
        worker.window_fix_width = target_width
        driver.set_window_size(target_width, worker.window_init_height)
