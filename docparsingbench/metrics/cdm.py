import sys
import os
from typing import Tuple

# If FASTCDM_SRC env var is set, load fastcdm source from that path (for local development)
# Example: export FASTCDM_SRC=/path/to/fastcdm
_fastcdm_src = os.environ.get("FASTCDM_SRC", "")
if _fastcdm_src and os.path.isdir(_fastcdm_src) and _fastcdm_src not in sys.path:
    sys.path.insert(0, _fastcdm_src)

try:
    from fastcdm.core import FastCDM, postprocess, preprocess
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
        gt_latex, gt_color_map = preprocess(gt or "")
        pred_latex, pred_color_map = preprocess(pred or "")
        imgs, render_errors = self._render_with_dom_status([gt_latex, pred_latex])
        if len(imgs) < 2 or imgs[0] is None or imgs[1] is None or any(render_errors):
            self._impl.render_failure_count += 1
            return 0.0, 0.0, 0.0, None
        result = postprocess(imgs[0], imgs[1], gt_color_map, pred_color_map, visualize)
        if visualize:
            f1, recall, precision, vis_img = result
        else:
            f1, recall, precision = result
            vis_img = None
        return float(f1), float(recall), float(precision), vis_img

    def _render_with_dom_status(self, latex_list):
        worker = self._impl.render_worker
        if worker is None:
            return [], [True] * len(latex_list)
        contents = [f"$$${value}$$" if not value.startswith("$$") else value for value in latex_list]
        imgs = worker.render(contents)
        render_errors = worker.driver.execute_script(
            "return [...document.querySelectorAll('.screenshot')]"
            ".map(e => Boolean(e.querySelector('.katex-error')))"
        )
        if len(render_errors) != len(contents):
            render_errors = [True] * len(contents)
        return imgs, [bool(value) for value in render_errors]
