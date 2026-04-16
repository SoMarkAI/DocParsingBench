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
        if visualize:
            f1, recall, precision, vis_img = self._impl.compute(gt or "", pred or "", visualize=True)
        else:
            f1, recall, precision = self._impl.compute(gt or "", pred or "")
            vis_img = None
        return float(f1), float(recall), float(precision), vis_img
