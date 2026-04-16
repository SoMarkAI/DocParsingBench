"""PaddleOCR VL 1.5 model runner."""

import argparse
import os
import tempfile
from typing import Optional
from paddleocr import PaddleOCRVL

from .base import BaseModelRunner


class PaddleRunner(BaseModelRunner):
    name = "paddle"

    def __init__(self, vllm_base_url: str) -> None:
        self.vllm_base_url = vllm_base_url

        self.pipeline = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=self.vllm_base_url,
        )

    @staticmethod
    def _read_first_markdown_file(root_dir: str) -> Optional[str]:
        for current_root, _, files in os.walk(root_dir):
            for filename in sorted(files):
                if not filename.endswith(".md"):
                    continue
                path = os.path.join(current_root, filename)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    return content
        return None

    def parse_md(self, img_path: str) -> str:
        output = self.pipeline.predict(img_path)

        try:
            result = next(iter(output))
        except StopIteration as exc:
            raise ValueError("PaddleOCR VL 1.5 returned no results") from exc

        with tempfile.TemporaryDirectory(prefix="paddleocr_vl_") as tmp_dir:
            result.save_to_markdown(save_path=tmp_dir)
            markdown = self._read_first_markdown_file(tmp_dir)

        if not markdown:
            raise ValueError("PaddleOCR VL 1.5 did not generate valid markdown")

        return markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaddleOCR VL 1.5")
    parser.add_argument("img_dir", help="Image directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")
    parser.add_argument(
        "--base-url",
        required=True,
        help="PaddleOCR-VL vLLM service endpoint (required)",
    )
    args = parser.parse_args()

    runner = PaddleRunner(vllm_base_url=args.base_url)
    runner.run([args.img_dir] + ([args.output_dir] if args.output_dir else []))

    """
    Example:
    python -m scripts.paddle ./images ./output/paddle_md \
      --base-url http://paddleocr-vl15-vllm:8080/v1 \
    """
