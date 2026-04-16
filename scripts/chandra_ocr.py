"""Chandra OCR 2 model runner."""

import argparse

from .base import BaseModelRunner

PROMPT_TYPE = "ocr_layout"
MAX_OUTPUT_TOKENS = 12384
INCLUDE_HEADERS_FOOTERS = False
INCLUDE_IMAGES = False
TEMPERATURE = 0.0
TOP_P = 0.1


class ChandraOcrRunner(BaseModelRunner):
    name = "chandra_ocr"

    def __init__(
            self,
            vllm_api_base: str,
    ) -> None:
        self.vllm_api_base = vllm_api_base

        self._bootstrap_chandra()

    def _bootstrap_chandra(self) -> None:

        try:
            from chandra.input import load_image
            from chandra.model import InferenceManager
            from chandra.model.schema import BatchInputItem

        except ImportError as exc:
            raise ImportError(
                "Missing chandra-ocr dependency. Install it first: pip install chandra-ocr"
            ) from exc

        self._load_image = load_image
        self._BatchInputItem = BatchInputItem
        self._manager = InferenceManager(method="vllm")

    def parse_md(self, img_path: str) -> str:
        image = self._load_image(img_path)

        batch_item = self._BatchInputItem(
            image=image,
            prompt_type=PROMPT_TYPE,
        )

        result = self._manager.generate(
            [batch_item],
            max_output_tokens=MAX_OUTPUT_TOKENS,
            include_images=INCLUDE_IMAGES,
            include_headers_footers=INCLUDE_HEADERS_FOOTERS,
            vllm_api_base=self.vllm_api_base,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )[0]

        if getattr(result, "error", None):
            raise ValueError(f"Chandra OCR vLLM inference failed: {result.error}")

        markdown = getattr(result, "markdown", None)
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError("Chandra OCR response is missing valid markdown")

        return markdown.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chandra OCR 2")
    parser.add_argument("img_dir", help="Image directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")
    parser.add_argument(
        "--base-url",
        required=True,
        help="Chandra OpenAI-compatible vLLM endpoint (required)",
    )

    args = parser.parse_args()

    runner = ChandraOcrRunner(
        vllm_api_base=args.base_url,
    )
    runner.run([args.img_dir] + ([args.output_dir] if args.output_dir else []))

    """
    Example:
    python -m scripts.chandra_ocr ./images ./output/chandra_ocr_md \
      --base-url http://chandraocr2-vllm:8000/v1 \
      --model-name chandra
    """
