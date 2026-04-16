"""Dots OCR model runner."""

"""
Example vLLM startup command for Dots OCR.

docker run -d \
  --name dotsocr-vllm \
  --network dpb \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  vllm/vllm-openai:v0.11.0 \
  --model rednote-hilab/dots.ocr \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name dots-ocr \
  --chat-template-content-format string
"""

import argparse
from .base import BaseModelRunner

PROMPT_MODE = "prompt_layout_all_en"
TEMPERATURE = 0.1
TOP_P = 1.0
DPI = 200
MAX_COMPLETION_TOKENS = 16384


class DotsOcrRunner(BaseModelRunner):
    name = "dots_ocr"

    def __init__(
            self,
            protocol: str,
            ip: str,
            port: int,
            model_name: str = "dots-ocr",

    ) -> None:
        self.protocol = protocol
        self.ip = ip
        self.port = port
        self.model_name = model_name

        self._bootstrap_official_modules()

    def _bootstrap_official_modules(self) -> None:
        try:
            from dots_ocr.model.inference import inference_with_vllm
            from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
            from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image
            from dots_ocr.utils.prompts import dict_promptmode_to_prompt
            from dots_ocr.utils.layout_utils import (
                post_process_output,
                pre_process_bboxes,
            )
            from dots_ocr.utils.format_transformer import layoutjson2md
        except ImportError as exc:
            raise ImportError(
                "Missing official dots.ocr dependency. Install dots.ocr from the official repository first."
            ) from exc

        self._inference_with_vllm = inference_with_vllm

        self._get_image_by_fitz_doc = get_image_by_fitz_doc
        self._fetch_image = fetch_image
        self._dict_promptmode_to_prompt = dict_promptmode_to_prompt
        self._post_process_output = post_process_output
        self._layoutjson2md = layoutjson2md

        if PROMPT_MODE not in self._dict_promptmode_to_prompt:
            raise ValueError(
                f"Unsupported prompt_mode: {PROMPT_MODE}. "
                f"Available: {list(self._dict_promptmode_to_prompt.keys())}"
            )

    def parse_md(self, img_path: str) -> str:
        origin_image = self._fetch_image(img_path)
        image = self._get_image_by_fitz_doc(origin_image, target_dpi=DPI)
        image = self._fetch_image(image, min_pixels=None, max_pixels=None)

        prompt = self._dict_promptmode_to_prompt[PROMPT_MODE]

        response = self._inference_with_vllm(
            image,
            prompt,
            protocol=self.protocol,
            ip=self.ip,
            port=self.port,
            model_name=self.model_name,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

        cells, filtered = self._post_process_output(
            response,
            PROMPT_MODE,
            origin_image,
            image,
            min_pixels=None,
            max_pixels=None,
        )

        if filtered:
            if not isinstance(cells, str) or not cells.strip():
                raise ValueError("Dots OCR filtered output is empty")
            return cells

        markdown = self._layoutjson2md(origin_image, cells, text_key="text")
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError("Dots OCR official layoutjson2md output is empty")
        return markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dots OCR")
    parser.add_argument("img_dir", help="Image directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")

    parser.add_argument(
        "--protocol",
        choices=["http", "https"],
        required=True,
        help="vLLM protocol, choices: http, https (required)",
    )
    parser.add_argument(
        "--ip",
        required=True,
        help="vLLM service IP or hostname (required)",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="vLLM service port (required)",
    )
    parser.add_argument(
        "--model-name",
        default="dots-ocr",
        help="Model name on the vLLM service (default: dots-ocr)",
    )

    args = parser.parse_args()

    runner = DotsOcrRunner(
        protocol=args.protocol,
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
    )
    runner.run([args.img_dir] + ([args.output_dir] if args.output_dir else []))

    """
    Example:
    python -m scripts.dots_ocr ./images ./output/dots_ocr_md \
      --protocol http \
      --ip dotsocr-vllm \
      --port 8000 \
      --model-name dots-ocr
    """
