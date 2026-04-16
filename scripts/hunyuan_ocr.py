"""Hunyuan OCR model runner."""

import argparse
import base64
import mimetypes
from typing import Any

from openai import OpenAI
from .base import BaseModelRunner

"""
Example vLLM startup command for Hunyuan OCR.

docker run -d \
  --name hunyuanocr-vllm \
  --network dpb \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  vllm/vllm-openai:v0.19.0-cu130-ubuntu2404 \
  --model tencent/HunyuanOCR \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name HunyuanOCR \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
"""

DOC_TO_MD_PROMPT = (
    "• Recognize formulas in the image and represent them in LaTeX.\n"
    "• Parse tables in the image into HTML.\n"
    "• Parse charts in the image: use Mermaid for flowcharts and Markdown for other charts.\n"
    "• Extract all main-body document content as Markdown, ignore headers and footers, represent tables in HTML and formulas in LaTeX, and preserve reading order."
)

MAX_TOKENS = 16384
TEMPERATURE = 0.0


class HunyuanOcrRunner(BaseModelRunner):
    name = "hunyuan_ocr"

    def __init__(
            self,
            vllm_base_url: str,
            model_name: str = "HunyuanOCR",
            timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name

        self.client = OpenAI(
            api_key="EMPTY",
            base_url=vllm_base_url,
            timeout=timeout,
        )

    @staticmethod
    def _encode_img(img_path: str) -> str:
        """
        Encode an image to base64.
        :param img_path: Image path.
        :return: Base64-encoded image payload.
        """

        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        mime = mimetypes.guess_type(img_path)[0] or "application/octet-stream"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)

                if isinstance(text, str) and text.strip():
                    parts.append(text)

            merged = "\n".join(parts).strip()
            if merged:
                return merged

        raise ValueError(f"Unexpected Hunyuan OCR content format: {type(content)!r}")

    def parse_md(self, img_path: str) -> str:
        """
        Parse image content into Markdown.
        :param img_path: Image path.
        :return: Markdown content.
        """
        image_data_url = self._encode_img(img_path)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[

                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                        {
                            "type": "text",
                            "text": DOC_TO_MD_PROMPT,
                        },
                    ],
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        if not resp.choices:
            raise ValueError("Hunyuan OCR response has no choices")

        markdown = self._extract_text_content(resp.choices[0].message.content)
        if not markdown:
            raise ValueError("Hunyuan OCR response content is empty")

        return markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan OCR")
    parser.add_argument("img_dir", help="Image directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible vLLM endpoint (required)",
    )

    parser.add_argument(
        "--model-name",
        default="HunyuanOCR",
        help="Model name on the vLLM service (default: HunyuanOCR)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120.0)",
    )
    args = parser.parse_args()

    runner = HunyuanOcrRunner(
        vllm_base_url=args.base_url,
        model_name=args.model_name,
        timeout=args.timeout,
    )
    runner.run([args.img_dir] + ([args.output_dir] if args.output_dir else []))

    """
    Example:
    python -m scripts.hunyuan_ocr ./images ./output/hunyuan_ocr_md \
      --base-url http://hunyuanocr-vllm:8000/v1 \
      --model-name HunyuanOCR \
      --timeout 120.0 \
    """
