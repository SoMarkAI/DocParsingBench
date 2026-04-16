"""DeepSeek OCR 2 model runner."""

from typing import Any
import argparse
import base64
import mimetypes

from openai import OpenAI
from .base import BaseModelRunner

DOC_TO_MD_PROMPT = "\nFree OCR."

TEMPERATURE = 0.0
MAX_TOKENS = 4096

# Match the official vLLM no-repeat configuration.
VLLM_XARGS = {
    "ngram_size": 20,
    "window_size": 90,
    "whitelist_token_ids": [128821, 128822],
}


class DeepSeekOcrRunner(BaseModelRunner):
    name = "deepseek_ocr"

    def __init__(
            self,
            vllm_base_url: str,
            model_name: str = "DeepSeek-OCR-2",
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

        raise ValueError(f"Unexpected DeepSeek OCR 2 content format: {type(content)!r}")

    def parse_md(self, img_path: str) -> str:
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
            extra_body={
                "vllm_xargs": VLLM_XARGS,
            },
        )

        if not resp.choices:
            raise ValueError("DeepSeek OCR 2 response has no choices")

        markdown = self._extract_text_content(resp.choices[0].message.content)
        if not markdown:
            raise ValueError("DeepSeek OCR 2 response content is empty")

        return markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek OCR 2")
    parser.add_argument("img_dir", help="Image directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible vLLM endpoint (required)",
    )
    parser.add_argument(
        "--model-name",
        default="DeepSeek-OCR-2",
        help="Model name on the vLLM service (default: DeepSeek-OCR-2)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120.0)",
    )
    args = parser.parse_args()

    runner = DeepSeekOcrRunner(
        vllm_base_url=args.base_url,
        model_name=args.model_name,
        timeout=args.timeout,
    )
    runner.run([args.img_dir] + ([args.output_dir] if args.output_dir else []))

    """
    Example:
    python -m scripts.deepseek_ocr ./images ./output/deepseek_ocr_md \
      --base-url http://deepseekocr2-vllm:8000/v1 \
      --model-name DeepSeek-OCR-2 \
      --timeout 120.0
    """
