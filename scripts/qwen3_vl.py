"""Qwen3-VL-235B-A22B-Instruct model runner."""
import argparse
import base64
import mimetypes
from typing import Any

from openai import OpenAI
from .base import BaseModelRunner

"""
Example vLLM startup command for Qwen3-VL-235B-A22B-Instruct.

docker run -d \
  --name qwen3vl-235b-vllm \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  -e OMP_NUM_THREADS=1 \
  -p 8000:8000 \
  vllm/vllm-openai:v0.11.0 \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling
"""

DOC_TO_MD_PROMPT = """
Please recognize all text content in the image and output it directly in Markdown.
Strictly follow these rules:

    **Basic Requirements**:
    1. Output recognized content directly, without any explanation or commentary.
    2. Preserve the original paragraph and line-break structure.

    **Math Formulas**:
    - Wrap inline formulas with single `$`.
    - Wrap standalone display formulas with double `$$` on separate lines.

    Example:
    $$
    \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}
    $$

    **Code Blocks**:
    - If the image contains code, wrap it with triple backticks.

    **Tables**:
    - Output tables in HTML format, not Markdown table syntax.
    - Use tags such as `<table>`, `<tr>`, `<td>`, and `<th>`.

    If the image contains no text, output an empty string.

    Important: output the content itself directly. Do not wrap the whole
    response in a fenced code block like ```markdown.
"""
TEMPERATURE = 0.0
MAX_TOKENS = 4096
TOP_P = 1.0


class Qwen3VlRunner(BaseModelRunner):
    name = "qwen3_vl"

    def __init__(
            self,
            vllm_base_url: str,
            model_name: str = "Qwen3-VL-235B-A22B-Instruct",
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

        raise ValueError(f"Unexpected Qwen3-VL content format: {type(content)!r}")

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
            top_p=TOP_P,
        )

        if not resp.choices:
            raise ValueError("Qwen3-VL response has no choices")

        markdown = self._extract_text_content(resp.choices[0].message.content)
        if not markdown:
            raise ValueError("Qwen3-VL response content is empty")

        return markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument("img_dir", help="Image directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible vLLM endpoint (required)",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen3-VL-235B-A22B-Instruct",
        help="Model name on the vLLM service (default: Qwen3-VL-235B-A22B-Instruct)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120.0)",
    )
    args = parser.parse_args()

    runner = Qwen3VlRunner(
        vllm_base_url=args.base_url,
        model_name=args.model_name,
        timeout=args.timeout,
    )
    runner.run([args.img_dir] + ([args.output_dir] if args.output_dir else []))

    """
    Example:
    python -m scripts.qwen3_vl ./images ./output/qwen3_vl_md \
      --base-url http://qwen3vl-235b-vllm:8000/v1 \
      --model-name Qwen3-VL-235B-A22B-Instruct \
      --timeout 120.0
    """
