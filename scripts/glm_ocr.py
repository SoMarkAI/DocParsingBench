"""GLM OCR model runner."""

import argparse

from .base import BaseModelRunner


class GlmOcrRunner(BaseModelRunner):
    name = "glm_ocr"

    def __init__(self, config_path: str):
        self.config_path = config_path

        try:
            from glmocr import GlmOcr
        except ImportError as exc:
            raise ImportError(
                "Missing glmocr dependency. Install glmocr first, for example: pip install \"glmocr[selfhosted]\""
            ) from exc

        self._GlmOcr = GlmOcr

    def parse_md(self, img_path: str) -> str:
        glmocr_kwargs = {
            "config_path": self.config_path,
        }

        with self._GlmOcr(**glmocr_kwargs) as parser:
            result = parser.parse(img_path)

        markdown = getattr(result, "markdown_result", None)
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError("GLM OCR response is missing a valid markdown_result")
        return markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM OCR")
    parser.add_argument("img_dir", help="Image directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")
    parser.add_argument(
        "--config",
        default="glmocr.config.yaml",
        help="GLM-OCR SDK config file path (default: glmocr.config.yaml)",
    )

    args = parser.parse_args()

    runner = GlmOcrRunner(
        config_path=args.config,
    )
    runner.run([args.img_dir] + ([args.output_dir] if args.output_dir else []))

    """
    Example:
    python -m scripts.glm_ocr ./images ./output/glm_ocr_md \
      --config ./scripts/glmocr.config.yaml
    """
