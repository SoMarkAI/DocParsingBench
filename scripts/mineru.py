"""MinerU 2.5 model runner."""

import argparse
import os
import shutil
import subprocess
import tempfile
from typing import Optional

from .base import BaseModelRunner


class MineruRunner(BaseModelRunner):
    name = "mineru"

    def __init__(self, server_url: str, timeout: float = 1800.0) -> None:
        self.server_url = server_url
        self.timeout = timeout

        if shutil.which("mineru") is None:
            raise ImportError(
                "Missing MinerU CLI. Install it first, for example: pip install -U \"mineru[all]\""
            )

    @staticmethod
    def _read_first_markdown_file(root_dir: str) -> Optional[str]:
        candidates = []
        for current_root, _, files in os.walk(root_dir):
            for filename in files:
                if filename.endswith(".md"):
                    path = os.path.join(current_root, filename)
                    candidates.append(path)

        if not candidates:
            return None

        candidates.sort(key=lambda p: (p.count(os.sep), len(p), p))

        for path in candidates:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                return content

        return None

    def parse_md(self, img_path: str) -> str:
        with tempfile.TemporaryDirectory(prefix="mineru_") as tmp_dir:
            cmd = [
                "mineru",
                "-p",
                img_path,
                "-o",
                tmp_dir,
                "-b",
                "hybrid-http-client",
                "-u",
                self.server_url,
            ]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                detail = stderr or stdout or str(exc)
                raise ValueError(f"MinerU CLI execution failed: {detail}") from exc
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"MinerU CLI timed out: {img_path}") from exc

            markdown = self._read_first_markdown_file(tmp_dir)

        if not markdown:
            raise ValueError("MinerU did not generate valid markdown")
        return markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinerU 2.5")
    parser.add_argument("img_dir", help="Image directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")
    parser.add_argument(
        "--base-url",
        required=True,
        help="MinerU OpenAI server endpoint without /v1 (required)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Per-image timeout in seconds (default: 1800.0)",
    )
    args = parser.parse_args()

    runner = MineruRunner(
        server_url=args.base_url,
        timeout=args.timeout,
    )
    runner.run([args.img_dir] + ([args.output_dir] if args.output_dir else []))

    """
    Example:
    python -m scripts.mineru ./images ./output/mineru_md \
      --base-url http://mineru-vllm:8000 \
      --timeout 1800.0 \
    """
