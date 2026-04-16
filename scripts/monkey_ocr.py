"""Monkey OCR Pro 3B model runner."""

import argparse
import io
import mimetypes
import os
import zipfile
from typing import Optional
from urllib.parse import urljoin

import requests

from .base import BaseModelRunner


class MonkeyOcrRunner(BaseModelRunner):
    name = "monkey_ocr"

    def __init__(
            self,
            base_url: str,
            timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

    def _resolve_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        return urljoin(f"{self.base_url}/", path_or_url.lstrip("/"))

    @staticmethod
    def _pick_markdown_name(names: list[str]) -> Optional[str]:
        candidates = []
        for name in names:
            normalized = name.replace("\\", "/")
            if normalized.lower().endswith(".md"):
                candidates.append(normalized)

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item.count("/"), len(item), item))
        return candidates[0]

    def _download_markdown_from_zip(self, content: bytes) -> str:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            md_name = self._pick_markdown_name(zf.namelist())
            if md_name is None:
                raise ValueError("No markdown file found in MonkeyOCR returned archive")

            with zf.open(md_name) as fp:
                markdown = fp.read().decode("utf-8").strip()

        if not markdown:
            raise ValueError("MonkeyOCR returned markdown file is empty")
        return markdown

    def _raise_api_error(self, resp: requests.Response) -> None:
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise ValueError(f"MonkeyOCR API request failed: HTTP {resp.status_code}, {payload}")

    def parse_md(self, img_path: str) -> str:
        mime = mimetypes.guess_type(img_path)[0] or "application/octet-stream"
        filename = os.path.basename(img_path)

        with open(img_path, "rb") as f:
            resp = self.session.post(
                self._resolve_url("/parse"),
                files={"file": (filename, f, mime)},
                timeout=self.timeout,
            )

        if not resp.ok:
            self._raise_api_error(resp)

        data = resp.json()
        if not data.get("success", False):
            raise ValueError(f"MonkeyOCR parse failed: {data.get('message') or data}")

        download_url = data.get("download_url")
        if not isinstance(download_url, str) or not download_url.strip():
            raise ValueError(f"MonkeyOCR response is missing download_url: {data}")

        download_resp = self.session.get(
            self._resolve_url(download_url),
            timeout=self.timeout,
        )
        if not download_resp.ok:
            self._raise_api_error(download_resp)

        return self._download_markdown_from_zip(download_resp.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monkey OCR Pro 3B")
    parser.add_argument("img_dir", help="Image directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")
    parser.add_argument(
        "--base-url",
        required=True,
        help="MonkeyOCR FastAPI service endpoint (required)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Request timeout in seconds (default: 600.0)",
    )
    args = parser.parse_args()

    runner = MonkeyOcrRunner(
        base_url=args.base_url,
        timeout=args.timeout,
    )
    runner.run([args.img_dir] + ([args.output_dir] if args.output_dir else []))

    """
    Example:
    python -m scripts.monkey_ocr ./images ./output/monkey_ocr_md \
      --base-url http://monkeyocr-vllm:7861 \
      --timeout 600.0
    """
