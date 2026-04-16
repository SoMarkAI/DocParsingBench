"""
Base class for model invocation scripts.

Each model script should inherit `BaseModelRunner` and implement:
  - `name`: model name (class attribute)
  - `parse_md(img_path) -> str`: convert an image to markdown
  - (optional) `postprocess(md) -> str`: post-process markdown output
"""

import abc
import argparse
import os
from typing import List, Optional

from tqdm import tqdm

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class BaseModelRunner(abc.ABC):

    name: str  # Set in subclasses as a class attribute.

    @abc.abstractmethod
    def parse_md(self, img_path: str) -> str:
        """Convert an image to a markdown string."""
        ...

    def postprocess(self, md: str) -> str:
        """Post-process markdown. Returns input as-is by default."""
        return md

    def _scan_images(self, img_dir: str) -> List[str]:
        images = sorted(
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No image files found in image directory: {img_dir}")
        return images

    def run(self, argv: Optional[List[str]] = None) -> None:
        parser = argparse.ArgumentParser(description=f"{self.name} model runner")
        parser.add_argument("img_dir", help="Image directory")
        parser.add_argument("output_dir", nargs="?", default=None, help="Output directory")
        args = parser.parse_args(argv)

        output_dir = args.output_dir or os.path.join("output", f"{self.name}_md")
        images = self._scan_images(args.img_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[{self.name}] {len(images)} images found, writing outputs to {output_dir}")

        failures: List[str] = []
        skipped = 0

        for img_path in tqdm(images, desc=self.name, unit="img"):
            img_name = os.path.basename(img_path)
            stem = os.path.splitext(img_name)[0]
            output_path = os.path.join(output_dir, f"{stem}.md")

            if os.path.exists(output_path):
                skipped += 1
                continue

            try:
                md = self.parse_md(img_path)
                md = self.postprocess(md)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(md)
            except Exception as e:
                print(f"[{self.name}] failed on {img_name}: {e}")
                failures.append(img_name)
                continue

        total = len(images)
        success = total - len(failures) - skipped
        print(
            f"[{self.name}] done: success {success} / skipped {skipped}"
            f" / failed {len(failures)} / total {total}"
        )
        if failures:
            print(f"[{self.name}] failed files: {failures}")
