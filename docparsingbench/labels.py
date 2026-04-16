import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


LABELS_FILENAME = "labels.json"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
PathLike = Union[str, Path]

_GT_NAME_RE = re.compile(r"^(?P<industry>[^_]+)_(?P<sub_industry>.+)_(?P<sample_id>\d+)$")


class LabelsError(ValueError):
    """Raised when labels cannot be resolved or generated from GT inputs."""


def resolve_gt_markdown_dir(gt_path: Optional[PathLike]) -> Path:
    if gt_path is None:
        raise LabelsError(
            "Labels were not provided.\n"
            "Fix: pass --gt <ground_truth_markdown_dir> (or dataset root containing markdowns/) "
            "so labels can be auto-generated."
        )

    path = Path(gt_path).expanduser()
    candidates: List[Path] = []
    if path.is_dir():
        candidates.append(path / "markdowns")
        candidates.append(path)

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.md")):
            return candidate

    raise LabelsError(
        "\n".join(
            [
                f"GT dir has no top-level .md files: {path}",
                "Fix: pass the ground truth markdown directory, or the dataset root containing markdowns/.",
            ]
        )
    )


def build_image_index(images_dir: Path) -> Dict[str, str]:
    image_index: Dict[str, str] = {}
    if not images_dir.is_dir():
        return image_index

    for image_path in sorted(images_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            image_index[image_path.stem] = image_path.name
    return image_index


def build_labels_payload(gt_path: PathLike) -> Dict[str, Any]:
    gt_dir = resolve_gt_markdown_dir(gt_path)
    md_files = sorted(gt_dir.glob("*.md"))
    if not md_files:
        raise LabelsError(
            "\n".join(
                [
                    f"GT dir has no top-level .md files: {gt_dir}",
                    "Fix: pass the ground truth markdown directory, or the dataset root containing markdowns/.",
                ]
            )
        )

    image_index = build_image_index(gt_dir.parent / "images")
    schema_map: Dict[str, set[str]] = {}
    items: List[Dict[str, str]] = []

    for md_path in md_files:
        match = _GT_NAME_RE.match(md_path.stem)
        if not match:
            raise LabelsError(
                "\n".join(
                    [
                        f"GT filename does not match <industry>_<sub-industry>_<id>: {md_path.name}",
                        "Fix: rename the GT markdown file to match the dataset naming convention, or provide --labels explicitly.",
                    ]
                )
            )

        industry = match.group("industry")
        sub_industry = match.group("sub_industry")
        schema_map.setdefault(industry, set()).add(sub_industry)
        items.append(
            {
                "img": image_index.get(md_path.stem, ""),
                "md": md_path.name,
                "industry": industry,
                "sub-industry": sub_industry,
            }
        )

    label_schema = [
        {
            "industry": industry,
            "sub_industries": [{"sub-industry": sub_industry} for sub_industry in sorted(schema_map[industry])],
        }
        for industry in sorted(schema_map)
    ]
    return {"label_schema": label_schema, "data": items}


def generate_labels_file(gt_path: PathLike, output_path: Optional[PathLike] = None) -> Path:
    gt_dir = resolve_gt_markdown_dir(gt_path)
    target = Path(output_path).expanduser() if output_path is not None else gt_dir.parent / LABELS_FILENAME
    payload = build_labels_payload(gt_dir)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_text(encoding="utf-8") != serialized:
        target.write_text(serialized, encoding="utf-8")
    return target


def resolve_labels_path(labels_path: Optional[PathLike], gt_path: Optional[PathLike]) -> Path:
    if labels_path is not None and str(labels_path).strip():
        explicit_path = Path(labels_path).expanduser()
        if not explicit_path.exists():
            raise LabelsError(
                "\n".join(
                    [
                        f"Explicit labels path does not exist: {explicit_path}",
                        "Fix: provide an existing labels.json, or omit --labels and provide --gt to auto-generate it.",
                    ]
                )
            )
        return explicit_path

    return generate_labels_file(gt_path)
