"""Prepare a new prediction directory without access to ground truth."""

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .display_math_v1 import DISPLAY_RE, TABLE_RE, isolate_display_math
from .integrity import check_count, inventory, sha256
from .surya_html_v1 import serialize_page

PROFILES = ("identity-v1", "display-math-v1", "surya-html-v1")


def prepare(source: Path, output: Path, profile: str, expected_count: int) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output directories must not overlap")
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    inputs = inventory(source, ".json" if profile == "surya-html-v1" else ".md")
    check_count(inputs, expected_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    changed = []
    totals = Counter()
    try:
        for name in inputs["files"]:
            path = source / name
            payload = path.read_bytes()
            if profile == "identity-v1":
                prepared = payload
            elif profile == "display-math-v1":
                original = path.read_text(encoding="utf-8")
                text = isolate_display_math(original)
                if TABLE_RE.findall(original) != TABLE_RE.findall(text):
                    raise ValueError(f"table content changed: {name}")
                if DISPLAY_RE.findall(TABLE_RE.sub("", original)) != DISPLAY_RE.findall(
                    TABLE_RE.sub("", text)
                ):
                    raise ValueError(f"formula payload changed: {name}")
                if re.sub(r"\s+", "", original) != re.sub(r"\s+", "", text):
                    raise ValueError(f"non-whitespace content changed: {name}")
                prepared = text.encode("utf-8")
            else:
                response = json.loads(payload)
                if not isinstance(response, dict) or not isinstance(
                    response.get("blocks"), list
                ):
                    raise ValueError(
                        f"expected a Surya response with a blocks list: {name}"
                    )
                if any(not isinstance(block, dict) for block in response["blocks"]):
                    raise ValueError(f"invalid Surya block: {name}")
                text, stats = serialize_page(response)
                totals.update(stats)
                totals["skipped_or_error_blocks"] += sum(
                    bool(b.get("skipped") or b.get("error")) for b in response["blocks"]
                )
                if not text.strip():
                    raise ValueError(f"empty serialized prediction: {name}")
                prepared = (text + "\n").encode("utf-8")
            target_name = Path(name).stem + ".md"
            (stage / target_name).write_bytes(prepared)
            if profile != "surya-html-v1" and payload != prepared:
                changed.append(name)
        # Detect changed inputs, including added/deleted pages, before publishing.
        if (
            inventory(source, ".json" if profile == "surya-html-v1" else ".md")
            != inputs
        ):
            raise ValueError("source changed during preparation")
        predictions = inventory(stage, ".md")
        check_count(predictions, expected_count)
        audit = {
            "profile": profile,
            "implementation_sha256": {
                p.name: sha256(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("display_math_v1.py"),
                    Path(__file__).with_name("surya_html_v1.py"),
                    Path(__file__).with_name("integrity.py"),
                )
            },
            "source": inputs,
            "predictions": predictions,
            "changed_markdown_files": changed,
            "serializer_counts": dict(sorted(totals.items())),
            "ground_truth_used": False,
            "scoring_modified": False,
        }
        (stage / "preparation.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # Output must remain absent; a concurrent process may have created it.
        if output.exists():
            raise ValueError(f"output appeared during preparation: {output}")
        stage.rename(output)
        return audit
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--expected-count", type=int, default=1400)
    args = parser.parse_args()
    try:
        audit = prepare(args.source, args.output, args.profile, args.expected_count)
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f"preparation failed: {error}\n")
    print(
        json.dumps(
            {
                "profile": audit["profile"],
                "files": audit["predictions"]["count"],
                "changed": len(audit["changed_markdown_files"]),
                "collection_sha256": audit["predictions"]["collection_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
