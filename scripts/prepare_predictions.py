import argparse
import json
from pathlib import Path

from .postprocess.prepare import prepare

MODEL_PROFILES = {
    "oxr-1.0": "display-math-v1",
    "surya-ocr-2": "surya-html-v1",
    "paddleocr-vl-1.6": "identity-v1",
    "mineru": "identity-v1",
    "teleocr": "identity-v1",
    "dots-mocr": "identity-v1",
    "infinity-parser2-pro": "identity-v1",
    "ovisocr2": "identity-v1",
    "qianfan-ocr": "identity-v1",
}


def prepare_model(source: Path, output: Path, model: str, expected_count: int = 1400):
    try:
        profile = MODEL_PROFILES[model]
    except KeyError as error:
        raise ValueError(f"unsupported model: {model}") from error
    return prepare(source, output, profile, expected_count, model=model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_PROFILES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1400)
    args = parser.parse_args()
    try:
        audit = prepare_model(args.source, args.output, args.model, args.expected_count)
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f"preparation failed: {error}\n")
    print(
        json.dumps(
            {
                "model": audit["model"],
                "profile": audit["profile"],
                "files": audit["predictions"]["count"],
                "collection_sha256": audit["predictions"]["collection_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
