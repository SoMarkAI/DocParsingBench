"""Content fingerprints shared by preparation and replay checks (stdlib only)."""

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root: Path, suffix: str) -> dict:
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    files = sorted(root.glob(f"*{suffix}"))
    if not files:
        raise ValueError(f"no {suffix} files: {root}")
    collection = hashlib.sha256()
    hashes = {}
    for path in files:
        if not path.is_file():
            raise ValueError(f"not a file: {path}")
        payload = path.read_bytes()
        # UTF-8 and non-whitespace content are required even in identity mode.
        if not payload.decode("utf-8").strip():
            raise ValueError(f"empty input: {path}")
        digest = hashlib.sha256(payload).digest()
        hashes[path.name] = digest.hex()
        collection.update(path.name.encode("utf-8") + b"\0" + digest + b"\n")
    return {
        "files": hashes,
        "count": len(files),
        "collection_sha256": collection.hexdigest(),
    }


def check_count(item: dict, expected: int) -> None:
    if item["count"] != expected:
        raise ValueError(f"expected {expected} files, found {item['count']}")
