"""Serialize saved Surya OCR 2 blocks into DPB-compatible Markdown.

This adapter is deterministic and ground-truth blind. It changes only markup
that the original runner incorrectly copied from Surya's HTML response into a
Markdown prediction file.
"""

from __future__ import annotations

import html as html_module
import re
from collections import Counter

MATH_PATTERN = re.compile(
    r"<math\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</math>", re.IGNORECASE
)
ANY_HEADING_PATTERN = re.compile(
    r"<h(?P<level>[1-6])\b[^>]*>(?P<body>[\s\S]*?)</h(?P=level)>",
    re.IGNORECASE,
)
PARAGRAPH_PATTERN = re.compile(
    r"^\s*<p\b[^>]*>(?P<body>[\s\S]*?)</p>\s*$", re.IGNORECASE
)
TABLE_PATTERN = re.compile(r"^\s*<table\b", re.IGNORECASE)


def convert_math_tags(source: str) -> tuple[str, int, int]:
    """Convert Surya math tags while leaving table-internal markup intact."""
    if TABLE_PATTERN.match(source):
        return source, 0, 0

    display_count = 0
    inline_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal display_count, inline_count
        attrs = match.group("attrs")
        body = html_module.unescape(match.group("body").strip())
        if re.search(r"\bdisplay\s*=\s*(['\"]?)block\1", attrs, re.IGNORECASE):
            display_count += 1
            return f"$$\n{body}\n$$"
        inline_count += 1
        return f"${body}$"

    return MATH_PATTERN.sub(replace, source), display_count, inline_count


def convert_heading_tags(source: str) -> tuple[str, int]:
    """Convert every non-table HTML heading while preserving its level."""
    if TABLE_PATTERN.match(source):
        return source, 0

    heading_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal heading_count
        heading_count += 1
        level = int(match.group("level"))
        body = html_module.unescape(match.group("body").strip())
        return f"\n\n{'#' * level} {body}\n\n"

    converted = ANY_HEADING_PATTERN.sub(replace, source)
    return converted.strip(), heading_count


def serialize_block(block: dict) -> tuple[str, Counter]:
    source = (block.get("html") or "").strip()
    stats: Counter = Counter()
    if not source or block.get("skipped") or block.get("error"):
        return "", stats

    converted, display_count, inline_count = convert_math_tags(source)
    stats["display_math_tags_converted"] += display_count
    stats["inline_math_tags_converted"] += inline_count

    converted, heading_count = convert_heading_tags(converted)
    stats["html_heading_tags_converted"] += heading_count

    if block.get("label") == "SectionHeader":
        if re.match(r"^\s*#{1,6}\s+", converted):
            stats["section_headers_converted"] += heading_count
        else:
            level = 3
            paragraph_match = PARAGRAPH_PATTERN.match(converted)
            body = (
                paragraph_match.group("body") if paragraph_match else converted
            ).strip()
            if body:
                converted = f"{'#' * level} {body}"
                stats["section_headers_converted"] += 1

    return converted.strip(), stats


def serialize_page(payload: dict) -> tuple[str, Counter]:
    parts = []
    stats: Counter = Counter()
    blocks = sorted(
        payload.get("blocks", []), key=lambda block: block.get("reading_order", 0)
    )
    for block in blocks:
        text, block_stats = serialize_block(block)
        stats.update(block_stats)
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip(), stats
