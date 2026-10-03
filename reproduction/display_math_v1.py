r"""GT-blind normalization for complete Markdown display-math delimiters.

The transformer only inserts blank-line boundaries around already complete
``$$...$$`` and ``\[...\]`` expressions outside HTML tables. It never edits
formula payloads, table payloads, or non-whitespace model content.
"""

from __future__ import annotations

import re

TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
DISPLAY_RE = re.compile(r"\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]")
ONLY_DISPLAY_RE = re.compile(r"(?:\$\$|\\\[)([\s\S]+?)(?:\$\$|\\\])")
BLANK_RE = re.compile(r"(\n[ \t]*\n+)")


def _fix_region(region: str) -> str:
    output: list[str] = []
    for chunk in BLANK_RE.split(region):
        if not chunk or BLANK_RE.fullmatch(chunk):
            output.append(chunk)
            continue

        stripped = chunk.strip()
        display_matches = list(DISPLAY_RE.finditer(chunk))
        nonempty_lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if (
            not display_matches
            or (len(display_matches) == 1 and ONLY_DISPLAY_RE.fullmatch(stripped))
            or (
                nonempty_lines
                and all(
                    len(DISPLAY_RE.findall(line)) == 1
                    and ONLY_DISPLAY_RE.fullmatch(line)
                    for line in nonempty_lines
                )
            )
        ):
            output.append(chunk)
            continue

        leading = chunk[: len(chunk) - len(chunk.lstrip())]
        trailing = chunk[len(chunk.rstrip()) :]
        core = chunk.strip()
        pieces: list[str] = []
        position = 0
        for match in DISPLAY_RE.finditer(core):
            text = core[position : match.start()].strip()
            if text:
                pieces.append(text)
            pieces.append(match.group(0).strip())
            position = match.end()
        text = core[position:].strip()
        if text:
            pieces.append(text)
        output.append(leading + "\n\n".join(pieces) + trailing)
    return "".join(output)


def isolate_display_math(markdown: str) -> str:
    output: list[str] = []
    position = 0
    for table in TABLE_RE.finditer(markdown):
        output.append(_fix_region(markdown[position : table.start()]))
        output.append(table.group(0))
        position = table.end()
    output.append(_fix_region(markdown[position:]))
    return "".join(output)
