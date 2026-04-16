import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any, Tuple

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from docparsingbench.format import TableFormater


SegmentType = Literal["text", "display_formula", "table", "image"]


@dataclass
class Segment:
    """Markdown segment structure.
    - `type`: segment category: text|display_formula|table|image
    - `raw`: raw text of the segment (HTML string for table; formula content for display_formula)
    - `text_no_formula`: text with inline formulas replaced by `[FORMULA]` placeholder
    - `inline_formulas`: list of inline formulas extracted from text (delimiters stripped)
    """
    type: SegmentType
    raw: str
    text_no_formula: Optional[str] = None
    inline_formulas: Optional[List[str]] = None


INLINE_PATTERNS = [
    (r"\$(.+?)\$", False),
    (r"\\\((.+?)\\\)", False),
]


DISPLAY_PATTERNS = [
    (r"\$\$(.+?)\$\$", True),
    (r"\\\[(.+?)\\\]", True),
]


TABLE_PATTERN = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
IMG_PATTERN_MD = re.compile(r"!\[[^\]]*\]\([^\)]*\)")
IMG_PATTERN_HTML = re.compile(r"<img\b[\s\S]*?>", re.IGNORECASE)
IMG_PATTERN_HTML_NONSTANDARD = re.compile(r"<image\b[\s\S]*?>", re.IGNORECASE)
IMG_PATTERN_HTML_BROKEN = re.compile(r"<(?:img|image)\b[^>\n]*(?:>|$)", re.IGNORECASE)
IMG_PATTERN_MD_BROKEN_LINK = re.compile(r"!\[[^\]]*\]\([^)\n]*(?:\)|$)")
IMG_PATTERN_MD_BROKEN_OPEN = re.compile(r"!\[[^\n]*$")

# Markdown pipe table separator row: `|---|`, `| --- |`, `|:---:|` and variants
_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s\|\-:]+\|[\s\|\-:]*$")
_PRESENTATION_WRAPPERS = {"div", "p", "center", "span"}

_table_formater = TableFormater()


def extract_inline_formulas(text: str, placeholder: str = "[FORMULA]") -> Tuple[str, List[str]]:
    """Extract inline formulas and replace them with a placeholder.
    Returns `(text_with_placeholders, formula_list)`.
    The `placeholder` parameter is controlled by `paragraph.formula_placeholder` in config.
    """
    formulas: List[str] = []
    replaced = text
    for pat, _ in INLINE_PATTERNS:
        def _sub(m):
            formulas.append(m.group(1))
            return placeholder
        replaced = re.sub(pat, _sub, replaced)
    return replaced, formulas


def is_only_display_formula(block: str) -> bool:
    for pat, _ in DISPLAY_PATTERNS:
        if re.fullmatch(pat, block.strip(), flags=re.DOTALL):
            return True
    return False


def is_markdown_table(block: str) -> bool:
    """Check whether a text block is a Markdown pipe table (loose matching, supports variants without standard header row).

    Conditions:
    - At least 2 non-empty lines
    - More than half of the lines contain `|`
    - At least 1 separator row (containing only `|`, `-`, `:`, spaces)
    """
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    pipe_count = sum(1 for ln in lines if "|" in ln)
    sep_count = sum(1 for ln in lines if _MD_TABLE_SEP.match(ln) and "-" in ln)
    return pipe_count >= max(2, len(lines) * 0.5) and sep_count >= 1


def _meaningful_children(tag: Tag) -> List[object]:
    children: List[object] = []
    for child in tag.contents:
        if isinstance(child, NavigableString):
            if child.strip():
                children.append(child)
        elif isinstance(child, Tag):
            children.append(child)
    return children


def _normalize_presentation_tag(tag: Tag) -> Optional[str]:
    name = tag.name.lower()
    if name == "img":
        return str(tag).strip()
    if name not in _PRESENTATION_WRAPPERS:
        return None

    children = _meaningful_children(tag)
    if len(children) != 1:
        return None

    child = children[0]
    if isinstance(child, NavigableString):
        text = str(child).strip()
        return text or None

    nested = _normalize_presentation_tag(child)
    if nested is not None:
        return nested.strip()
    if child.name.lower() == "img":
        return str(child).strip()
    return None


def normalize_block(block: str) -> str:
    stripped = block.strip()
    if not stripped.startswith("<") or not stripped.endswith(">"):
        return block

    soup = BeautifulSoup(stripped, "html.parser")
    roots = []
    for node in soup.contents:
        if isinstance(node, NavigableString):
            if node.strip():
                roots.append(node)
        elif isinstance(node, Tag):
            roots.append(node)

    if len(roots) != 1 or not isinstance(roots[0], Tag):
        return block

    normalized = _normalize_presentation_tag(roots[0])
    return normalized if normalized is not None else block


def normalize_markdown(md: str) -> str:
    normalized_lines: List[str] = []
    for line in md.splitlines():
        if not line.strip():
            normalized_lines.append("")
            continue
        normalized_lines.append(normalize_block(line))
    return "\n".join(normalized_lines)


def drop_image_tokens(text: str) -> str:
    """Remove image tokens and leave a space in place to prevent text from merging."""
    original = text
    out = text
    out = IMG_PATTERN_MD.sub(" ", out)
    out = IMG_PATTERN_MD_BROKEN_LINK.sub(" ", out)
    out = IMG_PATTERN_MD_BROKEN_OPEN.sub(" ", out)
    out = IMG_PATTERN_HTML.sub(" ", out)
    out = IMG_PATTERN_HTML_NONSTANDARD.sub(" ", out)
    out = IMG_PATTERN_HTML_BROKEN.sub(" ", out)
    # Common placeholders in datasets, treated as image tokens.
    out = re.sub(r"\b(?:image_url|img_url)\b", " ", out, flags=re.IGNORECASE)
    if out != original:
        out = re.sub(r" {2,}", " ", out)
    return out


def split_markdown(md: str, placeholder: str = "[FORMULA]", drop_img: bool = True) -> List[Segment]:
    r"""Split Markdown text into a list of segments.
    - Table: `<table>...</table>` or Markdown pipe table as a single segment
    - Display formula: `$$...$$`, `\[...\]`
    - Text: aggregated by blank lines/newlines, inline formula placeholders preserved
    - Image: kept or dropped based on drop_img flag
    """
    segments: List[Segment] = []
    # Normalize once at markdown-line level to avoid repeated per-block normalization.
    normalized_md = normalize_markdown(md)

    def append_segment(block: str):
        if not block.strip():
            return

        stripped = block.strip()
        if (
            IMG_PATTERN_MD.fullmatch(stripped)
            or IMG_PATTERN_HTML.fullmatch(stripped)
            or IMG_PATTERN_HTML_NONSTANDARD.fullmatch(stripped)
            or IMG_PATTERN_HTML_BROKEN.fullmatch(stripped)
            or IMG_PATTERN_MD_BROKEN_LINK.fullmatch(stripped)
            or IMG_PATTERN_MD_BROKEN_OPEN.fullmatch(stripped)
        ):
            if not drop_img:
                segments.append(Segment(type="image", raw=stripped))
            return

        if drop_img:
            block = drop_image_tokens(block)
            if not block.strip():
                return

        if is_only_display_formula(block):
            m = re.fullmatch(r"(?:\$\$|\\\[)([\s\S]+?)(?:\$\$|\\\])", block)
            formula = m.group(1) if m else block
            segments.append(Segment(type="display_formula", raw=formula))
            return

        if is_markdown_table(block):
            table_html = _table_formater.to_html(block)
            segments.append(Segment(type="table", raw=table_html))
            return

        text_no_formula, inline_formulas = extract_inline_formulas(block, placeholder=placeholder)
        segments.append(Segment(type="text", raw=block, text_no_formula=text_no_formula, inline_formulas=inline_formulas))

    def process_chunk(chunk: str):
        lines = chunk.splitlines()
        buffer: List[str] = []

        def flush_buffer():
            if not buffer:
                return
            block = "\n".join(buffer)
            buffer.clear()
            if not block.strip():
                return

            if (
                IMG_PATTERN_MD.fullmatch(block)
                or IMG_PATTERN_HTML.fullmatch(block)
                or is_only_display_formula(block)
                or is_markdown_table(block)
            ):
                append_segment(block)
                return

            for line in block.splitlines():
                if line.strip():
                    append_segment(line)

        for ln in lines:
            if not ln.strip():
                flush_buffer()
            else:
                buffer.append(ln)
        flush_buffer()

    pos = 0
    for m in TABLE_PATTERN.finditer(normalized_md):
        pre = normalized_md[pos:m.start()]
        if pre:
            process_chunk(pre)
        table_html = _table_formater.to_html(m.group(0))
        segments.append(Segment(type="table", raw=table_html))
        pos = m.end()
    tail = normalized_md[pos:]
    if tail:
        process_chunk(tail)

    return segments
