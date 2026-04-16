import html
import re
from typing import Callable, Optional

import mistune

from docparsingbench.utils.otsl2html import convert_otsl_to_html


class TableFormater:
    """
    TableFormater provides conversion from multiple table formats to HTML.

    Supported input formats:
    - Markdown table
    - LaTeX table (tabular or table environment)
    - OTSL table string
    - HTML table (normalized and cleaned)
    """

    def __init__(self) -> None:
        """
        Initialize TableFormater and create the Markdown parser.
        """
        if hasattr(mistune, "create_markdown"):
            self._markdown_parser = mistune.create_markdown(plugins=["table"])
        else:
            self._markdown_parser = mistune.markdown

        # Dispatch table: maps detected format type to corresponding conversion function.
        self._dispatch: dict[str, Callable[[str], str]] = {
            "markdown": self._markdown_to_html,
            "latex": self._latex_to_html,
            "otsl": self._otsl_to_html,
            "html": self._normalize_html_table,
        }

    def to_html(self, content: str, fmt: Optional[str] = None) -> str:
        """
        Convert input table content to normalized HTML.

        Args:
            content: Raw table content string.
            fmt: Optional format type; auto-detected if None.
                 Options: 'markdown', 'latex', 'otsl', 'html'.

        Returns:
            Converted HTML string containing only <table>, <tr>, <td> tags.
        """
        if not content:
            return ""

        fmt_detected = fmt or self._detect_format(content)
        handler = self._dispatch.get(fmt_detected)
        if handler is None:
            # When format is unrecognized, try normalizing as HTML.
            return self._normalize_html_table(content)

        html_table = handler(content)
        # All branches go through normalization to ensure uniform output.
        return self._normalize_html_table(html_table)

    def _detect_format(self, content: str) -> str:
        """
        Detect the format type of the input table content.

        Detection order:
        1. OTSL: contains specific OTSL tags.
        2. HTML: contains <table> tag.
        3. LaTeX: contains \\begin{tabular} or \\begin{table} structure.
        4. Markdown: contains typical Markdown table separators.

        Args:
            content: Raw table content.

        Returns:
            Format type string.
        """
        text = content.strip()
        if not text:
            return "html"

        # OTSL format detection: contains any OTSL tag.
        if any(tag in text for tag in ("<fcel>", "<ecel>", "<nl>", "<lcel>", "<ucel>", "<xcel>")):
            return "otsl"

        # HTML format detection: contains table opening tag.
        if re.search(r"<\s*table\b", text, flags=re.IGNORECASE):
            return "html"

        # LaTeX table detection: tabular or table environment.
        if re.search(r"\\begin\{tabular\}", text) or re.search(r"\\begin\{table\}", text):
            return "latex"

        # Markdown table detection: contains pipes and separator row.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2 and "|" in lines[0] and re.match(r"^\s*[:\-|\s]+\s*$", lines[1]):
            return "markdown"

        # Fallback: default to Markdown.
        return "markdown"

    def _markdown_to_html(self, content: str) -> str:
        """
        Convert a Markdown table to HTML.

        Args:
            content: Text containing a Markdown table.

        Returns:
            HTML string.
        """
        text = content.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2 and "|" in lines[0] and re.match(
            r"^\s*[:\-|\s]+\s*$", lines[1]
        ):
            header = lines[0]
            data_lines = lines[2:]

            def split_row(row: str):
                return [c.strip() for c in row.strip("|").split("|")]

            rows = [split_row(header)]
            for line in data_lines:
                rows.append(split_row(line))

            html_rows = []
            for cols in rows:
                tds = "".join(
                    f"<td>{html.escape(col)}</td>" for col in cols
                )
                html_rows.append(f"<tr>{tds}</tr>")
            return f"<table>{''.join(html_rows)}</table>"

        html_fragment = self._markdown_parser(text)
        match = re.search(
            r"<table\b.*?</table>",
            html_fragment,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(0)
        return html_fragment

    def _latex_to_html(self, content: str) -> str:
        """
        Convert a LaTeX table environment to HTML.

        Supports \\begin{tabular}...\\end{tabular} structure, parses \\multicolumn and \\multirow.

        Args:
            content: LaTeX table string.

        Returns:
            HTML table string.
        """
        tabular_match = re.search(
            r"\\begin\{tabular\}.*?\}(?P<body>.*?)\\end\{tabular\}",
            content,
            flags=re.DOTALL,
        )
        if tabular_match:
            body = tabular_match.group("body")
        else:
            body = content

        lines = []
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            lines.append(line)

        html_rows = []
        active_multirows = {}

        for line in lines:
            line = re.sub(r"\\hline", "", line)
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\\\\", line)
            for part in parts:
                segment = part.strip()
                if not segment:
                    continue
                raw_cols = [c.strip() for c in segment.split("&")]
                if not raw_cols:
                    continue

                row_cells = []
                col_index = 0

                for col in raw_cols:
                    has_active_multirow = (
                        col_index in active_multirows
                        and active_multirows[col_index] > 0
                    )
                    if has_active_multirow and col == "":
                        active_multirows[col_index] -= 1
                        if active_multirows[col_index] <= 0:
                            del active_multirows[col_index]
                        col_index += 1
                        continue

                    m_multirow = re.match(
                        r"\\multirow\{(\d+)\}\{[^}]*\}\{(.*)\}$", col
                    )
                    if m_multirow:
                        row_span = int(m_multirow.group(1))
                        text_value = m_multirow.group(2).strip()
                        attr = (
                            f' rowspan="{row_span}"' if row_span > 1 else ""
                        )
                        row_cells.append(
                            f"<td{attr}>{html.escape(text_value)}</td>"
                        )
                        if row_span > 1:
                            active_multirows[col_index] = row_span - 1
                        col_index += 1
                        continue

                    m_multicol = re.match(
                        r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", col
                    )
                    if m_multicol:
                        span = int(m_multicol.group(1))
                        text_value = m_multicol.group(2).strip()
                        attr = f' colspan="{span}"' if span > 1 else ""
                        row_cells.append(
                            f"<td{attr}>{html.escape(text_value)}</td>"
                        )
                        col_index += span
                        continue

                    text_value = col
                    row_cells.append(
                        f"<td>{html.escape(text_value)}</td>"
                    )
                    col_index += 1

                if row_cells:
                    html_rows.append(f"<tr>{''.join(row_cells)}</tr>")

        if not html_rows:
            return "<table></table>"
        return f"<table>{''.join(html_rows)}</table>"

    def _otsl_to_html(self, content: str) -> str:
        """
        Convert an OTSL string to HTML.

        Args:
            content: OTSL table string.

        Returns:
            HTML table string.
        """
        return convert_otsl_to_html(content)

    def _normalize_html_table(self, content: str) -> str:
        """
        Normalize an HTML table, keeping only table, tr, td tags and removing all attributes.

        Processing steps:
        1. Extract the first <table>...</table> fragment.
        2. Remove thead, tbody, etc. wrappers, keeping only inner content.
        3. Convert th tags to td.
        4. Remove all attributes from table/tr/td.
        5. Remove all tags other than table/tr/td.

        Args:
            content: HTML string.

        Returns:
            Normalized HTML table string.
        """
        if not content:
            return ""

        text = content.strip()

        # Extract the first table fragment; return empty table if not found.
        table_match = re.search(
            r"<table\b.*?</table>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if table_match:
            table_html = table_match.group(0)
        else:
            return "<table></table>"

        # Convert all th tags to td.
        table_html = re.sub(
            r"</?\s*th\b",
            lambda m: m.group(0).lower().replace("th", "td"),
            table_html,
            flags=re.IGNORECASE,
        )

        # Remove thead, tbody, tfoot, colgroup wrapper tags but keep their inner content.
        for tag in ("thead", "tbody", "tfoot", "colgroup"):
            pattern = rf"</?\s*{tag}\b[^>]*>"
            table_html = re.sub(pattern, "", table_html, flags=re.IGNORECASE)

        # Strip all attributes from table and tr; keep rowspan and colspan on td.
        def _clean_tag(match: re.Match) -> str:
            tag = match.group(1).lower()
            full = match.group(0)
            if tag in ("table", "tr"):
                return f"<{tag}>"
            rowspan_match = re.search(
                r'rowspan\s*=\s*("|\')(\d+)\1', full, flags=re.IGNORECASE
            )
            colspan_match = re.search(
                r'colspan\s*=\s*("|\')(\d+)\1', full, flags=re.IGNORECASE
            )
            attrs = ""
            if rowspan_match:
                attrs += f' rowspan="{rowspan_match.group(2)}"'
            if colspan_match:
                attrs += f' colspan="{colspan_match.group(2)}"'
            return f"<td{attrs}>"

        table_html = re.sub(
            r"<\s*(table|tr|td)\b[^>]*>",
            _clean_tag,
            table_html,
            flags=re.IGNORECASE,
        )

        # Remove all tags except table/tr/td, keeping only their inner text.
        table_html = re.sub(
            r"</?(?!table\b|tr\b|td\b)[a-zA-Z][^>]*>",
            "",
            table_html,
        )

        # Collapse extra whitespace to keep structure compact without altering content.
        table_html = re.sub(r">\s+<", "><", table_html)
        table_html = table_html.strip()

        # Ensure a valid table structure is always returned.
        if not table_html.lower().startswith("<table"):
            table_html = f"<table>{table_html}</table>"
        return table_html


__all__ = ["TableFormater"]
