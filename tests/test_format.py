from docparsingbench.format import TableFormater


def test_markdown_table_to_html_basic():
    md = """
| A | B |
|---|---|
| 1 | 2 |
""".strip()
    formatter = TableFormater()
    html = formatter.to_html(md)
    assert "<table>" in html
    assert "<tr>" in html
    assert html.count("<tr>") == 2
    assert "<td>A</td>" in html
    assert "<td>1</td>" in html
    assert "thead" not in html.lower()
    assert "tbody" not in html.lower()


def test_latex_table_to_html_basic():
    latex = r"""
\begin{tabular}{cc}
a & b \\
c & d \\
\end{tabular}
""".strip()
    formatter = TableFormater()
    html = formatter.to_html(latex)
    assert html == "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"


def test_latex_table_with_multicolumn():
    latex = r"""
\begin{tabular}{ccc}
\multicolumn{2}{c}{A} & B \\
C & D & E \\
\end{tabular}
""".strip()
    formatter = TableFormater()
    html = formatter.to_html(latex)
    assert (
        html
        == '<table><tr><td colspan="2">A</td><td>B</td></tr><tr><td>C</td><td>D</td><td>E</td></tr></table>'
    )


def test_latex_table_with_multirow():
    latex = r"""
\begin{tabular}{cc}
\multirow{2}{*}{A} & B \\
 & C \\
\end{tabular}
""".strip()
    formatter = TableFormater()
    html = formatter.to_html(latex)
    assert (
        html
        == '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'
    )


def test_otsl_table_to_html_basic():
    # Single-column table, two rows, using basic OTSL tags.
    otsl = "<fcel>a<ecel><nl><fcel>b<ecel><nl>"
    formatter = TableFormater()
    html = formatter.to_html(otsl)
    assert "<table>" in html
    assert "a" in html
    assert "b" in html


def test_html_normalization_removes_thead_tbody_and_attrs():
    raw_html = '''
<table class="x" style="border:1px">
  <thead>
    <tr style="color:red"><th>H1</th></tr>
  </thead>
  <tbody>
    <tr class="row"><td style="color:blue">V1</td></tr>
  </tbody>
</table>
'''.strip()
    formatter = TableFormater()
    html = formatter.to_html(raw_html)
    # Keep only table/tr/td tags without attributes.
    assert html == "<table><tr><td>H1</td></tr><tr><td>V1</td></tr></table>"
