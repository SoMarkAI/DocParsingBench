from docparsingbench.markdown_segmenter import normalize_markdown, split_markdown


def test_split_basic():
    md = """
Text with $a+b$ inline.

$$c=d$$

<table><tr><td>x</td></tr></table>

![](img.png)
""".strip()
    segs = split_markdown(md)
    types = [s.type for s in segs]
    assert types == ["text", "display_formula", "table"]
    assert segs[0].text_no_formula and "[FORMULA]" in segs[0].text_no_formula
    assert segs[0].inline_formulas == ["a+b"]


def test_normalize_presentation_wrappers():
    md = """
<div style="text-align: center;"><img src="imgs/foo.jpg" alt="Image" width="78%" /></div>

<div style="text-align: center;">Figure 3</div>

<p align="center">Figure 1</p>
""".strip()
    normalized = normalize_markdown(md)
    assert '<div style="text-align: center;"><img' not in normalized
    assert "<img" in normalized
    assert "<div" not in normalized
    assert "<p" not in normalized
    assert "Figure 3" in normalized
    assert "Figure 1" in normalized


def test_drop_img_after_normalization_keeps_caption():
    md = """
<div style="text-align: center;"><img src="imgs/foo.jpg" alt="Image" width="78%" /></div>

<div style="text-align: center;">Figure 3</div>
""".strip()
    segs = split_markdown(md, drop_img=True)
    assert [seg.type for seg in segs] == ["text"]
    assert segs[0].raw == "Figure 3"


def test_drop_img_removes_whitespace_padded_image_line():
    md = """
![](imgs/foo.jpg)  
keep
""".strip()
    segs_drop = split_markdown(md, drop_img=True)
    assert [seg.type for seg in segs_drop] == ["text"]
    assert [seg.raw for seg in segs_drop] == ["keep"]

    segs_keep = split_markdown(md, drop_img=False)
    assert [seg.type for seg in segs_keep] == ["image", "text"]


def test_drop_img_keeps_single_text_for_inline_markdown_image():
    md = "prefix![](imgs/foo.jpg)suffix"
    segs = split_markdown(md, drop_img=True)
    assert [seg.type for seg in segs] == ["text"]
    assert [seg.raw for seg in segs] == ["prefix suffix"]
    assert "![" not in segs[0].raw


def test_drop_img_keeps_single_text_for_inline_html_image():
    md = "Note: <img src=image_url>indicates malformed mode."
    segs = split_markdown(md, drop_img=True)
    assert [seg.type for seg in segs] == ["text"]
    assert [seg.raw for seg in segs] == ["Note: indicates malformed mode."]
    assert "<img" not in segs[0].raw.lower()
    assert "image_url" not in segs[0].raw.lower()


def test_drop_img_handles_nonstandard_and_broken_image_tags():
    md = """
<image src=img_url>
X <img src=image_url
Y
""".strip()
    segs = split_markdown(md, drop_img=True)
    assert [seg.type for seg in segs] == ["text", "text"]
    assert [seg.raw for seg in segs] == ["X ", "Y"]


def test_bibliography_is_split_line_by_line():
    md = "\n".join(
        [
            "[60] Ref A.  ",
            "[61] Ref B.  ",
            "[62] Ref C.",
        ]
    )
    segs = split_markdown(md)
    assert [seg.type for seg in segs] == ["text", "text", "text"]
    assert [seg.raw for seg in segs] == ["[60] Ref A.  ", "[61] Ref B.  ", "[62] Ref C."]


def test_plain_multiline_text_is_split_line_by_line():
    md = """
Line one
Line two

Line three
""".strip()
    segs = split_markdown(md)
    assert [seg.raw for seg in segs] == ["Line one", "Line two", "Line three"]


def test_preserve_plain_text_whitespace():
    md = "  indented line  \nsecond line"
    segs = split_markdown(md)
    assert [seg.raw for seg in segs] == ["  indented line  ", "second line"]


def test_tables_and_display_formulas_stay_whole():
    md = """
| A | B |
| - | - |
| 1 | 2 |

$$
a=b
$$
""".strip()
    segs = split_markdown(md)
    assert [seg.type for seg in segs] == ["table", "display_formula"]
