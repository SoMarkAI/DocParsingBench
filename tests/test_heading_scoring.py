import pytest

from docparsingbench.config.schema import Config
from docparsingbench.core import evaluate_single
from docparsingbench.markdown_segmenter import split_markdown
from docparsingbench.metrics.text_distance import ned


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("# Title", "Title"),
        ("## Title", "Title"),
        ("###### Title", "Title"),
        ("   ### Title", "Title"),
        ("## Title ##", "Title"),
        ("### C# ###   ", "C#"),
    ],
)
def test_atx_heading_markers_are_removed_from_scoring_text_only(heading, expected):
    segment = split_markdown(heading)[0]

    assert segment.raw == heading
    assert segment.text_no_formula == expected


@pytest.mark.parametrize(
    "text",
    [
        "C# language",
        "Issue #42",
        "#hashtag",
        "####### not an ATX heading",
        "    # indented code",
        r"\# escaped marker",
    ],
)
def test_non_heading_hashes_are_preserved(text):
    assert split_markdown(text)[0].text_no_formula == text


@pytest.mark.parametrize(
    "markdown",
    [
        "```c\n# include <stdio.h>\n```",
        "~~~python\n# N: number of nodes\n~~~",
        "```shell\n# install dependencies",
    ],
)
def test_atx_like_code_lines_are_preserved_inside_fences(markdown):
    segment = next(s for s in split_markdown(markdown) if s.raw.startswith("# "))

    assert segment.text_no_formula == segment.raw
    assert segment.inline_formulas == []


def test_heading_level_differences_do_not_reduce_end_to_end_text_score():
    result = evaluate_single("# Same title", "### Same title ###", Config(), "text")

    assert result["score"] == pytest.approx(1.0)


def test_global_ned_still_counts_heading_markers():
    assert ned("# Same title", "### Same title") > 0.0


def test_markdown_constructs_inside_fence_remain_code_text():
    markdown = "```md\n$$x$$\n<table><tr><td># value</td></tr></table>\n```"
    by_raw = {segment.raw: segment for segment in split_markdown(markdown)}

    assert by_raw["$$x$$"].type == "text"
    html = "<table><tr><td># value</td></tr></table>"
    assert by_raw[html].type == "text"

