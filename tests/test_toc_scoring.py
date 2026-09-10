import pytest

from docparsingbench.config.schema import Config
from docparsingbench.core import evaluate_single
from docparsingbench.markdown_segmenter import split_markdown
from docparsingbench.metrics.text_distance import ned


@pytest.mark.parametrize(
    ("toc_line", "expected"),
    [
        ("1. Introduction .... 3", "1. Introduction 3"),
        ("Chapter 2.............12", "Chapter 2 12"),
        ("Appendix . . . . . iv", "Appendix iv"),
        ("Methods ........ 12-15", "Methods 12-15"),
        ("Results ........ IX–XII", "Results IX–XII"),
    ],
)
def test_confirmed_toc_dot_leaders_are_removed(toc_line, expected):
    segment = split_markdown("# Contents\n" + toc_line)[1]

    assert segment.raw == toc_line
    assert segment.text_no_formula == expected


@pytest.mark.parametrize(
    "text",
    [
        "field....10",
        "Status ........ 10",
        "The measured value ........ 10",
        "Wait... 3",
        "Version 1.2.3",
    ],
)
def test_isolated_toc_like_ordinary_text_is_preserved(text):
    segment = split_markdown(text)[0]

    assert segment.raw == text
    assert segment.text_no_formula == text


@pytest.mark.parametrize(
    "markdown",
    [
        "```text\nfield....10\n```",
        "~~~shell\nfield....10\n~~~",
        "```python\nfield....10",
    ],
)
def test_toc_like_text_inside_fenced_code_is_preserved(markdown):
    segment = next(s for s in split_markdown(markdown) if s.raw == "field....10")
    assert segment.text_no_formula == "field....10"


def test_two_consecutive_rows_confirm_toc_without_heading():
    segments = split_markdown("Introduction ........ 1\nMethods ............ 5")
    assert [segment.text_no_formula for segment in segments] == [
        "Introduction 1",
        "Methods 5",
    ]


def test_toc_context_stops_at_code_fence_and_ordinary_text():
    markdown = "\n".join(
        [
            "### Contents ###",
            "Introduction ........ 1",
            "```text",
            "# field....10",
            "```",
            "field....10",
            "## Chapter ##",
        ]
    )
    scoring = {segment.raw: segment.text_no_formula for segment in split_markdown(markdown)}

    assert scoring["### Contents ###"] == "Contents"
    assert scoring["Introduction ........ 1"] == "Introduction 1"
    assert scoring["# field....10"] == "# field....10"
    assert scoring["field....10"] == "field....10"
    assert scoring["## Chapter ##"] == "Chapter"


def test_heading_levels_and_toc_leaders_normalize_together_end_to_end():
    gt = "# Contents\nIntroduction ............ 3\n# Chapter"
    pred = "### Contents ###\nIntroduction .... 3\n#### Chapter ####"

    result = evaluate_single(gt, pred, Config(), "text")

    assert result["score"] == pytest.approx(1.0)


def test_global_ned_still_counts_toc_dot_leader_differences():
    assert ned("Introduction ........ 3", "Introduction .... 3") > 0.0
