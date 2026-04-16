from docparsingbench.metrics.text_distance import ned, cer
from docparsingbench.metrics.teds import calculate_teds, calculate_teds_structure_only


def test_ned_cer():
    assert ned("abc", "abc") == 0.0
    assert cer("abc", "abc") == 0.0
    assert 0.0 < ned("abc", "abd") <= 1.0


def test_teds():
    t1 = "<table><tr><td>a</td></tr></table>"
    t2 = "<table><tr><td>a</td></tr></table>"
    assert calculate_teds(t1, t2) == 1.0
    assert calculate_teds_structure_only(t1, t2) == 1.0


def test_teds_span_attribute_diff_affects_scores():
    t1 = '<table><tr><td colspan="2">a</td></tr></table>'
    t2 = '<table><tr><td colspan="3">a</td></tr></table>'
    assert calculate_teds(t1, t2) < 1.0
    assert calculate_teds_structure_only(t1, t2) < 1.0


def test_teds_text_diff_only_affects_teds_not_structure_only():
    t1 = "<table><tr><td>a</td></tr></table>"
    t2 = "<table><tr><td>b</td></tr></table>"
    assert calculate_teds(t1, t2) < 1.0
    assert calculate_teds_structure_only(t1, t2) == 1.0


def test_teds_non_cell_attrs_do_not_affect_scores():
    t1 = '<table data-x="left"><tr><td>a</td></tr></table>'
    t2 = '<table data-x="right"><tr><td>a</td></tr></table>'
    assert calculate_teds(t1, t2) == 1.0
    assert calculate_teds_structure_only(t1, t2) == 1.0
