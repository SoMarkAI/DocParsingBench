import logging
from typing import Dict, Iterable

from bs4 import BeautifulSoup
from apted import APTED, Config  # type: ignore


logger = logging.getLogger(__name__)


class TreeNode:
    def __init__(self, name: str, text: str, attrs: Dict[str, str], *children: "TreeNode"):
        self.name = name
        self.text = text
        self.attrs = attrs
        self.children_ = list(children)


class TableTreeConfig(Config):
    def rename(self, node1: TreeNode, node2: TreeNode):
        if node1.name != node2.name:
            return 1
        if node1.attrs != node2.attrs:
            return 1
        if (node1.text or "").strip() != (node2.text or "").strip():
            return 1
        return 0

    def children(self, node: TreeNode):
        return node.children_


class TableTreeStructureOnlyConfig(Config):
    def rename(self, node1: TreeNode, node2: TreeNode):
        if node1.name != node2.name:
            return 1
        if node1.attrs != node2.attrs:
            return 1
        return 0

    def children(self, node: TreeNode):
        return node.children_


def _node_attrs(element) -> Dict[str, str]:
    if element.name in ("td", "th"):
        return {
            "colspan": str(element.get("colspan", "1")),
            "rowspan": str(element.get("rowspan", "1")),
        }
    return {}


def html_to_tree(element) -> TreeNode:
    children = [html_to_tree(child) for child in element.find_all(recursive=False)]
    text_content = element.get_text(strip=True)
    return TreeNode(element.name, text_content, _node_attrs(element), *children)


def preorder_traversal(node: TreeNode) -> Iterable[TreeNode]:
    yield node
    for child in node.children_:
        yield from preorder_traversal(child)


def _calculate_teds_core(html1_str: str, html2_str: str, structure_only: bool = False) -> float:
    if not html1_str or not html2_str:
        return 1.0 if html1_str == html2_str else 0.0

    try:
        if not html1_str.strip().lower().startswith("<table"):
            html1_str = f"<table>{html1_str}</table>"
        if not html2_str.strip().lower().startswith("<table"):
            html2_str = f"<table>{html2_str}</table>"

        soup1 = BeautifulSoup(html1_str, "html.parser")
        soup2 = BeautifulSoup(html2_str, "html.parser")
        table1 = soup1.find("table")
        table2 = soup2.find("table")
        if not table1 or not table2:
            return 1.0 if table1 == table2 else 0.0
        tree1 = html_to_tree(table1)
        tree2 = html_to_tree(table2)
        config = TableTreeStructureOnlyConfig() if structure_only else TableTreeConfig()
        apted = APTED(tree1, tree2, config)
        distance = apted.compute_edit_distance()
        size1 = sum(1 for _ in preorder_traversal(tree1))
        size2 = sum(1 for _ in preorder_traversal(tree2))
        if size1 + size2 == 0:
            return 1.0
        return 1.0 - (distance / (size1 + size2))
    except Exception as exc:
        logger.debug("TEDS calculation failed: %s", exc, exc_info=True)
        return 0.0


def calculate_teds(html1_str: str, html2_str: str) -> float:
    """Public API for TEDS scoring; kept for backward compatibility."""
    return _calculate_teds_core(html1_str, html2_str, structure_only=False)


def calculate_teds_structure_only(html1_str: str, html2_str: str) -> float:
    """Public API for TEDS-S scoring; kept for backward compatibility."""
    return _calculate_teds_core(html1_str, html2_str, structure_only=True)
