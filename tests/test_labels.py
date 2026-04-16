import json

import pytest

from docparsingbench.labels import LabelsError, resolve_labels_path


def test_resolve_labels_path_generates_from_dataset_root(tmp_path):
    dataset_dir = tmp_path / "DocParsingBench"
    gt_dir = dataset_dir / "markdowns"
    images_dir = dataset_dir / "images"
    gt_dir.mkdir(parents=True)
    images_dir.mkdir()
    (dataset_dir / "README.md").write_text("dataset readme", encoding="utf-8")

    (gt_dir / "education_chemistry_00001.md").write_text("hello", encoding="utf-8")
    (gt_dir / "finance_research_report_00002.md").write_text("world", encoding="utf-8")
    (images_dir / "education_chemistry_00001.png").write_text("img", encoding="utf-8")
    (images_dir / "finance_research_report_00002.jpg").write_text("img", encoding="utf-8")

    labels_path = resolve_labels_path(None, dataset_dir)

    assert labels_path == dataset_dir / "labels.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    assert payload["label_schema"] == [
        {
            "industry": "education",
            "sub_industries": [{"sub-industry": "chemistry"}],
        },
        {
            "industry": "finance",
            "sub_industries": [{"sub-industry": "research_report"}],
        },
    ]
    assert payload["data"] == [
        {
            "img": "education_chemistry_00001.png",
            "md": "education_chemistry_00001.md",
            "industry": "education",
            "sub-industry": "chemistry",
        },
        {
            "img": "finance_research_report_00002.jpg",
            "md": "finance_research_report_00002.md",
            "industry": "finance",
            "sub-industry": "research_report",
        },
    ]


def test_resolve_labels_path_allows_missing_images(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "education_chemistry_00001.md").write_text("hello", encoding="utf-8")

    labels_path = resolve_labels_path(None, gt_dir)

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    assert payload["data"][0]["img"] == ""


def test_resolve_labels_path_errors_for_missing_explicit_path(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "education_chemistry_00001.md").write_text("hello", encoding="utf-8")

    with pytest.raises(LabelsError) as exc_info:
        resolve_labels_path(tmp_path / "missing.json", gt_dir)

    assert "Explicit labels path does not exist" in str(exc_info.value)
