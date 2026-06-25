from __future__ import annotations

from pathlib import Path

from PIL import Image

from agent.skill_retriever import SkillRetriever
from data.prepare_zoomearth_parquet import build_record, normalize_row


def test_dataset_record_schema_and_no_label_leak(tmp_path: Path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (1000, 800), "white").save(image_path)
    row = normalize_row(
        {
            "question_id": "q1",
            "image_name": str(image_path),
            "category": "Object color",
            "question": "What color is the left-most bridge?",
            "ground_truth": "white",
            "bbox": [1, 2, 100, 120],
            "type": "object",
            "label": "right-most forbidden label",
            "label_cleaned": "right forbidden",
        }
    )
    assert row is not None
    record = build_record(row, 0, SkillRetriever(enabled=False), tmp_path / "cache")
    assert record["agent_name"] == "zoomearth_full_agent"
    assert record["extra_info"]["index"] == 0
    assert "label" not in record["extra_info"]
    assert "label_cleaned" not in record["extra_info"]


def test_skillbank_uses_question_not_external_label(tmp_path: Path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (1000, 800), "white").save(image_path)
    retriever = SkillRetriever(enabled=True)
    no_locator = retriever.build("What color is the bridge?")
    with_locator = retriever.build("What color is the left-most bridge?")
    assert no_locator["retrieved_skill_ids"] != with_locator["retrieved_skill_ids"]

