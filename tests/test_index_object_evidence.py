"""Preview → RT-DETR object evidence without touching FAISS / originals."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.entity_intelligence import (
    EntityRecord,
    match_score,
    parse_entity_query,
)
from core.global_object_intelligence import DetectedObject
from core.index_object_evidence import attach_preview_object_evidence
from core.object_index import ObjectIndexStore


def test_preview_rtdetr_writes_object_index_not_original(tmp_path):
    preview = tmp_path / "feat.webp"
    Image.new("RGB", (64, 64), (20, 80, 40)).save(preview, "WEBP")
    orig = tmp_path / "source.tif"
    orig.write_bytes(b"not-a-preview")
    obj_db = tmp_path / "object_index.db"

    def fake_detect(_path: str):
        assert Path(_path) == preview
        return [
            DetectedObject(
                instance_id="handbag_01",
                label="handbag",
                label_tr="çanta",
                confidence=0.94,
                bbox=(8, 10, 40, 50),
                area_ratio=0.3,
                center=(0.4, 0.5),
            )
        ]

    goi = attach_preview_object_evidence(
        file_id=7,
        preview_path=str(preview),
        object_db_path=str(obj_db),
        detector=fake_detect,
    )
    assert goi and goi["clip_used_as_detector"] is False
    assert goi["objects"][0]["label"] == "handbag"
    assert goi["objects"][0]["bbox"] == [8, 10, 40, 50]
    store = ObjectIndexStore(obj_db, readonly=True)
    rows = store.instances_for_file(7)
    assert len(rows) == 1
    assert rows[0]["label"] == "handbag"
    assert orig.read_bytes() == b"not-a-preview"


def test_second_pass_skips_completed_scan(tmp_path):
    preview = tmp_path / "feat.webp"
    Image.new("RGB", (32, 32), (10, 10, 10)).save(preview, "WEBP")
    obj_db = tmp_path / "object_index.db"
    calls = {"n": 0}

    def fake_detect(_path: str):
        calls["n"] += 1
        return [
            DetectedObject(
                instance_id="p1",
                label="person",
                label_tr="insan",
                confidence=0.9,
                bbox=(1, 1, 10, 10),
                area_ratio=0.2,
                center=(0.5, 0.5),
            )
        ]

    a = attach_preview_object_evidence(
        file_id=3, preview_path=str(preview), object_db_path=str(obj_db), detector=fake_detect
    )
    b = attach_preview_object_evidence(
        file_id=3, preview_path=str(preview), object_db_path=str(obj_db), detector=fake_detect
    )
    assert a and a["object_count"] == 1
    assert b is None
    assert calls["n"] == 1


def test_multi_entity_detector_and_fusion():
    q = parse_entity_query("kadın ve çanta ve çiçek")
    recs = [
        EntityRecord("Kadın", "person", confidence=0.91, evidence_source="object_detector", bbox=(1, 1, 20, 40)),
        EntityRecord("Çanta", "handbag", confidence=0.94, evidence_source="object_detector", bbox=(21, 8, 50, 40)),
        EntityRecord("Çiçek", "flower", confidence=0.87, evidence_source="openclip"),
    ]
    score, kind, hits = match_score(q, recs)
    assert kind == "fusion"
    assert score >= 0.70
    assert {h.canonical_name for h in hits} >= {"person", "handbag", "flower"}
