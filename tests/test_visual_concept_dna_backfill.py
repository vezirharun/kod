"""Backfill object/concept DNA from previews; no CLIP/DINO rerun."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.db import Database
from core.global_object_intelligence import DetectedObject
from core.object_index import ObjectIndexStore
from core.settings import AppSettings
from core.visual_concept_dna_backfill import backfill_visual_concept_dna


def test_backfill_twenty_cap_uses_preview_not_original(tmp_path: Path):
    cache = tmp_path / "cache"
    data = tmp_path / "data"
    cache.mkdir()
    data.mkdir()
    prev = cache / "p.webp"
    Image.new("RGB", (48, 48), (200, 20, 20)).save(prev, "WEBP")
    orig = tmp_path / "NAS.tif"
    orig.write_bytes(b"do-not-read")
    settings = AppSettings(
        db_path=str(data / "patterns.db"),
        cache_dir=str(cache),
        object_db_path=str(data / "object_index.db"),
        faiss_dino_path=str(data / "d.index"),
        faiss_clip_path=str(data / "c.index"),
        ai_embedding_enabled=False,
    )
    db = Database(settings.db_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    db.upsert_file(
        {
            "path": str(orig),
            "filename": "NAS.tif",
            "source_id": 1,
            "status": "pending",
            "feature_preview_path": str(prev),
            "physical_preview_ready": 1,
        }
    )
    fid = db.get_file_by_path(str(orig))["id"]
    db.upsert_features(
        fid,
        {
            "phash": "aa",
            "clip_embedding": b"\x00" * 16,
            "dino_embedding": b"\x00" * 16,
            "texture_map": {"pattern_dna": {"family": "floral"}},
        },
    )

    def fake_detect(path: str):
        assert Path(path) == prev
        return [
            DetectedObject(
                instance_id="apple_1",
                label="apple",
                label_tr="elma",
                confidence=0.91,
                bbox=(2, 2, 20, 20),
                area_ratio=0.25,
                center=(0.4, 0.4),
            )
        ]

    rep = backfill_visual_concept_dna(settings, limit=20, detector=fake_detect)
    assert orig.read_bytes() == b"do-not-read"
    assert rep["preview_found"] == 1
    assert rep["visual_concept_dna_written"] == 1
    assert rep["objects_detected"] == 1
    assert rep["object_index_files"] == 1
    feat = db.get_features(fid)
    tm = feat["texture_map"]
    assert tm["visual_concept_dna"]
    assert feat.get("clip_embedding") == b"\x00" * 16
    store = ObjectIndexStore(settings.object_db_path, readonly=True)
    assert store.has_scan(fid)
    assert store.concepts_for_file(fid) or store.instances_for_file(fid)
