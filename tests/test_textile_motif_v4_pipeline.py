"""Textile Motif V4 pipeline: isolated dataset, dry-run, acceptance. INDEX FROZEN."""
from __future__ import annotations

import json
from pathlib import Path

from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.settings import DEFAULT_DATA_DIR
from core.textile_motif_v4 import (
    FIXTURE_DIR,
    GATE_CLASSES,
    SPATIAL_EVIDENCE_UNAVAILABLE,
    ensure_dummy_images,
    load_annotations,
    run_acceptance,
    spatial_status,
)

REPORT = Path(__file__).resolve().parents[1] / "data" / "reports" / "textile_motif_v4.json"


def _snap() -> dict:
    return snapshot_index_artifacts(
        db_path=DEFAULT_DATA_DIR / "patterns.db",
        faiss_dino_path=DEFAULT_DATA_DIR / "faiss_dino.index",
        faiss_clip_path=DEFAULT_DATA_DIR / "faiss_clip.index",
        cache_dir="",
    )


def _faiss_part(fp: dict) -> dict:
    return {k: v for k, v in fp.items() if "faiss" in k.replace("\\", "/").lower()}


def test_v4_pipeline_not_ready(tmp_path: Path) -> None:
    before = freeze_fingerprint(_snap())
    ensure_dummy_images(FIXTURE_DIR)
    data = load_annotations(FIXTURE_DIR / "annotations.json")
    assert data["dummy"] is True
    assert data["isolated_from_pattern_index"] is True

    payload = run_acceptance(FIXTURE_DIR, work=tmp_path, pred_mode="none")
    after = freeze_fingerprint(_snap())

    assert payload["dataset"]["dummy"] is True
    assert payload["model"]["clip_as_detector"] is False
    assert payload["model"]["fine_tune_required"] is True
    assert payload["training"]["compiled"] is True
    assert payload["benchmark"]["metrics_are_textile_quality"] is False
    assert payload["spatial"]["status"] == SPATIAL_EVIDENCE_UNAVAILABLE
    assert payload["spatial"]["invented_boxes"] is False
    assert spatial_status(reliable_boxes=False) == SPATIAL_EVIDENCE_UNAVAILABLE
    assert payload["production_gate"]["status"] == "NOT READY"
    assert payload["production_gate"]["production_ready"] is False
    assert payload["wiring"]["coco_as_motif"] is False
    assert payload["wiring"]["production_activated"] is False
    assert payload["benchmark"]["independent_gt"] is True
    assert payload["benchmark"]["gt_is_model_output"] is False
    for c in GATE_CLASSES:
        assert payload["production_gate"]["class_ready"][c] is False
    assert payload["index_freeze"]["archive_116k_scanned"] is False
    assert payload["index_freeze"]["reindex"] is False
    assert payload["index_freeze"]["this_task_wrote_pattern_index"] is False
    assert _faiss_part(after) == _faiss_part(before)

    payload["index_freeze"]["fingerprint_changed"] = after != before
    payload["index_freeze"]["faiss_changed"] = _faiss_part(after) != _faiss_part(before)
    payload["index_freeze"]["fingerprint_stable"] = after == before
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    v2 = REPORT.with_name("textile_motif_v2.json")
    v2.write_text(json.dumps(payload, indent=2), encoding="utf-8")
