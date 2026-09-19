"""Archive Intelligence vertical-slice tests (A–H)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from core.archive_intelligence.consistency import score_file_consistency
from core.archive_intelligence.family_graph import (
    family_snapshot,
    may_merge_as_same_family,
)
from core.archive_intelligence.scanner import ArchiveIntelligenceScanner, _search_busy
from core.archive_intelligence.store import ArchiveIntelligenceStore
from core.archive_intelligence.time_fields import labeled_time_fields
from core.background_index import is_search_active, set_search_active


def test_a_consistency_coherent_vs_conflict():
    ok = score_file_consistency(
        {"pattern_family": "leopard", "pattern_confidence": 0.9},
        {"pattern_family": "leopard", "classification_confidence": 0.9},
    )
    assert ok.kind == "ok"
    assert ok.score >= 0.55

    bad = score_file_consistency(
        {"pattern_family": "leopard", "pattern_confidence": 0.9},
        {
            "pattern_family": "leopard",
            "animal_print_type": "tiger",
            "classification_confidence": 0.9,
            "pattern_dna": {"motif_class": "tiger"},
        },
    )
    assert bad.kind in ("outlier", "suspicious")
    assert bad.score < ok.score


def test_b_outlier_candidate_store(tmp_path: Path):
    store = ArchiveIntelligenceStore(str(tmp_path / "ai.db"))
    store.upsert_candidate(
        file_id=7,
        kind="outlier",
        reason="Çelişen kavram: leopard ≠ tiger",
        consistency=0.2,
        guess="leopard",
    )
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0]["kind"] == "outlier"
    assert pending[0]["file_id"] == 7


def test_c_concept_identity_leopard_ne_tiger():
    assert may_merge_as_same_family("Leopard", "Tiger") is False
    assert may_merge_as_same_family("Leopard", "leopard") is True


def test_d_family_variants_same_concept():
    snap = family_snapshot(
        {"filename": "lv_brown.tif", "pattern_family": "Louis Vuitton"},
        {
            "pattern_family": "Louis Vuitton",
            "color_family": "brown_tan",
            "pattern_dna": {"scale": "large", "confidence": 0.8},
        },
    )
    assert snap["auto_merge_allowed"] is False
    assert snap["family"]
    assert "variant" in snap
    # Same concept scale variants are not distinct-concept merges
    assert may_merge_as_same_family("tiger", "Tiger") is True


def test_e_time_query_labeling():
    fields = labeled_time_fields(
        {
            "mtime": 1_700_000_000.0,
            "indexed_at": "2024-01-01T00:00:00+00:00",
            "feature_preview_mtime": 1_700_000_100.0,
        }
    )
    by_key = {f.key: f for f in fields}
    assert by_key["mtime"].source == "filesystem_mtime"
    assert by_key["mtime"].is_usage_date is False
    assert "kullanım" in by_key["mtime"].note.lower() or "kullanim" in by_key["mtime"].note.lower()
    assert by_key["indexed_at"].source == "index_clock"
    assert by_key["usage_date"].source == "unavailable"
    assert by_key["usage_date"].is_usage_date is True


def test_f_review_to_teach_resolve(tmp_path: Path):
    store = ArchiveIntelligenceStore(str(tmp_path / "ai.db"))
    store.upsert_candidate(
        file_id=3,
        kind="suspicious",
        reason="Güven düşük",
        consistency=0.3,
        guess="flower",
    )
    n = store.resolve_files([3], status="resolved")
    assert n == 1
    assert store.list_pending() == []


def test_g_background_does_not_block_when_search_active():
    set_search_active(True)
    try:
        assert is_search_active() is True
        assert _search_busy() is True
        # Scanner start must be non-blocking and respect flag
        class _S:
            archive_intelligence_enabled = True
            db_path = ""
            archive_intelligence_db_path = ""

        sc = ArchiveIntelligenceScanner(
            _S(), batch_size=5, idle_sleep=0.05, busy_sleep=0.05, startup_delay=0.0
        )
        assert sc.start() is True
        time.sleep(0.15)
        assert sc._thread is not None and sc._thread.is_alive()
        sc.stop(timeout=1.0)
    finally:
        set_search_active(False)


def test_h_generalization_multi_concept():
    for concept in ("flower", "tiger", "zebra", "marble"):
        report = score_file_consistency(
            {"pattern_family": concept, "pattern_confidence": 0.4},
            {"pattern_family": concept, "classification_confidence": 0.4},
        )
        assert report.kind in ("suspicious", "ok")
        # low conf alone → suspicious for any concept (no concept-specific branch)
        assert report.score < 0.9
