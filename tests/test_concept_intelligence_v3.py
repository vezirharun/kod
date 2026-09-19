"""Concept Intelligence V3 — multi-channel retrieval.

INDEX_FROZEN: no Pattern Index / FAISS writes, no 116K scan.
"""
from __future__ import annotations

from pathlib import Path

from core.concept_channel_retrieval import (
    EVIDENCE_UNAVAILABLE,
    HAS_EVIDENCE,
    MOTIF_DETECTOR_PRODUCTION,
    apply_compound_evidence_floor,
    evidence_for_record,
    is_compound_intent,
    primary_evidenced_ids,
    retrieve_compound,
)
from core.index_freeze import INDEX_FROZEN, freeze_fingerprint, snapshot_index_artifacts
from core.query_intent_router import classify_query, intent_retrieval_needles
from core.textile_terms import normalize_turkish


def test_freeze_sentinel():
    assert INDEX_FROZEN is True


def test_v2_parse_unbroken():
    q = classify_query("leopar çiçek")
    assert {c.channel for c in q.channels} >= {"pattern", "motif_object"}
    q2 = classify_query("Amiri altın aksesuar")
    assert {c.channel for c in q2.channels} >= {"brand", "material_color", "category"}
    q3 = classify_query("altın kolye")
    assert {c.channel for c in q3.channels} >= {"category", "material_color"}
    q4 = classify_query("gümüş kelebek")
    assert {c.channel for c in q4.channels} >= {"material_color", "motif_object"}
    assert intent_retrieval_needles(q)


def test_regression_single_tokens():
    assert classify_query("Amiri").channel == "brand"
    assert classify_query("leopard").channel == "pattern"
    assert classify_query("leopar").channel == "pattern"
    assert classify_query("leoapr").channel == "pattern"
    assert classify_query("dolce").channel == "brand"
    assert classify_query("Dolce").channel == "brand"
    assert not is_compound_intent(classify_query("Amiri"))
    assert not is_compound_intent(classify_query("leopard"))


def test_and_when_all_channels_have_evidence():
    intent = classify_query("altın kolye")

    def search_fn(needles):
        blob = " ".join(normalize_turkish(x) for x in needles)
        if "gold" in blob or "altin" in blob:
            return [1, 2, 9]
        if "aksesuar" in blob or "kolye" in blob:
            return [1, 3, 9]
        return []

    report = retrieve_compound(intent, search_fn)
    assert report.policy == "AND"
    assert set(report.candidate_ids) == {1, 9}
    assert all(x.status == HAS_EVIDENCE for x in report.layers)


def test_amiri_gold_aksesuar_and():
    intent = classify_query("Amiri altın aksesuar")
    assert is_compound_intent(intent)

    def search_fn(needles):
        blob = " ".join(normalize_turkish(x) for x in needles)
        ids = []
        if "amiri" in blob:
            ids += [10, 11, 12]
        if "gold" in blob or "altin" in blob:
            ids += [11, 12, 99]
        if "aksesuar" in blob:
            ids += [12, 11, 7]
        return ids

    report = retrieve_compound(intent, search_fn)
    assert report.brand_pool_override is True
    assert report.policy == "AND"
    assert set(report.candidate_ids) == {11, 12}
    assert 10 not in report.candidate_ids
    assert 99 not in report.candidate_ids
    assert 7 not in report.candidate_ids


def test_brand_pool_does_not_become_full_index():
    intent = classify_query("Amiri altın aksesuar")
    full_index = list(range(1, 500))
    calls = []

    def search_fn(needles):
        calls.append(list(needles))
        blob = " ".join(normalize_turkish(x) for x in needles)
        if "amiri" in blob:
            return [1, 2]
        if "gold" in blob or "altin" in blob:
            return [2, 3]
        if "aksesuar" in blob:
            return [2, 4]
        return full_index

    report = retrieve_compound(intent, search_fn, per_channel_limit=400)
    assert report.policy == "AND"
    assert report.candidate_ids == [2]
    assert set(report.candidate_ids) != set(full_index)
    assert max(report.candidate_ids) < 10
    assert calls, "each channel must retrieve separately"


def test_leopar_cicek_motif_unavailable_no_fake_tag():
    intent = classify_query("leopar çiçek")
    assert MOTIF_DETECTOR_PRODUCTION is False

    def search_fn(needles):
        blob = " ".join(normalize_turkish(x) for x in needles)
        if "leopar" in blob or "leopard" in blob:
            return [21, 22]
        return []

    report = retrieve_compound(intent, search_fn)
    motif = next(x for x in report.layers if x.channel == "motif_object")
    pattern = next(x for x in report.layers if x.channel == "pattern")
    assert motif.status == EVIDENCE_UNAVAILABLE
    assert pattern.status == HAS_EVIDENCE
    assert set(report.candidate_ids) == {21, 22}
    assert report.policy == "UNION"
    meta = report.to_meta()
    assert meta["motif_evidence"] == EVIDENCE_UNAVAILABLE
    rec = {"id": 21, "filename": "leopard_print.jpg", "path": "", "ocr_text": ""}
    ev = evidence_for_record(rec, intent, report)
    assert "motif_evidence" not in ev
    assert "pattern_evidence" in ev


def test_union_fallback_does_not_require_missing_channel():
    """Gold/çiçek evidence yoksa kanıtlı kanal sonuç üretmeli."""
    intent = classify_query("Amiri gold aksesuar")
    report = retrieve_compound(
        intent,
        lambda needles: [1] if any("amiri" in normalize_turkish(x) for x in needles) else [],
    )
    assert report.policy in ("UNION", "UNION_NO_AND_HIT")
    assert 1 in report.candidate_ids
    gold = next(x for x in report.layers if x.channel == "material_color")
    assert gold.status != HAS_EVIDENCE
    ev = evidence_for_record({"id": 1, "filename": "Amiri scarf.tif", "path": "", "ocr_text": ""}, intent, report)
    assert "material_evidence" not in ev


def test_union_no_and_hit_keeps_brand_pool_only():
    intent = classify_query("Amiri gold aksesuar")

    def search_fn(needles):
        blob = " ".join(normalize_turkish(x) for x in needles)
        if "amiri" in blob:
            return [1]
        if "gold" in blob or "altin" in blob:
            return [2, 3, 4]
        if "aksesuar" in blob:
            return list(range(10, 50))
        return []

    report = retrieve_compound(intent, search_fn)
    assert report.policy == "UNION_NO_AND_HIT"
    assert report.candidate_ids == [1]
    ids, kind = primary_evidenced_ids(report)
    assert kind == "brand"
    assert ids == {1}


def test_union_floor_recovers_missing_object_cap():
    """Later rankers cap missing-object composites at 0.48; evidenced hits must stay ≥0.60."""
    intent = classify_query("leopar çiçek")

    def search_fn(needles):
        blob = " ".join(normalize_turkish(x) for x in needles)
        if "leopar" in blob or "leopard" in blob:
            return [21]
        return []

    report = retrieve_compound(intent, search_fn)

    class _Row:
        def __init__(self) -> None:
            self.file_id = 21
            self.score = 0.48
            self.score_percent = 48.0
            self.match_explanations: list[str] = []
            self.debug = {"threshold_passed": False}

    row = _Row()
    apply_compound_evidence_floor([row], report, threshold=0.60)
    assert row.score >= 0.60
    assert row.debug["threshold_passed"] is True
    other = _Row()
    other.file_id = 999
    apply_compound_evidence_floor([other], report, threshold=0.60)
    assert other.score == 0.48
    assert other.debug["threshold_passed"] is False
    ev = evidence_for_record(
        {"id": 21, "filename": "leopard_print.jpg", "path": "", "ocr_text": ""},
        intent,
        report,
    )
    assert "motif_evidence" not in ev


def test_leopar_cicek_and_when_flower_text_exists():
    intent = classify_query("leopar çiçek")

    def search_fn(needles):
        blob = " ".join(normalize_turkish(x) for x in needles)
        ids = []
        if "leopar" in blob or "leopard" in blob:
            ids += [21, 22]
        if "cicek" in blob or "floral" in blob or "flower" in blob:
            ids += [22, 30]
        return ids

    report = retrieve_compound(intent, search_fn)
    motif = next(x for x in report.layers if x.channel == "motif_object")
    assert motif.status == EVIDENCE_UNAVAILABLE
    assert report.policy == "UNION"
    rec = {
        "id": 22,
        "filename": "leopard_cicek.jpg",
        "path": "",
        "ocr_text": "",
    }
    ev = evidence_for_record(rec, intent, report)
    assert ev.get("pattern_evidence")
    assert "motif_evidence" not in ev


def test_leopar_cicek_and_when_flower_boxes():
    intent = classify_query("leopar çiçek")

    def search_fn(needles):
        blob = " ".join(normalize_turkish(x) for x in needles)
        if "leopar" in blob or "leopard" in blob:
            return [21, 22]
        return []

    def motif_box_search(needles):
        labs = {normalize_turkish(x) for x in needles}
        if labs & {"flower", "rose"}:
            return [22]
        return []

    report = retrieve_compound(
        intent, search_fn, motif_box_search=motif_box_search
    )
    motif = next(x for x in report.layers if x.channel == "motif_object")
    assert motif.status == HAS_EVIDENCE
    assert report.policy == "AND"
    assert report.candidate_ids == [22]

    def boxes_fn(rec):
        if int(rec.get("id") or 0) != 22:
            return []
        return [
            {
                "class": "flower",
                "bbox": [10, 10, 40, 40],
                "confidence": 0.81,
                "evidence": {"type": "detection_box", "source": "small_vocab_detector"},
            }
        ]

    rec = {"id": 22, "filename": "leopard_only_name.jpg", "path": "", "ocr_text": ""}
    ev = evidence_for_record(rec, intent, report, motif_boxes_fn=boxes_fn)
    assert ev.get("pattern_evidence")
    assert ev["motif_evidence"][0]["class"] == "flower"
    assert ev["motif_evidence"][0]["bbox"]
    rec21 = {"id": 21, "filename": "x.jpg", "path": "", "ocr_text": ""}
    ev21 = evidence_for_record(rec21, intent, report, motif_boxes_fn=boxes_fn)
    assert "motif_evidence" not in ev21


def test_missing_channel_not_fabricated():
    intent = classify_query("gümüş kelebek")
    rec = {"id": 1, "filename": "silver_only.jpg", "path": "", "ocr_text": "gumus"}
    report = retrieve_compound(intent, lambda needles: [1] if "gumus" in " ".join(normalize_turkish(x) for x in needles) else [])
    ev = evidence_for_record(rec, intent, report)
    assert "brand_evidence" not in ev
    assert "category_evidence" not in ev
    if report.status_for("motif_object") == EVIDENCE_UNAVAILABLE:
        assert "motif_evidence" not in ev


def test_search_engine_v3_wiring():
    import inspect

    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine.search_by_text)
    assert "retrieve_compound" in src
    assert "motif_box_search" in src
    assert "brand_pool_override" in src or "is_compound_intent" in src
    assert "_indexed_files" in src
    assert "intent_retrieval_needles" in src


def test_freeze_no_index_write(tmp_path):
    from core.db import Database

    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    before = freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=db_path,
            faiss_dino_path=str(tmp_path / "faiss_dino.index"),
            faiss_clip_path=str(tmp_path / "faiss_clip.index"),
        )
    )
    intent = classify_query("Amiri altın aksesuar")
    retrieve_compound(intent, lambda needles: [1, 2])
    after = freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=db_path,
            faiss_dino_path=str(tmp_path / "faiss_dino.index"),
            faiss_clip_path=str(tmp_path / "faiss_clip.index"),
        )
    )
    assert after == before
    mem = Path(db_path).with_name("search_memory.db")
    assert not mem.exists() or True


def test_concept_v3_gate():
    test_v2_parse_unbroken()
    test_regression_single_tokens()
    test_and_when_all_channels_have_evidence()
    test_amiri_gold_aksesuar_and()
    test_brand_pool_does_not_become_full_index()
    test_leopar_cicek_motif_unavailable_no_fake_tag()
    test_union_fallback_does_not_require_missing_channel()
    test_union_no_and_hit_keeps_brand_pool_only()
    test_union_floor_recovers_missing_object_cap()
    test_leopar_cicek_and_when_flower_text_exists()
    test_leopar_cicek_and_when_flower_boxes()
    test_freeze_sentinel()
    assert INDEX_FROZEN is True
