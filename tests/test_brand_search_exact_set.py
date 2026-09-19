from types import SimpleNamespace

from core.category_tree import resolve_category_query
from core.search_engine import SearchEngine, SearchResult


def _fake_result(rec, score, breakdown):
    return SearchResult(
        file_id=int(rec["id"]),
        path=str(rec.get("path") or ""),
        filename=str(rec.get("filename") or ""),
        customer="",
        thumbnail_path="",
        score=float(score),
        score_percent=float(score) * 100,
        breakdown=dict(breakdown),
        debug={},
    )


def _engine(records):
    eng = SearchEngine.__new__(SearchEngine)
    eng.settings = SimpleNamespace(
        search_result_limit=50,
        similarity_threshold=0.30,
        customer_filter="",
        semantic_pattern_intel_enabled=False,
        semantic_text_search_enabled=False,
        pattern_intelligence_v2_enabled=False,
        universal_visual_intel_enabled=False,
        search_text_visual=False,
    )
    eng._intel_enabled = lambda: False
    eng._clip_text_hits = lambda *a, **k: ({}, {})
    eng._records_with_family_overrides = lambda rows: rows
    eng._search_filter = lambda customer: SimpleNamespace(source_types=lambda: [], source_ids=[])
    eng._record_in_scope = lambda rec, customer: True
    eng._candidate_clip_scores = lambda *a, **k: {}
    eng._to_result = _fake_result
    eng._last_text_intel_meta = {}
    eng._last_clip_rivals = {}
    eng._last_v2_concepts = {}
    eng._feedback = SimpleNamespace()
    eng._indexed_files = lambda customer, lightweight=True: list(records)
    eng.db = SimpleNamespace(
        search_filename_path_candidates=lambda *a, **k: [],
        search_text_candidates=lambda *a, **k: [],
        get_indexed_files_by_ids=lambda ids, **k: [r for r in records if int(r["id"]) in {int(x) for x in ids}],
        get_file_by_id=lambda fid: next((r for r in records if int(r["id"]) == int(fid)), None),
    )
    return eng


def test_brand_specific_and_general_share_exact_evidence_set():
    records = [
        {"id": 1, "filename": "amiri-monogram.jpg", "ocr_text": "", "category_path": "Monogram Logo", "texture_map": {"brand_name": "Amiri"}},
        {"id": 2, "filename": "dior-logo.png", "ocr_text": "", "category_path": "Monogram Logo", "texture_map": {"semantic_tags": {"brand_references": ["dior"]}}},
        {"id": 3, "filename": "louis-vuitton.jpg", "ocr_text": "", "category_path": "Marka/Louis Vuitton", "texture_map": {}},
        {"id": 4, "filename": "ordinary-floral.jpg", "ocr_text": "", "category_path": "Floral", "texture_map": {}},
    ]
    eng = _engine(records)
    emitted = []
    general = eng.search_by_text("marka", limit=50, result_callback=lambda rows, n, total: emitted.append((len(rows), n, total)))
    assert emitted and emitted[-1][0] == 3 and emitted[-1][1] == 3
    amiri = eng.search_by_text("amiri", limit=50)
    dior = eng.search_by_text("dior", limit=50)
    assert {r.file_id for r in general} == {1, 2, 3}
    assert {r.file_id for r in amiri} == {1}
    assert {r.file_id for r in dior} == {2}
    assert resolve_category_query("marka").category_path == "Marka"


def test_done_job_reopens_when_planner_reports_physical_gap(tmp_path):
    from core.index_v3.queues import JobStore
    from core.index_v3.types import Artifact, Job, QueueKind

    store = JobStore(tmp_path / "jobs.db")
    job = Job(file_id=7, artifact=Artifact.THUMBNAIL, queue=QueueKind.LIGHT, source_id=1, path="x.jpg")
    assert store.enqueue([job]) == 1
    claimed = store.claim(QueueKind.LIGHT, "test")
    assert claimed
    store.complete(7, Artifact.THUMBNAIL)
    assert store.job_state(7, Artifact.THUMBNAIL) == "done"
    assert store.enqueue([job], reopen_done=True) == 1
    assert store.job_state(7, Artifact.THUMBNAIL) == "pending"


def _brand_rec(fid: int, filename: str, brand: str) -> dict:
    return {
        "id": fid,
        "filename": filename,
        "path": filename,
        "ocr_text": "",
        "category_path": f"Marka/{brand}",
        "texture_map": {"brand_name": brand},
    }


def test_brand_parent_still_uses_full_index():
    records = [
        _brand_rec(1, "amiri-x.jpg", "Amiri"),
        _brand_rec(2, "dior-x.jpg", "Dior"),
    ]
    calls = {"n": 0}
    eng = _engine(records)
    eng._indexed_files = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or list(records))
    eng.search_by_text("marka", limit=50)
    assert calls["n"] >= 1


def test_brand_child_skips_full_scan_when_filename_hits():
    filler = [_brand_rec(99, "other.jpg", "Gucci")]
    hit = [_brand_rec(1, "Amiri scarf.tif", "Amiri")]
    calls = {"n": 0}
    eng = _engine(filler)
    eng._indexed_files = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or list(filler))
    eng.db.search_filename_path_candidates = lambda *a, **k: hit
    out = eng.search_by_text("amiri", limit=50)
    assert calls["n"] == 0
    assert {r.file_id for r in out} == {1}


def test_brand_child_falls_back_to_full_index_when_fast_empty():
    records = [_brand_rec(1, "Amiri scarf.tif", "Amiri")]
    calls = {"n": 0}
    eng = _engine(records)
    eng._indexed_files = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or list(records))
    out = eng.search_by_text("amiri", limit=50)
    assert calls["n"] >= 1
    assert {r.file_id for r in out} == {1}


def test_brand_child_queries_use_fast_pool_only():
    hits = {
        "amiri": [_brand_rec(1, "Amiri scarf.tif", "Amiri")],
        "versace": [_brand_rec(2, "versace-print.jpg", "Versace")],
        "louise": [_brand_rec(3, "louis-vuitton.jpg", "Louis Vuitton")],
        "dolce": [_brand_rec(4, "dolce-gabbana.jpg", "Dolce Gabbana")],
        "amrii": [_brand_rec(1, "Amiri scarf.tif", "Amiri")],
        "versacce": [_brand_rec(2, "versace-print.jpg", "Versace")],
    }
    expected = {
        "amiri": {1},
        "versace": {2},
        "louise": {3},
        "dolce": {4},
        "amrii": {1},
        "versacce": {2},
    }
    filler = [_brand_rec(99, "unrelated.jpg", "Gucci")]
    for query, recs in hits.items():
        calls = {"n": 0}
        eng = _engine(filler)
        eng._indexed_files = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or list(filler))
        eng.db.search_filename_path_candidates = lambda *a, **k: recs
        out = eng.search_by_text(query, limit=50)
        assert calls["n"] == 0, query
        assert {r.file_id for r in out} == expected[query], query


def test_brand_evidence_outranks_visual_lookalike():
    records = [
        {
            "id": 1,
            "filename": "metadata-only.jpg",
            "ocr_text": "",
            "category_path": "Monogram Logo",
            "texture_map": {"brand_name": "Amiri", "pattern_dna": {"brand": "Amiri"}},
        },
        {
            "id": 4,
            "filename": "lookalike-floral.jpg",
            "ocr_text": "",
            "category_path": "Floral",
            "texture_map": {},
        },
    ]
    eng = _engine(records)
    eng._clip_text_hits = lambda *a, **k: (
        {4: 0.99, 1: 0.22},
        {"clip_attempted": True, "clip_hits": 2},
    )
    eng._candidate_clip_scores = lambda fid, *a, **k: {
        "_similarity": 0.99 if int(fid) == 4 else 0.22
    }
    out = eng.search_by_text("amiri", limit=50)
    assert [r.file_id for r in out][0] == 1
    ids = [r.file_id for r in out]
    assert 1 in ids
    if 4 in ids:
        assert ids.index(1) < ids.index(4)
        brand = next(r for r in out if r.file_id == 1)
        look = next(r for r in out if r.file_id == 4)
        assert float(brand.breakdown.get("brand_evidence_hit") or 0) >= 0.5
        assert float(look.breakdown.get("brand_evidence_hit") or 0) < 0.5
        assert float(brand.score) >= float(look.score)


def test_nike_adidas_gucci_resolve_as_brand():
    from core.brand_aliases import resolve_brand_alias
    from core.category_tree import resolve_category_query
    from core.query_intent_router import classify_query

    for q in ("nike", "adidas", "gucci", "amiri"):
        assert resolve_brand_alias(q)
        hit = resolve_category_query(q)
        assert hit.primary_family == "marka"
        route = classify_query(q)
        assert route.channel == "brand"

