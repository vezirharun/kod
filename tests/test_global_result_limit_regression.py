from types import SimpleNamespace

from core.pattern_intelligence_v2 import parse_pattern_query_v2, rank_v2_results


def _row(fid, score=0.7):
    return SimpleNamespace(
        file_id=fid,
        score=score,
        score_percent=score * 100,
        filename=f"pattern_{fid}.jpg",
        pattern_family="",
        debug={},
    )


def test_v2_no_hidden_24_cap_for_leaf_categories():
    for query_text in ("leopar", "yılan", "zebra", "çiçek", "geometrik", "papatya", "logo"):
        q = parse_pattern_query_v2(query_text)
        rows = [_row(i) for i in range(1, 61)]
        clips = {i: {q.primary: 0.35} for i in range(1, 61)} if q.primary else {}
        kept, stats = rank_v2_results(q, rows, clip_by_id=clips, max_results=60)
        assert len(kept) == 60, (query_text, stats)


def test_v2_uses_requested_limit_not_category_constant():
    q = parse_pattern_query_v2("leopar")
    rows = [_row(i) for i in range(1, 101)]
    clips = {i: {"leopard": 0.35} for i in range(1, 101)}
    kept, _ = rank_v2_results(q, rows, clip_by_id=clips, max_results=75)
    assert len(kept) == 75


def test_semantic_merge_receives_requested_limit_and_has_no_undefined_limit():
    import inspect
    from core.search_engine import SearchEngine
    src = inspect.getsource(SearchEngine._merge_semantic_intel)
    assert "max_results: int | None = None" in src
    assert "max_results=limit" in inspect.getsource(SearchEngine.search_by_text)

def test_uvi_has_no_hidden_24_fallback():
    import inspect
    from core.universal_visual_intel import apply_universal_ranking
    src = inspect.getsource(apply_universal_ranking)
    assert "leaf_cap = 24" not in src


def test_learned_leaf_retrieval_prioritizes_exact_leopard_over_animal_family(tmp_path):
    """Regression: generic Animal Print FTS hits must not crowd out Leopard."""
    import json
    from core.db import Database

    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        for fid in range(1, 301):
            conn.execute(
                """INSERT INTO files(
                    path, filename, status, pattern_family, pattern_type,
                    pattern_subtype, category_path, manual_category_path
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    f"/patterns/leopard_{fid}.jpg",
                    f"leopard_{fid}.jpg",
                    "indexed",
                    "animal_print",
                    "leopard",
                    "leopard",
                    "Animal Print/Leopard",
                    "Animal Print/Leopard",
                ),
            )
            real_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO features(file_id, texture_map) VALUES(?,?)",
                (
                    real_id,
                    json.dumps(
                        {
                            "pattern_family": "animal_print",
                            "animal_print_type": "leopard",
                            "user_labeled": True,
                        }
                    ),
                ),
            )
        for fid in range(301, 601):
            conn.execute(
                """INSERT INTO files(
                    path, filename, status, pattern_family, pattern_type,
                    pattern_subtype, category_path
                ) VALUES(?,?,?,?,?,?,?)""",
                (
                    f"/patterns/zebra_{fid}.jpg",
                    f"zebra_{fid}.jpg",
                    "indexed",
                    "animal_print",
                    "zebra",
                    "zebra",
                    "Animal Print/Zebra",
                ),
            )
            real_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO features(file_id, texture_map) VALUES(?,?)",
                (
                    real_id,
                    json.dumps(
                        {
                            "pattern_family": "animal_print",
                            "animal_print_type": "zebra",
                        }
                    ),
                ),
            )

    rows = db.search_text_candidates(["leopard"], limit=75)
    assert len(rows) == 75
    assert all(r["pattern_subtype"] == "leopard" for r in rows)


def test_category_leopard_is_valid_leaf_evidence_without_legacy_animal_field():
    from core.semantic_pattern_intel import collect_evidence, gate_result, parse_pattern_intent

    intent = parse_pattern_intent("leopard")
    rec = {
        "id": 1,
        "filename": "unnamed.jpg",
        "status": "indexed",
        "pattern_family": "",
        "pattern_type": "",
        "pattern_subtype": "",
        "category_path": "Animal Print/Leopard",
        "manual_category_path": "Animal Print/Leopard",
        "texture_map": {},
    }
    ev = collect_evidence(intent, rec, clip_score=0.0)
    ok, tier, score = gate_result(intent, ev, text_score=0.0)
    assert ev.animal_type == "leopard"
    assert ev.subtype_hit is True
    assert ok is True
    assert tier == "Very Similar"
    assert score >= 0.78


def test_text_retrieval_cache_key_changes_when_learned_leaf_logic_changes():
    from core.search_cache import TEXT_RETRIEVAL_SCHEMA, search_cache_key
    from core.search_models import SearchQuery
    from types import SimpleNamespace

    settings = SimpleNamespace(
        search_mode="standard",
        color_weight_mode="default",
        customer_filter="",
        search_scope="all",
        selected_source_ids=[],
        ai_embedding_enabled=True,
        semantic_text_search_enabled=True,
        semantic_pattern_intel_enabled=True,
    )
    q = SearchQuery(mode="text", text="leopard")
    key = search_cache_key(settings, q)
    assert TEXT_RETRIEVAL_SCHEMA in key


def test_text_search_rebuilds_semantic_intent_after_fuzzy_correction():
    import inspect
    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine.search_by_text)
    assert "intent_text = corrected_text if was_corrected and corrected_text else text" in src
    assert "intent = parse_pattern_intent(intent_text)" in src
