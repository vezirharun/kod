"""Concept Intelligence V2 — compound query channels.

INDEX_FROZEN: no Pattern Index / FAISS writes, no 116K scan.
"""
from __future__ import annotations

from pathlib import Path

from core.index_freeze import INDEX_FROZEN, freeze_fingerprint, snapshot_index_artifacts
from core.query_intent_router import classify_query, intent_retrieval_needles
from core.search_memory import learn_concept_token, memory_db_path
from core.textile_terms import normalize_turkish


def _kinds(q) -> set[str]:
    return {c.channel for c in q.channels}


def _values(q) -> set[str]:
    return {normalize_turkish(c.value) for c in q.channels}


_COMPOUND_TABLE = [
    (
        "altın kolye",
        {"category", "material_color"},
        {"aksesuar", "aksesuar/kolye", "gold"},
    ),
    (
        "gümüş kelebek",
        {"material_color", "motif_object"},
        {"gumus", "kelebek"},
    ),
    (
        "Amiri altın aksesuar",
        {"brand", "material_color", "category"},
        {"amiri", "gold", "aksesuar"},
    ),
    (
        "Dolce dantel",
        {"brand", "texture_style"},
        {"dolce gabbana", "dantel"},
    ),
    (
        "leopar çiçek",
        {"pattern", "motif_object"},
        {"leopard", "floral"},
    ),
]


def test_compound_parse_table():
    rows = []
    for query, exp_kinds, exp_vals in _COMPOUND_TABLE:
        q = classify_query(query)
        got_k, got_v = _kinds(q), _values(q)
        rows.append((query, sorted(exp_kinds), sorted(got_k), sorted(exp_vals), sorted(got_v)))
        assert exp_kinds <= got_k, f"{query}: kinds {got_k} expected {exp_kinds}"
        assert exp_vals <= got_v, f"{query}: values {got_v} expected {exp_vals}"
        assert len(q.channels) >= 2, f"{query}: not compound"
        needles = [normalize_turkish(x) for x in intent_retrieval_needles(q)]
        assert needles, f"{query}: empty handoff needles"
        for kind in exp_kinds:
            assert any(c.channel == kind for c in q.channels)


def test_oov_not_invented():
    q = classify_query("blorple kolye")
    assert "category" in _kinds(q)
    assert "blorple" in q.unknown_tokens
    assert not any("blorple" == c.value for c in q.channels)


def test_user_taught_token_persists(tmp_path):
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
    cid = learn_concept_token(db_path, "zumrut", "material_color")
    assert cid
    q = classify_query("zumrut kolye", db_path=db_path)
    assert "material_color" in _kinds(q)
    assert "category" in _kinds(q)
    assert "zumrut" in _values(q)
    after = freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=db_path,
            faiss_dino_path=str(tmp_path / "faiss_dino.index"),
            faiss_clip_path=str(tmp_path / "faiss_clip.index"),
        )
    )
    assert after == before
    assert Path(memory_db_path(db_path)).is_file()


def test_freeze_sentinel():
    assert INDEX_FROZEN is True


def test_regression_amiri_leopard_leoapr():
    assert classify_query("Amiri").channel == "brand"
    assert classify_query("leopard").channel == "pattern"
    assert classify_query("leopar").channel == "pattern"
    assert classify_query("leoapr").channel == "pattern"


def test_search_engine_handoff_wiring():
    import inspect

    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine.search_by_text)
    assert "intent_retrieval_needles" in src
    assert "concept_needles" in src
    assert "short_terms += _ch_needles" in src or "_ch_needles" in src


def test_concept_v2_gate():
    """Parse+wiring PASS. Live 116K retrieval is not claimed (index frozen)."""
    test_compound_parse_table()
    test_oov_not_invented()
    test_regression_amiri_leopard_leoapr()
    test_search_engine_handoff_wiring()
    test_freeze_sentinel()
    assert INDEX_FROZEN is True
