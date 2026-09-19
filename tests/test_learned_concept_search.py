"""Learned concepts participate in text search; count matches the rendered list."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.concept_registry import concepts, learn
from core.db import Database
from core.learned_concept_search import (
    LEARNED_REASON,
    apply_learned_to_results,
    collect_learned_hits,
    conflicting_taught_ids,
    filter_taught_conflicts,
    max_clip_to_examples,
    query_matches_verified,
    resolve_learned_concept,
)
from core.search_models import SearchResponse, results_for_display
from core.teach_me import teach_files


def _db(tmp_path):
    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db


def _add(db: Database, tmp_path, name: str, *, clip=None, family: str = "") -> int:
    img = tmp_path / name
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": name,
                "source_id": 1,
                "status": "indexed",
                "pattern_family": family,
            }
        )
    )
    payload = {}
    if clip is not None:
        payload["clip_embedding"] = clip
    if payload:
        db.upsert_features(fid, payload)
    return fid


def test_ananas_exact_examples_are_authoritative(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    ids = [_add(db, tmp_path, f"a{i}.jpg") for i in range(10)]
    teach_files(db, path, ids, "Ananas")
    pack = collect_learned_hits(path, "ananas", db=db)
    assert pack["canonical"] == "Ananas"
    assert set(pack["exact_ids"]) == set(ids)
    for fid in ids:
        assert pack["file_scores"][fid] >= 0.96


def test_clip_neighbors_are_not_limited_to_training_set(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    proto = np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
    near = np.array([0.95, 0.05, 0, 0], dtype=np.float32).tobytes()
    far = np.array([0, 1, 0, 0], dtype=np.float32).tobytes()
    taught = [_add(db, tmp_path, f"z{i}.jpg", clip=proto) for i in range(3)]
    extra = _add(db, tmp_path, "other_chain.jpg", clip=near)
    noise = _add(db, tmp_path, "unrelated.jpg", clip=far)
    teach_files(db, path, taught, "Zincir")
    pack = collect_learned_hits(path, "zincir", db=db)
    assert extra in pack["file_scores"]
    assert extra not in pack["exact_ids"]
    assert noise not in pack["file_scores"]
    assert max_clip_to_examples(near, [proto]) >= 0.72


def test_learned_reason_survives_generic_score():
    pack = {
        "exact_ids": [1],
        "neighbor_scores": {2: 0.88},
        "file_scores": {1: 0.96, 2: 0.88},
    }
    weak = SimpleNamespace(
        file_id=1, score=0.2, score_percent=20, debug={}, match_explanations=[]
    )
    neigh = SimpleNamespace(
        file_id=2, score=0.1, score_percent=10, debug={}, match_explanations=["FTS"]
    )
    out = apply_learned_to_results([weak, neigh], pack)
    assert out[0].debug["learned_concept_exact"] is True
    assert LEARNED_REASON in out[0].match_explanations
    assert out[0].score >= 0.96
    assert out[1].debug["learned_concept"] is True
    assert LEARNED_REASON in out[1].match_explanations


def test_edit_relabel_changes_search_concept(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    fid = _add(db, tmp_path, "x.jpg")
    teach_files(db, path, [fid], "Zebra")
    assert resolve_learned_concept(path, "zebra")
    learn(path, "Leopard", file_id=fid, concept_type="visual_concept")
    assert resolve_learned_concept(path, "leopard")
    pack = collect_learned_hits(path, "leopard", db=db)
    assert fid in pack["exact_ids"]


def test_compound_kucuk_kaplan_is_tiger_exact(tmp_path):
    """Stage 2C: 'küçük kaplan' must resolve Tiger EXACT, not RELATED."""
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    tiger = _add(db, tmp_path, "tiger.jpg")
    leopard = _add(db, tmp_path, "leo.jpg")
    teach_files(db, path, [tiger], "Tiger")
    teach_files(db, path, [leopard], "Leopard")
    from core.concept_registry import upsert

    upsert(path, "Tiger", aliases=["kaplan", "Tiger"], parent="Animal Print")
    for q in ("kaplan", "küçük kaplan", "yoğun küçük siyah krem kaplan"):
        pack = collect_learned_hits(path, q, db=db)
        assert pack.get("canonical") in {"Tiger", "kaplan"} or "Tiger" in str(
            pack.get("canonical")
        )
        assert tiger in (pack.get("exact_ids") or []), q
        assert leopard not in (pack.get("exact_ids") or []), q
        # Exact floor — not RELATED 0.88
        assert float(pack["file_scores"][tiger]) >= 0.96, q


def test_concepts_survive_reopen_for_search(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    fid = _add(db, tmp_path, "p.jpg")
    teach_files(db, path, [fid], "Parfüm")
    again = concepts(path)
    assert any(str(r["canonical"]) == "Parfüm" for r in again)
    assert resolve_learned_concept(path, "parfüm")
    assert resolve_learned_concept(path, "parfum") or resolve_learned_concept(path, "Parfüm")


def test_taught_kedi_is_kept_for_kedi_and_dropped_for_elma(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    fid = _add(db, tmp_path, "cat.jpg")
    unlabeled = _add(db, tmp_path, "plain.jpg")
    teach_files(db, path, [fid], "kedi")
    assert fid in collect_learned_hits(path, "kedi", db=db)["exact_ids"]
    assert fid in collect_learned_hits(path, "KEDİ", db=db)["exact_ids"]
    elma_pack = collect_learned_hits(path, "elma", db=db)
    assert fid not in (elma_pack.get("exact_ids") or [])
    assert fid not in (elma_pack.get("file_scores") or {})
    blocked = conflicting_taught_ids(path, "elma")
    assert fid in blocked
    assert unlabeled not in blocked
    assert fid not in conflicting_taught_ids(path, "kedi")
    row = SimpleNamespace(file_id=fid, score=0.8, debug={})
    other = SimpleNamespace(file_id=unlabeled, score=0.7, debug={})
    assert [r.file_id for r in filter_taught_conflicts([row, other], path, "elma")] == [
        unlabeled
    ]
    assert fid in {r.file_id for r in filter_taught_conflicts([row], path, "kedi")}


def test_taught_barok_and_elma_do_not_cross_queries(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    barok = _add(db, tmp_path, "baroque.jpg")
    apple = _add(db, tmp_path, "apple.jpg")
    teach_files(db, path, [barok], "barok")
    teach_files(db, path, [apple], "elma")
    assert barok in collect_learned_hits(path, "barok", db=db)["exact_ids"]
    assert apple in collect_learned_hits(path, "elma", db=db)["exact_ids"]
    assert barok in conflicting_taught_ids(path, "elma")
    assert apple in conflicting_taught_ids(path, "barok")
    assert barok not in conflicting_taught_ids(path, "barok")
    assert apple not in conflicting_taught_ids(path, "elma")
    assert query_matches_verified("elma", "Elma", ["apple"])
    assert query_matches_verified("APPLE", "elma", ["apple"])
    assert not query_matches_verified("elma", "kedi", [])


def test_conflicting_label_is_not_a_clip_neighbor(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    proto = np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
    near = np.array([0.95, 0.05, 0, 0], dtype=np.float32).tobytes()
    taught = [_add(db, tmp_path, f"z{i}.jpg", clip=proto) for i in range(3)]
    extra = _add(db, tmp_path, "looks_like_chain.jpg", clip=near)
    teach_files(db, path, taught, "Zincir")
    teach_files(db, path, [extra], "Elma")
    pack = collect_learned_hits(path, "zincir", db=db)
    assert extra not in pack["file_scores"]
    assert extra in conflicting_taught_ids(path, "zincir")


def test_unlabeled_files_are_not_blocked():
    assert conflicting_taught_ids("", "elma") == set()
    row = SimpleNamespace(file_id=9, score=0.4, debug={})
    assert filter_taught_conflicts([row], "", "elma") == [row]


def test_no_learned_concept_leaves_general_search():
    assert resolve_learned_concept("", "ananas") is None
    assert collect_learned_hits("missing.db", "ananas") == {}


def test_display_count_matches_rendered_list():
    rows = [SimpleNamespace(score=0.9, debug={}) for _ in range(17)]
    ghost = SearchResponse(results=[], all_results=[], stats=SimpleNamespace(above_threshold=17), meta={"relevant_count": 17})
    assert results_for_display(ghost, 0.6) == []
    filled = SearchResponse(results=rows, all_results=rows)
    shown = results_for_display(filled, 0.6)
    assert len(shown) == 17
    below = [SimpleNamespace(score=0.2, debug={"learned_concept_exact": True}) ]
    mixed = SearchResponse(results=below, all_results=below)
    shown2 = results_for_display(mixed, 0.6)
    assert len(shown2) == 1


def _teach_label(db, path, tmp_path, name: str, label: str) -> int:
    fid = _add(db, tmp_path, name)
    teach_files(db, path, [fid], label)
    return fid


def test_hypernym_and_typo_cover_taught_concepts(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    woman_face = _teach_label(db, path, tmp_path, "wf.jpg", "kadın yüzü")
    human_face = _teach_label(db, path, tmp_path, "hf.jpg", "insan yüzü")
    woman_shoe = _teach_label(db, path, tmp_path, "ws.jpg", "kadın ayakkabısı")
    butterfly = _teach_label(db, path, tmp_path, "bf.jpg", "kelebek")
    cat = _teach_label(db, path, tmp_path, "cat2.jpg", "kedi")
    perfume = _teach_label(db, path, tmp_path, "pf.jpg", "parfüm")

    assert woman_face in collect_learned_hits(path, "insan", db=db)["exact_ids"]
    assert woman_face in collect_learned_hits(path, "kadın", db=db)["exact_ids"]
    assert human_face in collect_learned_hits(path, "yüz", db=db)["exact_ids"]
    assert woman_shoe in collect_learned_hits(path, "ayakkabı", db=db)["exact_ids"]
    assert butterfly in collect_learned_hits(path, "hayvan", db=db)["exact_ids"]
    assert cat in collect_learned_hits(path, "kedi", db=db)["exact_ids"]
    assert butterfly not in collect_learned_hits(path, "kedi", db=db).get("exact_ids", [])
    assert cat not in conflicting_taught_ids(path, "kedi")

    face_pack = collect_learned_hits(path, "kadın yüzü", db=db)
    assert woman_face in face_pack["exact_ids"]
    parent_ids = face_pack.get("neighbor_scores") or {}
    assert human_face not in face_pack["exact_ids"] or face_pack["file_scores"][woman_face] >= face_pack["file_scores"].get(human_face, 0)

    assert perfume in collect_learned_hits(path, "parfum", db=db)["exact_ids"]
    assert woman_shoe in collect_learned_hits(path, "ayakabısı", db=db)["exact_ids"]
    assert woman_face in collect_learned_hits(path, "kadın yuzu", db=db)["exact_ids"]
    assert butterfly in collect_learned_hits(path, "kelebekk", db=db)["exact_ids"]


def test_new_taught_labels_join_the_same_graph(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    giraffe = _teach_label(db, path, tmp_path, "gf.jpg", "zürafa")
    red = _teach_label(db, path, tmp_path, "rf.jpg", "kırmızı flurp")
    blue = _teach_label(db, path, tmp_path, "bf2.jpg", "mavi flurp")
    assert giraffe in collect_learned_hits(path, "hayvan", db=db)["exact_ids"]
    pack = collect_learned_hits(path, "flurp", db=db)
    assert red in pack["exact_ids"]
    assert blue in pack["exact_ids"]
    specific = collect_learned_hits(path, "kırmızı flurp", db=db)
    assert red in specific["exact_ids"]
    assert specific["file_scores"][red] >= specific["file_scores"].get(blue, 0)
    assert red in collect_learned_hits(path, "kirmizi flurp", db=db)["exact_ids"]
