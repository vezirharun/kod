"""Parent→child hierarchical concept learning & search (non-destructive)."""
from __future__ import annotations

import numpy as np

from core.concept_registry import (
    _conn,
    add_example,
    children_of_parent,
    learn,
)
from core.db import Database
from core.learned_concept_search import (
    LEARNED_CHILD_SCORE,
    collect_learned_hits,
    concept_query_relation,
    match_taught_concepts,
)
from core.teach_me import teach_files


def _db(tmp_path):
    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db, str(tmp_path / "patterns.db")


def _unit(vals):
    v = np.asarray(vals, dtype=np.float32)
    n = float(np.linalg.norm(v) or 1.0)
    return (v / n).tobytes()


def _add(db, tmp_path, name, *, clip=None):
    img = tmp_path / name
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": name,
                "source_id": 1,
                "status": "indexed",
            }
        )
    )
    if clip is not None:
        db.upsert_features(fid, {"clip_embedding": clip})
    return fid


def _pos_ids(db_path, label: str) -> set[int]:
    want = " ".join(str(label or "").strip().casefold().split())
    c = _conn(db_path)
    rows = c.execute(
        """SELECT e.file_id, r.canonical, e.source FROM concept_examples e
           JOIN concept_registry r ON r.id=e.concept_id
           WHERE e.role='positive'"""
    ).fetchall()
    c.close()
    out = set()
    for r in rows:
        can = " ".join(str(r["canonical"] or "").strip().casefold().split())
        if can == want:
            out.add(int(r["file_id"]))
    return out


def _user_pos(db_path, label: str) -> set[int]:
    want = " ".join(str(label or "").strip().casefold().split())
    c = _conn(db_path)
    rows = c.execute(
        """SELECT e.file_id, r.canonical, e.source FROM concept_examples e
           JOIN concept_registry r ON r.id=e.concept_id
           WHERE e.role='positive'
             AND IFNULL(e.source,'user') NOT IN ('auto','autonomous','candidate')"""
    ).fetchall()
    c.close()
    out = set()
    for r in rows:
        can = " ".join(str(r["canonical"] or "").strip().casefold().split())
        if can == want:
            out.add(int(r["file_id"]))
    return out


def test_miki_child_learned_and_searchable(tmp_path):
    db, path = _db(tmp_path)
    miki_v = _unit([1, 0, 0, 0])
    fids = [_add(db, tmp_path, f"miki{i}.jpg", clip=miki_v) for i in range(3)]
    teach_files(db, path, fids, "Miki Mouse", overlay={"parent": "Cartoon"})
    assert _user_pos(path, "Miki Mouse") == set(fids)
    kids = children_of_parent(path, "Cartoon", user_verified_only=True)
    assert any("miki" in str(k.get("canonical") or "").casefold() for k in kids)
    pack = collect_learned_hits(path, "Miki Mouse", db=db)
    assert set(pack["exact_ids"]) == set(fids)
    assert pack.get("hierarchical_parent") is not True


def test_cartoon_parent_finds_miki_user_evidence(tmp_path):
    db, path = _db(tmp_path)
    miki_v = _unit([1, 0, 0, 0])
    fids = [_add(db, tmp_path, f"m{i}.jpg", clip=miki_v) for i in range(2)]
    teach_files(db, path, fids, "Miki Mouse", overlay={"parent": "Cartoon"})
    pack = collect_learned_hits(path, "Cartoon", db=db)
    assert pack.get("hierarchical_parent") is True
    assert set(fids).issubset(set(pack["exact_ids"]))
    for fid in fids:
        assert pack["file_scores"][fid] >= LEARNED_CHILD_SCORE - 1e-6


def test_cizgi_film_synonym_parent_same_children(tmp_path):
    db, path = _db(tmp_path)
    miki_v = _unit([1, 0, 0, 0])
    fids = [_add(db, tmp_path, f"m{i}.jpg", clip=miki_v) for i in range(2)]
    teach_files(db, path, fids, "Miki Mouse", overlay={"parent": "Cartoon"})
    a = collect_learned_hits(path, "Cartoon", db=db)
    b = collect_learned_hits(path, "Çizgi film", db=db)
    assert a.get("hierarchical_parent") and b.get("hierarchical_parent")
    assert set(fids).issubset(set(a["exact_ids"]))
    assert set(fids).issubset(set(b["exact_ids"]))
    assert concept_query_relation("Çizgi film", "Miki Mouse", parent="Cartoon") == "child"


def test_twenty_adds_without_breaking_miki(tmp_path):
    db, path = _db(tmp_path)
    miki_v = _unit([1, 0, 0, 0])
    twenty_v = _unit([0, 1, 0, 0])
    miki = [_add(db, tmp_path, f"m{i}.jpg", clip=miki_v) for i in range(3)]
    twenty = [_add(db, tmp_path, f"t{i}.jpg", clip=twenty_v) for i in range(2)]
    teach_files(db, path, miki, "Miki Mouse", overlay={"parent": "Cartoon"})
    before = _user_pos(path, "Miki Mouse")
    teach_files(db, path, twenty, "Twenty", overlay={"parent": "Cartoon"})
    assert _user_pos(path, "Miki Mouse") == before == set(miki)
    assert _user_pos(path, "Twenty") == set(twenty)
    pack = collect_learned_hits(path, "Cartoon", db=db)
    assert set(miki).issubset(set(pack["exact_ids"]))
    assert set(twenty).issubset(set(pack["exact_ids"]))
    miki_pack = collect_learned_hits(path, "Miki Mouse", db=db)
    assert set(miki_pack["exact_ids"]) == set(miki)
    assert not set(twenty).intersection(set(miki_pack["exact_ids"]))


def test_auto_candidate_do_not_overwrite_user(tmp_path):
    db, path = _db(tmp_path)
    v = _unit([1, 0, 0, 0])
    fid = _add(db, tmp_path, "u.jpg", clip=v)
    teach_files(db, path, [fid], "Miki Mouse", overlay={"parent": "Cartoon"})
    cid = learn(path, "Miki Mouse", parent="Cartoon")
    assert add_example(path, cid, file_id=fid, role="positive", source="auto") is False
    assert add_example(path, cid, file_id=fid, role="positive", source="candidate") is False
    c = _conn(path)
    src = c.execute(
        "SELECT source FROM concept_examples WHERE concept_id=? AND file_id=? AND role='positive'",
        (cid, fid),
    ).fetchone()["source"]
    c.close()
    assert str(src) == "user"
    assert fid in _user_pos(path, "Miki Mouse")


def test_clip_fabric_does_not_dominate_parent_results(tmp_path):
    db, path = _db(tmp_path)
    miki_v = _unit([1, 0, 0, 0])
    near_fabric = _unit([0.98, 0.2, 0, 0])  # CLIP-near miki, not taught
    far = _unit([0, 0, 1, 0])
    miki = [_add(db, tmp_path, f"m{i}.jpg", clip=miki_v) for i in range(2)]
    fabric = _add(db, tmp_path, "fabric.jpg", clip=near_fabric)
    _add(db, tmp_path, "noise.jpg", clip=far)
    teach_files(db, path, miki, "Miki Mouse", overlay={"parent": "Cartoon"})
    pack = collect_learned_hits(path, "Cartoon", db=db)
    assert set(miki).issubset(set(pack["exact_ids"]))
    for fid in miki:
        assert pack["file_scores"][fid] >= LEARNED_CHILD_SCORE - 1e-6
    if fabric in pack.get("file_scores", {}):
        assert pack["file_scores"][fabric] < pack["file_scores"][miki[0]]
        assert pack["file_scores"][fabric] <= LEARNED_CHILD_SCORE - 0.04


def test_flower_rose_tulip_parent(tmp_path):
    db, path = _db(tmp_path)
    rose_v = _unit([1, 0, 0, 0])
    tulip_v = _unit([0, 1, 0, 0])
    roses = [_add(db, tmp_path, f"r{i}.jpg", clip=rose_v) for i in range(2)]
    tulips = [_add(db, tmp_path, f"t{i}.jpg", clip=tulip_v) for i in range(2)]
    teach_files(db, path, roses, "Rose", overlay={"parent": "Flower"})
    teach_files(db, path, tulips, "Tulip", overlay={"parent": "Flower"})
    pack = collect_learned_hits(path, "Flower", db=db)
    assert pack.get("hierarchical_parent") is True
    assert set(roses + tulips).issubset(set(pack["exact_ids"]))
    assert set(collect_learned_hits(path, "Rose", db=db)["exact_ids"]) == set(roses)


def test_animal_leopard_zebra_parent(tmp_path):
    db, path = _db(tmp_path)
    leo_v = _unit([1, 0, 0, 0])
    zeb_v = _unit([0, 1, 0, 0])
    leos = [_add(db, tmp_path, f"l{i}.jpg", clip=leo_v) for i in range(2)]
    zebs = [_add(db, tmp_path, f"z{i}.jpg", clip=zeb_v) for i in range(2)]
    teach_files(db, path, leos, "Leopard", overlay={"parent": "Animal"})
    teach_files(db, path, zebs, "Zebra", overlay={"parent": "Animal"})
    pack = collect_learned_hits(path, "Animal", db=db)
    assert pack.get("hierarchical_parent") is True
    assert set(leos + zebs).issubset(set(pack["exact_ids"]))
    # animal print synonym should also surface children
    pack2 = collect_learned_hits(path, "Animal Print", db=db)
    assert set(leos + zebs).issubset(set(pack2["exact_ids"]))


def test_parent_shell_created_non_destructive(tmp_path):
    db, path = _db(tmp_path)
    fid = _add(db, tmp_path, "m.jpg", clip=_unit([1, 0, 0, 0]))
    teach_files(db, path, [fid], "Miki Mouse", overlay={"parent": "Cartoon"})
    c = _conn(path)
    parent = c.execute(
        "SELECT canonical, concept_type FROM concept_registry WHERE lower(canonical)=?",
        ("cartoon",),
    ).fetchone()
    child = c.execute(
        "SELECT parent FROM concept_registry WHERE lower(canonical)=?",
        ("miki mouse",),
    ).fetchone()
    c.close()
    assert parent is not None
    assert str(parent["concept_type"]) == "parent_group"
    assert "cartoon" in str(child["parent"] or "").casefold()
    assert fid in _user_pos(path, "Miki Mouse")


def test_match_lists_children_for_parent_query(tmp_path):
    db, path = _db(tmp_path)
    a = _add(db, tmp_path, "a.jpg", clip=_unit([1, 0, 0, 0]))
    b = _add(db, tmp_path, "b.jpg", clip=_unit([0, 1, 0, 0]))
    teach_files(db, path, [a], "Miki Mouse", overlay={"parent": "Cartoon"})
    teach_files(db, path, [b], "Twenty", overlay={"parent": "Cartoon"})
    ms = match_taught_concepts(path, "Cartoon")
    cans = {str(m["canonical"]).casefold(): m["relation"] for m in ms}
    assert cans.get("miki mouse") == "child"
    assert cans.get("twenty") == "child"
