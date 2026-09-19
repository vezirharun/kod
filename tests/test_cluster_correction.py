"""Cluster correction: yanlış üyeyi ayır, yeni concept anchor, eski positives korunur."""
from __future__ import annotations

import numpy as np

from core.autonomous_learn import correct_reviews, upsert_review
from core.concept_registry import _conn
from core.db import Database
from core.learned_concept_search import collect_learned_hits, conflicting_taught_ids
from core.teach_me import correct_concept_membership, teach_files


def _db(tmp_path):
    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db, str(tmp_path / "patterns.db")


def _unit(vals):
    v = np.array(vals, dtype=np.float32)
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
        """SELECT e.file_id, r.canonical FROM concept_examples e
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


def test_correct_reviews_only_selected_not_whole_cluster(tmp_path):
    db, path = _db(tmp_path)
    lip = _unit([1, 0, 0, 0])
    rose = _unit([0, 1, 0, 0])
    a = _add(db, tmp_path, "dudak_a.jpg", clip=lip)
    b = _add(db, tmp_path, "dudak_b.jpg", clip=lip)
    x = _add(db, tmp_path, "gul_x.jpg", clip=rose)
    teach_files(db, path, [a, b, x], "DUDAK")
    c = _conn(path)
    cur = c.execute(
        "INSERT INTO autonomous_clusters(suggested_name,file_count,status,reason,created_at) "
        "VALUES('DUDAK',3,'pending','t','')"
    )
    cid = int(cur.lastrowid)
    c.commit()
    c.close()
    for fid in (a, b, x):
        upsert_review(
            path, fid, lane="undecided", suggested="DUDAK", cluster_id=cid, status="pending"
        )
    stats = correct_reviews(db, path, [x], "GÜL")
    assert stats["taught"] >= 1
    assert a in _pos_ids(path, "DUDAK")
    assert b in _pos_ids(path, "DUDAK")
    assert x not in _pos_ids(path, "DUDAK")
    assert x in _pos_ids(path, "GÜL")
    assert a not in _pos_ids(path, "GÜL")
    assert b not in _pos_ids(path, "GÜL")
    c = _conn(path)
    rows = c.execute(
        """SELECT e.file_id, r.canonical FROM concept_examples e
           JOIN concept_registry r ON r.id=e.concept_id
           WHERE e.role='negative'"""
    ).fetchall()
    c.close()
    negs = {
        int(r["file_id"])
        for r in rows
        if " ".join(str(r["canonical"] or "").strip().casefold().split()) == "dudak"
    }
    assert x in negs


def test_other_dudak_positives_untouched(tmp_path):
    db, path = _db(tmp_path)
    v = _unit([1, 0, 0, 0])
    lips = [_add(db, tmp_path, f"d{i}.jpg", clip=v) for i in range(5)]
    gul = _add(db, tmp_path, "g.jpg", clip=_unit([0, 1, 0, 0]))
    teach_files(db, path, lips + [gul], "DUDAK")
    before = _pos_ids(path, "DUDAK")
    correct_concept_membership(db, path, [gul], "GÜL", rival_labels=["DUDAK"])
    after = _pos_ids(path, "DUDAK")
    assert after == (before - {gul})
    assert all(fid in after for fid in lips)


def test_gul_anchor_expansion_enqueues_neighbors(tmp_path):
    db, path = _db(tmp_path)
    anchor_v = _unit([0, 1, 0, 0])
    near_v = _unit([0.05, 0.9987, 0, 0])
    far_v = _unit([1, 0, 0, 0])
    lips = [_add(db, tmp_path, f"lip{i}.jpg", clip=far_v) for i in range(3)]
    teach_files(db, path, lips, "DUDAK")
    gul = _add(db, tmp_path, "gul.jpg", clip=anchor_v)
    near = _add(db, tmp_path, "gul2.jpg", clip=near_v)
    teach_files(db, path, [gul], "DUDAK")
    stats = correct_reviews(db, path, [gul], "GÜL")
    assert stats.get("expanded", 0) >= 1
    c = _conn(path)
    row = c.execute(
        "SELECT suggested, status FROM autonomous_review WHERE file_id=?", (near,)
    ).fetchone()
    c.close()
    assert row is not None
    assert str(row["suggested"]) == "GÜL"
    assert str(row["status"]) == "pending"
    # Expansion candidate evidence yazar; user verified değildir.
    c = _conn(path)
    src = c.execute(
        """SELECT e.source FROM concept_examples e
           JOIN concept_registry r ON r.id=e.concept_id
           WHERE e.file_id=? AND e.role='positive'""",
        (near,),
    ).fetchone()
    c.close()
    assert src is not None
    assert str(src["source"]) == "candidate"
    assert all(fid in _pos_ids(path, "DUDAK") for fid in lips)


def test_search_dudak_penalizes_gul_and_gul_rises(tmp_path):
    db, path = _db(tmp_path)
    lip_v = _unit([1, 0, 0, 0])
    gul_v = _unit([0, 1, 0, 0])
    lips = [_add(db, tmp_path, f"l{i}.jpg", clip=lip_v) for i in range(3)]
    gul = _add(db, tmp_path, "rose.jpg", clip=gul_v)
    teach_files(db, path, lips, "DUDAK")
    teach_files(db, path, [gul], "DUDAK")
    correct_concept_membership(db, path, [gul], "GÜL", rival_labels=["DUDAK"])
    dudak = collect_learned_hits(path, "DUDAK", db=db)
    assert set(dudak["exact_ids"]) == set(lips)
    assert gul not in dudak.get("exact_ids", [])
    assert gul not in dudak.get("file_scores", {})
    assert gul in conflicting_taught_ids(path, "DUDAK")
    gul_pack = collect_learned_hits(path, "GÜL", db=db)
    assert gul in gul_pack["exact_ids"]
    assert gul_pack["file_scores"][gul] >= 0.9


def test_kalp_third_boundary(tmp_path):
    db, path = _db(tmp_path)
    d = _add(db, tmp_path, "d.jpg", clip=_unit([1, 0, 0, 0]))
    g = _add(db, tmp_path, "g.jpg", clip=_unit([0, 1, 0, 0]))
    k = _add(db, tmp_path, "k.jpg", clip=_unit([0, 0, 1, 0]))
    teach_files(db, path, [d, g, k], "DUDAK")
    correct_concept_membership(db, path, [g], "GÜL", rival_labels=["DUDAK"])
    correct_concept_membership(db, path, [k], "KALP", rival_labels=["DUDAK"])
    assert _pos_ids(path, "DUDAK") == {d}
    assert _pos_ids(path, "GÜL") == {g}
    assert _pos_ids(path, "KALP") == {k}


def test_double_correction_no_duplicate_positive(tmp_path):
    db, path = _db(tmp_path)
    gul = _add(db, tmp_path, "g.jpg", clip=_unit([0, 1, 0, 0]))
    teach_files(db, path, [gul], "DUDAK")
    correct_concept_membership(db, path, [gul], "GÜL", rival_labels=["DUDAK"])
    correct_concept_membership(db, path, [gul], "GÜL", rival_labels=["DUDAK"])
    assert gul in _pos_ids(path, "GÜL")
    c = _conn(path)
    pos_roles = c.execute(
        """SELECT COUNT(*) n FROM concept_examples e
           JOIN concept_registry r ON r.id=e.concept_id
           WHERE e.file_id=? AND e.role='positive'""",
        (gul,),
    ).fetchone()["n"]
    c.close()
    assert int(pos_roles) == 1
