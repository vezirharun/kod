"""Faz A — Self-learning brain: evidence tiers, self-check, learning_events."""
from __future__ import annotations

import numpy as np

from core.autonomous_learn import (
    correct_reviews,
    run_autonomous_pass,
    self_check_decision,
    upsert_review,
)
from core.concept_registry import (
    _conn,
    add_example,
    list_learning_events,
    positive_example_vectors,
    record_learning_event,
    upsert,
)
from core.db import Database
from core.teach_me import correct_concept_membership, match_clip_to_concepts, teach_files


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


def _pos_sources(db_path, label: str) -> list[tuple[int, str]]:
    c = _conn(db_path)
    rows = c.execute(
        """SELECT e.file_id, e.source, r.canonical FROM concept_examples e
           JOIN concept_registry r ON r.id=e.concept_id
           WHERE e.role='positive'"""
    ).fetchall()
    c.close()
    want = " ".join(label.strip().casefold().split())
    out = []
    for r in rows:
        can = " ".join(str(r["canonical"] or "").strip().casefold().split())
        if can == want:
            out.append((int(r["file_id"]), str(r["source"] or "user")))
    return out


def test_clear_leader_writes_auto_evidence(tmp_path):
    db, path = _db(tmp_path)
    proto = _unit([1, 0, 0, 0])
    seed = _add(db, tmp_path, "seed.jpg", clip=proto)
    teach_files(db, path, [seed], "DUDAK")
    near = _unit([0.99, 0.141067, 0, 0])
    target = _add(db, tmp_path, "near.jpg", clip=near)
    stats = run_autonomous_pass(db, path, limit=50)
    assert stats["high_applied"] >= 1
    sources = dict(_pos_sources(path, "DUDAK"))
    assert sources.get(seed) in ("user", "")
    assert sources.get(target) == "auto"
    # auto matching prototipine girmez
    assert all(ex["file_id"] != target for ex in positive_example_vectors(path))
    ev = list_learning_events(path, file_id=target)
    assert any(e["action"] == "auto_evidence" for e in ev)


def test_rival_close_is_candidate_not_auto(tmp_path):
    db, path = _db(tmp_path)
    v = _unit([1, 0, 0, 0])
    a = _add(db, tmp_path, "d.jpg", clip=v)
    b = _add(db, tmp_path, "g.jpg", clip=v)
    teach_files(db, path, [a], "DUDAK")
    teach_files(db, path, [b], "GÜL")
    mix = _add(db, tmp_path, "mix.jpg", clip=v)
    hits = match_clip_to_concepts(path, v)
    check = self_check_decision(path, hits, clip_blob=v, file_id=mix)
    assert check["decision"] == "candidate"
    stats = run_autonomous_pass(db, path, limit=50)
    assert stats.get("high_applied", 0) == 0 or mix not in dict(_pos_sources(path, "DUDAK"))
    # mix should be candidate or pending, not auto for a single concept as verified user
    sources_d = dict(_pos_sources(path, "DUDAK"))
    sources_g = dict(_pos_sources(path, "GÜL"))
    assert sources_d.get(mix) != "auto"
    assert sources_g.get(mix) != "auto"
    assert sources_d.get(mix) == "candidate" or sources_g.get(mix) == "candidate" or stats.get("candidates", 0) >= 1


def test_auto_does_not_mutate_user_evidence(tmp_path):
    db, path = _db(tmp_path)
    proto = _unit([1, 0, 0, 0])
    seed = _add(db, tmp_path, "seed.jpg", clip=proto)
    teach_files(db, path, [seed], "DUDAK")
    before = _pos_sources(path, "DUDAK")
    near = _unit([0.99, 0.141067, 0, 0])
    _add(db, tmp_path, "near.jpg", clip=near)
    run_autonomous_pass(db, path, limit=50)
    after = _pos_sources(path, "DUDAK")
    assert (seed, "user") in [(f, s if s else "user") for f, s in after] or any(
        f == seed and s == "user" for f, s in after
    )
    assert any(f == seed for f, _s in before)


def test_gul_correction_preserves_dudak_users(tmp_path):
    db, path = _db(tmp_path)
    lip = _unit([1, 0, 0, 0])
    rose = _unit([0, 1, 0, 0])
    lips = [_add(db, tmp_path, f"d{i}.jpg", clip=lip) for i in range(4)]
    gul = _add(db, tmp_path, "gul.jpg", clip=rose)
    teach_files(db, path, lips + [gul], "DUDAK")
    correct_concept_membership(db, path, [gul], "GÜL", rival_labels=["DUDAK"])
    assert set(f for f, _ in _pos_sources(path, "DUDAK")) == set(lips)
    assert set(f for f, s in _pos_sources(path, "GÜL") if s == "user") == {gul}


def test_gul_anchor_expansion_candidates(tmp_path):
    db, path = _db(tmp_path)
    lip = _unit([1, 0, 0, 0])
    rose = _unit([0, 1, 0, 0])
    near = _unit([0.05, 0.9987, 0, 0])
    lips = [_add(db, tmp_path, f"l{i}.jpg", clip=lip) for i in range(3)]
    teach_files(db, path, lips, "DUDAK")
    gul = _add(db, tmp_path, "g.jpg", clip=rose)
    neigh = _add(db, tmp_path, "g2.jpg", clip=near)
    teach_files(db, path, [gul], "DUDAK")
    stats = correct_reviews(db, path, [gul], "GÜL")
    assert stats.get("expanded", 0) >= 1
    sources = dict(_pos_sources(path, "GÜL"))
    assert sources.get(gul) == "user"
    assert sources.get(neigh) == "candidate"
    assert neigh not in {ex["file_id"] for ex in positive_example_vectors(path)}


def test_candidate_not_in_matching_prototypes(tmp_path):
    db, path = _db(tmp_path)
    v = _unit([0, 1, 0, 0])
    fid = _add(db, tmp_path, "c.jpg", clip=v)
    cid = upsert(path, "GÜL", source="candidate")
    add_example(path, cid, file_id=fid, role="positive", source="candidate", embedding=v)
    hits = match_clip_to_concepts(path, v)
    assert hits == []
    assert all(ex["file_id"] != fid for ex in positive_example_vectors(path))


def test_negative_boundary_blocks_auto(tmp_path):
    db, path = _db(tmp_path)
    lip = _unit([1, 0, 0, 0])
    rose = _unit([0, 1, 0, 0])
    seed = _add(db, tmp_path, "d.jpg", clip=lip)
    teach_files(db, path, [seed], "DUDAK")
    cid = upsert(path, "DUDAK")
    bad = _add(db, tmp_path, "bad.jpg", clip=rose)
    add_example(
        path, cid, file_id=bad, role="negative", source="user", embedding=rose
    )
    target = _add(db, tmp_path, "looks_rose.jpg", clip=rose)
    # rose won't match DUDAK positives highly — use near-lip that is also near negative?
    # Better: negative has lip-like embedding so target near lip hits both positive and negative.
    neg_like = _unit([0.95, 0.3122, 0, 0])
    add_example(
        path, cid, file_id=_add(db, tmp_path, "neg.jpg", clip=neg_like),
        role="negative", source="user", embedding=neg_like,
    )
    target2 = _add(db, tmp_path, "t2.jpg", clip=neg_like)
    hits = match_clip_to_concepts(path, neg_like)
    check = self_check_decision(path, hits, clip_blob=neg_like, file_id=target2)
    assert check["decision"] in ("candidate", "ignore")
    assert check["decision"] != "auto" or "negative" in str(check.get("reason"))


def test_learning_event_recorded(tmp_path):
    db, path = _db(tmp_path)
    eid = record_learning_event(
        path,
        file_id=7,
        concept_id=3,
        action="candidate_created",
        confidence=0.84,
        source="candidate",
        anchor_id=998,
        rival_concept_id=1,
        reason="anchor_similarity_high_but_rival_close",
    )
    assert eid > 0
    rows = list_learning_events(path, file_id=7)
    assert len(rows) == 1
    row = rows[0]
    assert row["action"] == "candidate_created"
    assert abs(float(row["confidence"]) - 0.84) < 1e-6
    assert row["source"] == "candidate"
    assert int(row["anchor_id"]) == 998
    assert "rival" in str(row["reason"])


def test_learning_event_no_duplicate(tmp_path):
    db, path = _db(tmp_path)
    a = record_learning_event(
        path, file_id=1, concept_id=2, action="auto_evidence", source="auto", confidence=0.9
    )
    b = record_learning_event(
        path, file_id=1, concept_id=2, action="auto_evidence", source="auto", confidence=0.91
    )
    assert a > 0
    assert b == 0
    assert len(list_learning_events(path, file_id=1)) == 1


def test_triple_separation_dudak_gul_kalp(tmp_path):
    db, path = _db(tmp_path)
    d = _add(db, tmp_path, "d.jpg", clip=_unit([1, 0, 0, 0]))
    g = _add(db, tmp_path, "g.jpg", clip=_unit([0, 1, 0, 0]))
    k = _add(db, tmp_path, "k.jpg", clip=_unit([0, 0, 1, 0]))
    teach_files(db, path, [d, g, k], "DUDAK")
    correct_concept_membership(db, path, [g], "GÜL", rival_labels=["DUDAK"])
    correct_concept_membership(db, path, [k], "KALP", rival_labels=["DUDAK"])
    assert {f for f, s in _pos_sources(path, "DUDAK") if s == "user"} == {d}
    assert {f for f, s in _pos_sources(path, "GÜL") if s == "user"} == {g}
    assert {f for f, s in _pos_sources(path, "KALP") if s == "user"} == {k}
    from core.learned_concept_search import collect_learned_hits

    assert d in collect_learned_hits(path, "DUDAK", db=db)["exact_ids"]
    assert g in collect_learned_hits(path, "GÜL", db=db)["exact_ids"]
    assert k in collect_learned_hits(path, "KALP", db=db)["exact_ids"]
    assert g not in collect_learned_hits(path, "DUDAK", db=db).get("exact_ids", [])
