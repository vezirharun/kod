"""Otonom öğrenme: eminsen kaydet, emin değilsen sor."""
from __future__ import annotations

import numpy as np

from core.autonomous_learn import (
    HIGH_MIN,
    LOW_MIN,
    confirm_reviews,
    correct_reviews,
    decide_lane,
    reject_reviews,
    run_autonomous_pass,
)
from core.concept_registry import add_example, positive_example_vectors, upsert
from core.db import Database
from core.teach_me import list_inbox_pools, match_clip_to_concepts, teach_files


def _db(tmp_path):
    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db, str(tmp_path / "patterns.db")


def _add(db, tmp_path, name, *, clip=None, tm=None, family=""):
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
    if tm is not None:
        payload["texture_map"] = tm
    if payload:
        db.upsert_features(fid, payload)
    return fid


def _unit(x):
    return np.array(x, dtype=np.float32).tobytes()


def test_decide_lane_high_undecided_low():
    high, _ = decide_lane([{"canonical": "Kedi", "score": HIGH_MIN + 0.05}])
    assert high == "high"
    mid, reason = decide_lane(
        [
            {"canonical": "Kedi", "score": 0.91},
            {"canonical": "Tilki", "score": 0.88},
        ]
    )
    assert mid == "undecided"
    assert "Kedi" in reason and "Tilki" in reason
    low, _ = decide_lane([{"canonical": "Kedi", "score": (HIGH_MIN + LOW_MIN) / 2}])
    assert low == "low"
    none, _ = decide_lane([{"canonical": "Kedi", "score": LOW_MIN - 0.05}])
    assert none == "none"


def test_undecided_candidate_labels_from_signals_only():
    from core.autonomous_learn import undecided_candidate_labels

    labels = undecided_candidate_labels(
        [["GÜL", 0.91], ["DUDAK", 0.88], ["GÜL", 0.87]],
        suggested="GÜL",
    )
    assert labels == ["GÜL", "DUDAK"]
    # No rivals → suggested only; junk filtered
    assert undecided_candidate_labels([], suggested="Belirsiz") == []
    assert undecided_candidate_labels(None, suggested="Kaplan") == ["Kaplan"]
    # Does not invent random concepts
    assert undecided_candidate_labels([], suggested="") == []


def test_one_click_confirm_reviews_teaches_chosen_rival(tmp_path):
    """One-click GÜL → confirm_reviews(label=GÜL) → user teach; no second dialog."""
    db, path = _db(tmp_path)
    proto_g = _unit([1, 0, 0, 0])
    proto_d = _unit([0, 1, 0, 0])
    seed_g = _add(db, tmp_path, "seed_gul.jpg", clip=proto_g)
    seed_d = _add(db, tmp_path, "seed_dudak.jpg", clip=proto_d)
    teach_files(db, path, [seed_g], "GÜL")
    teach_files(db, path, [seed_d], "DUDAK")
    # Ambiguous file closer to both
    amb = _add(db, tmp_path, "amb.jpg", clip=_unit([0.7, 0.7, 0, 0]))
    run_autonomous_pass(db, path, limit=50)
    pools = list_inbox_pools(db, path)
    undecided = pools.get("undecided") or []
    # Ensure amb is pending or force review
    from core.autonomous_learn import review_for_file, upsert_review

    if not review_for_file(path, amb):
        upsert_review(
            path,
            amb,
            lane="undecided",
            suggested="GÜL",
            confidence=0.9,
            rivals=[["GÜL", 0.9], ["DUDAK", 0.88]],
            reason="test_rivals",
        )
    stats = confirm_reviews(db, path, [amb], label="GÜL")
    assert int(stats.get("taught") or 0) >= 1
    # Taught as GÜL — not re-asked as pending
    row = review_for_file(path, amb)
    assert row is None or str(row.get("status") or "") != "pending"
    from core.concept_registry import positives_for_file

    pos = positives_for_file(path, amb)
    canons = {str(p.get("canonical") or "") for p in pos}
    assert "GÜL" in canons or any("GÜL" in str(p) for p in pos)


def test_high_confidence_auto_learns_without_user(tmp_path):
    db, path = _db(tmp_path)
    proto = _unit([1, 0, 0, 0])
    seed = _add(db, tmp_path, "seed.jpg", clip=proto)
    teach_files(db, path, [seed], "Kedi")
    near = _unit([0.99, 0.141067, 0, 0])
    target = _add(db, tmp_path, "auto.jpg", clip=near, family="")
    stats = run_autonomous_pass(db, path, limit=50)
    assert stats["high_applied"] >= 1
    feat = db.get_features(target)
    tm = feat["texture_map"] if feat else {}
    assert tm.get("autonomous_label") == "Kedi"
    assert float(tm.get("autonomous_confidence") or 0) >= HIGH_MIN
    assert not tm.get("user_labeled")
    pools = list_inbox_pools(db, path)
    ids = {c.file_id for bucket in pools.values() for c in bucket}
    assert target not in ids


def test_medium_confidence_goes_to_undecided(tmp_path):
    db, path = _db(tmp_path)
    vec = _unit([1, 0, 0, 0])
    a = _add(db, tmp_path, "kedi.jpg", clip=vec)
    b = _add(db, tmp_path, "tilki.jpg", clip=vec)
    teach_files(db, path, [a], "Kedi")
    teach_files(db, path, [b], "Tilki")
    mix = _add(db, tmp_path, "mix.jpg", clip=vec, family="")
    stats = run_autonomous_pass(db, path, limit=50)
    assert stats["undecided"] >= 1
    feat = db.get_features(mix)
    tm = (feat or {}).get("texture_map") or {}
    assert not tm.get("autonomous_label")
    pools = list_inbox_pools(db, path)
    undecided_ids = {c.file_id for c in pools.get("undecided") or []}
    assert mix in undecided_ids
    card = next(c for c in pools["undecided"] if c.file_id == mix)
    assert "Kedi" in card.reason and "Tilki" in card.reason


def test_low_confidence_asks_user_not_saved_as_truth(tmp_path):
    db, path = _db(tmp_path)
    proto = _unit([1, 0, 0, 0])
    seed = _add(db, tmp_path, "seed.jpg", clip=proto)
    teach_files(db, path, [seed], "Kedi")
    weak = _unit([0.75, 0.661437827, 0, 0])
    target = _add(db, tmp_path, "maybe.jpg", clip=weak, family="")
    stats = run_autonomous_pass(db, path, limit=50)
    assert stats["low"] >= 1
    feat = db.get_features(target)
    tm = (feat or {}).get("texture_map") or {}
    assert not tm.get("autonomous_label")
    pools = list_inbox_pools(db, path)
    sus_ids = {c.file_id for c in pools.get("suspicious") or []}
    assert target in sus_ids


def test_new_concept_cluster_needs_user(tmp_path):
    db, path = _db(tmp_path)
    proto = _unit([1, 0, 0, 0])
    seed = _add(db, tmp_path, "seed.jpg", clip=proto)
    teach_files(db, path, [seed], "Kedi")
    group = _unit([0, 1, 0, 0])
    ids = [
        _add(db, tmp_path, f"g{i}.jpg", clip=group, family="")
        for i in range(3)
    ]
    stats = run_autonomous_pass(db, path, limit=50)
    assert stats["new_concept"] >= 3
    pools = list_inbox_pools(db, path)
    new_ids = {c.file_id for c in pools.get("new_concept") or []}
    assert set(ids) <= new_ids
    for fid in ids:
        feat = db.get_features(fid)
        tm = (feat or {}).get("texture_map") or {}
        assert not tm.get("autonomous_label")
        assert not tm.get("user_labeled")


def test_user_reject_does_not_learn_concept(tmp_path):
    db, path = _db(tmp_path)
    vec = _unit([1, 0, 0, 0])
    a = _add(db, tmp_path, "kedi.jpg", clip=vec)
    b = _add(db, tmp_path, "tilki.jpg", clip=vec)
    teach_files(db, path, [a], "Kedi")
    teach_files(db, path, [b], "Tilki")
    mix = _add(db, tmp_path, "mix.jpg", clip=vec, family="")
    run_autonomous_pass(db, path, limit=50)
    n = reject_reviews(path, [mix])
    assert n >= 1
    pools = list_inbox_pools(db, path)
    all_ids = {c.file_id for bucket in pools.values() for c in bucket}
    assert mix not in all_ids
    feat = db.get_features(mix)
    tm = (feat or {}).get("texture_map") or {}
    assert not tm.get("user_labeled")
    assert not tm.get("autonomous_label")


def test_user_correction_is_verified_truth(tmp_path):
    db, path = _db(tmp_path)
    vec = _unit([1, 0, 0, 0])
    a = _add(db, tmp_path, "kedi.jpg", clip=vec)
    b = _add(db, tmp_path, "tilki.jpg", clip=vec)
    teach_files(db, path, [a], "Kedi")
    teach_files(db, path, [b], "Tilki")
    mix = _add(db, tmp_path, "mix.jpg", clip=vec, family="")
    run_autonomous_pass(db, path, limit=50)
    stats = correct_reviews(db, path, [mix], "Köpek")
    assert stats["taught"] >= 1
    hits = match_clip_to_concepts(path, vec)
    names = [h["canonical"] for h in hits]
    assert "Köpek" in names


def test_confirm_persists_after_reopen(tmp_path):
    db, path = _db(tmp_path)
    proto = _unit([1, 0, 0, 0])
    seed = _add(db, tmp_path, "seed.jpg", clip=proto)
    teach_files(db, path, [seed], "Kedi")
    other = _add(db, tmp_path, "tilki.jpg", clip=proto)
    teach_files(db, path, [other], "Tilki")
    near = _unit([0.99, 0.141067, 0, 0])
    target = _add(db, tmp_path, "ask.jpg", clip=near, family="")
    run_autonomous_pass(db, path, limit=50)
    db2 = Database(path)
    pools = list_inbox_pools(db2, path)
    pending = {c.file_id for bucket in pools.values() for c in bucket}
    assert target in pending
    confirm_reviews(db2, path, [target], label="Kedi")
    db3 = Database(path)
    pools2 = list_inbox_pools(db3, path)
    left = {c.file_id for bucket in pools2.values() for c in bucket}
    assert target not in left
    hits = match_clip_to_concepts(path, near)
    assert hits and hits[0]["canonical"] == "Kedi"


def test_auto_example_does_not_seed_matching(tmp_path):
    db, path = _db(tmp_path)
    vec = _unit([0, 1, 0, 0])
    fid = _add(db, tmp_path, "auto_only.jpg", clip=vec)
    cid = upsert(path, "YanlışKavram", source="auto")
    add_example(
        path,
        cid,
        file_id=fid,
        role="positive",
        source="auto",
        embedding=vec,
        embedding_backend="clip",
    )
    hits = match_clip_to_concepts(path, vec)
    assert hits == []
    assert all(ex.get("file_id") != fid for ex in positive_example_vectors(path))


def test_user_label_outranks_auto(tmp_path):
    db, path = _db(tmp_path)
    proto = _unit([1, 0, 0, 0])
    seed = _add(db, tmp_path, "seed.jpg", clip=proto)
    teach_files(db, path, [seed], "Kedi")
    near = _unit([0.99, 0.141067, 0, 0])
    target = _add(
        db,
        tmp_path,
        "manual.jpg",
        clip=near,
        tm={
            "user_labeled": True,
            "user_label_source": "teach_me",
            "pattern_family": "floral",
        },
        family="floral",
    )
    run_autonomous_pass(db, path, limit=50)
    feat = db.get_features(target)
    tm = (feat or {}).get("texture_map") or {}
    assert tm.get("pattern_family") == "floral"
    assert tm.get("user_labeled") is True
    assert tm.get("autonomous_label") in (None, "")
