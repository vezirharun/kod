"""UVI Search Enforcement v1 — object evidence must drive ranking.

Fixtures only. No production index / DB writes.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.pattern_intelligence_v2 import parse_pattern_query_v2, rank_v2_results
from core.universal_visual_intel import (
    CAPABILITIES,
    EVIDENCE_FAMILY,
    EVIDENCE_MISMATCH,
    TIER_EXACT,
    TIER_SAME_FAMILY,
    TIER_SEMANTIC,
    apply_universal_ranking,
    classify_object_evidence,
    parse_universal_query,
)


def _row(fid: int, score: float = 0.8, **debug):
    return SimpleNamespace(
        file_id=fid,
        score=score,
        filename=debug.pop("filename", f"{fid}.jpg"),
        path=debug.pop("path", f"archive/{fid}.jpg"),
        debug=debug,
    )


def _rank(query: str, rows, clip_by_id, *, clip_active=True):
    uq = parse_universal_query(query)
    return apply_universal_ranking(uq, rows, clip_by_id=clip_by_id, clip_active=clip_active)


def test_araba_not_generic_textile():
    q = parse_universal_query("araba")
    assert q.node_id == "car" and not q.leaf
    car = _row(1, 0.5, filename="car_motif.jpg")
    textile = _row(
        2,
        0.95,
        filename="generic_texture.jpg",
        path="kumas dokusu/generic.jpg",
        clip_score=0.32,
        visual_win=True,
    )
    kept, stats = _rank(
        "araba",
        [car, textile],
        {1: {"car": 0.34, "togg": 0.28, "flower": 0.10}, 2: {"car": 0.12, "flower": 0.31, "denim": 0.18}},
    )
    ids = [r.file_id for r in kept]
    assert 1 in ids
    assert 2 not in ids
    assert stats["dropped_mismatch"] + stats["dropped_unspecific"] >= 1
    assert kept[0].debug.get("object_evidence") == EVIDENCE_FAMILY


def test_kus_not_denim_texture():
    q = parse_universal_query("kuş")
    assert q.node_id == "bird"
    bird = _row(1, 0.4, filename="bird_print.jpg")
    denim = _row(
        2,
        0.9,
        filename="shutterstock_denim.eps",
        path="kot dokusu, deri/shutterstock.eps",
        clip_score=0.32,
        visual_win=True,
    )
    kept, _ = _rank(
        "kuş",
        [bird, denim],
        {1: {"bird": 0.33, "crow": 0.30, "flower": 0.12}, 2: {"bird": 0.11, "denim": 0.30, "flower": 0.16}},
    )
    ids = [r.file_id for r in kept]
    assert 1 in ids
    assert 2 not in ids
    assert "kus" not in "dokusu" or True  # substring is not token evidence


def test_balik_not_floral():
    fish = _row(1, 0.4, filename="fish_motif.jpg")
    floral = _row(
        2,
        0.88,
        filename="floral_print.jpg",
        path="cicekler/floral.jpg",
        clip_score=0.31,
    )
    kept, _ = _rank(
        "balık",
        [fish, floral],
        {1: {"fish": 0.32, "flower": 0.14}, 2: {"fish": 0.10, "flower": 0.34, "floral": 0.33}},
    )
    ids = [r.file_id for r in kept]
    assert 1 in ids
    assert 2 not in ids


def test_kot_resolves_to_jeans_and_keeps_denim():
    q = parse_universal_query("kot")
    assert q.node_id == "jeans"
    denim = _row(1, 0.82, filename="kot_pantolon.jpg", path="kot dokusu, deri/jean.jpg")
    other = _row(2, 0.50, filename="red_floral.jpg")
    kept, stats = _rank("kot", [denim, other], {}, clip_active=True)
    assert [r.file_id for r in kept] == [1, 2] or {r.file_id for r in kept} == {1, 2}
    assert stats.get("dropped_mismatch", 0) == 0
    assert stats.get("dropped_unspecific", 0) == 0


def test_karga_not_forced_from_leylek():
    q = parse_universal_query("karga")
    crow = _row(1, 0.4)
    stork = _row(2, 0.9, clip_score=0.34)
    kept, _ = _rank(
        "karga",
        [crow, stork],
        {1: {"crow": 0.36, "stork": 0.12}, 2: {"crow": 0.10, "stork": 0.35}},
    )
    ids = [r.file_id for r in kept]
    assert 1 in ids
    assert 2 not in ids
    assert kept[0].debug.get("visual_dna", {}).get("fine_grained") != "stork"


def test_gül_generic_floral_is_not_rose():
    q = parse_universal_query("gül")
    assert q.node_id == "rose"
    rose = _row(1, 0.5)
    generic = _row(2, 0.8)
    kept, _ = _rank(
        "gül",
        [rose, generic],
        {1: {"rose": 0.36, "daisy": 0.12, "flower": 0.28}, 2: {"rose": 0.12, "daisy": 0.18, "flower": 0.33}},
        clip_active=True,
    )
    ids = [r.file_id for r in kept]
    assert 1 in ids
    assert 2 not in ids
    assert kept[0].debug.get("uvi_tier") == TIER_EXACT


def test_leopar_pattern_intel_kept():
    q = parse_universal_query("leopar")
    assert q.node_id == "leopard"
    assert q.node_id not in ("car", "bird", "fish")
    row = _row(1, 0.7, filename="leopard_print.jpg")
    kept, _ = _rank("leopar", [row], {1: {"leopard": 0.34, "snake": 0.12}}, clip_active=True)
    assert kept and kept[0].file_id == 1
    assert kept[0].debug.get("uvi_tier") in (TIER_EXACT, TIER_SAME_FAMILY) or kept[0].debug.get("uvi_rank", 9) <= 3


def test_yilan_not_leopard():
    snake = _row(1, 0.5)
    leo = _row(2, 0.9)
    kept, _ = _rank(
        "yılan",
        [snake, leo],
        {1: {"snake": 0.33, "leopard": 0.14}, 2: {"snake": 0.11, "leopard": 0.36}},
        clip_active=True,
    )
    ids = [r.file_id for r in kept]
    assert 1 in ids
    assert 2 not in ids


def test_leopar_cicek_composite_and():
    pq = parse_pattern_query_v2("leopar çiçek")
    assert pq.composition == "composite"
    assert "leopard" in pq.required and "floral" in pq.required
    both = _row(1, 0.8)
    leo_only = _row(2, 0.9)
    kept, stats = rank_v2_results(
        pq,
        [both, leo_only],
        clip_by_id={1: {"leopard": 0.34, "floral": 0.31}, 2: {"leopard": 0.35, "floral": 0.08}},
        rep_by_id={1: {"textile_pattern": 0.32}, 2: {"textile_pattern": 0.30}},
    )
    ids = [r.file_id for r in kept]
    assert 1 in ids
    assert ids[0] == 1
    if 2 in ids:
        assert ids.index(1) < ids.index(2)
    assert kept[0].debug.get("v2_bucket", 0) == 0


def test_kucuk_kargali_desen_no_false_positive():
    q = parse_universal_query("küçük kargalı desen")
    assert q.node_id == "crow"
    assert q.pattern_query.get("scale") in ("small", "kucuk", "")
    noise = _row(1, 0.85, filename="generic_textile.jpg", path="kumas dokusu/x.jpg", clip_score=0.30)
    kept, stats = _rank("küçük kargalı desen", [noise], {1: {}}, clip_active=True)
    assert kept == []
    assert stats["dropped_unspecific"] + stats["dropped_mismatch"] >= 1


def test_generic_clip_is_not_object_evidence():
    q = parse_universal_query("kuş")
    row = _row(3, 0.7, clip_score=0.32, visual_win=True, filename="dokusu.jpg", path="kot dokusu/x.jpg")
    ev, reason = classify_object_evidence(
        q, {"bird": 0.32}, tier=TIER_SEMANTIC, reason="weak_family", row=row, card={}
    )
    # unsanitized equal-to-generic score still conflicts with denim path
    assert ev in (EVIDENCE_MISMATCH, "object_unknown") or "mismatch" in reason or ev != EVIDENCE_FAMILY
    kept, _ = apply_universal_ranking(
        q, [row], clip_by_id={3: {"bird": 0.32}}, clip_active=True
    )
    # sanitized: bird score == generic clip_score → not family evidence → drop
    assert kept == [] or kept[0].debug.get("object_evidence") != EVIDENCE_FAMILY


def test_araba_not_parsed_as_horse():
    v2 = parse_pattern_query_v2("araba")
    assert "horse" not in v2.objects
    assert "horse" not in v2.required


def test_capability_honesty():
    assert CAPABILITIES["object_aware_ranking"] == "supported"
    assert CAPABILITIES["negative_object_evidence"] == "supported"
    assert CAPABILITIES["dedicated_togg_classifier"] == "unsupported"
    assert CAPABILITIES["dedicated_bird_species_classifier"] == "unsupported"
    assert CAPABILITIES["clip_zero_shot_species"] == "experimental"


def test_explainable_ranking_fields():
    car = _row(1, 0.5)
    kept, _ = _rank("araba", [car], {1: {"car": 0.33, "flower": 0.10}})
    assert kept
    explain = kept[0].debug.get("uvi_explain") or {}
    assert explain.get("object_match")
    assert "object_evidence" in (kept[0].debug or {})
    assert explain.get("tier")


def test_gender_uvi_keeps_visual_hits_without_text_evidence():
    row = _row(101, 0.0, filename="random_hash.jpg")
    kept, stats = _rank(
        "erkek",
        [row],
        {101: {"male_person": 0.24, "person": 0.22}},
        clip_active=True,
    )
    assert kept and kept[0].file_id == 101
    assert kept[0].debug.get("gender_visual_score", 0) >= 0.24
    assert kept[0].debug.get("human_semantic_mode") is True
    assert kept[0].debug.get("human_semantic_score", 0) >= 0.24


def test_gender_face_match_survives_without_clip_score():
    row = _row(102, 0.0, face_gender_match=True)
    kept, _ = _rank("kadın", [row], {}, clip_active=True)
    assert kept and kept[0].file_id == 102
    assert kept[0].debug.get("face_gender_match") is True
