"""Universal Visual Intelligence fixtures — query-time, no index writes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.pattern_intelligence_v2 import (
    REP_PHOTO,
    REP_TEXTILE,
    build_pattern_card,
    parse_pattern_query_v2,
)
from core.universal_visual_intel import (
    CAPABILITIES,
    FORBIDDEN_LABELS,
    ONTOLOGY,
    TIER_EXACT,
    TIER_SAME_FAMILY,
    UNSUPPORTED,
    apply_universal_ranking,
    build_visual_dna,
    cluster_clip_identities,
    fine_grained_tier,
    node_path,
    parse_universal_query,
    siblings,
)


def test_hierarchy_paths():
    assert node_path("crow") == ["entity", "animal", "bird", "crow"]
    assert node_path("rose") == ["entity", "plant", "flower", "rose"]
    assert node_path("togg") == ["entity", "vehicle", "car", "togg"]
    assert "stork" in siblings("crow")
    assert "bmw" in siblings("togg")


def test_open_ontology_not_tiny_closed_list():
    assert len(ONTOLOGY) >= 30
    for nid in ("person", "building", "insect", "ant", "tulip", "motorcycle"):
        assert nid in ONTOLOGY


def test_nl_parse_daisy_navy():
    q = parse_universal_query("küçük beyaz papatyalı lacivert kumaş")
    assert q.node_id == "daisy"
    pq = q.pattern_query
    assert pq.get("scale") == "small"
    q2 = parse_universal_query("küçük kargalı desen")
    assert q2.node_id == "crow"
    q3 = parse_universal_query("leopar ve yılan")
    assert q3.pattern_query.get("composition") == "composite" or q3.leaf == ""


def test_karga_not_leylek():
    q = parse_universal_query("karga")
    assert q.node_id == "crow" and q.leaf == "crow"
    crow_win, _ = fine_grained_tier(q, {"crow": 0.36, "stork": 0.12})
    stork_as_crow, reason = fine_grained_tier(q, {"crow": 0.11, "stork": 0.34})
    no_rival, nr_reason = fine_grained_tier(q, {"crow": 0.36})
    assert crow_win == TIER_EXACT
    assert stork_as_crow != TIER_EXACT
    assert "unspecific" in reason or stork_as_crow == TIER_SAME_FAMILY
    assert no_rival != TIER_EXACT
    assert nr_reason == "no_rival_scores"

    crow = SimpleNamespace(file_id=1, score=0.4, debug={})
    stork = SimpleNamespace(file_id=2, score=0.9, debug={})
    kept, stats = apply_universal_ranking(
        q,
        [crow, stork],
        clip_by_id={1: {"crow": 0.36, "stork": 0.12}, 2: {"crow": 0.10, "stork": 0.35}},
        clip_active=True,
    )
    ids = [r.file_id for r in kept]
    assert 1 in ids
    assert 2 not in ids
    assert stats["dropped_unspecific"] >= 1


def test_togg_above_other_cars():
    q = parse_universal_query("Togg")
    assert q.node_id == "togg"
    exact, _ = fine_grained_tier(q, {"togg": 0.34, "bmw": 0.14, "car": 0.22})
    family, _ = fine_grained_tier(q, {"togg": 0.12, "bmw": 0.33, "car": 0.28})
    assert exact == TIER_EXACT
    assert family != TIER_EXACT

    q_car = parse_universal_query("araba")
    assert q_car.node_id == "car" and not q_car.leaf
    fam, reason = fine_grained_tier(q_car, {"togg": 0.31, "car": 0.20})
    assert fam == TIER_SAME_FAMILY
    assert reason == "family_query"


def test_filename_is_not_identity_or_brand_proof():
    q = parse_universal_query("Togg")
    row = SimpleNamespace(file_id=9, score=0.8, debug={"filename": "togg.jpg"})
    kept, stats = apply_universal_ranking(q, [row], clip_by_id={9: {}}, clip_active=True)
    assert kept == []
    assert stats["dropped_unspecific"] >= 1


def test_visual_dna_merges_v2_card():
    pq = parse_pattern_query_v2("leopar üzerine gül")
    card = build_pattern_card(
        pq,
        clip_scores={"leopard": 0.33, "rose": 0.30},
        rep_scores={REP_TEXTILE: 0.32, REP_PHOTO: 0.14},
    )
    uq = parse_universal_query("leopar üzerine gül")
    dna = build_visual_dna(uq, card=card.to_dict(), search_tier=TIER_EXACT)
    d = dna.to_dict()
    assert d["pattern_family"]
    assert "race" not in d and "ethnicity" not in d
    assert d["relationships"]
    assert d["representation"] in (REP_TEXTILE, "unknown") or d["textile"] is True


def test_representation_photo_vs_textile_capability():
    uq = parse_universal_query("gerçek leopar fotoğrafı")
    assert uq.pattern_query.get("representation") == REP_PHOTO
    assert CAPABILITIES["representation_textile_vs_photo"] == "experimental"
    assert CAPABILITIES["face_recognition"] == "experimental"
    assert CAPABILITIES["race_ethnicity_classification"] == UNSUPPORTED
    assert CAPABILITIES["clip_zero_shot_species"] == "experimental"
    assert CAPABILITIES["dedicated_togg_classifier"] == UNSUPPORTED


def test_no_ethnicity_in_ontology():
    blob = " ".join(
        [str(k) for k in ONTOLOGY]
        + [a for meta in ONTOLOGY.values() for a in meta.get("aliases") or ()]
    ).lower()
    for bad in FORBIDDEN_LABELS:
        assert bad not in blob


def test_identity_cluster_similar_vectors():
    rng = np.random.default_rng(7)
    base = rng.normal(size=32).astype(np.float32)
    other = rng.normal(size=32).astype(np.float32)
    labels = cluster_clip_identities(
        {
            1: base,
            2: base + 0.02,
            3: other * 3.0,
        },
        threshold=0.88,
    )
    assert labels[1] == labels[2]
    assert labels[3] != labels[1]
    assert CAPABILITIES["identity_clip_clustering"] == "experimental"
    assert CAPABILITIES["person_reidentification_gallery"] == UNSUPPORTED


def test_bird_family_keeps_species_without_calling_them_crow():
    q = parse_universal_query("kuş")
    assert q.node_id == "bird" and not q.leaf
    crow = SimpleNamespace(file_id=1, score=0.5, debug={})
    stork = SimpleNamespace(file_id=2, score=0.5, debug={})
    kept, _ = apply_universal_ranking(
        q,
        [crow, stork],
        clip_by_id={1: {"crow": 0.34, "bird": 0.30}, 2: {"stork": 0.33, "bird": 0.29}},
        clip_active=True,
    )
    assert {r.file_id for r in kept} == {1, 2}
    assert all(r.debug.get("uvi_tier") == TIER_SAME_FAMILY for r in kept)


def test_object_index_hit_keeps_bird_without_clip():
    q = parse_universal_query("kuş")
    bird = SimpleNamespace(file_id=7, score=0.56, debug={"object_index_hit": True})
    kept, _ = apply_universal_ranking(
        q, [bird], clip_by_id={}, clip_active=True,
    )
    assert [r.file_id for r in kept] == [7]


def test_parent_object_index_is_not_crow_evidence():
    q = parse_universal_query("karga")
    parent = SimpleNamespace(file_id=3, score=0.34, debug={"object_index_parent_hit": True})
    kept, stats = apply_universal_ranking(
        q, [parent], clip_by_id={}, clip_active=True,
    )
    assert 3 not in {r.file_id for r in kept}
    assert stats["dropped_unspecific"] >= 1


def test_search_bags_crow_retrieves_bird_parent_only():
    from core.visual_concept_dna import parse_search_bags

    crow = parse_search_bags("karga")
    assert "crow" in crow.objects
    assert "bird" in crow.parent_objects
    assert "bird" not in crow.objects
    bird = parse_search_bags("kuş")
    assert "bird" in bird.objects
    car = parse_search_bags("araba")
    assert "car" in car.objects
    assert car.query_sense == "both"
    animal = parse_search_bags("yılan hayvanı")
    assert animal.query_sense == "animal"
    skin = parse_search_bags("yılan derisi")
    assert skin.query_sense == "pattern"
    assert skin.pattern_weighted is True
