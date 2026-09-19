from types import SimpleNamespace

from core.fuzzy_correct import correct_term
from core.search_evidence_gate import (
    ZERO_SHOT_FLOOR,
    ZERO_SHOT_TYPE,
    apply_search_evidence_gate,
    has_zero_shot_visual,
    score_zero_shot_gate,
    stamp_zero_shot,
    zero_shot_spec,
)
from core.search_engine import SearchEngine
from core.visual_concept import DEFAULT_FUSION_WEIGHTS, detector_labels_for_lemma


def test_cherry_is_zero_shot_not_detector():
    spec = zero_shot_spec("kiraz")
    assert spec is not None
    assert spec["concept"] == "cherry"
    assert spec["type"] == ZERO_SHOT_TYPE
    assert spec["source"] == "openclip"
    assert not detector_labels_for_lemma("cherry")


def test_apple_stays_detector_backed():
    assert detector_labels_for_lemma("apple")
    assert zero_shot_spec("elma") is None


def test_gender_is_not_zero_shot():
    assert zero_shot_spec("kadın") is None
    assert zero_shot_spec("erkek") is None
    row = SimpleNamespace(
        score=0.91,
        debug={"clip_score": 0.42, "clip_as_detector": False},
        breakdown={"clip": 0.42},
    )
    assert has_zero_shot_visual(row, "kadın") is False


def test_pattern_lane_is_not_zero_shot():
    assert zero_shot_spec("leopar") is None
    assert zero_shot_spec("çiçek") is None
    assert zero_shot_spec("kuş deseni") is None
    assert zero_shot_spec("ekose") is None
    assert zero_shot_spec("barok") is None
    assert zero_shot_spec("suluboya") is None
    assert zero_shot_spec("etnik") is None
    assert zero_shot_spec("fırça etkisi") is None


def test_jewelry_is_zero_shot_object():
    spec = zero_shot_spec("takı")
    assert spec is not None
    assert spec["concept"] == "jewelry"
    assert "textile" in spec["prompt"] or "jewelry" in spec["prompt"]



def test_crow_and_lips_are_zero_shot_concepts():
    assert zero_shot_spec("karga")["concept"] == "crow"
    assert zero_shot_spec("dudak")["concept"] == "lips"
    assert zero_shot_spec("timsah")["concept"] == "crocodile"


def test_gate_keeps_zero_shot_cherry_without_calling_it_detector():
    row = SimpleNamespace(
        score=0.31,
        debug={"clip_score": 0.31, "clip_as_detector": False},
        breakdown={"clip": 0.31},
    )
    out, meta = apply_search_evidence_gate([row], "kiraz")
    assert out
    assert out[0].debug.get("evidence_type") == ZERO_SHOT_TYPE
    assert out[0].debug.get("clip_as_detector") is False
    assert out[0].debug.get("object_index_hit") in (None, False)
    assert meta.get("clip_as_detector") is False


def test_gate_still_rejects_weak_clip_cherry():
    row = SimpleNamespace(
        score=0.12,
        debug={"clip_score": 0.12},
        breakdown={"clip": 0.12},
    )
    out, meta = apply_search_evidence_gate([row], "kiraz")
    assert out == []
    assert "Kiraz" in str(meta.get("empty_state") or "")


def test_zero_shot_survives_threshold_without_becoming_exact():
    row = SimpleNamespace(
        file_id=9,
        score=0.31,
        debug={
            "evidence_type": ZERO_SHOT_TYPE,
            "zero_shot_visual": True,
            "zero_shot_score": 0.31,
            "clip_score": 0.31,
        },
    )
    assert SearchEngine._passes_search_threshold(row, 0.60) is True
    weak = SimpleNamespace(
        file_id=10,
        score=0.20,
        debug={
            "evidence_type": ZERO_SHOT_TYPE,
            "zero_shot_visual": True,
            "zero_shot_score": 0.20,
            "clip_score": 0.20,
        },
    )
    assert SearchEngine._passes_search_threshold(weak, 0.60) is False
    assert ZERO_SHOT_FLOOR >= 0.27


def test_stamp_does_not_override_exact_or_face():
    spec = {"concept": "cherry", "type": ZERO_SHOT_TYPE, "source": "openclip"}
    exact = SimpleNamespace(debug={"object_index_hit": True})
    stamp_zero_shot(exact, spec, 0.4)
    assert exact.debug.get("evidence_type") != ZERO_SHOT_TYPE
    face = SimpleNamespace(debug={"face_gender_match": True})
    stamp_zero_shot(face, spec, 0.4)
    assert face.debug.get("evidence_type") != ZERO_SHOT_TYPE


def test_fusion_weights_unchanged():
    assert DEFAULT_FUSION_WEIGHTS["clip"] == 0.34
    assert DEFAULT_FUSION_WEIGHTS["object_index"] == 0.16


def test_gate_rejects_floral_print_as_cherry():
    row = SimpleNamespace(
        score=0.31,
        pattern_family="floral",
        debug={"clip_score": 0.31, "clip_as_detector": False},
        breakdown={"clip": 0.31},
    )
    out, _meta = apply_search_evidence_gate([row], "kiraz")
    assert out == []


def test_clip_margin_rejects_crow_when_butterfly_wins():
    row = SimpleNamespace(
        score=0.31,
        pattern_family="unknown",
        debug={
            "clip_score": 0.31,
            "zs_clip_rivals": {"butterfly": 0.32, "cherry": 0.18, "textile": 0.20},
            "clip_as_detector": False,
        },
        breakdown={"clip": 0.31},
    )
    out, _meta = apply_search_evidence_gate([row], "karga")
    assert out == []


def test_clip_margin_keeps_crow_over_butterfly():
    row = SimpleNamespace(
        score=0.31,
        pattern_family="unknown",
        debug={
            "clip_score": 0.305,
            "zs_clip_rivals": {"butterfly": 0.211, "cherry": 0.177, "textile": 0.226},
            "clip_as_detector": False,
        },
        breakdown={"clip": 0.305},
    )
    out, _meta = apply_search_evidence_gate([row], "karga")
    assert out
    assert out[0].debug.get("evidence_type") == ZERO_SHOT_TYPE
    assert out[0].debug.get("zs_verified") is True


def test_cherry_loses_when_textile_clip_is_close():
    row = SimpleNamespace(
        score=0.29,
        pattern_family="unknown",
        debug={
            "clip_score": 0.288,
            "zs_clip_rivals": {"floral": 0.246, "textile": 0.246, "butterfly": 0.22},
            "clip_as_detector": False,
        },
        breakdown={"clip": 0.288},
    )
    out, _meta = apply_search_evidence_gate([row], "kiraz")
    assert out == []


def test_cilek_is_not_fuzzy_corrected_to_cicek():
    for term in ("çilek", "cilek", "Çilek"):
        res = correct_term(term)
        assert res.was_corrected is False
        assert "cicek" not in res.corrected.lower()
        assert "çiçek" not in res.corrected.lower()
    flower = correct_term("çiçek")
    assert flower.was_corrected is False
    assert flower.corrected in ("çiçek", "cicek")


def test_strawberry_loses_when_flower_clip_wins():
    ok, meta = score_zero_shot_gate(
        "strawberry",
        0.31,
        {"flower": 0.32, "butterfly": 0.18, "jewelry": 0.10, "floral": 0.20},
    )
    assert ok is False
    assert meta.get("clip_best_rival") == "flower"
    keep, _ = score_zero_shot_gate(
        "strawberry",
        0.31,
        {"flower": 0.20, "butterfly": 0.18, "jewelry": 0.10, "floral": 0.20},
    )
    assert keep is True
