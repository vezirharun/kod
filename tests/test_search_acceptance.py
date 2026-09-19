from types import SimpleNamespace

from core.search_acceptance import result_passes_threshold


def r(score, **debug):
    return SimpleNamespace(score=score, debug=debug)


def test_human_uses_human_floor_not_global_threshold():
    assert result_passes_threshold(
        r(0.25, human_semantic_mode=True, human_semantic_score=0.25),
        0.60,
    )


def test_human_below_floor_is_rejected():
    assert not result_passes_threshold(
        r(0.17, human_semantic_mode=True, human_semantic_score=0.17),
        0.60,
    )


def test_normal_result_uses_global_threshold():
    assert result_passes_threshold(r(0.60), 0.60)
    assert not result_passes_threshold(r(0.59), 0.60)


def test_protected_exact_is_kept():
    assert result_passes_threshold(r(0.10, protected_exact=True), 0.60)


def test_gender_match_is_kept():
    assert result_passes_threshold(r(0.10, face_gender_match=True), 0.60)


def test_entity_uses_entity_floor_not_global_threshold():
    assert result_passes_threshold(
        r(0.20, entity_semantic_mode=True, entity_semantic_score=0.20),
        0.60,
    )


def test_entity_below_floor_is_rejected():
    assert not result_passes_threshold(
        r(0.17, entity_semantic_mode=True, entity_semantic_score=0.17),
        0.60,
    )
