from types import SimpleNamespace

from core.search_engine import SearchEngine


def _row(fid, score, **debug):
    return SimpleNamespace(file_id=fid, score=score, debug=debug)


def test_human_semantic_result_survives_global_text_threshold():
    row = _row(
        101,
        0.24,
        human_semantic_only=True,
        gender_visual_score=0.24,
    )
    assert SearchEngine._passes_search_threshold(row, 0.60) is True


def test_face_gender_result_survives_global_text_threshold():
    row = _row(102, 0.10, face_gender_match=True)
    assert SearchEngine._passes_search_threshold(row, 0.60) is True


def test_normal_text_result_still_respects_global_threshold():
    row = _row(103, 0.59)
    assert SearchEngine._passes_search_threshold(row, 0.60) is False


def test_human_semantic_canonical_marker_survives_global_text_threshold():
    row = _row(
        105,
        0.16,
        human_semantic_mode=True,
        human_semantic_score=0.19,
    )
    assert SearchEngine._passes_search_threshold(row, 0.60) is True


def test_human_semantic_canonical_marker_below_floor_is_rejected():
    row = _row(
        106,
        0.16,
        human_semantic_mode=True,
        human_semantic_score=0.17,
    )
    assert SearchEngine._passes_search_threshold(row, 0.60) is False


def test_weak_human_semantic_result_is_still_rejected():
    row = _row(
        104,
        0.12,
        human_semantic_only=True,
        gender_visual_score=0.12,
    )
    assert SearchEngine._passes_search_threshold(row, 0.60) is False


def test_object_index_exact_survives_global_text_threshold():
    row = _row(201, 0.56, object_index_hit=True)
    assert SearchEngine._passes_search_threshold(row, 0.60) is True


def test_object_index_parent_does_not_bypass_threshold():
    row = _row(202, 0.34, object_index_parent_hit=True)
    assert SearchEngine._passes_search_threshold(row, 0.60) is False


def test_gender_still_rejects_generic_clip_without_face():
    row = _row(203, 0.88, human_semantic_mode=True, human_semantic_score=0.0)
    assert SearchEngine._passes_search_threshold(row, 0.60) is False
