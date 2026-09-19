from core.search_engine import _visual_candidate_ids


def test_uvi_visual_hits_are_promoted_when_not_in_text_candidates():
    candidates = [{"id": 10}, {"id": 20}]
    visual_hits = {30: 0.71, 20: 0.69, 40: 0.66}
    assert _visual_candidate_ids(candidates, visual_hits) == [30, 40]


def test_visual_candidate_union_ignores_invalid_ids_and_respects_limit():
    candidates = [{"id": 1}]
    visual_hits = {"x": 0.8, 2: 0.7, 3: 0.6, 4: 0.5}
    assert _visual_candidate_ids(candidates, visual_hits, limit=2) == [2, 3]
