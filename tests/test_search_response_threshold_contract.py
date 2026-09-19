from core.search_models import SearchResponse


class R:
    def __init__(self, score, **debug):
        self.score = score
        self.debug = debug


def test_human_semantic_result_survives_ui_threshold():
    r = R(0.22, human_semantic_mode=True, human_semantic_score=0.31)
    response = SearchResponse(all_results=[r])
    assert response.filter_by_threshold(0.60) == [r]
    assert response.count_above(0.60) == 1


def test_human_semantic_result_below_floor_is_rejected():
    r = R(0.22, human_semantic_mode=True, human_semantic_score=0.17)
    response = SearchResponse(all_results=[r])
    assert response.filter_by_threshold(0.60) == []
    assert response.count_above(0.60) == 0


def test_face_gender_match_is_authoritative():
    r = R(0.05, face_gender_match=True)
    response = SearchResponse(all_results=[r])
    assert response.filter_by_threshold(0.60) == [r]


def test_normal_results_still_use_global_threshold():
    low = R(0.59)
    high = R(0.61)
    response = SearchResponse(all_results=[low, high])
    assert response.filter_by_threshold(0.60) == [high]
