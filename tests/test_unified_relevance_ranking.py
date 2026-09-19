from dataclasses import replace


def _result(file_id, score, breakdown=None, debug=None):
    from core.search_engine import SearchResult
    return SearchResult(
        file_id=file_id,
        path=f"/x/{file_id}.jpg",
        filename=f"{file_id}.jpg",
        customer="",
        thumbnail_path="",
        score=score,
        score_percent=score * 100,
        breakdown=dict(breakdown or {}),
        debug=dict(debug or {}),
    )


def test_text_visual_evidence_can_outrank_generic_text_score():
    from core.unified_relevance_ranker import apply_unified_text_ranking
    weak = _result(1, 0.92, {"filename_score": 0.92}, {"clip_score": 0.18})
    strong = _result(2, 0.72, {"family_score": 0.72}, {"clip_score": 0.96, "visual_grade": "visual_strong"})
    out = apply_unified_text_ranking([weak, strong])
    assert out[0].file_id == 2
    assert out[0].score > out[1].score


def test_exact_metadata_survives_when_ai_visual_signal_is_missing():
    from core.unified_relevance_ranker import apply_unified_text_ranking
    exact = _result(10, 0.88, {"filename_score": 0.95}, {"clip_score": 0.0})
    generic = _result(11, 0.70, {"family_score": 0.70}, {"clip_score": 0.0})
    out = apply_unified_text_ranking([generic, exact])
    assert out[0].file_id == 10
    assert out[0].score >= 0.88


def test_gender_visual_score_is_the_primary_text_signal():
    from core.unified_relevance_ranker import apply_unified_text_ranking
    generic = _result(20, 0.99, {"filename_score": 0.99}, {"clip_score": 0.10})
    woman = _result(21, 0.76, {"semantic_score": 0.76}, {"gender_visual_score": 0.99, "human_semantic_score": 0.99})
    out = apply_unified_text_ranking([generic, woman], human_query=True)
    assert out[0].file_id == 21
    assert out[0].debug["ranking_engine"] == "unified_relevance_v1"


def test_unified_ranker_contract_is_wired_into_text_search():
    import inspect
    from core.search_engine import SearchEngine
    src = inspect.getsource(SearchEngine.search_by_text)
    assert "apply_unified_text_ranking" in src
    assert "unified_relevance_ranking" in src


def test_leaf_motif_clip_floor_survives_unified_caps():
    from core.unified_relevance_ranker import apply_unified_text_ranking, compute_unified_relevance

    r = _result(
        30,
        0.48,
        {},
        {
            "clip_score": 0.28,
            "semantic_intent": {"motif": "leaf"},
            "visual_win": False,
        },
    )
    score, _ = compute_unified_relevance(r, mode="text")
    assert score >= 0.60
    out = apply_unified_text_ranking([r], query_text="yaprak")
    assert out[0].score >= 0.60


def test_unified_ranker_keeps_unlimited_results_contract():
    import inspect
    from core.search_engine import SearchEngine
    src = inspect.getsource(SearchEngine.search_by_text)
    assert "unlimited = int(requested_limit) <= 0" in src
    assert "limit = 0 if unlimited" in src
