from core.display_match_percent import (
    display_match_percent,
    display_sort_tuple,
    merge_results_in_engine_order,
    stamp_display_match_percent,
)
from core.search_engine import SearchResult
from core.visual_pattern_query import apply_visual_pattern_priority


def _row(
    file_id: int,
    *,
    score: float,
    dna: float,
    family: float = 0.0,
    texture: float = 0.0,
    dino: float = 0.0,
    clip: float = 0.0,
    visual_positive: bool = True,
    negative: bool = False,
) -> SearchResult:
    return SearchResult(
        file_id=file_id,
        path=f"/x/{file_id}.jpg",
        filename=f"{file_id}.jpg",
        customer="",
        thumbnail_path="",
        score=score,
        score_percent=score * 100,
        breakdown={
            "dna_score": dna,
            "family_score": family,
            "texture_score": texture,
            "dino": dino,
        },
        debug={
            "clip_score": clip,
            "dino_score": dino,
            "pattern_visual_positive": visual_positive,
            "pattern_visual_negative": negative,
            "pattern_visual_signals": {
                "positive": visual_positive,
                "negative": negative,
                "dna": dna,
                "family": family,
                "texture": texture,
                "dino": dino,
                "clip": clip,
            },
        },
    )


def test_display_percent_monotone_when_clip_inverted():
    # Engine DNA-first order; CLIP / score_percent inverted vs that order.
    rows = [
        _row(1, score=0.20, dna=0.95, family=0.9, texture=0.8, dino=0.4, clip=0.15),
        _row(2, score=0.80, dna=0.55, family=0.5, texture=0.4, dino=0.7, clip=0.92),
        _row(3, score=0.99, dna=0.10, family=0.2, texture=0.1, dino=0.9, clip=0.99),
    ]
    ids_before = [r.file_id for r in rows]
    scores_before = [r.score for r in rows]
    keys_before = [display_sort_tuple(r) for r in rows]
    stamp_display_match_percent(rows)
    pcts = [display_match_percent(r) for r in rows]
    assert ids_before == [r.file_id for r in rows]
    assert scores_before == [r.score for r in rows]
    assert [r.score_percent for r in rows] == [20.0, 80.0, 99.0]
    assert keys_before == [display_sort_tuple(r) for r in rows]
    assert pcts[0] >= pcts[1] >= pcts[2]
    assert 70 <= pcts[2] <= pcts[0] <= 99
    assert pcts != [r.score_percent for r in rows]


def test_stamping_does_not_change_engine_scores_or_sort_keys():
    rows = [
        _row(10, score=0.41, dna=0.9, clip=0.1),
        _row(11, score=0.88, dna=0.2, clip=0.99),
    ]
    score0 = rows[0].score
    stamp_display_match_percent(rows)
    rows[0].debug["display_match_percent"] = 1.0
    assert rows[0].score == score0
    assert rows[1].score == 0.88
    assert SearchEngine_sort_untouched(rows)


def SearchEngine_sort_untouched(rows):
    from core.search_engine import SearchEngine

    k0 = SearchEngine._engine_sort_key(rows[0])
    rows[0].debug["display_match_percent"] = 50.0
    assert SearchEngine._engine_sort_key(rows[0]) == k0
    return True


def test_stamp_preserves_file_id_order():
    rows = [
        _row(7, score=0.3, dna=0.99, clip=0.01),
        _row(3, score=0.9, dna=0.4, clip=0.95),
        _row(9, score=0.1, dna=0.2, clip=0.5),
    ]
    stamp_display_match_percent(rows)
    assert [r.file_id for r in rows] == [7, 3, 9]


def test_merge_keeps_engine_file_order_not_percent():
    full = [
        _row(1, score=0.2, dna=0.9, clip=0.1),
        _row(2, score=0.9, dna=0.2, clip=0.99),
        _row(3, score=0.5, dna=0.5, clip=0.5),
    ]
    incoming = [full[1], full[0]]
    merged = merge_results_in_engine_order(full, incoming)
    assert [r.file_id for r in merged] == [2, 1, 3]


def test_visual_pattern_priority_order_stays_dna_first_after_stamp():
    low_clip = _row(1, score=0.22, dna=0.92, family=0.88, clip=0.12)
    high_clip = _row(2, score=0.91, dna=0.21, family=0.20, clip=0.98)
    ranked = apply_visual_pattern_priority([high_clip, low_clip], "leopar")
    assert ranked[0].file_id == 1
    stamp_display_match_percent(ranked)
    assert [r.file_id for r in ranked] == [1, 2]
    pcts = [display_match_percent(r) for r in ranked]
    assert pcts[0] >= pcts[1]
    assert ranked[0].score_percent < ranked[1].score_percent
