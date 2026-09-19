"""Text → visual retrieval quality. Fixtures only; no index/DB writes."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from core.pattern_intelligence_v2 import (
    DISCOVERY_RESULT_CAP,
    LEAF_RESULT_CAP,
    discovery_result_cap,
    parse_pattern_query_v2,
    rank_v2_results,
)
from core.unified_relevance_ranker import (
    apply_unified_text_ranking,
    build_query_profile,
    compute_unified_relevance,
)
from core.universal_visual_intel import (
    apply_universal_ranking,
    is_textile_discovery_query,
    parse_universal_query,
)


def _row(fid, score, breakdown=None, debug=None, filename=""):
    from core.search_engine import SearchResult

    return SearchResult(
        file_id=fid,
        path=f"/x/{fid}.jpg",
        filename=filename or f"{fid}.jpg",
        customer="",
        thumbnail_path="",
        score=score,
        score_percent=score * 100,
        breakdown=dict(breakdown or {}),
        debug=dict(debug or {}),
    )


def test_text_query_uses_clip_visual_candidate_union():
    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine.search_by_text)
    assert "if clip_by_id and not brand_search_mode:" in src
    assert "_visual_candidate_ids(candidates, clip_by_id)" in src
    assert "v2q.composition == \"composite\" and clip_by_id" not in src


def test_leopar_is_textile_discovery_not_24_leaf_cap():
    v2q = parse_pattern_query_v2("leopar")
    assert v2q.primary == "leopard"
    assert discovery_result_cap(v2q) == DISCOVERY_RESULT_CAP
    assert discovery_result_cap(v2q) > LEAF_RESULT_CAP

    uq = parse_universal_query("leopar")
    assert is_textile_discovery_query(uq)
    rows = [
        SimpleNamespace(
            file_id=i,
            score=0.5,
            filename=f"{i}.jpg",
            path="",
            debug={"clip_score": 0.40, "visual_win": True, "visual_grade": "visual_strong"},
        )
        for i in range(1, 40)
    ]
    clip = {i: {"leopard": 0.40} for i in range(1, 40)}
    kept, _ = apply_universal_ranking(uq, rows, clip_by_id=clip, clip_active=True)
    assert len(kept) > 24


def test_text_and_image_share_unified_fusion_function():
    visual = _row(1, 0.80, {"clip": 0.80}, {"clip_score": 0.80})
    text_score, text_comp = compute_unified_relevance(visual, mode="text")
    image_score, image_comp = compute_unified_relevance(visual, mode="image")
    assert "visual_similarity" in text_comp and "keyword_match" in text_comp
    assert "final_score" in text_comp
    assert text_comp.keys() == image_comp.keys()
    assert text_score > 0 and image_score > 0


def test_filename_only_can_rank_low_when_visual_evidence_exists():
    weak = _row(1, 0.92, {"filename_score": 0.92}, {"clip_score": 0.12})
    strong = _row(2, 0.50, {"filename_score": 0.0}, {"clip_score": 0.96, "visual_grade": "visual_strong"})
    out = apply_unified_text_ranking([weak, strong], visual_retrieval=True)
    assert out[0].file_id == 2
    assert out[1].file_id == 1
    assert out[1].score <= 0.58


def test_visual_match_can_beat_filename_match():
    fname = _row(10, 0.90, {"filename_score": 0.90}, {"clip_score": 0.20})
    visual = _row(11, 0.55, {"family_score": 0.40}, {"clip_score": 0.91, "visual_grade": "visual_exact"})
    out = apply_unified_text_ranking([fname, visual], visual_retrieval=True)
    assert out[0].file_id == 11


def test_image_search_path_unchanged():
    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine)
    inner = inspect.getsource(SearchEngine._execute_search_inner)
    assert "textile_rerank_v2" in src
    assert "search_dino" in inspect.getsource(SearchEngine._adaptive_expand_scored) or "search_dino" in inner
    img_branch = inner.split("if query.mode == \"text\"")[0]
    assert "apply_unified_text_ranking" not in img_branch


def test_exact_search_path_unchanged():
    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine)
    inner = inspect.getsource(SearchEngine._execute_search_inner)
    assert "collect_protected_exact_ids" in inner
    assert "is_protected_exact_match" in src
    assert "protected_score_floor" in src


def test_pattern_family_path_unchanged():
    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine.search_by_text)
    assert "_apply_family_overrides" in src
    assert "query_family_hints" in src


def test_faiss_search_does_not_mutate_indexes():
    from core.search_engine import SearchEngine
    from core.faiss_store import FaissStore

    text_src = inspect.getsource(SearchEngine.search_by_text)
    clip_src = inspect.getsource(SearchEngine._clip_text_hits)
    store_src = inspect.getsource(FaissStore.search_clip)
    assert "search_clip" in clip_src
    assert "write_index" not in text_src
    assert "add_with_ids" not in clip_src
    assert "write_index" not in store_src


def test_db_cache_not_written_on_text_search():
    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine.search_by_text)
    assert "DELETE FROM" not in src
    assert "DROP TABLE" not in src
    assert "rebuild" not in src.lower()


def test_v2_object_leaf_cap_still_applies():
    q = parse_pattern_query_v2("kuş")
    assert discovery_result_cap(q) == LEAF_RESULT_CAP


def test_query_profile_is_general_not_keyword_special_case():
    src = inspect.getsource(build_query_profile)
    assert "leopar" not in src
    p = build_query_profile(mode="text", visual_retrieval=True)
    assert p.keyword_is_weak is True


def test_rank_v2_keeps_visual_leopard_pool():
    q = parse_pattern_query_v2("leopar")
    rows = []
    clip = {}
    for i in range(1, 50):
        rows.append(
            SimpleNamespace(
                file_id=i,
                score=0.4,
                filename=f"print_{i}.jpg",
                pattern_family="animal_print",
                debug={"clip_score": 0.35, "semantic_evidence": {}},
            )
        )
        clip[i] = {"leopard": 0.35}
    kept, stats = rank_v2_results(q, rows, clip_by_id=clip)
    assert stats["kept"] > 24
    assert len(kept) > 24


def test_unified_ranker_wired_into_production_text_search():
    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine.search_by_text)
    assert "apply_unified_text_ranking" in src
    assert "unified_relevance_ranking" in src
    assert "from core.unified_relevance_ranker import apply_unified_text_ranking" in src


def test_weak_clip_confusion_cannot_display_as_98():
    from core.semantic_pattern_intel import (
        calibrate_text_clip,
        collect_evidence,
        gate_result,
        parse_pattern_intent,
    )

    intent = parse_pattern_intent("leopard")
    rec = {
        "id": 1,
        "filename": "geo_baklava.jpg",
        "path": "/x/geo_baklava.jpg",
        "ocr_text": "",
        "pattern_family": "geometric",
        "pattern_subtype": "",
        "texture_map": {},
        "feedback_labels": [],
    }
    ev = collect_evidence(
        intent,
        rec,
        clip_score=0.25,
        rival_clip={
            "leopard": 0.25,
            "geometric": 0.32,
            "paisley": 0.28,
            "zebra": 0.18,
        },
    )
    kept, _tier, score = gate_result(intent, ev, text_score=0.92)
    assert ev.visual_win is False
    assert score < 0.70
    geo = _row(
        1,
        0.92,
        {"filename_score": 0.10, "family_score": 0.80},
        {"clip_score": 0.25, "visual_win": False, "visual_grade": ""},
        filename="geo_baklava.jpg",
    )
    real = _row(
        2,
        0.55,
        {"filename_score": 0.0},
        {
            "clip_score": 0.34,
            "visual_win": True,
            "visual_grade": "visual_strong",
        },
        filename="desen_123.jpg",
    )
    out = apply_unified_text_ranking([geo, real], visual_retrieval=True)
    assert out[0].file_id == 2
    assert out[1].score < 0.70
    assert calibrate_text_clip(0.25) < 0.70
    assert calibrate_text_clip(0.34) > calibrate_text_clip(0.25)


def test_text_clip_calibration_is_monotonic_not_percent_stretch():
    from core.semantic_pattern_intel import calibrate_text_clip

    xs = [0.18, 0.22, 0.25, 0.30, 0.33, 0.38, 0.42, 0.80]
    ys = [calibrate_text_clip(x) for x in xs]
    assert ys == sorted(ys)
    # Typical text–image cosine must not display as 75–99%.
    assert calibrate_text_clip(0.33) < 0.70
    assert calibrate_text_clip(0.38) < 0.78
    assert calibrate_text_clip(0.38) > calibrate_text_clip(0.30)
    # Image-like cosine stays high so true visual hits are not crushed.
    assert calibrate_text_clip(0.80) >= 0.88
    assert calibrate_text_clip(0.25) < 0.45


def test_low_confidence_clip_cannot_display_as_99():
    from core.confidence_engine import LEVEL_LOW, compute_confidence

    geo = _row(
        1,
        0.38,
        {"family_score": 0.20},
        {
            "clip_score": 0.38,
            "visual_win": False,
            "visual_grade": "",
            "semantic_intent": {"family": "animal_print", "motif": "leopard"},
        },
        filename="geo_lattice.jpg",
    )
    geo.pattern_family = "geometric"
    out = apply_unified_text_ranking([geo], visual_retrieval=True)
    assert out[0].score < 0.70
    assert out[0].score_percent < 70
    conf = compute_confidence(out[0], mode="text")
    if conf.level == LEVEL_LOW:
        assert out[0].score_percent < 90


def test_unknown_family_cannot_outrank_known_from_clip_alone():
    intent = {"family": "animal_print", "motif": "leopard"}
    unknown = _row(
        1,
        0.40,
        {},
        {
            "clip_score": 0.37,
            "visual_win": False,
            "semantic_intent": intent,
        },
    )
    unknown.pattern_family = "unknown"
    known = _row(
        2,
        0.34,
        {"family_score": 0.80, "dna_score": 0.70},
        {
            "clip_score": 0.34,
            "visual_win": True,
            "visual_grade": "visual_strong",
            "semantic_intent": intent,
        },
    )
    known.pattern_family = "animal_print"
    strong_unknown = _row(
        3,
        0.41,
        {"dna_score": 0.72, "texture_score": 0.60},
        {
            "clip_score": 0.36,
            "visual_win": True,
            "visual_grade": "visual_strong",
            "semantic_intent": intent,
        },
    )
    strong_unknown.pattern_family = "unknown"
    out = apply_unified_text_ranking(
        [unknown, known, strong_unknown], visual_retrieval=True
    )
    assert out[0].file_id in (2, 3)
    assert unknown.file_id != out[0].file_id or out[0].score < known.score


def test_filename_cannot_create_strong_visual_match_alone():
    intent = {"family": "animal_print", "motif": "leopard"}
    named = _row(
        1,
        0.95,
        {"filename_score": 0.95},
        {"clip_score": 0.14, "semantic_intent": intent},
        filename="leopar_archive.jpg",
    )
    visual = _row(
        2,
        0.40,
        {"family_score": 0.70},
        {
            "clip_score": 0.35,
            "visual_win": True,
            "visual_grade": "visual_medium",
            "semantic_intent": intent,
        },
        filename="scan_8821.jpg",
    )
    visual.pattern_family = "animal_print"
    out = apply_unified_text_ranking([named, visual], visual_retrieval=True)
    assert out[0].file_id == 2
    assert out[1].score <= 0.58


def test_query_normalization_shares_profile_not_results():
    from core.semantic_pattern_intel import parse_pattern_intent

    leo = [parse_pattern_intent(q) for q in ("leopar", "leopard", "leopard print")]
    assert {i.concept_id for i in leo} == {"leopard"}
    assert {i.family for i in leo} == {"animal_print"}
    assert {i.specificity for i in leo} == {"leaf"}
    parent = parse_pattern_intent("animal print")
    assert parent.family == "animal_print"
    assert parent.specificity == "parent"
    assert parent.concept_id != "leopard"
    floral = parse_pattern_intent("flower")
    rose = parse_pattern_intent("rose")
    assert floral.family == "floral" and rose.family == "floral"
    assert floral.specificity == "parent" and rose.specificity == "leaf"


def test_image_mode_rank_order_unchanged_by_text_calibration():
    rows = [
        _row(1, 0.91, {"clip": 0.91, "semantic_score": 0.10}, {"clip_score": 0.91}),
        _row(2, 0.80, {"clip": 0.80, "semantic_score": 0.10}, {"clip_score": 0.80}),
        _row(3, 0.70, {"clip": 0.70, "semantic_score": 0.10}, {"clip_score": 0.70}),
    ]
    image_scores = [compute_unified_relevance(r, mode="image")[0] for r in rows]
    assert image_scores[0] > image_scores[1] > image_scores[2]
    # Image formula stays base-dominant (0.72 / 0.18 / 0.10).
    expected = [
        0.72 * 0.91 + 0.18 * 0.91 + 0.10 * 0.10,
        0.72 * 0.80 + 0.18 * 0.80 + 0.10 * 0.10,
        0.72 * 0.70 + 0.18 * 0.70 + 0.10 * 0.10,
    ]
    for got, exp in zip(image_scores, expected):
        assert abs(got - exp) < 1e-6
    text_scores = [compute_unified_relevance(r, mode="text")[0] for r in rows]
    assert text_scores != image_scores


def _ndcg(rels, k=20):
    import math

    rels = list(rels[:k])
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def _fixture_pool_for_query(query: str) -> list:
    from core.semantic_pattern_intel import parse_pattern_intent

    intent = parse_pattern_intent(query)
    intent_d = {
        "family": intent.family,
        "motif": intent.motif,
        "concept_id": intent.concept_id,
        "specificity": intent.specificity,
    }
    rows = []
    fid = 1

    def add(family, clip, win, extra=None, fname="", grade=""):
        nonlocal fid
        bd = dict(extra or {})
        if family and family == intent.family:
            bd.setdefault("family_score", 0.75)
        dbg = {
            "clip_score": clip,
            "visual_win": win,
            "visual_grade": grade,
            "semantic_intent": intent_d,
            "query_family": intent.family,
        }
        row = _row(fid, clip, bd, dbg, filename=fname or f"{family}_{fid}.jpg")
        row.pattern_family = family
        rows.append(row)
        fid += 1

    qfam = intent.family
    qmot = intent.motif or intent.concept_id
    # Relevant
    for i in range(8):
        add(
            qfam or "unknown",
            0.33 + i * 0.008,
            True,
            {"dna_score": 0.55 + i * 0.02, "texture_score": 0.50},
            grade="visual_strong",
        )
    # Confusion families (CLIP cosine in the misleading 0.33–0.38 band)
    for fam in ("geometric", "paisley", "camouflage", "marble_abstract", "floral", "abstract"):
        if fam == qfam:
            continue
        add(fam, 0.34 + (fid % 4) * 0.01, False, {"family_score": 0.80})
    # Unknown, CLIP-only
    add("unknown", 0.37, False, fname="scan_unknown.jpg")
    # Unknown with real extra visual
    add(
        "unknown",
        0.36,
        True,
        {"dna_score": 0.68, "texture_score": 0.55},
        grade="visual_strong",
    )
    return rows, intent


def test_fixture_ranking_metrics_and_confusion():
    from core.confidence_engine import LEVEL_LOW, compute_confidence
    from core.semantic_pattern_intel import parse_pattern_intent

    queries = [
        "leopard",
        "leopar",
        "animal print",
        "flower",
        "floral",
        "rose",
        "leaf",
        "geometric",
        "paisley",
        "camouflage",
        "marble",
        "abstract",
        "zebra",
        "snake print",
    ]
    report = {}
    for q in queries:
        rows, intent = _fixture_pool_for_query(q)
        ranked = apply_unified_text_ranking(rows, visual_retrieval=True)
        rels = []
        for r in ranked:
            fam = r.pattern_family or ""
            win = bool((r.debug or {}).get("visual_win"))
            if fam and fam == intent.family and win:
                rels.append(3)
            elif fam and fam == intent.family:
                rels.append(2)
            elif fam in ("", "unknown") and win:
                rels.append(1)
            else:
                rels.append(0)
        p10 = sum(1 for x in rels[:10] if x >= 2) / 10
        p20 = sum(1 for x in rels[:20] if x >= 2) / min(20, len(rels))
        ndcg20 = _ndcg(rels, 20)
        wrong = sum(
            1
            for r, rel in zip(ranked[:10], rels[:10])
            if rel == 0 and r.pattern_family not in ("", "unknown")
        )
        scores = [r.score for r in ranked[:10]]
        confs = [compute_confidence(r, mode="text") for r in ranked[:10]]
        low_high_pct = sum(
            1 for r, c in zip(ranked[:10], confs) if c.level == LEVEL_LOW and r.score_percent >= 90
        )
        report[q] = {
            "p10": p10,
            "p20": p20,
            "ndcg20": ndcg20,
            "wrong_family_top10": wrong,
            "score_max": max(scores),
            "score_min": min(scores),
            "low_conf_as_99": low_high_pct,
        }
        assert p10 >= 0.6, (q, report[q])
        assert ndcg20 >= 0.7, (q, report[q])
        assert wrong <= 3, (q, report[q])
        assert max(scores) < 0.95, (q, report[q])
        assert low_high_pct == 0, (q, report[q])
    # butterfly stays an object-leaf query; profile must not collapse into floral.
    from core.pattern_intelligence_v2 import parse_pattern_query_v2

    b = parse_pattern_query_v2("butterfly")
    assert b.primary == "butterfly"
    assert parse_pattern_intent("butterfly").family != "floral"
