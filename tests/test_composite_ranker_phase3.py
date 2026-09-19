
from types import SimpleNamespace
from core.composite_ranker import apply_composite_ranking, composite_rank_score


def row(score, *, q=0.0, comp=0.0, visual=0.0, coverage=2, bucket=0, clips=None, contradiction=0, concept_ids=None):
    ids = list(concept_ids or [str(i) for i in range(coverage)])
    concepts = {str(i): {"state": "SUPPORTED"} for i in ids}
    return SimpleNamespace(
        score=score,
        score_percent=score*100,
        debug={
            "query_evidence_report": {
                "query_evidence": q,
                "composite_evidence": comp,
                "visual_composite": visual,
                "contradiction_penalty": contradiction,
                "concepts": concepts,
            },
            "pattern_intel_v2": {"bucket": bucket},
            "v2_clip": clips or {},
        },
    )


def test_complete_composite_beats_leopard_only_high_base():
    complete = row(.78, q=.85, comp=.86, visual=.88, coverage=2, concept_ids=["rose", "leopard"], clips={"rose": .80, "leopard": .82, "_composite": .88})
    leopard_only = row(.95, q=.45, comp=0, visual=0, coverage=1, concept_ids=["leopard"], bucket=1, clips={"leopard": .95})
    ranked, _ = apply_composite_ranking(
        [leopard_only, complete],
        query_is_composite=True,
        required=["rose", "leopard"],
        required_count=2,
    )
    assert ranked[0] is complete


def test_partial_composite_is_penalized():
    partial = row(.90, q=.65, comp=.25, visual=.80, coverage=1, concept_ids=["leopard"], clips={"leopard": .80, "_composite": .80})
    score, features = composite_rank_score(
        partial, query_is_composite=True, required=["rose", "leopard"], required_count=2
    )
    assert features["coverage"] < 1.0
    assert score < .75


def test_single_term_keeps_base_dominant():
    a = row(.90, q=.10, bucket=2)
    b = row(.70, q=.90, bucket=0)
    ranked, _ = apply_composite_ranking([a, b], query_is_composite=False)
    assert ranked[0] is a


def test_ranker_does_not_touch_index_files():
    # Architectural contract is represented by this module having no index/DB
    # imports and only receiving candidate rows.
    import core.composite_ranker as m
    assert not hasattr(m, "sqlite3")


def test_realistic_rose_leopard_concepts_beat_leopard_only():
    complete = row(.76, q=.35, comp=.62, visual=.68, coverage=0, bucket=0, clips={
        "rose": .72, "leopard": .78, "_composite": .74,
    })
    complete.debug["query_evidence_report"]["concepts"] = {
        "rose": {"state": "UNKNOWN", "score": .0},
        "leopard": {"state": "SUPPORTED", "score": .78},
    }
    partial = row(.93, q=.40, comp=0.0, visual=0.0, coverage=0, bucket=0, clips={
        "leopard": .94,
    })
    partial.debug["query_evidence_report"]["concepts"] = {
        "rose": {"state": "UNKNOWN", "score": .0},
        "leopard": {"state": "SUPPORTED", "score": .94},
    }
    ranked, _ = apply_composite_ranking(
        [partial, complete], query_is_composite=True, required=["rose", "leopard"], required_count=2
    )
    assert ranked[0] is complete
    assert ranked[0].debug["composite_ranker"]["coverage"] == 1.0


def test_ranker_does_not_import_index_stack():
    import core.composite_ranker as m
    assert not hasattr(m, "sqlite3")


def test_soft_floral_family_support_helps_rose_leopard_without_claiming_exact_rose():
    complete = row(.76, q=.30, comp=.72, visual=.78, clips={
        "leopard": .80, "floral": .84, "_composite": .82,
    })
    complete.debug["query_evidence_report"]["concepts"] = {
        "rose": {"state": "UNKNOWN", "score": .0},
        "leopard": {"state": "SUPPORTED", "score": .80},
    }
    partial = row(.96, q=.35, comp=0.0, visual=0.0, clips={"leopard": .96})
    partial.debug["query_evidence_report"]["concepts"] = {
        "rose": {"state": "UNKNOWN", "score": .0},
        "leopard": {"state": "SUPPORTED", "score": .96},
    }
    ranked, _ = apply_composite_ranking(
        [partial, complete], query_is_composite=True,
        required=["rose", "leopard"], required_count=2,
    )
    assert ranked[0] is complete
    cr = ranked[0].debug["composite_ranker"]
    assert "rose" in cr["soft_supported"]
    assert "rose" in cr["missing"] or "rose" not in cr["supported"]


def test_composite_ranker_runs_after_final_query_evidence_in_search_engine():
    from pathlib import Path
    src = Path("core/search_engine.py").read_text(encoding="utf-8")
    assert src.count("apply_composite_ranking(") == 1
    assert src.index("apply_universal_ranking") < src.index("apply_composite_ranking")
    assert src.index("query_evidence_applied") < src.index("apply_composite_ranking")


def test_soft_flower_family_bridge_beats_leopard_only():
    complete = row(
        .72, q=.08, comp=0.0, visual=.75, coverage=1,
        concept_ids=["leopard", "rose"],
        clips={"leopard": .80, "floral": .80, "_composite": .75},
    )
    complete.debug["query_evidence_report"]["concepts"] = {
        "leopard": {"state": "SUPPORTED", "score": .90, "channels": {"visual": .80}},
        "rose": {
            "state": "UNKNOWN",
            "score": .35,
            "channels": {"family_bridge": .62},
        },
    }
    partial = row(
        .95, q=.08, coverage=1, concept_ids=["leopard"],
        clips={"leopard": .95},
    )
    partial.debug["query_evidence_report"]["concepts"] = {
        "leopard": {"state": "SUPPORTED", "score": .95},
        "rose": {"state": "UNKNOWN", "score": 0.0},
    }
    ranked, _ = apply_composite_ranking(
        [partial, complete],
        query_is_composite=True,
        required=["leopard", "rose"],
        required_count=2,
    )
    assert ranked[0] is complete


def test_rank_badge_numbering_is_not_limited_to_ranked_mode():
    from pathlib import Path
    src = Path("ui/results_panel.py").read_text(encoding="utf-8")
    block = src[src.index("def _rebuild_virtual_entries"):src.index("def _select_result")]
    assert "rank = 1" in block
    assert 'entries.append(("card", result, rank))' in block


def test_search_engine_merges_v2_required_with_query_plan():
    from pathlib import Path
    src = Path("core/search_engine.py").read_text(encoding="utf-8")
    block_start = src.index("required = list(getattr(v2q, \"required\", [])")
    block = src[block_start:block_start + 1200]
    assert "plan_required" in block
    assert "if cid and cid not in required" in block


def test_rank_numbers_are_inline_only():
    from pathlib import Path
    src = Path("ui/result_card.py").read_text(encoding="utf-8")
    assert 'score_text = f"#{self._rank} · {score_text}"' in src
    assert 'self.rank_lbl = QLabel' not in src


def test_multi_concept_relative_contribution_is_exposed():
    from core.composite_ranker import composite_rank_score
    # Concept evidence is deliberately unequal: leopard 0.8, leaf 0.2.
    r = row(.80, q=.80, comp=.90, visual=.90, coverage=2,
            clips={"_composite": .90})
    r.debug["query_evidence_report"]["concepts"] = {
        "leopard": {"state": "SUPPORTED", "score": .80},
        "yaprak": {"state": "SUPPORTED", "score": .20},
    }
    # Inject channel-level evidence consumed by _concept_signals.
    r.debug["query_evidence_report"]["concepts"]["leopard"]["channels"] = {"visual": .80}
    r.debug["query_evidence_report"]["concepts"]["yaprak"]["channels"] = {"visual": .20}
    score, f = composite_rank_score(
        r, query_is_composite=True, required=["leopard", "yaprak"], required_count=2
    )
    assert abs(f["relative_contribution"]["leopard"] - .8) < .05
    assert abs(f["relative_contribution"]["yaprak"] - .2) < .05
    assert f["coverage"] == 1.0
    assert f["same_composition"] >= .55


def test_multi_concept_missing_one_never_gets_full_coverage():
    from core.composite_ranker import composite_rank_score
    r = row(.99, q=.90, comp=.90, visual=.90, coverage=1,
            clips={"_composite": .90})
    r.debug["query_evidence_report"]["concepts"] = {
        "leopard": {"state": "SUPPORTED", "score": .99, "channels": {"visual": .99}},
        "yaprak": {"state": "UNKNOWN", "score": 0.0, "channels": {}},
    }
    score, f = composite_rank_score(
        r, query_is_composite=True, required=["leopard", "yaprak"], required_count=2
    )
    assert f["supported_count"] == 1
    assert f["coverage"] == .5
    assert score < .80


def test_query_order_does_not_change_required_set():
    from core.composite_ranker import composite_rank_score
    r = row(.8, q=.8, comp=.9, visual=.9, coverage=2, clips={"_composite": .9})
    r.debug["query_evidence_report"]["concepts"] = {
        "leopard": {"state": "SUPPORTED", "channels": {"visual": .8}},
        "gül": {"state": "SUPPORTED", "channels": {"visual": .2}},
    }
    a, fa = composite_rank_score(r, query_is_composite=True,
                                  required=["leopard", "gül"], required_count=2)
    b, fb = composite_rank_score(r, query_is_composite=True,
                                  required=["gül", "leopard"], required_count=2)
    assert abs(a-b) < 1e-9
    assert set(fa["required"]) == set(fb["required"])


def test_multi_pro_v3_exact_weight_contract():
    from core.composite_ranker import composite_rank_score
    r = row(.90, q=.80, comp=.80, visual=.80, coverage=2,
            clips={"_composite": .80})
    r.debug["query_evidence_report"]["concepts"] = {
        "leopard": {"state":"SUPPORTED","score":.8,"channels":{"visual":.8}},
        "yaprak": {"state":"SUPPORTED","score":.2,"channels":{"visual":.2}},
    }
    r.debug.update({
        "multi_scale_patch_score": .70,
        "patch_score": .60,
        "semantic_score": .80,
        "ai_score": .90,
        "visual_score": .85,
        "dna_score": .70,
        "texture_score": .60,
    })
    score, f = composite_rank_score(
        r, query_is_composite=True,
        required=["leopard","yaprak"], required_count=2
    )
    assert f["weights"] == {
        "concept_coverage": .30,
        "same_composition": .25,
        "patch": .15,
        "semantic": .10,
        "visual_embedding": .10,
        "pattern_dna": .05,
        "texture": .05,
    }
    assert f["concept_coverage"] == 1.0
    assert f["same_composition"] == .8
    assert f["patch"] == .7
    assert f["semantic"] == .8
    assert f["visual_embedding"] == .9
    assert f["pattern_dna"] == .7
    assert f["texture"] == .6


def test_multi_pro_v3_missing_concept_is_penalized():
    from core.composite_ranker import composite_rank_score
    r = row(.99, q=.90, comp=.90, visual=.90, coverage=1,
            clips={"_composite": .90})
    r.debug["query_evidence_report"]["concepts"] = {
        "leopard": {"state":"SUPPORTED","score":.99,"channels":{"visual":.99}},
        "yaprak": {"state":"UNKNOWN","score":0.0,"channels":{}},
    }
    r.debug.update({
        "multi_scale_patch_score": .95,
        "semantic_score": .95,
        "ai_score": .99,
        "dna_score": .95,
        "texture_score": .95,
    })
    score, f = composite_rank_score(
        r, query_is_composite=True,
        required=["leopard","yaprak"], required_count=2
    )
    assert f["concept_coverage"] == .5
    assert score < .80


def test_multi_pro_v3_preserves_single_query_path():
    from core.composite_ranker import composite_rank_score
    r = row(.73, q=.99, comp=.99, visual=.99, coverage=1,
            clips={"_composite": .99})
    score, f = composite_rank_score(r, query_is_composite=False)
    assert score == .73
    assert f["query_is_composite"] is False
