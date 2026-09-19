from core.query_evidence import build_search_plan, apply_query_evidence, CandidateEvidenceReport, ConceptEvidence, STATE_SUPPORTED, STATE_UNKNOWN
from core.natural_language_query import parse_natural_query


def test_multi_object_query_builds_and_group():
    parsed = parse_natural_query("gül leopard")
    plan = build_search_plan("gül leopard", parsed=parsed)
    assert any(len(g) >= 2 for g in plan.and_groups)


def test_multi_object_query_requires_all_required_objects():
    parsed = parse_natural_query("gül leopard")
    plan = build_search_plan("gül leopard", parsed=parsed)
    group = next(g for g in plan.and_groups if len(g) >= 2)
    report = CandidateEvidenceReport()
    for cid in group:
        report.concepts[cid] = ConceptEvidence(cid, "object", state=STATE_UNKNOWN)
    # yalnızca ilk kavram destekleniyor -> AND geçmemeli
    report.concepts[group[0]].state = STATE_SUPPORTED
    final, rep = apply_query_evidence(0.9, plan, report)
    assert rep.composite != "PASS"
    assert final <= 0.48


def test_multi_object_query_passes_when_all_supported():
    parsed = parse_natural_query("gül leopard")
    plan = build_search_plan("gül leopard", parsed=parsed)
    group = next(g for g in plan.and_groups if len(g) >= 2)
    report = CandidateEvidenceReport()
    for cid in group:
        report.concepts[cid] = ConceptEvidence(cid, "object", state=STATE_SUPPORTED)
    final, rep = apply_query_evidence(0.8, plan, report)
    assert rep.composite == "PASS"
    assert final > 0.8


def test_composite_visual_evidence_can_open_and_gate():
    from core.query_evidence import collect_candidate_evidence
    parsed = parse_natural_query("gül leopard")
    plan = build_search_plan("gül leopard", parsed=parsed)
    rec = {"filename": "DOLCE CICEKLI LEOPAR.tif", "path": "x/DOLCE CICEKLI LEOPAR.tif"}
    report = collect_candidate_evidence(plan, rec, texture_map={}, clip_scores={"_composite": 0.30})
    final, rep = apply_query_evidence(0.60, plan, report)
    assert rep.composite != "PASS"
    assert final <= 0.60


def test_composite_family_bridge_supports_flower_family_without_claiming_exact_species():
    from core.query_evidence import collect_candidate_evidence
    parsed = parse_natural_query("gül leopard")
    plan = build_search_plan("gül leopard", parsed=parsed)
    rec = {
        "filename": "DOLCE CICEKLI LEOPAR.tif",
        "path": "x/DOLCE CICEKLI LEOPAR.tif",
        "pattern_family": "animal_print",
    }
    texture = {
        "pattern_family": "floral",
        "semantic_tags": {"animal": ["leopard"]},
    }
    report = collect_candidate_evidence(
        plan,
        rec,
        texture_map=texture,
        clip_scores={"leopard": 0.80, "floral": 0.80, "_composite": 0.75},
    )
    rose = report.concepts["rose"]
    assert rose.state == STATE_UNKNOWN
    assert rose.channels["family_bridge"] >= 0.30
    final, rep = apply_query_evidence(0.72, plan, report)
    assert rep.composite == "PARTIAL_PRIMARY"
    assert final > 0.60


def test_generic_daisy_leopard_builds_same_composite_plan():
    parsed = parse_natural_query("papatya leopard")
    plan = build_search_plan("papatya leopard", parsed=parsed)
    assert any(set(g) == {"daisy", "leopard"} for g in plan.and_groups)
