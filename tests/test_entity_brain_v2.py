from core.entity_brain import build_entity_query_plan, score_record


def test_single_entity_plan():
    p = build_entity_query_plan("şahin")
    assert "hawk" in p.required
    assert not p.composite


def test_composite_entity_requires_all():
    p = build_entity_query_plan("yılan ve leopar karışık desen")
    assert "snake" in p.required
    assert "leopard" in p.required
    assert p.composite

    rec = {"texture_map": {"global_object_intelligence": {
        "objects": [{"label": "snake", "confidence": 0.92}],
    }}}
    score, matched, missing = score_record(rec, p)
    assert score == 0.0
    assert "snake" in matched
    assert "leopard" in missing


def test_composite_entity_passes_when_all_present():
    p = build_entity_query_plan("yılan ve leopar")
    rec = {"texture_map": {
        "global_object_intelligence": {
            "objects": [
                {"label": "snake", "confidence": 0.92},
                {"label": "leopard", "confidence": 0.88},
            ]
        }
    }}
    score, matched, missing = score_record(rec, p)
    assert score >= 0.88
    assert set(matched) == {"snake", "leopard"}
    assert not missing
