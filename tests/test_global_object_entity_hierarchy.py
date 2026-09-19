from core.entity_aliases import resolve_entity_alias
from core.entity_evidence import extract_entity_evidence, entity_evidence_strength
from core.universal_visual_intel import node_path, parse_universal_query


def test_global_entity_hierarchy_aliases():
    assert resolve_entity_alias("araba") == "car"
    assert resolve_entity_alias("TOGG") == "togg"
    assert resolve_entity_alias("şahin") == "hawk"
    assert resolve_entity_alias("martı") == "seagull"
    assert resolve_entity_alias("çocuk") == "child"
    assert resolve_entity_alias("kız") == "girl"
    assert resolve_entity_alias("broş") == "brooch"
    assert resolve_entity_alias("barok desen") == "baroque_pattern"


def test_specific_is_narrower_than_family():
    assert node_path("hawk")[-2:] == ["bird", "hawk"]
    assert node_path("togg")[-2:] == ["car", "togg"]
    assert node_path("t10x")[-3:] == ["car", "togg", "t10x"]
    assert parse_universal_query("şahin").node_id == "hawk"
    assert parse_universal_query("TOGG").node_id == "togg"


def test_persistent_concept_evidence():
    rec = {"texture_map": {"global_object_intelligence": {
        "objects": [{"label": "car", "label_tr": "araba", "confidence": 0.91}],
        "concepts": [{"label": "brooch", "label_tr": "broş", "confidence": 0.82}],
    }}}
    ev = extract_entity_evidence(rec)
    assert "car" in ev and "vehicle" in ev and "entity" in ev
    assert "brooch" in ev and "accessory" in ev
    assert entity_evidence_strength(rec, "car") >= 0.90
