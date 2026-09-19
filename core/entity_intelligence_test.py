from core.entity_aliases import resolve_entity_alias, expand_entity_terms
from core.entity_evidence import extract_entity_evidence, entity_evidence_strength
from core.search_acceptance import result_passes_threshold


def main():
    assert resolve_entity_alias("kedi") == "cat"
    assert resolve_entity_alias("kuş") == "bird"
    assert resolve_entity_alias("insan") == "person"
    assert "cat" in expand_entity_terms("kedi")

    rec = {"texture_map": {"semantic_tags": {"objects": ["cat"]}}}
    assert "cat" in extract_entity_evidence(rec)
    assert entity_evidence_strength(rec, "cat") >= 0.9

    class R:
        score = 0.05
        debug = {"entity_evidence": True, "entity_evidence_score": 0.94}
    assert result_passes_threshold(R(), 0.60)
    print("entity_intelligence_test: OK")


if __name__ == "__main__":
    main()
