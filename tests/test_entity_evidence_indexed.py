from core.entity_evidence import extract_entity_evidence, entity_evidence_strength


def test_global_object_detection_creates_entity_and_ancestry():
    rec = {
        "texture_map": {
            "global_object_intelligence": {
                "objects": [
                    {"label": "cat", "label_tr": "kedi", "confidence": 0.91}
                ]
            }
        }
    }
    ev = extract_entity_evidence(rec)
    assert {"cat", "mammal", "animal", "entity"} <= ev
    assert entity_evidence_strength(rec, "cat") >= 0.90
    assert entity_evidence_strength(rec, "animal") >= 0.90
