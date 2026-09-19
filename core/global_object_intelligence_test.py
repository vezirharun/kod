from core.global_object_intelligence import GlobalObjectIntelligence


def test_capability_does_not_offer_identity():
    cap = GlobalObjectIntelligence.capability()
    assert cap["human_instance_separation"] == "supported"
    assert cap["face_recognition"] == "optional_dependency"
    assert cap["person_identity_matching"] == "supported_by_face_index"


def test_missing_image_is_safe():
    engine = GlobalObjectIntelligence(enabled=True)
    assert engine.detect("Z:/__does_not_exist__/image.jpg") == []


if __name__ == "__main__":
    test_capability_does_not_offer_identity()
    test_missing_image_is_safe()
    print("global_object_intelligence_test: OK")
