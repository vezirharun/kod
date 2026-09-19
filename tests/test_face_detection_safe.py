from core.face_detection import FaceDetection
from core.global_object_intelligence import GlobalObjectIntelligence
from core.universal_visual_intel import CAPABILITIES


def test_face_capability_is_detection_only():
    cap = FaceDetection.capability()
    assert cap["face_identity_matching"] in {"unsupported", "supported"}
    assert cap["gender_classification_from_face"] == "unsupported"


def test_global_capability_keeps_identity_and_gender_off():
    cap = GlobalObjectIntelligence.capability()
    assert cap["person_identity_matching"] == "supported_by_face_index"
    assert cap["gender_classification_from_image"] in {"unsupported", "optional_dependency"}


def test_uvi_capability_is_honest():
    assert CAPABILITIES["face_detection"] == "supported"
    assert CAPABILITIES["face_counting"] == "supported"
    assert CAPABILITIES["face_recognition"] == "experimental"
    assert CAPABILITIES["gender_classification_from_face"] in {"unsupported", "supported", "experimental"}
