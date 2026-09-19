import numpy as np
from core.face_identity import FaceIdentityEngine, confidence_band


def test_identity_backend_is_optional_and_honest():
    cap = FaceIdentityEngine.capability()
    assert cap["gender_classification_from_face"] in {"supported", "optional_dependency"}
    assert cap["face_recognition"] in {"supported", "optional_dependency"}


def test_confidence_bands():
    assert confidence_band(.80) == "VERY_HIGH"
    assert confidence_band(.65) == "HIGH"
    assert confidence_band(.55) == "MEDIUM"
    assert confidence_band(.45) == "LOW"
    assert confidence_band(.20) == "UNKNOWN"


def test_cluster_same_vectors():
    a = np.ones(8, dtype=np.float32)
    b = a.copy()
    c = -a
    labels = FaceIdentityEngine.cluster({"a": a, "b": b, "c": c}, threshold=.9)
    assert labels["a"] == labels["b"]
    assert labels["c"] != labels["a"]
