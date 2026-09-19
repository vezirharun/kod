import numpy as np

from core.face_identity import FaceIdentityEngine, _cosine
from core.face_index import FaceIndexStore


def test_cosine_dimension_mismatch_is_safe():
    assert _cosine(np.ones(4), np.ones(5)) == 0.0


def test_backend_is_explicit():
    e = FaceIdentityEngine()
    assert e.backend in {"insightface_arcface_v11", "vision_face_crop_v11", "unavailable"}
    if e.backend.startswith("insightface"):
        assert e.effective_threshold == e.requested_threshold
    else:
        assert e.effective_threshold >= 0.66


def test_face_db_engine_isolated_and_migratable(tmp_path):
    db = tmp_path / "face.db"
    s1 = FaceIndexStore(db, identity_engine="engine_a")
    with s1._connect() as con:
        con.execute("INSERT INTO persons(person_id,sample_count,centroid) VALUES(?,?,?)", ("person_0001", 1, np.ones(4, dtype=np.float32).tobytes()))
    s2 = FaceIndexStore(db, identity_engine="engine_b")
    assert s2.stats()["persons"] == 0


def test_face_hidden_cluster_exemption():
    from core.search_display import filter_ranked_visual_results
    class R:
        score = 0.91
        cluster_group = "unrelated"
        category = "style"
        same_pattern_family = False
        same_animal_family = False
        pattern_family = ""
        animal_print_type = ""
        debug = {"face_match": True}
    out = filter_ranked_visual_results([R()], 0.40, hidden_clusters=frozenset({"unrelated"}), normalize_cluster_key=lambda x: x)
    assert len(out) == 1


def test_fallback_embedding_converts_bgr_to_rgb():
    from core.face_identity import FaceIdentityEngine
    class FakeFeatures:
        dino_embedding = np.asarray([1.0, 0.0], dtype=np.float32).tobytes()
        clip_embedding = np.asarray([0.0, 1.0], dtype=np.float32).tobytes()
    class FakeExtractor:
        def __init__(self):
            self.seen = []
        def extract_from_array(self, image, **kwargs):
            self.seen.append(tuple(int(x) for x in image[0,0]))
            return FakeFeatures()
    e = FaceIdentityEngine()
    fake = FakeExtractor()
    e._fallback_extractor = fake
    crop = np.zeros((40,40,3), dtype=np.uint8)
    crop[...,2] = 255  # BGR red
    out = e._fallback_embedding(crop)
    assert out is not None
    assert fake.seen[0] == (255, 0, 0)  # RGB red
    assert len(out) == 12  # 3 views x (2 DINO + 2 CLIP)


def test_filename_alias_search_expands_person_gallery(tmp_path):
    from core.face_identity import FaceObservation
    s = FaceIndexStore(tmp_path / "face.db", threshold=0.60, min_margin=0.01)
    s.replace_file_faces(1, "/archive/hulya-kocyigit.jpg.webp", 1, 10,
                         [FaceObservation("f1",(0,0,20,20),np.asarray([1,0,0,0],dtype=np.float32))])
    s.replace_file_faces(2, "/archive/hulya-kocyigit-portrait.jpg", 1, 10,
                         [FaceObservation("f2",(0,0,20,20),np.asarray([.99,.02,0,0],dtype=np.float32))])
    ids = s.search_by_name("Hülya Koçyiğit", limit=50)
    assert 1 in ids and 2 in ids
