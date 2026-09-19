import numpy as np

from core.face_identity import FaceObservation
from core.face_index import FaceIndexStore


def obs(v, idx=1):
    return FaceObservation(f"face_{idx}", (0, 0, 100, 100), np.asarray(v, dtype=np.float32), "UNKNOWN", 0.0, "UNKNOWN")


def test_assignment_uses_exemplars_not_only_centroid(tmp_path):
    s = FaceIndexStore(tmp_path / "face.db", threshold=0.56, min_margin=0.02)
    s.replace_file_faces(1, "a.jpg", 1, 10, [obs([1, 0, 0, 0], 1)])
    # This appearance is deliberately far enough from the centroid to expose
    # the old centroid-only assignment, but still very close to a real exemplar.
    s.replace_file_faces(2, "b.jpg", 1, 10, [obs([0.78, 0.62, 0, 0], 2)])
    assert len(s.list_persons()) == 1


def test_split_people_can_be_consolidated(tmp_path):
    s = FaceIndexStore(tmp_path / "face.db", threshold=0.90, min_margin=0.0)
    s.replace_file_faces(1, "a.jpg", 1, 10, [obs([1, 0, 0, 0], 1)])
    s.replace_file_faces(2, "b.jpg", 1, 10, [obs([0.8, 0.6, 0, 0], 2)])
    assert len(s.list_persons()) == 2
    merged = s.consolidate_identities(merge_threshold=0.79, max_pairs=100)
    assert merged == 1
    assert len(s.list_persons()) == 1
    assert set(s.search(person_id=s.list_persons()[0]["person_id"])) == {1, 2}
