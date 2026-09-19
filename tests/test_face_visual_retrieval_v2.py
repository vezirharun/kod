import numpy as np
from core.face_index import FaceIndexStore
from core.face_identity import FaceObservation


def ob(v, gender='UNKNOWN', idx=1):
    return FaceObservation(f'face_{idx:03d}', (1, 2, 30, 40), np.asarray(v, dtype=np.float32), gender, .9 if gender != 'UNKNOWN' else 0.0, 'VERY_HIGH' if gender != 'UNKNOWN' else 'UNKNOWN')


def test_image_search_keeps_other_gender_as_lower_face_result(tmp_path):
    store = FaceIndexStore(tmp_path / 'face.db', threshold=.72, min_margin=.01)
    store.replace_file_faces(1, 'woman.jpg', 1, 10, [ob([1, 0, 0, 0], 'FEMALE')])
    store.replace_file_faces(2, 'man.jpg', 1, 10, [ob([.96, .28, 0, 0], 'MALE')])
    result = store.search_by_embeddings([np.asarray([1, 0, 0, 0], dtype=np.float32)], limit=20)
    assert 1 in result
    assert 2 in result
    assert result[1]['face_match_type'] == 'same_person'
    assert result[2]['person_gender'] == 'MALE'


def test_explicit_gender_filter_still_works(tmp_path):
    store = FaceIndexStore(tmp_path / 'face.db', threshold=.72, min_margin=.01)
    store.replace_file_faces(1, 'woman.jpg', 1, 10, [ob([1, 0, 0, 0], 'FEMALE')])
    store.replace_file_faces(2, 'man.jpg', 1, 10, [ob([.96, .28, 0, 0], 'MALE')])
    result = store.search_by_embeddings(
        [np.asarray([1, 0, 0, 0], dtype=np.float32)],
        gender_filter='FEMALE', limit=20,
    )
    assert 1 in result
    assert 2 not in result


def test_face_match_payload_distinguishes_identity_from_similarity(tmp_path):
    store = FaceIndexStore(tmp_path / 'face.db', threshold=.80, min_margin=.01)
    store.replace_file_faces(1, 'same.jpg', 1, 10, [ob([1, 0, 0, 0])])
    store.replace_file_faces(2, 'similar.jpg', 1, 10, [ob([.78, .62, 0, 0])])
    result = store.search_by_embeddings([np.asarray([1, 0, 0, 0], dtype=np.float32)], limit=20)
    assert result[1]['face_match_type'] == 'same_person'
    assert result[2]['face_match_type'] == 'similar_face'
