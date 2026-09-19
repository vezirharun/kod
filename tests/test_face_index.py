import numpy as np
from core.face_index import FaceIndexStore
from core.face_identity import FaceObservation


def _obs(vec, gender="UNKNOWN", idx=1):
    return FaceObservation(f"face_{idx:03d}",(1,2,20,30),np.asarray(vec,dtype=np.float32),gender,0.9,"VERY_HIGH")


def test_persistent_gallery_assignment_and_search(tmp_path):
    db=tmp_path/'face.db'
    store=FaceIndexStore(db, threshold=.80, min_margin=.01)
    store.replace_file_faces(1,'a.jpg',1,10,[_obs([1,0,0,0], 'FEMALE')])
    store.replace_file_faces(2,'b.jpg',1,10,[_obs([.99,.01,0,0], 'FEMALE')])
    persons=store.list_persons()
    assert len(persons)==1
    pid=persons[0]['person_id']
    assert store.search(person_id=pid)==[1,2]
    assert store.search(gender='FEMALE')==[1,2]


def test_ambiguous_match_creates_new_person(tmp_path):
    db=tmp_path/'face.db'
    store=FaceIndexStore(db, threshold=.80, min_margin=.10)
    store.replace_file_faces(1,'a.jpg',1,10,[_obs([1,0,0,0], 'MALE')])
    store.replace_file_faces(2,'b.jpg',1,10,[_obs([.99,.01,0,0], 'MALE')])
    assert len(store.list_persons())==1


def test_rename_and_delete_reconciliation(tmp_path):
    db=tmp_path/'face.db'
    store=FaceIndexStore(db)
    store.replace_file_faces(1,'a.jpg',1,10,[_obs([1,0,0,0])])
    pid=store.list_persons()[0]['person_id']
    store.rename_person(pid,'Kişi Test')
    assert store.person(pid)['display_name']=='Kişi Test'
    assert store.reconcile_missing(set())==1
    assert store.person(pid) is not None


def test_known_gender_conflict_does_not_merge_people(tmp_path):
    db = tmp_path / 'face.db'
    store = FaceIndexStore(db, threshold=.80, min_margin=.01)
    store.replace_file_faces(1, 'woman.jpg', 1, 10, [_obs([1,0,0,0], 'FEMALE')])
    store.replace_file_faces(2, 'man.jpg', 1, 10, [_obs([.99,.01,0,0], 'MALE')])
    people = store.list_persons()
    assert len(people) == 2
    assert {p['gender'] for p in people} == {'FEMALE', 'MALE'}


def test_multi_person_filename_does_not_label_every_face(tmp_path):
    db = tmp_path / 'face.db'
    store = FaceIndexStore(db, threshold=.80, min_margin=.01)
    obs = [
        _obs([1,0,0,0], 'FEMALE', idx=1),
        _obs([0,1,0,0], 'MALE', idx=2),
    ]
    store.replace_file_faces(1, 'Hulya Kocyigit and Partner.jpg', 1, 10, obs)
    with store._connect() as con:
        aliases = con.execute('SELECT alias FROM person_aliases').fetchall()
    assert aliases == []
