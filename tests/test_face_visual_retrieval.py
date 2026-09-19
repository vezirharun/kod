import numpy as np
from core.face_index import FaceIndexStore
from core.face_identity import FaceObservation


def obs(v, idx=1):
    return FaceObservation(f"face_{idx:03d}",(1,2,30,40),np.asarray(v,dtype=np.float32),'UNKNOWN',0.0,'UNKNOWN')


def test_visual_face_retrieval_expands_person_to_all_files(tmp_path):
    store=FaceIndexStore(tmp_path/'face.db', threshold=.60, min_margin=.01)
    store.replace_file_faces(10,'a.jpg',1,10,[obs([1,0,0,0])])
    store.replace_file_faces(11,'b.jpg',1,10,[obs([.99,.02,0,0])])
    store.replace_file_faces(12,'other.jpg',1,10,[obs([0,1,0,0])])
    result=store.search_by_embeddings([np.asarray([.995,.01,0,0],dtype=np.float32)], threshold=.60, limit=50)
    assert 10 in result and 11 in result
    assert 12 not in result
    assert result[10]['person_id']==result[11]['person_id']
