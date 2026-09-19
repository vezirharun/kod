import numpy as np
from core.face_index import FaceIndexStore
from core.face_identity import FaceObservation


def _o(v, i=1, gender="UNKNOWN"):
    return FaceObservation(f"face_{i}",(0,0,50,50),np.asarray(v,dtype=np.float32),gender,0.9,"VERY_HIGH")


def test_v8_exemplar_recovers_person_when_centroid_drifts(tmp_path):
    s=FaceIndexStore(tmp_path/'face.db', threshold=.60, min_margin=.01)
    # Three different appearances of the same person; centroid is deliberately
    # less similar than one of the stored exemplars.
    s.replace_file_faces(1,'p1.jpg',1,10,[_o([1,0,0,0],1)])
    s.replace_file_faces(2,'p2.jpg',1,10,[_o([.72,.69,0,0],2)])
    s.replace_file_faces(3,'p3.jpg',1,10,[_o([.98,.20,0,0],3)])
    s.replace_file_faces(4,'other.jpg',1,10,[_o([0,1,0,0],4)])
    r=s.search_by_embeddings([np.asarray([1,.03,0,0],dtype=np.float32)], limit=100)
    assert {1,2,3}.issubset(r.keys())
    assert 4 not in r
    assert all(r[i]['face_match_type']=='same_person' for i in (1,2,3))


def test_v8_no_arbitrary_fifty_person_ceiling(tmp_path):
    s=FaceIndexStore(tmp_path/'face.db', threshold=.50, min_margin=.0)
    for i in range(1,61):
        v=np.zeros(64,dtype=np.float32); v[0]=1.0; v[i%63+1]=0.001*i
        s.replace_file_faces(i,f'{i}.jpg',1,10,[_o(v,i)])
    # The old implementation truncated candidate people to 50.
    # V8 must return more than 50 matched people when the gallery really has them.
    r=s.search_by_embeddings([np.r_[1.0,np.zeros(63,dtype=np.float32)]], threshold=.50, limit=200)
    assert len(r) > 50


def test_face_match_layer_is_first_class():
    from core.search_engine import SearchResult, SearchEngine
    def r(fid, face=False, score=.9):
        x=SearchResult(fid,'',str(fid),'','',score,score)
        x.debug={'face_match':True,'face_similarity':.95,'result_layer':'same_person'} if face else {'result_layer':'same_pattern_family','pattern_family_score':.99}
        return x
    ordered=sorted([r(2,False,.99),r(1,True,.80)], key=SearchEngine._engine_sort_key)
    assert ordered[0].file_id==1
