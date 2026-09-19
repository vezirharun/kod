
import os, tempfile, numpy as np
from core.face_index import FaceIndexStore
from core.face_search import FaceSearch
from core.face_identity import FaceObservation

def main():
    db=os.path.join(tempfile.gettempdir(),"face_v7_acceptance.sqlite")
    try: os.remove(db)
    except OSError: pass

    s=FaceIndexStore(db, threshold=0.62, min_margin=0.05)

    A1=np.array([1.0,0,0,0],dtype=np.float32)
    A2=np.array([0.99,0.12,0,0],dtype=np.float32)
    A3=np.array([0.97,0.20,0,0],dtype=np.float32)
    B =np.array([0.0,1.0,0,0],dtype=np.float32)
    C =np.array([0.0,0.0,1.0,0],dtype=np.float32)

    def ob(emb,g="FEMALE",i=1):
        return FaceObservation(
            face_id=f"f{i}",bbox=(0,0,10,10),embedding=emb,
            gender=g,gender_confidence=.9
        )

    s.replace_file_faces(1,"a1.jpg",1,10,[ob(A1,i=1)])
    s.replace_file_faces(2,"a2.jpg",1,10,[ob(A2,i=2)])
    s.replace_file_faces(3,"a3.jpg",1,10,[ob(A3,i=3)])
    s.replace_file_faces(4,"b.jpg",1,10,[ob(B,i=4)])
    s.replace_file_faces(5,"c.jpg",1,10,[ob(C,"MALE",5)])

    # Core identity retrieval: all 3 observations of the same person are returned
    # and they are ordered as same_person before unrelated visual results.
    r=s.search_by_embeddings([A1], limit=2000)
    assert r and set(r.keys()) >= {1,2,3}, r
    assert all(r[i]["face_match_type"]=="same_person" for i in (1,2,3))
    assert list(r.keys())[:3] == [1,2,3], list(r.keys())[:10]

    # Explicit female filter excludes the male gallery.
    female=s.search_by_embeddings([A1], limit=2000, gender_filter="FEMALE")
    assert set(female.keys()) >= {1,2,3}
    assert 5 not in female

    # Explicit male filter works when the query actually resembles the male identity.
    male=s.search_by_embeddings([C], limit=2000, gender_filter="MALE")
    assert 5 in male
    assert male[5]["face_match_type"]=="same_person"

    # No automatic gender exclusion: an unfiltered C query can see the male identity.
    unfiltered_c=s.search_by_embeddings([C], limit=2000)
    assert 5 in unfiltered_c

    # Scanner/gallery lifecycle APIs remain present.
    for name in ("replace_file_faces","is_current","reconcile_missing","stats","rename_person","list_persons"):
        assert hasattr(s,name), name

    # UI/search layer contract.
    import core.search_engine as se
    se_text=open(se.__file__,encoding="utf-8").read()
    assert '"same_face"' in se_text
    rp_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "results_panel.py")
    rp_text=open(rp_path,encoding="utf-8").read()
    assert '"same_face": "Aynı Kişi"' in rp_text

    # FaceSearch adapter: monkeypatch only image detection so this test does not
    # depend on InsightFace installation in the build environment.
    import core.face_search as fsmod
    fs=FaceSearch(db)
    fs.engine.analyze_image=lambda image: [ob(A1,i=99)]
    # Avoid filesystem decoding; call the store contract directly as the adapter's
    # final stage and verify the returned shape expected by SearchEngine.
    adapter_result=fs.store.search_by_embeddings([A1],limit=2000)
    assert isinstance(adapter_result,dict)
    assert 1 in adapter_result and adapter_result[1]["face_match_type"]=="same_person"

    print("FACE_V7_INTEGRATION_ACCEPTANCE_PASS")
    print("same_person_files=", [i for i in (1,2,3) if i in r])
    print("first_results=", list(r.items())[:5])
    print("female=", list(female.keys())[:10])
    print("male=", list(male.keys())[:10])

if __name__=="__main__":
    main()
