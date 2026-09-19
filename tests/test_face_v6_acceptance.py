
import os, tempfile
from core.face_index import FaceIndexStore

def run():
    db=os.path.join(tempfile.gettempdir(),"face_v6_acceptance.sqlite")
    try: os.remove(db)
    except OSError: pass

    s=FaceIndexStore(db)

    # Same person: two strong, slightly different observations.
    A1=[1.0,0.0,0.0,0.0]
    A2=[0.98,0.20,0.0,0.0]
    B =[0.0,1.0,0.0,0.0]
    C =[0.0,0.0,1.0,0.0]

    s.add_embedding("P_A","A1",A1,"a1.jpg","FEMALE","Person A",1.0)
    s.add_embedding("P_A","A2",A2,"a2.jpg","FEMALE","Person A",1.0)
    s.add_embedding("P_B","B1",B,"b.jpg","FEMALE","Person B",1.0)
    s.add_embedding("P_C","C1",C,"c.jpg","MALE","Person C",1.0)

    # Unfiltered: male is not silently removed.
    all_r=s.search_by_embeddings(A1,top_k=10)
    assert all_r and all(x["person_id"]!="UNKNOWN" for x in all_r)
    assert all_r[0]["person_id"]=="P_A"
    assert all_r[0]["evidence_count"]==2
    assert "identity_score" in all_r[0]
    assert any(x["gender"]=="MALE" for x in all_r)

    # Explicit gender filters.
    female=s.search_by_embeddings(A1,top_k=10,gender_filter="FEMALE")
    male=s.search_by_embeddings(A1,top_k=10,gender_filter="MALE")
    assert female and all(x["gender"]=="FEMALE" for x in female)
    assert male and all(x["gender"]=="MALE" for x in male)

    # No top-100 gate: source-level assertion.
    import inspect
    src=inspect.getsource(FaceIndexStore)
    assert "top_persons" not in src
    assert "[:100]" not in src

    print("FACE_V6_ACCEPTANCE_PASS")
    print("top_person=",all_r[0]["person_id"])
    print("evidence=",all_r[0]["evidence_count"])
    print("category=",all_r[0]["category"])
    print("female=", [x["person_id"] for x in female])
    print("male=", [x["person_id"] for x in male])

if __name__=="__main__":
    run()
