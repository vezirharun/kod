from core.face_retrieval_v3 import FaceRetrievalV3, FaceCandidate
from core.known_person_gallery_v3 import KnownPersonGallery

def test_categories():
    assert FaceRetrievalV3.classify(.90,.06)=="SAME_PERSON"
    assert FaceRetrievalV3.classify(.76,.03)=="PROBABLE_SAME_PERSON"
    assert FaceRetrievalV3.classify(.64,.01)=="SIMILAR_FACE"

def test_grouping():
    cs=[
      FaceCandidate("p1","f1",.90,.91,"SAME_PERSON","FEMALE"),
      FaceCandidate("p1","f2",.86,.87,"SAME_PERSON","FEMALE"),
      FaceCandidate("p2","f3",.64,.65,"SIMILAR_FACE","MALE")]
    g=FaceRetrievalV3.group_persons(cs)
    assert g[0]["person_id"]=="p1"
    assert g[0]["evidence_count"]==2
    assert g[1]["person_id"]=="p2"

def test_known_gallery():
    gal=KnownPersonGallery()
    gal.add("Test Person",[1.0,0.0])
    out=gal.identify([1.0,0.0])
    assert out["name"]=="Test Person"
    assert out["status"]=="PROBABLE_IDENTITY"

if __name__=="__main__":
    test_categories(); test_grouping(); test_known_gallery()
    print("3/3 passed")
