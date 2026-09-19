from core.face_index import FaceIndexV4

def test_categories():
    assert FaceIndexV4.category(.90,.06)=="SAME_PERSON"
    assert FaceIndexV4.category(.78,.03)=="PROBABLE_SAME_PERSON"
    assert FaceIndexV4.category(.64,.01)=="SIMILAR_FACE"

def test_no_top100_gate_source():
    import inspect
    s=inspect.getsource(FaceIndexV4.search)
    assert "top_persons" not in s
    assert "[:100]" not in s

if __name__=="__main__":
    test_categories()
    test_no_top100_gate_source()
    print("2/2 passed")
