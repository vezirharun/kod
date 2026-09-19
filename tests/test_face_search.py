from core.face_search import parse_face_query

def test_gender_queries():
    assert parse_face_query('kadın') == {'kind':'gender','value':'FEMALE'}
    assert parse_face_query('erkek') == {'kind':'gender','value':'MALE'}
    assert parse_face_query('bilinmiyor') == {'kind':'gender','value':'UNKNOWN'}

def test_person_queries():
    assert parse_face_query('Kişi 17') == {'kind':'person','value':'person_0017'}
    assert parse_face_query('person_17') == {'kind':'person','value':'person_0017'}
    assert parse_face_query('gül leopard') is None
    assert parse_face_query('kişi:Ahmet') == {'kind':'name','value':'ahmet'}

# image_matches is intentionally exercised only when cv2/InsightFace is installed;
# the persistent retrieval contract is covered by test_face_visual_retrieval.py.
