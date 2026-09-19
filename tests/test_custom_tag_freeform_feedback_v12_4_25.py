from core.db import Database
from core.feedback_learn import apply_wrong_match_correction
from core.user_feedback import UserFeedbackStore
from core.concept_registry import find
from core.settings import AppSettings

def test_freeform_tag_does_not_become_category_and_is_learned(tmp_path, monkeypatch):
    db_path = str(tmp_path / 'patterns.db')
    Database(db_path)
    settings = AppSettings(db_path=db_path)
    monkeypatch.setattr('core.feedback_learn._persist_tier_fields', lambda *a, **k: None)
    stats = apply_wrong_match_correction(settings, query_path='leopard', result_file_id=101, tag='fiori', propagate_exact=False)
    assert stats['category_path'] == ''
    store = UserFeedbackStore(Database(db_path))
    assert store.custom_tags_for_file(101) == ['fiori']
    learned = find(db_path, 'fiori')
    assert learned and learned[0]['canonical'] == 'fiori'
    assert learned[0]['concept_type'] == 'custom_tag'

def test_freeform_tag_keeps_category_separate(tmp_path, monkeypatch):
    db_path = str(tmp_path / 'patterns.db')
    settings = AppSettings(db_path=db_path)
    monkeypatch.setattr('core.feedback_learn._persist_tier_fields', lambda *a, **k: None)
    stats = apply_wrong_match_correction(settings, query_path='leopard', result_file_id=102, tag='fiori', propagate_exact=False)
    assert stats['category_path'] == ''

def test_ui_says_new_tag_not_category():
    from pathlib import Path
    src = Path(__file__).parents[1] / 'ui' / 'feedback_family_dialog.py'
    text = src.read_text(encoding='utf-8')
    assert 'Yeni etiket: {tag}' in text
    assert 'Yeni kategori: {tag}' not in text

def test_freeform_tag_is_part_of_text_retrieval_blob():
    from core.text_index import build_text_search_blob
    blob = build_text_search_blob(filename='unnamed.jpg', feedback_labels=['fiori'])
    assert 'fiori' in blob


def test_freeform_tag_is_retrieved_immediately_without_reindex(tmp_path, monkeypatch):
    db_path = str(tmp_path / 'patterns.db')
    db = Database(db_path)
    settings = AppSettings(db_path=db_path)
    monkeypatch.setattr('core.feedback_learn._persist_tier_fields', lambda *a, **k: None)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO files(path,filename,status) VALUES(?,?,?)",
            ('/accept/fiori.jpg','fiori.jpg','indexed')
        )
        fid = int(conn.execute('SELECT last_insert_rowid()').fetchone()[0])
        conn.commit()
    apply_wrong_match_correction(settings, query_path='leopard', result_file_id=fid, tag='fiori', propagate_exact=False)
    hits = db._label_text_search(['fiori'], 0)
    assert fid in hits
    assert hits[fid].get('_learned_custom_tag') is True


def test_custom_tag_never_becomes_category_even_through_category_helper(tmp_path):
    from core.category_learning import apply_manual_category
    db_path = str(tmp_path / 'patterns.db')
    settings = AppSettings(db_path=db_path)
    stats = apply_manual_category(settings, 501, '', custom_tag='fiori', propagate_exact=False, refresh_visual=False)
    assert stats['updated'] == 0
    with Database(db_path).connect() as conn:
        row = conn.execute('SELECT category_path, manual_category_path FROM files WHERE id=501').fetchone()
    assert row is None
