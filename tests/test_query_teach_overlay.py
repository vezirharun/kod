from core.search_memory import (
    _norm_query,
    overlay_adjustments,
    overlay_positive_ids,
    overlay_wrong_ids,
    record_feedback_overlay,
)
from core.user_feedback import teach_query_keys


def test_teach_query_keys_only_from_search_box():
    keys = teach_query_keys(
        "yaprak",
        {"child": "Yaprak", "category_path": "Bitki/Yaprak", "tags": ["leaf"]},
    )
    assert keys == [_norm_query("yaprak")]
    assert "leaf" not in keys
    assert "bitki" not in keys
    fare = teach_query_keys("fare", {"child": "Fare", "category_path": "Hayvan/Fare"})
    assert fare == [_norm_query("fare")]
    assert "mouse" not in fare
    assert teach_query_keys(r"C:\img\a.jpg", {}) == []
    assert teach_query_keys("", {"child": "Çilek"}) == []


def test_cilek_folds_cilek_not_cicek():
    assert _norm_query("çilek") == _norm_query("cilek")
    assert _norm_query("Çilek") == _norm_query("cilek")
    assert _norm_query("çilek") != _norm_query("çiçek")
    assert _norm_query("yaprak") != _norm_query("çiçek")


def test_overlay_upsert_and_query_isolation(tmp_path, monkeypatch):
    db = tmp_path / "patterns.db"
    db.write_text("")
    monkeypatch.setenv("VEZIR_SKIP", "1")
    from core.search_memory import get_search_memory

    path = str(db)
    get_search_memory(path)
    a = record_feedback_overlay(path, "yaprak", 10, "correct")
    b = record_feedback_overlay(path, "yaprak", 10, "correct")
    assert a == b
    assert overlay_positive_ids(path, "yaprak") == {10}
    assert overlay_positive_ids(path, "çilek") == set()
    assert overlay_positive_ids(path, "leaf") == set()
    record_feedback_overlay(path, "çilek", 11, "correct")
    assert overlay_positive_ids(path, "çilek") == {11}
    assert overlay_positive_ids(path, "cilek") == {11}
    assert overlay_positive_ids(path, "çiçek") == set()
    adj = overlay_adjustments(path, "yaprak")
    assert adj[10] > 0


def test_save_metadata_overlay_binds_search_box_not_category(tmp_path, monkeypatch):
    monkeypatch.setenv("VEZIR_SKIP", "1")
    from core.db import Database
    from core.user_feedback import UserFeedbackStore

    db_path = str(tmp_path / "patterns.db")
    store = UserFeedbackStore(Database(db_path))
    payload = {
        "parent": "Bitki",
        "child": "Yaprak",
        "category_path": "Bitki/Yaprak",
        "tags": ["leaf"],
    }
    store.save_metadata_overlay(42, payload, query_path="yaprak")
    assert overlay_positive_ids(db_path, "yaprak") == {42}
    assert overlay_positive_ids(db_path, "leaf") == set()
    assert overlay_positive_ids(db_path, "bitki") == set()
    assert overlay_positive_ids(db_path, "çilek") == set()
    store.save_metadata_overlay(42, payload, query_path="yaprak")
    assert overlay_positive_ids(db_path, "yaprak") == {42}

    store.save_metadata_overlay(
        7,
        {"parent": "Meyve", "child": "Çilek", "category_path": "Meyve/Çilek"},
        query_path="çilek",
    )
    assert overlay_positive_ids(db_path, "çilek") == {7}
    assert overlay_positive_ids(db_path, "cilek") == {7}
    assert overlay_positive_ids(db_path, "çiçek") == set()
    assert overlay_wrong_ids(db_path, "yaprak") == set()
