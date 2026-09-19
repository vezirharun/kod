import sqlite3

from core.category_memory import (
    register_root_category, register_category, dynamic_parents, dynamic_children,
    resolve_dynamic_category, category_suggestions,
)
from core.category_tree import parent_categories, child_categories, resolve_category_query


def test_dynamic_root_and_child_are_persistent(tmp_path):
    db = str(tmp_path / "cat.db")
    assert register_root_category(db, "Yeni Desen Türü", aliases=["ydt"])
    assert "Yeni Desen Türü" in dynamic_parents(db)
    assert "Yeni Desen Türü" in parent_categories(db)
    assert resolve_category_query("ydt", db_path=db).category_path == "Yeni Desen Türü"

    assert register_category(db, "Yeni Desen Türü", "Çatlak", aliases=["çatlak", "crack"])
    assert "Çatlak" in dynamic_children(db, "Yeni Desen Türü")
    assert "Çatlak" in child_categories("Yeni Desen Türü", db)
    assert resolve_category_query("crack", db_path=db).category_path == "Yeni Desen Türü/Çatlak"
    assert category_suggestions(db, "çatlak")


def test_broad_alias_does_not_force_specific_category():
    # "soyut" özel bir alt kategoriye zorla bağlanmaz; UI adayları ve
    # explicit "Ekle: soyut" akışını gösterir.
    match = resolve_category_query("soyut")
    assert not match.category_path
