
import sqlite3
from core.category_memory import register_root_category, register_category, resolve_dynamic_category, category_suggestions
from core.category_tree import resolve_category_query, parent_categories, child_categories

def test_new_category_is_persistent_and_searchable(tmp_path):
    db=str(tmp_path/"cat.db")
    # no existing category
    assert not resolve_category_query("sıçramış boya", db_path=db).category_path
    assert register_root_category(db, "Soyut")
    assert register_category(db, "Soyut", "Sıçramış Boya", aliases=["sıçramış boya"])
    assert "Soyut" in parent_categories(db)
    assert "Sıçramış Boya" in child_categories("Soyut", db)
    m=resolve_category_query("sıçramış boya", db_path=db)
    assert m.category_path=="Soyut/Sıçramış Boya"

def test_dynamic_category_suggestions_include_roots(tmp_path):
    db=str(tmp_path/"cat.db")
    assert register_root_category(db, "Yeni Kategori", aliases=["yeni kavram"])
    assert any(p=="Yeni Kategori" for p,_,_ in category_suggestions(db,"yeni kavram"))

def test_ambiguous_static_alias_is_not_auto_selected():
    # "soyut" is intentionally ambiguous/overloaded in the static tree.
    m=resolve_category_query("soyut")
    assert not m.category_path
