from core.category_tree import resolve_category_query


def test_resolve_category_query_without_db_path_is_safe():
    match = resolve_category_query("leopard")
    assert match.category_path == "Animal Print/Leopard"


def test_resolve_category_query_accepts_db_path():
    match = resolve_category_query("leopard", db_path="")
    assert match.category_path == "Animal Print/Leopard"
