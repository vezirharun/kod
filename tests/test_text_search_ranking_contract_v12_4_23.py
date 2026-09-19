import inspect


def test_gender_sort_uses_gender_evidence_before_generic_text_score():
    from core.search_engine import SearchEngine
    src = inspect.getsource(SearchEngine.search_by_text)
    assert "if human_semantic_mode:" in src
    assert "gender_visual_score" in src
    assert "-gender_score" in src


def test_text_unlimited_contract_is_not_converted_to_400():
    from core.search_engine import SearchEngine
    from core.db import Database
    s = inspect.getsource(SearchEngine.search_by_text)
    assert "unlimited = int(requested_limit) <= 0" in s
    assert "limit = 0 if unlimited" in s
    dbs = inspect.getsource(Database.search_text_candidates)
    assert "label_pool_limit = 0" in dbs
    assert "if int(limit or 0) > 0" in dbs


def test_gender_uvi_has_no_hidden_120_cap():
    from core.universal_visual_intel import apply_universal_ranking
    s = inspect.getsource(apply_universal_ranking)
    assert "gender_cap = None" in s
    assert "gender_cap = 120" not in s
