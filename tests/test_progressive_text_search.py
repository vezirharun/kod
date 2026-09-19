import inspect

from core.search_engine import SearchEngine


def test_search_by_text_accepts_progressive_callback():
    sig = inspect.signature(SearchEngine.search_by_text)
    assert "result_callback" in sig.parameters


def test_execute_search_accepts_progressive_callback():
    sig = inspect.signature(SearchEngine.execute_search)
    assert "result_callback" in sig.parameters
