from pathlib import Path

from core.search_engine import SearchResult
from core.search_memory import PersistentSearchMemory
from core.search_models import SearchResponse, SearchStats


def _response():
    r1 = SearchResult(
        file_id=1, path="/tmp/leopard1.jpg", filename="leopard1.jpg",
        customer="", thumbnail_path="", score=0.99, score_percent=99.0,
        pattern_family="animal_print", animal_print_type="leopard",
        debug={"text_mode": True},
    )
    r2 = SearchResult(
        file_id=2, path="/tmp/leopard2.jpg", filename="leopard2.jpg",
        customer="", thumbnail_path="", score=0.91, score_percent=91.0,
        pattern_family="animal_print", animal_print_type="leopard",
    )
    return SearchResponse(
        all_results=[r1, r2],
        results=[r1, r2],
        stats=SearchStats(displayed=2, text_query="leopard"),
        meta={"mode": "text", "text": "leopard"},
    )


def test_persistent_search_memory_survives_new_instance(tmp_path):
    db = tmp_path / "patterns.db"
    mem = PersistentSearchMemory(str(db), max_entries=32, max_results=20)
    key = ("", (), "text", "leopard", "style", "ignore", "", 0.6, "", "all", (), True, True, True)
    mem.put(key, _response())

    mem2 = PersistentSearchMemory(str(db), max_entries=32, max_results=20)
    got = mem2.get(key)
    assert got is not None
    assert [r.filename for r in got.results] == ["leopard1.jpg", "leopard2.jpg"]
    assert got.meta["search_memory_hit"] is True


def test_persistent_search_memory_caps_results_but_keeps_order(tmp_path):
    db = tmp_path / "patterns.db"
    mem = PersistentSearchMemory(str(db), max_entries=32, max_results=1)
    mem.put(("text", "x"), _response())
    got = mem.get(("text", "x"))
    assert got is not None
    assert [r.filename for r in got.all_results] == ["leopard1.jpg"]
