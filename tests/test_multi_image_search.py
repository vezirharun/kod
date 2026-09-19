"""Multi query-image collection, merge, and search orchestration."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.multi_image_search import (
    collect_query_images,
    merge_query_batches,
    result_identity,
    run_multi_image_queries,
    summarize_multi_search,
)


def _img(tmp_path, name: str) -> str:
    p = tmp_path / name
    p.write_bytes(b"x")
    return str(p)


def test_single_image_collect(tmp_path):
    a = _img(tmp_path, "a.jpg")
    assert collect_query_images([a]) == [a]


def test_two_images_collect(tmp_path):
    a = _img(tmp_path, "a.jpg")
    b = _img(tmp_path, "b.png")
    assert collect_query_images([a, b]) == [a, b]


def test_multi_select_dedupes_same_real_file(tmp_path):
    a = _img(tmp_path, "a.jpg")
    assert collect_query_images([a, a, str(tmp_path / "a.jpg")]) == [a]


def test_mixed_types_keep_images_only(tmp_path):
    img = _img(tmp_path, "ok.jpg")
    txt = tmp_path / "note.txt"
    txt.write_text("nope")
    missing = str(tmp_path / "gone.jpg")
    assert collect_query_images([txt, missing, img, str(txt)]) == [img]


def test_duplicate_hits_from_two_queries_collapse():
    a = SimpleNamespace(
        file_id=7, path="C:/x/hit.jpg", score=0.6, debug={}, match_explanations=[]
    )
    b = SimpleNamespace(
        file_id=7, path="C:/x/hit.jpg", score=0.9, debug={}, match_explanations=[]
    )
    other = SimpleNamespace(
        file_id=8, path="C:/x/other.jpg", score=0.5, debug={}, match_explanations=[]
    )
    merged = merge_query_batches(
        [("q1.jpg", [a]), ("q2.jpg", [b, other])],
        total=2,
    )
    ids = [result_identity(r) for r in merged]
    assert ids.count("id:7") == 1
    hit = next(r for r in merged if r.file_id == 7)
    assert float(hit.score) == 0.9
    assert hit.debug["query_sources"] == ["q1.jpg", "q2.jpg"]
    assert merged[0].file_id == 7


def test_one_query_keeps_single_search_order():
    exact = SimpleNamespace(
        file_id=1,
        path="e.jpg",
        score=0.99,
        is_self_match=True,
        hierarchy_score=0,
        debug={"result_layer": "same_files", "protected_exact": True, "exact_search_score": 0.99},
        match_explanations=[],
    )
    similar = SimpleNamespace(
        file_id=2,
        path="s.jpg",
        score=0.72,
        is_self_match=False,
        hierarchy_score=0,
        debug={"result_layer": "similar_patterns"},
        match_explanations=[],
    )
    family = SimpleNamespace(
        file_id=3,
        path="f.jpg",
        score=0.60,
        is_self_match=False,
        hierarchy_score=0,
        debug={"result_layer": "same_pattern_family", "pattern_family_score": 0.8},
        match_explanations=[],
    )
    merged = merge_query_batches([("q.jpg", [similar, family, exact])], total=1)
    assert [r.file_id for r in merged] == [1, 3, 2]


def test_two_and_four_queries_keep_each_exact_on_top(tmp_path):
    def exact(fid):
        return SimpleNamespace(
            file_id=fid,
            path=f"{fid}.jpg",
            score=0.99,
            is_self_match=True,
            hierarchy_score=0,
            debug={"result_layer": "same_files", "protected_exact": True},
            match_explanations=[],
        )

    def similar(fid):
        return SimpleNamespace(
            file_id=fid,
            path=f"{fid}.jpg",
            score=0.7,
            is_self_match=False,
            hierarchy_score=0,
            debug={"result_layer": "similar_patterns"},
            match_explanations=[],
        )

    class Engine:
        def execute_search(self, query):
            name = Path(query.image_path).stem
            fid = int(name[1:])
            rows = [exact(fid), similar(fid + 100)]
            return SimpleNamespace(results=rows[:1], all_results=rows)

    paths = [_img(tmp_path, f"q{i}.jpg") for i in (1, 2, 3, 4)]
    two = run_multi_image_queries(Engine(), paths[:2])
    assert [r.file_id for r in two.all_results[:2]] == [1, 2]
    assert {r.file_id for r in two.all_results} == {1, 2, 101, 102}
    four = run_multi_image_queries(Engine(), paths)
    assert [r.file_id for r in four.all_results[:4]] == [1, 2, 3, 4]
    assert four.meta["multi_matched"] == 4
    assert all("query_sources" in (r.debug or {}) for r in four.all_results)


def test_single_query_matches_execute_search_rows(tmp_path):
    path = _img(tmp_path, "only.jpg")
    rows = [
        SimpleNamespace(
            file_id=9,
            path="hit.jpg",
            score=0.99,
            is_self_match=True,
            hierarchy_score=0,
            debug={"result_layer": "same_files", "protected_exact": True},
            match_explanations=[],
        )
    ]

    class Engine:
        def execute_search(self, query):
            assert query.image_path == path
            return SimpleNamespace(results=[], all_results=rows)

    resp = run_multi_image_queries(Engine(), [path])
    assert [r.file_id for r in resp.all_results] == [9]
    assert resp.meta["multi_total"] == 1


def test_summary_text():
    assert (
        summarize_multi_search(processed=40, matched=38, missed=2)
        == "40 görsel işlendi — 38 eşleşti, 2 bulunamadı"
    )


def test_run_skips_missing_and_keeps_hits(tmp_path):
    ok = _img(tmp_path, "ok.jpg")
    miss = str(tmp_path / "nope.jpg")
    bad = tmp_path / "x.txt"
    bad.write_text("z")

    class Engine:
        def execute_search(self, query):
            row = SimpleNamespace(
                file_id=11,
                path="hit.jpg",
                score=0.77,
                debug={},
                match_explanations=[],
            )
            return SimpleNamespace(results=[row], all_results=[row])

    ticks = []
    resp = run_multi_image_queries(
        Engine(),
        [ok, miss, str(bad), ok],
        progress=lambda i, n, p: ticks.append((i, n, p)),
    )
    assert ticks == [(1, 1, ok)]
    assert resp.meta["multi_matched"] == 1
    assert resp.meta["multi_missed"] == 0
    assert resp.meta["multi_processed"] == 1
    assert len(resp.results) == 1
    assert "1 görsel işlendi" in resp.meta["multi_summary"]


def test_run_unreadable_query_continues(tmp_path):
    a = _img(tmp_path, "a.jpg")
    b = _img(tmp_path, "b.jpg")

    class Engine:
        def execute_search(self, query):
            if query.image_path.endswith("a.jpg"):
                raise RuntimeError("decode")
            row = SimpleNamespace(
                file_id=2, path="ok.jpg", score=0.5, debug={}, match_explanations=[]
            )
            return SimpleNamespace(results=[row], all_results=[row])

    resp = run_multi_image_queries(Engine(), [a, b])
    assert resp.meta["multi_matched"] == 1
    assert resp.meta["multi_missed"] == 1
    assert resp.meta["multi_processed"] == 2
    assert "2 görsel işlendi — 1 eşleşti, 1 bulunamadı" == resp.meta["multi_summary"]
