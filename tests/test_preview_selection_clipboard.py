import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.preview_selection import (
    PreviewSelectionStore,
    build_copy_plan,
    build_original_copy_plan,
    copy_originals_to_clipboard,
    copy_previews_to_clipboard,
    original_path_for_result,
    preview_context_menu_spec,
    preview_path_for_result,
)


def _rec(fid, thumb="", path="orig.jpg"):
    return SimpleNamespace(file_id=fid, thumbnail_path=thumb, path=path)


def test_select_count_and_unique():
    s = PreviewSelectionStore()
    s.set_selected(1, True)
    assert s.count() == 1
    s.select_many([2, 3, 4, 5, 1])
    assert s.count() == 5
    s.set_selected(3, True)
    assert s.count() == 5
    s.clear()
    assert s.count() == 0


def test_sequential_right_click_keeps_previous():
    s = PreviewSelectionStore()
    s.toggle(11)
    s.toggle(22)
    s.toggle(33)
    assert s.ids() == {11, 22, 33}
    s.toggle(22)
    assert s.ids() == {11, 33}


def test_context_menu_spec_turkish():
    labels = dict(preview_context_menu_spec(this_selected=False, count=0))
    assert labels["toggle"] == "Deseni seç"
    assert labels["visible"] == "Tüm görünenleri seç"
    assert labels["all"] == "Tüm sonuçları seç"
    assert "clear" not in labels
    assert "copy" not in labels
    assert "count" not in labels
    assert labels["copy_original"] == "Orijinal deseni kopyala"
    spec2 = dict(preview_context_menu_spec(this_selected=True, count=7))
    assert spec2["toggle"] == "Seçimi kaldır"
    assert spec2["copy"] == "Seçilen önizlemeleri kopyala"
    assert spec2["copy_original"] == "Orijinal deseni kopyala"
    assert spec2["edit"] == "Seçilenleri düzenle (7)"
    assert spec2["count"] == "Seçilen: 7"
    for _key, label in spec2.items():
        low = label.lower()
        assert "select" not in low
        assert "clipboard" not in low
        assert "export" not in low


def test_select_visible_vs_all():
    s = PreviewSelectionStore()
    visible = [1, 2, 3]
    all_ids = list(range(1, 114))
    s.select_many(visible)
    assert s.count() == 3
    s.select_many(all_ids)
    assert s.count() == 113
    assert 113 in s.ids()


def test_clear_and_new_search_store():
    s = PreviewSelectionStore()
    s.select_many([1, 2, 3])
    s.clear()
    assert s.count() == 0


def test_retain_keeps_old_when_more_loaded():
    s = PreviewSelectionStore()
    s.select_many([1, 2, 3])
    s.retain(list(range(1, 21)))
    assert s.ids() == {1, 2, 3}


def test_new_search_drops_previous_ids():
    s = PreviewSelectionStore()
    s.select_many([10, 20])
    s.clear()
    s.select_many([99])
    assert s.ids() == {99}


def test_preview_skips_original_file(tmp_path):
    orig = tmp_path / "pattern.tif"
    orig.write_bytes(b"orig")
    prev = tmp_path / "thumb.webp"
    prev.write_bytes(b"thumb")
    a = _rec(1, thumb=str(prev), path=str(orig))
    b = _rec(2, thumb=str(orig), path=str(orig))
    assert preview_path_for_result(a) == str(prev.resolve())
    assert preview_path_for_result(b) == ""
    plan = build_copy_plan([a, b], [1, 2])
    assert plan.paths == [str(prev.resolve())]
    assert plan.copied_ids == [1]
    assert 2 in plan.skipped_original_ids


def test_copy_plan_unique_and_order(tmp_path):
    files = []
    recs = []
    for i in range(1, 6):
        p = tmp_path / f"t{i}.jpg"
        p.write_bytes(b"x")
        files.append(str(p.resolve()))
        recs.append(_rec(i, thumb=str(p), path=str(tmp_path / f"o{i}.tif")))
    plan = build_copy_plan(recs, [5, 1, 5, 3])
    assert plan.copied_ids == [1, 3, 5]
    assert plan.paths == [files[0], files[2], files[4]]


def test_select_100_is_fast():
    s = PreviewSelectionStore()
    t0 = time.perf_counter()
    s.select_many(range(1, 201))
    elapsed = time.perf_counter() - t0
    assert s.count() == 200
    assert elapsed < 0.25


def test_windows_clipboard_gets_preview_files_not_original(tmp_path):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    prev = tmp_path / "card.webp"
    orig = tmp_path / "source.eps"
    prev.write_bytes(b"preview-bytes")
    orig.write_bytes(b"original-eps")
    rec = _rec(8, thumb=str(prev), path=str(orig))
    plan = build_copy_plan([rec], [8])
    msg = copy_previews_to_clipboard(plan, app.clipboard())
    assert "panoya kopyalandı" in msg
    mime = app.clipboard().mimeData()
    urls = mime.urls()
    assert len(urls) == 1
    assert urls[0].toLocalFile().replace("/", os.sep).lower() == str(prev.resolve()).lower()
    assert "source.eps" not in urls[0].toLocalFile().lower()
    if sys.platform == "win32":
        fmt = mime.formats()
        assert any("text/uri-list" in f or "x-qt-windows-mime" in f or "FileName" in f for f in fmt) or urls


def test_result_card_has_checkbox_and_emits_context_menu():
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication

    from core.search_engine import SearchResult
    from ui.result_card import ResultCard

    app = QApplication.instance() or QApplication([])
    rec = SearchResult(
        file_id=9,
        path="x.tif",
        filename="x.tif",
        customer="",
        thumbnail_path="",
        score=0.0,
        score_percent=0.0,
    )
    card = ResultCard(rec)
    assert card.chk_preview is not None
    card.show()
    assert card.chk_preview.isVisible()
    seen = []
    card.preview_menu_requested.connect(lambda r, pos: seen.append(int(r.file_id)))
    card._emit_preview_menu(QPoint(1, 1))
    assert seen == [9]
    app  # keep alive


def test_checkbox_single_and_multi():
    s = PreviewSelectionStore()
    s.apply_checkbox(1, True)
    assert s.ids() == {1}
    assert s.count() == 1
    s.apply_checkbox(2, True)
    s.apply_checkbox(3, True)
    s.apply_checkbox(4, True)
    assert s.ids() == {1, 2, 3, 4}
    s.apply_checkbox(2, False)
    assert s.ids() == {1, 3, 4}


def test_ctrl_toggle_add_and_remove():
    ordered = list(range(1, 12))
    s = PreviewSelectionStore()
    s.apply_card_click(1, ordered, ctrl=True)
    s.apply_card_click(2, ordered, ctrl=True)
    s.apply_card_click(4, ordered, ctrl=True)
    s.apply_card_click(7, ordered, ctrl=True)
    s.apply_card_click(10, ordered, ctrl=True)
    assert s.ids() == {1, 2, 4, 7, 10}
    s.apply_card_click(4, ordered, ctrl=True)
    assert s.ids() == {1, 2, 7, 10}


def test_shift_range_then_ctrl_then_shift():
    ordered = list(range(1, 21))
    s = PreviewSelectionStore()
    s.apply_card_click(1, ordered)
    s.apply_card_click(10, ordered, shift=True)
    assert s.ids() == set(range(1, 11))
    s.apply_card_click(15, ordered, ctrl=True)
    assert 15 in s.ids() and s.count() == 11
    s.apply_card_click(20, ordered, shift=True)
    assert s.ids() == set(range(1, 11)) | set(range(15, 21))


def test_shift_ctrl_adds_range_without_clearing():
    ordered = list(range(1, 21))
    s = PreviewSelectionStore()
    s.apply_card_click(1, ordered)
    s.apply_card_click(5, ordered, shift=True)
    s.apply_card_click(12, ordered, ctrl=True, shift=True)
    assert set(range(1, 6)).issubset(s.ids())
    assert set(range(5, 13)).issubset(s.ids())
    assert 19 not in s.ids()


def test_plain_click_replaces_selection():
    ordered = list(range(1, 8))
    s = PreviewSelectionStore()
    s.select_many([1, 2, 3])
    s.apply_card_click(6, ordered)
    assert s.ids() == {6}
    assert s.count() == 1


def test_retain_after_lazy_load_keeps_selection():
    s = PreviewSelectionStore()
    s.select_many([1, 3, 7, 12, 18])
    s.retain(list(range(1, 116)))
    assert s.ids() == {1, 3, 7, 12, 18}
    assert s.count() == 5


def test_select_all_115():
    s = PreviewSelectionStore()
    s.select_many(range(1, 116))
    assert s.count() == 115
    assert 115 in s.ids()


def test_copy_ten_selected_previews(tmp_path):
    recs = []
    wanted = [1, 3, 7, 12, 18, 22, 30, 41, 50, 61]
    for i in range(1, 70):
        p = tmp_path / f"t{i}.jpg"
        p.write_bytes(b"x")
        recs.append(_rec(i, thumb=str(p), path=str(tmp_path / f"o{i}.tif")))
    plan = build_copy_plan(recs, wanted)
    assert plan.copied_ids == wanted
    assert len(plan.paths) == 10


def test_selection_count_label_and_no_top_toolbar():
    from PySide6.QtWidgets import QApplication, QPushButton, QWidget

    from ui.results_panel import ResultsPanel

    app = QApplication.instance() or QApplication([])
    panel = ResultsPanel()
    assert panel.findChild(QWidget, "preview_selection_toolbar") is None
    texts = [b.text() for b in panel.findChildren(QPushButton)]
    assert "Tüm görünenleri seç" not in texts
    assert "Tüm sonuçları seç" not in texts
    assert "Önizlemeleri kopyala" not in texts
    assert "Seçilen önizlemeleri kopyala" not in texts
    panel._preview_sel.select_many([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    panel._refresh_preview_checks()
    assert panel.lbl_preview_count.text() == "Seçilen: 10"
    panel.begin_new_search()
    assert panel._preview_sel.count() == 0
    assert panel.lbl_preview_count.text() == "Seçilen: 0"
    app


def test_original_copy_uses_source_not_preview(tmp_path):
    prev = tmp_path / "card.webp"
    orig = tmp_path / "source.eps"
    prev.write_bytes(b"preview-bytes")
    orig.write_bytes(b"original-eps")
    rec = _rec(8, thumb=str(prev), path=str(orig))
    assert original_path_for_result(rec) == str(orig.resolve())
    plan = build_original_copy_plan([rec], [8])
    assert plan.paths == [str(orig.resolve())]
    assert plan.copied_ids == [8]


def test_original_copy_dedupes_same_source(tmp_path):
    orig = tmp_path / "same.tif"
    orig.write_bytes(b"orig")
    a = _rec(1, thumb=str(tmp_path / "a.webp"), path=str(orig))
    b = _rec(2, thumb=str(tmp_path / "b.webp"), path=str(orig))
    plan = build_original_copy_plan([a, b], [1, 2])
    assert plan.copied_ids == [1]
    assert plan.paths == [str(orig.resolve())]
    assert 2 in plan.skipped_original_ids


def test_original_copy_multi_and_clipboard(tmp_path):
    from PySide6.QtWidgets import QApplication

    recs = []
    wanted = [1, 3, 5]
    origs = []
    for i in range(1, 7):
        o = tmp_path / f"o{i}.ai"
        o.write_bytes(b"orig")
        origs.append(str(o.resolve()))
        recs.append(_rec(i, thumb=str(tmp_path / f"t{i}.webp"), path=str(o)))
    plan = build_original_copy_plan(recs, wanted + [1])
    assert plan.copied_ids == wanted
    assert plan.paths == [origs[0], origs[2], origs[4]]
    app = QApplication.instance() or QApplication([])
    msg = copy_originals_to_clipboard(plan, app.clipboard())
    assert "orijinal desen panoya kopyalandı" in msg
    urls = app.clipboard().mimeData().urls()
    assert len(urls) == 3
    got = [u.toLocalFile().replace("/", os.sep).lower() for u in urls]
    assert origs[0].lower() in got
    assert "t1.webp" not in got[0]
