"""Keyboard ↑/↓ navigation for result list active selection."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _result(fid: int, name: str = ""):
    from core.search_engine import SearchResult

    return SearchResult(
        file_id=fid,
        path=f"/t/{fid}.jpg",
        filename=name or f"{fid:02d}.jpg",
        customer="",
        thumbnail_path="",
        score=0.9,
        score_percent=90.0,
    )


class TestResultKeyboardNavigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def _list_with_results(self, n: int = 5):
        from ui.virtual_results_list import VirtualResultsList

        vl = VirtualResultsList()
        vl.resize(400, 320)
        entries = [("card", _result(i + 1), i) for i in range(n)]
        vl.set_entries(entries)
        self._app.processEvents()
        return vl

    def test_arrow_down_moves_to_next_result(self):
        vl = self._list_with_results(4)
        vl.set_selected_file_id(2)
        self.assertTrue(vl.navigate_by(1))
        self.assertEqual(vl._selected_file_id, 3)

    def test_arrow_up_moves_to_previous_result(self):
        vl = self._list_with_results(4)
        vl.set_selected_file_id(3)
        self.assertTrue(vl.navigate_by(-1))
        self.assertEqual(vl._selected_file_id, 2)

    def test_arrow_navigation_stops_at_first_result(self):
        vl = self._list_with_results(3)
        vl.set_selected_file_id(1)
        self.assertFalse(vl.navigate_by(-1))
        self.assertEqual(vl._selected_file_id, 1)

    def test_arrow_navigation_stops_at_last_result(self):
        vl = self._list_with_results(3)
        vl.set_selected_file_id(3)
        self.assertFalse(vl.navigate_by(1))
        self.assertEqual(vl._selected_file_id, 3)

    def test_arrow_navigation_updates_active_result(self):
        vl = self._list_with_results(3)
        seen = []
        vl.card_selected.connect(lambda r: seen.append(int(r.file_id)))
        vl.set_selected_file_id(1)
        vl.navigate_by(1)
        vl.navigate_by(1)
        self.assertEqual(seen, [2, 3])
        self.assertEqual(vl._selected_file_id, 3)

    def test_arrow_navigation_updates_detail_preview(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

        from ui.fixed_hover_preview import InspectorDockContent
        from ui.results_panel import ResultsPanel

        win = QMainWindow()
        win.resize(1100, 700)
        win.setCentralWidget(QWidget())
        panel = ResultsPanel()
        content = InspectorDockContent()
        dock = QDockWidget("Sonuç Detayı", win)
        dock.setWidget(content)
        win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        selected = []

        def on_sel(r):
            selected.append(int(r.file_id))
            content.hover_preview.set_selection_pixmap(
                __import__("PySide6.QtGui", fromlist=["QPixmap"]).QPixmap(200, 100),
                file_id=int(r.file_id),
                filename=r.filename,
            )

        panel.result_selected.connect(on_sel)
        # Drive via scroll entries + select
        entries = [("card", _result(i + 1), i) for i in range(3)]
        panel.scroll.set_entries(entries)
        panel._select_result(_result(1))
        panel.scroll.navigate_by(1)
        self._app.processEvents()
        self.assertEqual(panel._selected_file_id, 2)
        self.assertEqual(content.hover_preview._selection_token, 2)
        self.assertIn(2, selected)

    def test_arrow_navigation_updates_large_preview(self):
        from PySide6.QtGui import QPixmap

        from ui.fixed_hover_preview import FixedHoverPreviewPanel

        p = FixedHoverPreviewPanel()
        p.set_selection_pixmap(QPixmap(400, 200), file_id=1, filename="01.jpg")
        p.set_selection_pixmap(QPixmap(400, 200), file_id=2, filename="02.jpg")
        self.assertEqual(p._selection_token, 2)
        self.assertEqual(p._selection_name, "02.jpg")

    def test_arrow_navigation_scrolls_result_into_view(self):
        vl = self._list_with_results(20)
        vl.show()
        vl.resize(400, 240)
        self._app.processEvents()
        vl.refresh_layout()
        self._app.processEvents()
        vl.set_selected_file_id(1)
        vl.verticalScrollBar().setValue(0)
        for _ in range(15):
            vl.navigate_by(1)
        self._app.processEvents()
        self.assertEqual(vl._selected_file_id, 16)
        pos = vl._selected_card_pos()
        entry_i = vl._card_positions()[pos]
        y = vl._row_offsets[entry_i]
        h = vl._row_heights[entry_i]
        top = vl.verticalScrollBar().value()
        bottom = top + max(1, vl.viewport().height())
        # Active row intersects the viewport (auto-scrolled).
        self.assertLessEqual(y, bottom)
        self.assertGreaterEqual(y + h, top)
        self.assertGreater(vl.verticalScrollBar().value(), 0)


    def test_arrow_navigation_does_not_toggle_checkbox(self):
        from ui.preview_selection import PreviewSelectionStore

        store = PreviewSelectionStore()
        vl = self._list_with_results(3)
        vl.set_preview_selected_fn(store.is_selected)
        vl.set_selected_file_id(1)
        before = store.is_selected(1)
        vl.navigate_by(1)
        self.assertEqual(store.is_selected(1), before)
        self.assertFalse(store.is_selected(2))

    def test_arrow_navigation_ignored_in_search_input(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

        from ui.results_panel import ResultsPanel

        host = QWidget()
        lay = QVBoxLayout(host)
        search = QLineEdit()
        panel = ResultsPanel()
        lay.addWidget(search)
        lay.addWidget(panel)
        host.show()
        entries = [("card", _result(i + 1), i) for i in range(3)]
        panel.scroll.set_entries(entries)
        panel._select_result(_result(1))
        search.setFocus()
        self._app.processEvents()
        ev = QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier
        )
        panel.keyPressEvent(ev)
        self.assertEqual(panel._selected_file_id, 1)

    def test_arrow_navigation_ignored_in_text_inputs(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget

        from ui.results_panel import ResultsPanel

        host = QWidget()
        lay = QVBoxLayout(host)
        edit = QTextEdit()
        panel = ResultsPanel()
        lay.addWidget(edit)
        lay.addWidget(panel)
        host.show()
        panel.scroll.set_entries([("card", _result(1), 0), ("card", _result(2), 1)])
        panel._select_result(_result(1))
        edit.setFocus()
        self._app.processEvents()
        panel.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier
            )
        )
        self.assertEqual(panel._selected_file_id, 1)

    def test_arrow_navigation_preserves_hover_preview(self):
        """List-hover is temporary; selection apply ends it (keyboard priority)."""
        from PySide6.QtGui import QPixmap

        from ui.fixed_hover_preview import FixedHoverPreviewPanel

        p = FixedHoverPreviewPanel()
        p.set_selection_pixmap(QPixmap(100, 100), file_id=1, filename="sel.jpg")
        p.show_thumbnail(9, "/t/h.jpg", "hover.jpg")
        self.assertTrue(p._list_hovering)
        p.set_selection_pixmap(QPixmap(100, 100), file_id=2, filename="02.jpg")
        self.assertEqual(p._selection_token, 2)
        self.assertFalse(p._list_hovering)
        self.assertEqual(p.lbl_caption.text(), "02.jpg")
