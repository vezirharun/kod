"""Keyboard ↔ detail preview ↔ canvas hover final sync."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pix(w=200, h=100, color="#336699"):
    from PySide6.QtGui import QColor, QImage, QPixmap

    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(color))
    return QPixmap.fromImage(img)


def _result(fid: int, name: str = ""):
    from core.search_engine import SearchResult

    return SearchResult(
        file_id=fid,
        path=f"/t/{fid}.jpg",
        filename=name or f"{fid:02d}.jpg",
        customer="",
        thumbnail_path=f"/thumbs/{fid}.jpg",
        score=0.9,
        score_percent=90.0,
    )


class TestKeyboardPreviewSyncFinal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def _chain(self, n=5):
        """ResultsPanel + FixedHoverPreviewPanel wired like main window."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

        from ui.fixed_hover_preview import FixedHoverPreviewPanel, InspectorDockContent
        from ui.results_panel import ResultsPanel

        win = QMainWindow()
        win.resize(1280, 800)
        panel = ResultsPanel()
        win.setCentralWidget(panel)
        content = InspectorDockContent()
        preview = content.hover_preview
        dock = QDockWidget("Sonuc Detay", win)
        dock.setWidget(content)
        win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        win.show()
        dock.show()
        self._app.processEvents()
        win.resizeDocks([dock], [360], Qt.Orientation.Horizontal)
        self._app.processEvents()
        # keep refs so GC does not kill the window mid-test
        self._chain_win = win
        seq = []

        def on_sel(r):
            seq.append(int(r.file_id))
            preview.set_selection_pending(int(r.file_id), r.filename)
            preview.set_selection_pixmap(
                _pix(400, 200, f"#{int(r.file_id)*40:02x}0000"),
                file_id=int(r.file_id),
                filename=r.filename,
            )

        panel.result_selected.connect(on_sel)
        entries = [("card", _result(i + 1), i) for i in range(n)]
        panel.scroll.set_entries(entries)
        panel.scroll.show()
        panel.scroll.resize(400, 400)
        self._app.processEvents()
        return panel, preview, seq

    def test_arrow_down_updates_right_preview(self):
        panel, preview, seq = self._chain(3)
        panel._select_result(_result(1))
        panel.scroll.navigate_by(1)
        self.assertEqual(panel._selected_file_id, 2)
        self.assertEqual(preview._selection_token, 2)
        self.assertEqual(preview.lbl_caption.text(), "02.jpg")

    def test_arrow_up_updates_right_preview(self):
        panel, preview, seq = self._chain(3)
        panel._select_result(_result(3))
        panel.scroll.navigate_by(-1)
        self.assertEqual(preview._selection_token, 2)

    def test_arrow_navigation_preview_sequence(self):
        panel, preview, seq = self._chain(5)
        panel._select_result(_result(1))
        for expect in (2, 3, 4, 5):
            panel.scroll.navigate_by(1)
            self.assertEqual(preview._selection_token, expect)
            self.assertEqual(preview.lbl_caption.text(), f"{expect:02d}.jpg")
        self.assertEqual(seq, [1, 2, 3, 4, 5])

    def test_keyboard_active_result_is_preview_source(self):
        panel, preview, seq = self._chain(4)
        # Stale list hover on result 3, then keyboard to 4.
        panel._select_result(_result(1))
        preview.show_thumbnail(3, "/thumbs/3.jpg", "03.jpg")
        self.assertTrue(preview._list_hovering)
        panel.scroll.set_selected_file_id(1)
        panel.scroll.navigate_by(1)  # → 2
        panel.scroll.navigate_by(1)  # → 3
        panel.scroll.navigate_by(1)  # → 4
        self.assertEqual(panel._selected_file_id, 4)
        self.assertFalse(preview._list_hovering)
        self.assertEqual(preview._selection_token, 4)

    def test_keyboard_navigation_scrolls_active_card(self):
        """ensure_card_pos_visible is invoked on navigate (scroll covered in nav suite)."""
        panel, preview, seq = self._chain(8)
        calls = []
        orig = panel.scroll.ensure_card_pos_visible

        def wrap(pos):
            calls.append(pos)
            return orig(pos)

        panel.scroll.ensure_card_pos_visible = wrap  # type: ignore[method-assign]
        panel._select_result(_result(1))
        panel.scroll.navigate_by(1)
        panel.scroll.navigate_by(1)
        self.assertEqual(panel._selected_file_id, 3)
        self.assertEqual(calls, [1, 2])

    def test_list_hover_temporarily_overrides_preview(self):
        panel, preview, seq = self._chain(3)
        panel._select_result(_result(1))
        preview.show_thumbnail(2, "/thumbs/2.jpg", "02.jpg")
        self.assertTrue(preview._list_hovering)
        self.assertEqual(preview.lbl_caption.text(), "02.jpg")
        self.assertEqual(preview._selection_token, 1)

    def test_list_hover_returns_to_keyboard_active_result(self):
        panel, preview, seq = self._chain(3)
        panel._select_result(_result(2))
        preview.show_thumbnail(1, "/thumbs/1.jpg", "01.jpg")
        self.assertTrue(preview._list_hovering)
        preview._restore_selection()
        self.assertFalse(preview._list_hovering)
        self.assertEqual(preview.lbl_caption.text(), "02.jpg")
        self.assertEqual(preview._selection_token, 2)

    def test_canvas_hover_magnifies_current_preview(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        panel, preview, seq = self._chain(2)
        panel._select_result(_result(2))
        preview.resize(360, 280)
        self._app.processEvents()
        preview.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertTrue(preview._canvas_hovering)
        self.assertEqual(preview._selection_token, 2)

    def test_canvas_hover_uses_high_resolution_preview(self):
        from ui.fixed_hover_preview import FixedHoverPreviewPanel

        p = FixedHoverPreviewPanel()
        p.set_selection_pixmap(_pix(1024, 512), file_id=5, filename="05.jpg")
        p._source_pix = _pix(256, 128)
        best = p._overlay_source_pix()
        self.assertGreaterEqual(best.width(), 1024)

    def test_canvas_hover_stays_inside_detail_canvas(self):
        """Normal preview box unchanged; zoom overlay may spill left (right-anchored)."""
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QEnterEvent

        panel, preview, seq = self._chain(2)
        panel._select_result(_result(1))
        self._app.processEvents()
        win = preview.window()
        before = preview.lbl_image.geometry()
        normal_right = preview.lbl_image.mapTo(win, QPoint(preview.lbl_image.width(), 0)).x()
        preview.enterEvent(QEnterEvent(QPointF(4, 4), QPointF(4, 4), QPointF(4, 4)))
        self._app.processEvents()
        self.assertTrue(preview._canvas_hovering)
        self.assertEqual(preview.lbl_image.geometry(), before)
        self.assertTrue(preview._overlay.isVisible())
        self.assertAlmostEqual(
            preview._overlay.x() + preview._overlay.width(), normal_right, delta=3
        )
        self.assertLess(preview._overlay.x(), preview.lbl_image.mapTo(win, QPoint(0, 0)).x())

    def test_canvas_hover_does_not_open_popup(self):
        """Overlay is allowed but must be right-anchored, not a centered popup."""
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QEnterEvent
        from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

        from ui.fixed_hover_preview import InspectorDockContent

        win = QMainWindow()
        win.resize(1280, 800)
        win.setCentralWidget(QWidget())
        content = InspectorDockContent()
        p = content.hover_preview
        dock = QDockWidget("Sonuc Detay", win)
        dock.setWidget(content)
        win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        win.show()
        dock.show()
        self._app.processEvents()
        win.resizeDocks([dock], [360], Qt.Orientation.Horizontal)
        self._app.processEvents()
        p.set_selection_pixmap(_pix(800, 400), file_id=1, filename="a.jpg")
        self._app.processEvents()
        normal_right = p.lbl_image.mapTo(win, QPoint(p.lbl_image.width(), 0)).x()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertTrue(p._canvas_hovering)
        self.assertTrue(p._overlay.isVisible())
        self.assertAlmostEqual(p._overlay.x() + p._overlay.width(), normal_right, delta=3)
        central = win.centralWidget()
        ccenter = central.mapTo(win, central.rect().center())
        ocenter = p._overlay.mapTo(win, p._overlay.rect().center())
        self.assertNotEqual(ocenter.x(), ccenter.x())

    def test_canvas_leave_restores_active_result_preview(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        panel, preview, seq = self._chain(2)
        panel._select_result(_result(2))
        preview.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        preview.leaveEvent(QEvent(QEvent.Type.Leave))
        preview._end_canvas_hover(immediate=True)
        self.assertFalse(preview._canvas_hovering)
        self.assertEqual(preview._selection_token, 2)
        self.assertEqual(preview.lbl_caption.text(), "02.jpg")

    def test_keyboard_navigation_does_not_change_checkbox(self):
        from ui.preview_selection import PreviewSelectionStore

        panel, preview, seq = self._chain(3)
        store = PreviewSelectionStore()
        panel.scroll.set_preview_selected_fn(store.is_selected)
        panel._select_result(_result(1))
        panel.scroll.navigate_by(1)
        self.assertFalse(store.is_selected(1))
        self.assertFalse(store.is_selected(2))
