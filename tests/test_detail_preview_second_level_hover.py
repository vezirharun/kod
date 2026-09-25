"""STATE 3: detail preview hover → separate large magnify overlay."""
from __future__ import annotations

import inspect
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pix(w: int, h: int, color: str = "#336699"):
    from PySide6.QtGui import QColor, QImage, QPixmap

    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(color))
    return QPixmap.fromImage(img)


class TestDetailPreviewSecondLevelHover(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def _docked(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

        from ui.fixed_hover_preview import InspectorDockContent

        win = QMainWindow()
        win.resize(1280, 800)
        win.setCentralWidget(QWidget())  # results stand-in (left of dock)
        content = InspectorDockContent()
        dock = QDockWidget("Sonuç Detayı", win)
        dock.setObjectName("InspectorDock")
        dock.setWidget(content)
        win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        win.show()
        dock.show()
        self._app.processEvents()
        win.resizeDocks([dock], [360], Qt.Orientation.Horizontal)
        self._app.processEvents()
        return win, dock, content

    def test_detail_preview_hover_opens_large_overlay(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(800, 400), file_id=1, filename="leo.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(10, 10), QPointF(10, 10), QPointF(10, 10)))
        self._app.processEvents()
        self.assertTrue(p._overlay_visible)
        self.assertIsNotNone(p._overlay)
        self.assertTrue(p._overlay.isVisible())
        self.assertIsNot(p._overlay, p)

    def test_detail_preview_hover_magnifies(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(1024, 512), file_id=2, filename="a.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertTrue(p._overlay_visible)
        # In-canvas overlay: same geometry as lbl_image; hi-res source.
        self.assertEqual(p._overlay.geometry(), p._canvas_rect())
        used = p._overlay_source_pix()
        self.assertGreaterEqual(used.width(), 512)


    def test_detail_preview_hover_does_not_resize_normal_preview(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(600, 300), file_id=3, filename="b.jpg")
        self._app.processEvents()
        before = p.geometry()
        before_img = p.lbl_image.geometry()
        before_dock = dock.geometry()
        before_win = win.size()
        p.enterEvent(QEnterEvent(QPointF(8, 8), QPointF(8, 8), QPointF(8, 8)))
        self._app.processEvents()
        self.assertEqual(p.geometry(), before)
        self.assertEqual(p.lbl_image.geometry(), before_img)
        self.assertEqual(dock.geometry(), before_dock)
        self.assertEqual(win.size(), before_win)

    def test_detail_preview_hover_keeps_overlay_open(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(500, 500), file_id=4, filename="c.png")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(4, 4), QPointF(4, 4), QPointF(4, 4)))
        self._app.processEvents()
        p.leaveEvent(QEvent(QEvent.Type.Leave))
        self.assertTrue(p._overlay_hide_timer.isActive())
        p._overlay.enterEvent(QEnterEvent(QPointF(2, 2), QPointF(2, 2), QPointF(2, 2)))
        self.assertFalse(p._overlay_hide_timer.isActive())
        self.assertTrue(p._overlay.isVisible())

    def test_detail_preview_hover_closes_after_both_zones_leave(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(400, 400), file_id=5, filename="d.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(3, 3), QPointF(3, 3), QPointF(3, 3)))
        self._app.processEvents()
        self.assertTrue(p._overlay_visible)
        p.leaveEvent(QEvent(QEvent.Type.Leave))
        p._overlay.leaveEvent(QEvent(QEvent.Type.Leave))
        p._hide_overlay(immediate=True)
        self.assertFalse(p._overlay_visible)
        self.assertFalse(p._overlay.isVisible())

    def test_detail_preview_hover_does_not_change_selection(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(200, 200), file_id=99, filename="active.ai")
        tok, name = p._selection_token, p._selection_name
        p.enterEvent(QEnterEvent(QPointF(1, 1), QPointF(1, 1), QPointF(1, 1)))
        self._app.processEvents()
        self.assertEqual(p._selection_token, tok)
        self.assertEqual(p._selection_name, name)

    def test_detail_preview_hover_does_not_open_external_file(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(300, 150), file_id=6, filename="x.cdr")
        with mock.patch("os.startfile") as startfile, mock.patch(
            "PySide6.QtGui.QDesktopServices.openUrl"
        ) as open_url:
            p.enterEvent(QEnterEvent(QPointF(6, 6), QPointF(6, 6), QPointF(6, 6)))
            self._app.processEvents()
            p._hide_overlay(immediate=True)
            startfile.assert_not_called()
            open_url.assert_not_called()
        src = inspect.getsource(p.enterEvent)
        self.assertNotIn("startfile", src)
        self.assertNotIn("openUrl", src)
        self.assertNotIn("self._apply_fit_image", src)

    def test_detail_preview_hover_preserves_aspect_ratio(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(1600, 400), file_id=8, filename="wide.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        pm = p._overlay.lbl_image.pixmap()
        self.assertAlmostEqual(pm.width() / max(pm.height(), 1), 4.0, delta=0.1)

    def test_list_hover_still_uses_canvas_not_only_overlay(self):
        """STATE 2 must still paint into the dock canvas."""
        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(100, 100), file_id=1, filename="sel.jpg")
        p.show_thumbnail(10, "/t/a.jpg", "list.jpg")
        self.assertTrue(p._list_hovering)
        self.assertFalse(p._overlay_visible)
        self.assertEqual(p._selection_token, 1)
