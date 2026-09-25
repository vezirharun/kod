"""Canvas hover: right-anchored zoom expanding left (dock width unchanged)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pix(w: int, h: int, color: str = "#336699"):
    from PySide6.QtGui import QColor, QImage, QPixmap

    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(color))
    return QPixmap.fromImage(img)


class TestCanvasHoverRightAnchor(unittest.TestCase):
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
        win.setCentralWidget(QWidget())
        content = InspectorDockContent()
        dock = QDockWidget("Sonuc Detay", win)
        dock.setWidget(content)
        win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        win.show()
        dock.show()
        self._app.processEvents()
        win.resizeDocks([dock], [360], Qt.Orientation.Horizontal)
        self._app.processEvents()
        return win, dock, content

    def test_hover_right_edge_matches_preview(self):
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(1024, 512), file_id=1, filename="a.jpg")
        self._app.processEvents()
        normal_right = p.lbl_image.mapTo(win, QPoint(p.lbl_image.width(), 0)).x()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertTrue(p._overlay_visible)
        self.assertIsNotNone(p._overlay)
        self.assertTrue(p._overlay.isVisible())
        overlay_right = p._overlay.x() + p._overlay.width()
        self.assertAlmostEqual(overlay_right, normal_right, delta=3)

    def test_hover_expands_leftward(self):
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(1024, 512), file_id=2, filename="b.jpg")
        self._app.processEvents()
        normal_left = p.lbl_image.mapTo(win, p.lbl_image.rect().topLeft()).x()
        normal_w = p.lbl_image.width()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertGreater(p._overlay.width(), normal_w)
        self.assertLess(p._overlay.x(), normal_left)

    def test_hover_does_not_resize_dock(self):
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(800, 400), file_id=3, filename="c.jpg")
        before = dock.geometry()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertEqual(dock.geometry(), before)

    def test_hover_leave_restores_normal(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(600, 300), file_id=4, filename="d.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        p.leaveEvent(QEvent(QEvent.Type.Leave))
        p._overlay.leaveEvent(QEvent(QEvent.Type.Leave))
        p._end_canvas_hover(immediate=True)
        self.assertFalse(p._overlay_visible)
        self.assertFalse(p._overlay.isVisible())

    def test_hover_not_centered_popup(self):
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(900, 450), file_id=5, filename="e.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        ov = p._overlay
        central = win.centralWidget()
        ccenter = central.mapTo(win, central.rect().center())
        ocenter = ov.mapTo(win, ov.rect().center())
        # Anchored to preview — not free-centered in the window.
        self.assertNotEqual(ocenter.x(), ccenter.x())

    def test_portrait_aspect_keeps_right_anchor(self):
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(400, 900), file_id=7, filename="portrait.jpg")
        self._app.processEvents()
        normal_right = p.lbl_image.mapTo(win, QPoint(p.lbl_image.width(), 0)).x()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        overlay_right = p._overlay.x() + p._overlay.width()
        self.assertAlmostEqual(overlay_right, normal_right, delta=3)

    def test_overlay_stays_open_between_preview_and_overlay(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(700, 350), file_id=6, filename="f.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        p.leaveEvent(QEvent(QEvent.Type.Leave))
        self.assertTrue(p._overlay_hide_timer.isActive())
        p._overlay.enterEvent(QEnterEvent(QPointF(2, 2), QPointF(2, 2), QPointF(2, 2)))
        self.assertFalse(p._overlay_hide_timer.isActive())
        self.assertTrue(p._overlay.isVisible())


if __name__ == "__main__":
    unittest.main()
