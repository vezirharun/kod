"""Dual preview: normal dock preview geometry frozen; large overlay separate."""
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


class TestDualNormalHoverPreview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def _docked(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QMainWindow

        from ui.fixed_hover_preview import InspectorDockContent

        win = QMainWindow()
        win.resize(1280, 800)
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

    def test_normal_preview_geometry_unchanged_on_hover(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(800, 400), file_id=1, filename="leo.jpg")
        self._app.processEvents()
        before = p.geometry()
        before_img = p.lbl_image.geometry()
        before_pm = p.lbl_image.pixmap()
        before_pm_size = (
            (before_pm.width(), before_pm.height()) if before_pm else None
        )

        p.enterEvent(QEnterEvent(QPointF(10, 10), QPointF(10, 10), QPointF(10, 10)))
        self._app.processEvents()

        self.assertTrue(p._overlay_visible)
        self.assertEqual(p.geometry(), before)
        self.assertEqual(p.lbl_image.geometry(), before_img)
        after_pm = p.lbl_image.pixmap()
        after_pm_size = (after_pm.width(), after_pm.height()) if after_pm else None
        self.assertEqual(after_pm_size, before_pm_size)

    def test_large_preview_overlay_opens_on_hover(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(900, 450), file_id=2, filename="a.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertTrue(p._overlay_visible)
        self.assertIsNotNone(p._overlay)
        self.assertTrue(p._overlay.isVisible())
        self.assertIsNot(p._overlay, p)

    def test_large_preview_overlay_closes_after_hover_zone_leave(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(600, 600), file_id=3, filename="b.png")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        p.leaveEvent(QEvent(QEvent.Type.Leave))
        p._hide_overlay(immediate=True)
        self._app.processEvents()
        self.assertFalse(p._overlay_visible)
        self.assertFalse(p._overlay.isVisible())

    def test_large_preview_stays_open_between_normal_and_overlay(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(700, 350), file_id=4, filename="c.png")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(8, 8), QPointF(8, 8), QPointF(8, 8)))
        self._app.processEvents()
        p.leaveEvent(QEvent(QEvent.Type.Leave))
        self.assertTrue(p._overlay_hide_timer.isActive())
        p._overlay.enterEvent(QEnterEvent(QPointF(2, 2), QPointF(2, 2), QPointF(2, 2)))
        self.assertFalse(p._overlay_hide_timer.isActive())
        self.assertTrue(p._overlay.isVisible())

    def test_large_preview_does_not_resize_result_detail(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        before = dock.width()
        p.set_selection_pixmap(_pix(1200, 400), file_id=5, filename="w.tif")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(4, 4), QPointF(4, 4), QPointF(4, 4)))
        self._app.processEvents()
        self.assertEqual(dock.width(), before)

    def test_large_preview_does_not_resize_main_window(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        before = win.size()
        p.set_selection_pixmap(_pix(1000, 500), file_id=6, filename="m.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(4, 4), QPointF(4, 4), QPointF(4, 4)))
        self._app.processEvents()
        self.assertEqual(win.size(), before)

    def test_large_preview_does_not_open_external_file(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(400, 200), file_id=7, filename="x.ai")
        with mock.patch("os.startfile") as startfile, mock.patch(
            "PySide6.QtGui.QDesktopServices.openUrl"
        ) as open_url:
            p.enterEvent(QEnterEvent(QPointF(6, 6), QPointF(6, 6), QPointF(6, 6)))
            self._app.processEvents()
            p._hide_overlay(immediate=True)
            startfile.assert_not_called()
            open_url.assert_not_called()
        from ui.fixed_hover_preview import FixedHoverPreviewPanel

        src = inspect.getsource(FixedHoverPreviewPanel.enterEvent)
        self.assertNotIn("startfile", src)
        self.assertNotIn("openUrl", src)
        # No geometry mutation on enter — only overlay show.
        self.assertNotIn("self._apply_fit_image", src)
        self.assertIn("self._show_overlay", src)

    def test_large_preview_contain_fit(self):
        from PySide6.QtCore import QRect

        from ui.fixed_hover_preview import ResultDetailPreviewOverlay

        win, dock, content = self._docked()
        ov = ResultDetailPreviewOverlay(content.hover_preview, win)
        tw, th = ov.set_pixmap_contain(_pix(1600, 400), QRect(0, 0, 500, 400))
        self.assertLessEqual(tw, 500)
        self.assertLessEqual(th, 400)
        self.assertGreater(tw / max(th, 1), 2.0)

    def test_large_preview_inside_main_window(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(2000, 1200), file_id=8, filename="big.tif")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(3, 3), QPointF(3, 3), QPointF(3, 3)))
        self._app.processEvents()
        ov = p._overlay
        wr = win.rect()
        self.assertGreaterEqual(ov.x(), wr.left())
        self.assertGreaterEqual(ov.y(), wr.top())
        self.assertLessEqual(ov.x() + ov.width(), wr.right() + 1)
        self.assertLessEqual(ov.y() + ov.height(), wr.bottom() + 1)


if __name__ == "__main__":
    unittest.main()
