"""Result Detail hover → in-app overlay over results (not dock resize / not file open)."""
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


class TestResultDetailHoverOverlay(unittest.TestCase):
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
        # Force a known dock width
        win.resizeDocks([dock], [360], Qt.Orientation.Horizontal)
        self._app.processEvents()
        return win, dock, content

    def test_preview_hover_overlay_opens(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        dock_w_before = dock.width()
        win_w_before = win.width()
        p.set_selection_pixmap(_pix(900, 450), file_id=1, filename="leo.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(10, 10), QPointF(10, 10), QPointF(10, 10)))
        self._app.processEvents()
        self.assertTrue(p._overlay_visible)
        self.assertIsNotNone(p._overlay)
        self.assertTrue(p._overlay.isVisible())
        self.assertEqual(dock.width(), dock_w_before)
        self.assertEqual(win.width(), win_w_before)

    def test_preview_hover_overlay_closes_on_leave(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(600, 600), file_id=2, filename="a.png")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertTrue(p._overlay_visible)
        p.leaveEvent(QEvent(QEvent.Type.Leave))
        # Fire hide timer immediately
        p._overlay_hide_timer.stop()
        p._hide_overlay(immediate=True)
        self._app.processEvents()
        self.assertFalse(p._overlay_visible)
        self.assertFalse(p._overlay.isVisible())

    def test_preview_hover_overlay_keeps_visible_between_preview_and_overlay(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(700, 350), file_id=3, filename="b.png")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(8, 8), QPointF(8, 8), QPointF(8, 8)))
        self._app.processEvents()
        # Leave preview (starts hide timer) then enter overlay (cancels)
        p.leaveEvent(QEvent(QEvent.Type.Leave))
        self.assertTrue(p._overlay_hide_timer.isActive())
        p._overlay.enterEvent(QEnterEvent(QPointF(2, 2), QPointF(2, 2), QPointF(2, 2)))
        self.assertFalse(p._overlay_hide_timer.isActive())
        self.assertTrue(p._overlay_visible)
        self.assertTrue(p._overlay.isVisible())

    def test_preview_hover_overlay_does_not_resize_dock(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        before = dock.width()
        p.set_selection_pixmap(_pix(1200, 400), file_id=4, filename="wide.tif")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(4, 4), QPointF(4, 4), QPointF(4, 4)))
        self._app.processEvents()
        self.assertEqual(dock.width(), before)

    def test_preview_hover_overlay_does_not_open_file(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(400, 200), file_id=5, filename="x.ai")
        with mock.patch("os.startfile") as startfile, mock.patch(
            "PySide6.QtGui.QDesktopServices.openUrl"
        ) as open_url:
            p.enterEvent(QEnterEvent(QPointF(6, 6), QPointF(6, 6), QPointF(6, 6)))
            self._app.processEvents()
            p._hide_overlay(immediate=True)
            startfile.assert_not_called()
            open_url.assert_not_called()
        src = inspect.getsource(
            __import__("ui.fixed_hover_preview", fromlist=["FixedHoverPreviewPanel"]).FixedHoverPreviewPanel.enterEvent
        )
        self.assertNotIn("startfile", src)
        self.assertNotIn("openUrl", src)

    def test_preview_hover_overlay_contain_fit(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        from ui.fixed_hover_preview import ResultDetailPreviewOverlay
        from PySide6.QtCore import QRect

        ov = ResultDetailPreviewOverlay(p, win)
        tw, th = ov.set_pixmap_contain(_pix(1600, 400), QRect(0, 0, 500, 400))
        self.assertLessEqual(tw, 500)
        self.assertLessEqual(th, 400)
        self.assertGreater(tw / max(th, 1), 2.0)
        tw2, th2 = ov.set_pixmap_contain(_pix(400, 1600), QRect(0, 0, 500, 400))
        self.assertLessEqual(tw2, 500)
        self.assertLessEqual(th2, 400)
        self.assertLess(tw2 / max(th2, 1), 1.0)

    def test_preview_hover_overlay_stays_inside_main_window(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(2000, 1200), file_id=6, filename="big.tif")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(3, 3), QPointF(3, 3), QPointF(3, 3)))
        self._app.processEvents()
        ov = p._overlay
        self.assertTrue(ov.isVisible())
        # Overlay geometry is in main-window coords
        wr = win.rect()
        self.assertGreaterEqual(ov.x(), wr.left())
        self.assertGreaterEqual(ov.y(), wr.top())
        self.assertLessEqual(ov.x() + ov.width(), wr.right() + 1)
        self.assertLessEqual(ov.y() + ov.height(), wr.bottom() + 1)


# Keep aliases expected by older suite names if imported
class TestResultDetailHoverMagnify(TestResultDetailHoverOverlay):
    def test_result_detail_preview_hover_expands(self):
        self.test_preview_hover_overlay_opens()

    def test_result_detail_preview_hover_restores(self):
        self.test_preview_hover_overlay_closes_on_leave()

    def test_result_detail_hover_does_not_open_external_file(self):
        self.test_preview_hover_overlay_does_not_open_file()

    def test_result_detail_preview_contain_fit(self):
        self.test_preview_hover_overlay_contain_fit()

    def test_result_detail_preview_no_horizontal_overflow(self):
        self.test_preview_hover_overlay_stays_inside_main_window()


if __name__ == "__main__":
    unittest.main()
