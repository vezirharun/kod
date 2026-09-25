"""Result Detail canvas hover magnify — in-app scale only, no external open."""
from __future__ import annotations

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


class TestResultDetailHoverMagnify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def _panel(self):
        from ui.fixed_hover_preview import FixedHoverPreviewPanel

        p = FixedHoverPreviewPanel()
        p.resize(400, 360)
        self._app.processEvents()
        return p

    def test_result_detail_preview_hover_expands(self):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QEnterEvent
        from PySide6.QtCore import QPointF

        p = self._panel()
        pix = _pix(800, 400)
        p.set_selection_pixmap(pix, file_id=7, filename="tile.jpg", smooth=False)
        self._app.processEvents()
        normal = p.lbl_image.pixmap()
        self.assertIsNotNone(normal)
        nw, nh = normal.width(), normal.height()

        p.enterEvent(QEnterEvent(QPointF(10, 10), QPointF(10, 10), QPointF(10, 10)))
        self._app.processEvents()
        self.assertTrue(p._canvas_magnified)
        mag = p.lbl_image.pixmap()
        self.assertIsNotNone(mag)
        self.assertGreaterEqual(mag.width() * mag.height(), nw * nh)
        # Must stay inside canvas
        aw, ah = p._avail_box()
        self.assertLessEqual(mag.width(), aw + 1)
        self.assertLessEqual(mag.height(), ah + 1)

    def test_result_detail_preview_hover_restores(self):
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QEnterEvent

        p = self._panel()
        pix = _pix(600, 600)
        p.set_selection_pixmap(pix, file_id=3, filename="sq.png", smooth=False)
        self._app.processEvents()
        n = p.lbl_image.pixmap()
        n_area = n.width() * n.height()

        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        m = p.lbl_image.pixmap()
        self.assertGreaterEqual(m.width() * m.height(), n_area)

        leave = QEvent(QEvent.Type.Leave)
        p.leaveEvent(leave)
        self._app.processEvents()
        self.assertFalse(p._canvas_magnified)
        r = p.lbl_image.pixmap()
        self.assertAlmostEqual(r.width(), n.width(), delta=2)
        self.assertAlmostEqual(r.height(), n.height(), delta=2)

    def test_result_detail_hover_does_not_open_external_file(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        p = self._panel()
        p.set_selection_pixmap(_pix(400, 200), file_id=1, filename="x.ai", smooth=False)

        with mock.patch("os.startfile") as startfile, mock.patch(
            "PySide6.QtGui.QDesktopServices.openUrl"
        ) as open_url:
            p.enterEvent(QEnterEvent(QPointF(8, 8), QPointF(8, 8), QPointF(8, 8)))
            self._app.processEvents()
            p.leaveEvent(__import__("PySide6.QtCore", fromlist=["QEvent"]).QEvent(
                __import__("PySide6.QtCore", fromlist=["QEvent"]).QEvent.Type.Leave
            ))
            self._app.processEvents()
            startfile.assert_not_called()
            open_url.assert_not_called()

        # Source inspection: magnify path must not call open helpers
        import inspect

        from ui import fixed_hover_preview as mod

        src = inspect.getsource(mod.FixedHoverPreviewPanel.enterEvent)
        self.assertNotIn("startfile", src)
        self.assertNotIn("openUrl", src)
        self.assertNotIn("explorer", src.lower())

    def test_result_detail_preview_contain_fit(self):
        p = self._panel()
        # Landscape
        tw, th = p._apply_fit_image(_pix(1200, 300), smooth=False, fill=1.0)
        aw, ah = p._avail_box()
        self.assertLessEqual(tw, aw + 1)
        self.assertLessEqual(th, ah + 1)
        self.assertGreater(tw / max(th, 1), 2.0)
        # Portrait
        tw2, th2 = p._apply_fit_image(_pix(300, 1200), smooth=False, fill=1.0)
        self.assertLessEqual(tw2, aw + 1)
        self.assertLessEqual(th2, ah + 1)
        self.assertLess(tw2 / max(th2, 1), 1.0)

    def test_result_detail_preview_no_horizontal_overflow(self):
        from PySide6.QtCore import Qt, QPointF
        from PySide6.QtGui import QEnterEvent
        from PySide6.QtWidgets import QDockWidget, QMainWindow

        from ui.fixed_hover_preview import InspectorDockContent

        win = QMainWindow()
        win.resize(1000, 700)
        content = InspectorDockContent()
        dock = QDockWidget("Sonuç Detayı", win)
        dock.setWidget(content)
        win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        win.show()
        dock.resize(320, 640)
        dock.show()
        self._app.processEvents()

        self.assertEqual(
            content.detail_scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        p = content.hover_preview
        p.set_selection_pixmap(_pix(2000, 500), file_id=9, filename="wide.tif")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(12, 12), QPointF(12, 12), QPointF(12, 12)))
        self._app.processEvents()
        pm = p.lbl_image.pixmap()
        self.assertIsNotNone(pm)
        self.assertLessEqual(pm.width(), p.width() + 1)
        self.assertLessEqual(pm.width(), p._avail_box()[0] + 1)


if __name__ == "__main__":
    unittest.main()
