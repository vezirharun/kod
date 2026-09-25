"""Detail hover preview: in-canvas geometry + high-resolution source."""
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


class TestDetailHoverPositionResolution(unittest.TestCase):
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

    def test_detail_hover_preview_uses_preview_viewport(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(1024, 512), file_id=1, filename="a.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertIs(p._overlay.parentWidget(), p)

    def test_detail_hover_preview_geometry_matches_canvas(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(800, 400), file_id=2, filename="b.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertEqual(p._overlay.geometry(), p.lbl_image.geometry())

    def test_detail_hover_preview_not_centered_in_main_window(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(600, 300), file_id=3, filename="c.jpg")
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        ov = p._overlay
        # Overlay is child of panel — its top-left in window coords equals canvas.
        canvas_tl = p.lbl_image.mapTo(win, p.lbl_image.rect().topLeft())
        ov_tl = ov.mapTo(win, ov.rect().topLeft())
        self.assertEqual(ov_tl, canvas_tl)
        # Not floating over central widget center.
        central = win.centralWidget()
        if central is not None:
            ccenter = central.mapTo(win, central.rect().center())
            ocenter = ov.mapTo(win, ov.rect().center())
            self.assertNotEqual(ocenter, ccenter)

    def test_detail_hover_preview_uses_high_resolution_source(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        hi = _pix(1024, 512, "#00aa00")
        p.set_selection_pixmap(hi, file_id=4, filename="hi.jpg")
        # Simulate list-hover low-res overwrite of display source only.
        p._source_pix = _pix(256, 128, "#ff0000")
        p._list_hovering = False
        self._app.processEvents()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        used = p._overlay_source_pix()
        self.assertIsNotNone(used)
        self.assertGreaterEqual(used.width(), 1024)

    def test_detail_hover_preview_does_not_use_low_res_thumbnail_when_better_source_exists(
        self,
    ):
        win, dock, content = self._docked()
        p = content.hover_preview
        p._selection_pix = _pix(1024, 1024)
        p._source_pix = _pix(256, 256)
        p._selection_token = 7
        best = p._overlay_source_pix()
        self.assertEqual((best.width(), best.height()), (1024, 1024))

    def test_detail_hover_preview_preserves_aspect_ratio(self):
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

    def test_detail_hover_preview_does_not_resize_dock(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(500, 250), file_id=9, filename="d.jpg")
        before_d = dock.geometry()
        before_w = win.size()
        before_p = p.geometry()
        p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        self._app.processEvents()
        self.assertEqual(dock.geometry(), before_d)
        self.assertEqual(win.size(), before_w)
        self.assertEqual(p.geometry(), before_p)

    def test_detail_hover_preview_does_not_change_selection(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent

        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(400, 200), file_id=99, filename="sel.ai")
        tok, name = p._selection_token, p._selection_name
        with mock.patch("os.startfile") as sf:
            p.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
            self._app.processEvents()
            sf.assert_not_called()
        self.assertEqual(p._selection_token, tok)
        self.assertEqual(p._selection_name, name)
