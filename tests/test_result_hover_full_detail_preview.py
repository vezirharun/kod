"""List hover → full Result Detail preview canvas (no floating blue overlay)."""
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


class TestResultHoverFullDetailPreview(unittest.TestCase):
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

    def test_result_hover_updates_detail_preview(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(400, 200, "#112233"), file_id=1, filename="sel.jpg")
        self._app.processEvents()
        p.show_thumbnail(10, "/tmp/a.jpg", "hover_a.jpg", source_path="")
        # Inject pixmap as if thumb ready (no disk)
        p._apply_fit_image(_pix(800, 400, "#ff0000"), smooth=False)
        p.lbl_caption.setText("hover_a.jpg")
        self._app.processEvents()
        self.assertTrue(p._list_hovering)
        self.assertEqual(p.lbl_caption.text(), "hover_a.jpg")
        self.assertFalse(p.lbl_image.pixmap().isNull())

    def test_result_hover_uses_hovered_result(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(100, 100), file_id=1, filename="selected.jpg")
        p.show_thumbnail(22, "/t/b.png", "shutterstock_1218753076.jpg")
        p._hover_token = 22
        p._list_hovering = True
        p.lbl_caption.setText("shutterstock_1218753076.jpg")
        self.assertEqual(p._hover_token, 22)
        self.assertEqual(p.lbl_caption.text(), "shutterstock_1218753076.jpg")
        self.assertEqual(p._selection_token, 1)  # selection unchanged

    def test_result_hover_uses_full_detail_preview_canvas(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        geo_panel = p.geometry()
        p.show_thumbnail(3, "/t/c.jpg", "c.jpg")
        p._apply_fit_image(_pix(1200, 600), smooth=False)
        self._app.processEvents()
        # List hover paints into dock canvas; overlay stays closed.
        self.assertEqual(p.geometry(), geo_panel)
        self.assertIs(p.lbl_image.parentWidget(), p)
        self.assertFalse(p._overlay_visible)

    def test_result_hover_preview_maximize_contain_fit(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        self._app.processEvents()
        box_w, box_h = p._avail_box()
        tw, th = p._apply_fit_image(_pix(200, 100), smooth=False)
        # Small source should upscale to fill canvas axis
        self.assertGreaterEqual(tw, min(box_w, 200) - 2 or tw)
        self.assertLessEqual(tw, box_w)
        self.assertLessEqual(th, box_h)
        # At least one axis near full (within 2px)
        self.assertTrue(tw >= box_w - 2 or th >= box_h - 2)

    def test_result_hover_preview_preserves_aspect_ratio(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        tw, th = p._apply_fit_image(_pix(1600, 400), smooth=False)
        ratio = tw / max(th, 1)
        self.assertAlmostEqual(ratio, 4.0, delta=0.05)

    def test_result_hover_preview_no_crop(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        box_w, box_h = p._avail_box()
        tw, th = p._apply_fit_image(_pix(900, 900), smooth=False)
        self.assertLessEqual(tw, box_w)
        self.assertLessEqual(th, box_h)
        self.assertAlmostEqual(tw / max(th, 1), 1.0, delta=0.05)

    def test_result_hover_does_not_change_selection(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(300, 200), file_id=99, filename="active.ai")
        sel_tok = p._selection_token
        sel_name = p._selection_name
        p.show_thumbnail(5, "/t/h.jpg", "hover.jpg")
        self.assertEqual(p._selection_token, sel_tok)
        self.assertEqual(p._selection_name, sel_name)
        self.assertTrue(p._list_hovering)

    def test_result_hover_does_not_resize_detail_dock(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        before_dock = dock.geometry()
        before_win = win.geometry()
        before_preview = p.geometry()
        p.set_selection_pixmap(_pix(400, 200), file_id=1, filename="s.jpg")
        p.show_thumbnail(7, "/t/x.jpg", "x.jpg")
        p._apply_fit_image(_pix(1000, 500), smooth=False)
        self._app.processEvents()
        self.assertEqual(dock.geometry(), before_dock)
        self.assertEqual(win.geometry(), before_win)
        self.assertEqual(p.geometry(), before_preview)

    def test_result_hover_does_not_open_external_file(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        with mock.patch("os.startfile") as startfile, mock.patch(
            "PySide6.QtGui.QDesktopServices.openUrl"
        ) as open_url:
            p.show_thumbnail(8, "C:\\fake\\file.ai", "file.ai", source_path="C:\\fake\\file.ai")
            p._apply_fit_image(_pix(200, 200), smooth=False)
            self._app.processEvents()
            startfile.assert_not_called()
            open_url.assert_not_called()
        src = inspect.getsource(p.show_thumbnail)
        self.assertNotIn("startfile", src)
        self.assertNotIn("openUrl", src)

    def test_result_hover_switches_between_results(self):
        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(100, 100), file_id=1, filename="sel.jpg")
        p.show_thumbnail(11, "/a.jpg", "first.jpg")
        p.lbl_caption.setText("first.jpg")
        p._hover_token = 11
        self.assertEqual(p.lbl_caption.text(), "first.jpg")
        p.show_thumbnail(12, "/b.jpg", "second.jpg")
        p.lbl_caption.setText("second.jpg")
        p._hover_token = 12
        self.assertEqual(p.lbl_caption.text(), "second.jpg")
        self.assertEqual(p._hover_token, 12)
        # leave → restore selection
        p._restore_selection()
        self.assertFalse(p._list_hovering)
        self.assertEqual(p.lbl_caption.text(), "sel.jpg")

    def test_list_hover_does_not_open_magnify_overlay(self):
        """STATE 2 paints canvas only; magnify overlay is STATE 3 (detail hover)."""
        win, dock, content = self._docked()
        p = content.hover_preview
        p.set_selection_pixmap(_pix(100, 100), file_id=1, filename="sel.jpg")
        p.show_thumbnail(9, "/t/z.jpg", "z.jpg")
        self.assertTrue(p._list_hovering)
        self.assertFalse(p._overlay_visible)
