"""Result Detail: vertical scroll only, preview contain-fit, no horizontal overflow."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestResultDetailResponsive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def test_horizontal_scrollbar_always_off(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QMainWindow

        from ui.fixed_hover_preview import InspectorDockContent
        from ui.main_window import MainWindow

        win = QMainWindow()
        content = InspectorDockContent()
        scroll = MainWindow._scrollable_panel(content, horizontal=False)
        dock = QDockWidget("Sonuç Detayı", win)
        dock.setWidget(scroll)
        win.addDockWidget(
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.DockWidgetArea.RightDockWidgetArea,
            dock,
        )
        win.resize(1200, 800)
        win.show()
        dock.show()
        self._app.processEvents()

        self.assertEqual(
            scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertEqual(
            content.inspector_panel.preview_scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

    def test_long_filename_no_min_width_blowout(self):
        from ui.inspector_panel import InspectorPanel

        panel = InspectorPanel()
        panel.resize(320, 600)
        self._app.processEvents()
        long_name = "A" * 180 + "_leopard_pattern_tile.tif"
        panel.lbl_preview_name.setText(f"<b>{long_name}</b>")
        panel.lbl_preview_src.setText("Kaynak: " + "\\\\server\\share\\" + long_name)
        panel._constrain_text_widths()
        self._app.processEvents()
        self.assertTrue(panel.lbl_preview_name.wordWrap())
        self.assertEqual(panel.lbl_preview_name.minimumWidth(), 0)
        self.assertLessEqual(panel.lbl_preview_name.maximumWidth(), 320)
        # Must not demand full unwrapped string width
        self.assertLessEqual(panel.lbl_preview_name.maximumWidth(), 320)

    def test_preview_fit_within_viewport(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap, QColor, QImage
        from PySide6.QtWidgets import QScrollArea

        from ui.inspector_panel import InspectorPanel
        from ui.theme import configure_dock_scroll_area

        panel = InspectorPanel()
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.setWidget(panel)
        configure_dock_scroll_area(outer)
        outer.resize(400, 700)
        outer.show()
        self._app.processEvents()

        img = QImage(1200, 600, QImage.Format.Format_RGB32)
        img.fill(QColor("#336699"))
        pix = QPixmap.fromImage(img)
        tw, th = panel._apply_fit_preview(pix, smooth=False)
        self._app.processEvents()

        avail = panel._preview_avail_width()
        self.assertLessEqual(tw, avail + 1)
        self.assertLessEqual(tw, outer.viewport().width())
        self.assertAlmostEqual(tw / max(th, 1), 2.0, delta=0.15)

        outer.resize(280, 700)
        self._app.processEvents()
        tw2, _ = panel._apply_fit_preview(pix, smooth=False)
        self.assertLessEqual(tw2, panel._preview_avail_width() + 1)
        self.assertLessEqual(tw2, outer.viewport().width() + 1)

        outer.resize(600, 700)
        self._app.processEvents()
        tw3, _ = panel._apply_fit_preview(pix, smooth=False)
        self.assertGreaterEqual(tw3, tw2)
        self.assertLessEqual(tw3, outer.viewport().width() + 1)

    def test_floating_and_docked_no_hscroll(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QMainWindow

        from ui.fixed_hover_preview import InspectorDockContent
        from ui.main_window import MainWindow

        win = QMainWindow()
        win.resize(1280, 800)
        content = InspectorDockContent()
        scroll = MainWindow._scrollable_panel(content, horizontal=False)
        dock = QDockWidget("Sonuç Detayı", win)
        dock.setObjectName("InspectorDock")
        dock.setWidget(scroll)
        win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        win.show()

        for floating, w in ((False, 360), (True, 420), (False, 300)):
            dock.setFloating(floating)
            if floating:
                dock.resize(w, 640)
            dock.show()
            self._app.processEvents()
            self.assertEqual(
                scroll.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            self.assertGreater(scroll.viewport().width(), 0)
            # Buttons still present / not deleted
            for btn in (
                content.inspector_panel.btn_folder,
                content.inspector_panel.btn_file,
                content.inspector_panel.btn_similar,
                content.inspector_panel.btn_ai_edit,
            ):
                self.assertGreater(btn.height(), 0)

    def test_hover_preview_does_not_force_480(self):
        from PySide6.QtGui import QColor, QImage, QPixmap

        from ui.fixed_hover_preview import FixedHoverPreviewPanel

        panel = FixedHoverPreviewPanel()
        panel.resize(300, 200)
        img = QImage(900, 400, QImage.Format.Format_RGB32)
        img.fill(QColor("#112233"))
        panel._apply_fit_image(QPixmap.fromImage(img), smooth=False)
        pm = panel.lbl_image.pixmap()
        self.assertIsNotNone(pm)
        self.assertLessEqual(pm.width(), 300)


if __name__ == "__main__":
    unittest.main()
