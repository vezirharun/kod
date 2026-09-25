"""Result Detail: large top preview + no duplicate; category dock opaque/no-hscroll."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestResultDetailLargePreview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def test_main_preview_on_top_thumb_hidden(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QImage, QPixmap
        from PySide6.QtWidgets import QDockWidget, QMainWindow

        from ui.fixed_hover_preview import InspectorDockContent

        win = QMainWindow()
        win.resize(1100, 800)
        content = InspectorDockContent()
        dock = QDockWidget("Sonuç Detayı", win)
        dock.setWidget(content)
        win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        win.show()
        dock.show()
        self._app.processEvents()

        # Top preview always visible / has height
        self.assertTrue(content.hover_preview.isVisible())
        self.assertGreaterEqual(content.hover_preview.minimumHeight(), 220)
        # Inner detail scroll: no horizontal bar
        self.assertEqual(
            content.detail_scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        # No duplicate in-tab preview image
        self.assertFalse(content.inspector_panel.thumb.isVisible())

        img = QImage(800, 400, QImage.Format.Format_RGB32)
        img.fill(QColor("#445566"))
        tw, th = content.inspector_panel._apply_fit_preview(
            QPixmap.fromImage(img), smooth=False
        )
        self._app.processEvents()
        self.assertGreater(tw, 0)
        self.assertLessEqual(tw, content.hover_preview.width())
        pm = content.hover_preview.lbl_image.pixmap()
        self.assertIsNotNone(pm)
        self.assertFalse(pm.isNull())
        # Aspect ~2:1
        self.assertAlmostEqual(tw / max(th, 1), 2.0, delta=0.2)

    def test_selection_changes_main_preview(self):
        from PySide6.QtGui import QColor, QImage, QPixmap

        from ui.fixed_hover_preview import FixedHoverPreviewPanel

        panel = FixedHoverPreviewPanel()
        panel.resize(400, 300)
        self._app.processEvents()
        a = QImage(100, 50, QImage.Format.Format_RGB32)
        a.fill(QColor("#ff0000"))
        b = QImage(50, 100, QImage.Format.Format_RGB32)
        b.fill(QColor("#00ff00"))
        panel.set_selection_pixmap(QPixmap.fromImage(a), file_id=1, filename="a.png")
        self.assertEqual(panel._selection_token, 1)
        self.assertIn("a.png", panel.lbl_caption.text())
        panel.set_selection_pixmap(QPixmap.fromImage(b), file_id=2, filename="b.png")
        self.assertEqual(panel._selection_token, 2)
        self.assertIn("b.png", panel.lbl_caption.text())

    def test_portrait_landscape_fit(self):
        from PySide6.QtGui import QColor, QImage, QPixmap

        from ui.fixed_hover_preview import FixedHoverPreviewPanel

        panel = FixedHoverPreviewPanel()
        panel.resize(360, 400)
        self._app.processEvents()
        land = QImage(1200, 300, QImage.Format.Format_RGB32)
        land.fill(QColor("#112233"))
        tw, th = panel._apply_fit_image(QPixmap.fromImage(land), smooth=False)
        self.assertLessEqual(tw, 360)
        self.assertGreater(tw / max(th, 1), 2.0)

        port = QImage(300, 1200, QImage.Format.Format_RGB32)
        port.fill(QColor("#223344"))
        tw2, th2 = panel._apply_fit_image(QPixmap.fromImage(port), smooth=False)
        self.assertLessEqual(th2, 400)
        self.assertLess(tw2 / max(th2, 1), 1.0)


class TestCategoryDockShell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def test_category_scroll_opaque_no_hscroll(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QMainWindow

        from ui.category_tree_panel import CategoryTreePanel
        from ui.main_window import MainWindow
        from ui.theme import COLOR_BG, COLOR_SURFACE

        win = QMainWindow()
        panel = CategoryTreePanel()
        scroll = MainWindow._scrollable_panel(panel, horizontal=False)
        dock = QDockWidget("Kategori Filtresi", win)
        dock.setWidget(scroll)
        win.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        win.resize(1000, 700)
        win.show()
        dock.show()
        self._app.processEvents()

        self.assertEqual(
            scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertEqual(
            panel.tree.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        vp_bg = scroll.viewport().palette().color(
            scroll.viewport().palette().ColorRole.Window
        ).name().lower()
        self.assertEqual(vp_bg, COLOR_BG.lower())
        self.assertNotEqual(vp_bg, "#ffffff")
        self.assertIn(COLOR_SURFACE, __import__("ui.theme", fromlist=["APP_STYLESHEET"]).APP_STYLESHEET)
        self.assertIn("QTreeWidget", __import__("ui.theme", fromlist=["APP_STYLESHEET"]).APP_STYLESHEET)


if __name__ == "__main__":
    unittest.main()
