"""Dock shell opaque fill, floating restore sanitize, format open without forced refresh."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestDockShellOpaque(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def test_dock_shell_opaque(self):
        from PySide6.QtWidgets import QLabel

        from ui.main_window import MainWindow
        from ui.theme import APP_STYLESHEET, COLOR_BG

        scroll = MainWindow._scrollable_panel(QLabel("panel"))
        vp = scroll.viewport()
        self.assertTrue(scroll.autoFillBackground())
        self.assertTrue(vp.autoFillBackground())
        self.assertEqual(
            vp.palette().color(vp.palette().ColorRole.Window).name().lower(),
            COLOR_BG.lower(),
        )
        self.assertNotIn("background: transparent", APP_STYLESHEET)
        self.assertIn(f"background-color: {COLOR_BG}", APP_STYLESHEET)
        self.assertIn("QDockWidget", APP_STYLESHEET)


class TestFloatingRestore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def test_floating_restore(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QLabel, QMainWindow, QScrollArea

        from ui.layout_state import sanitize_dock_shell
        from ui.theme import COLOR_BG, configure_dock_scroll_area

        win = QMainWindow()
        win.resize(1000, 700)
        dock = QDockWidget("Format Durumu", win)
        dock.setObjectName("FormatStatusDock")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(QLabel("CONTENT"))
        dock.setWidget(scroll)
        win.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        win.show()
        dock.setFloating(True)
        dock.resize(80, 60)  # invalid tiny float
        sanitize_dock_shell(win)
        self.assertGreaterEqual(dock.width(), 320)
        self.assertGreaterEqual(dock.height(), 240)
        self.assertEqual(
            scroll.viewport().palette().color(scroll.viewport().palette().ColorRole.Window).name().lower(),
            COLOR_BG.lower(),
        )
        # Idempotent reconfigure
        configure_dock_scroll_area(scroll)
        self.assertTrue(scroll.viewport().autoFillBackground())


class TestPanelReopen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from ui.theme import apply_theme

        cls._app = QApplication.instance() or QApplication([])
        apply_theme(cls._app)

    def test_panel_reopen(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDockWidget, QLabel, QMainWindow

        from ui.main_window import MainWindow

        win = QMainWindow()
        win.resize(1100, 800)
        panel = QLabel("Kaynaklar chrome")
        scroll = MainWindow._scrollable_panel(panel)
        dock = QDockWidget("Kaynaklar ve İndeks", win)
        dock.setObjectName("SourceDock")
        dock.setWidget(scroll)
        win.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        win.show()

        for floating in (False, True, False):
            dock.setFloating(floating)
            dock.setVisible(True)
            dock.raise_()
            if floating:
                dock.resize(400, 360)
            self.assertTrue(panel.isVisible() or scroll.isVisible())
            self.assertGreater(scroll.viewport().width(), 0)
            self.assertGreater(scroll.viewport().height(), 0)
            dock.hide()
            dock.show()
            self.assertTrue(dock.isVisible())
            bg = scroll.viewport().palette().color(
                scroll.viewport().palette().ColorRole.Window
            ).name().lower()
            self.assertNotEqual(bg, "#ffffff")
            self.assertNotEqual(bg, "#efefef")


class TestFormatStatusNoForcedRefresh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_format_status_no_forced_refresh(self):
        """Open path must not call refresh via singleShot (source inspection + behavior)."""
        import inspect

        from ui import main_window as mw

        src = inspect.getsource(mw.MainWindow._set_panel_visible)
        self.assertNotIn(
            "format_status_panel.refresh",
            src,
            "format open must not force refresh",
        )
        # Panel still refreshes on demand via Yenile / set_db_path
        from ui.format_status_panel import FormatStatusPanel

        panel = FormatStatusPanel()
        with mock.patch.object(panel, "refresh") as refresh:
            panel.set_db_path("")
            refresh.assert_not_called()
            panel._db_path = "x"
            panel.table.setRowCount(2)
            panel.set_db_path("x")
            refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
