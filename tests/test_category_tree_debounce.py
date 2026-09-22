"""Regression: category tree set_db_path must not rebuild every status tick."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestCategoryTreeDebounce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_set_db_path_skips_rebuild_when_unchanged(self):
        from ui.category_tree_panel import CategoryTreePanel

        panel = CategoryTreePanel()
        with mock.patch.object(panel, "refresh", wraps=panel.refresh) as refresh:
            panel.set_db_path("C:/tmp/a.db")
            self.assertEqual(refresh.call_count, 1)
            panel.set_db_path("C:/tmp/a.db")
            self.assertEqual(refresh.call_count, 1, "same path must not rebuild")
            panel.set_db_path("C:/tmp/b.db")
            self.assertEqual(refresh.call_count, 2)


if __name__ == "__main__":
    unittest.main()
