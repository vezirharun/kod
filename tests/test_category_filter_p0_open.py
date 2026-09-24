"""Category Filter P0: no forced open refresh; one-pass tree build."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestCategoryFilterP0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_second_open_skips_refresh_when_tree_ready(self):
        from ui.category_tree_panel import CategoryTreePanel

        panel = CategoryTreePanel()
        panel.set_db_path("C:/tmp/cat_a.db")
        self.assertTrue(panel.is_tree_ready("C:/tmp/cat_a.db"))
        with mock.patch.object(panel, "refresh", wraps=panel.refresh) as refresh:
            panel.set_db_path("C:/tmp/cat_a.db")
            self.assertEqual(refresh.call_count, 0)

    def test_db_path_change_triggers_refresh(self):
        from ui.category_tree_panel import CategoryTreePanel

        panel = CategoryTreePanel()
        with mock.patch.object(panel, "refresh", wraps=panel.refresh) as refresh:
            panel.set_db_path("C:/tmp/cat_a.db")
            n1 = refresh.call_count
            panel.set_db_path("C:/tmp/cat_b.db")
            self.assertGreater(refresh.call_count, n1)

    def test_empty_tree_rebuilds_same_path(self):
        from ui.category_tree_panel import CategoryTreePanel

        panel = CategoryTreePanel()
        panel.set_db_path("C:/tmp/cat_a.db")
        panel.tree.clear()
        self.assertFalse(panel.is_tree_ready("C:/tmp/cat_a.db"))
        with mock.patch.object(panel, "refresh", wraps=panel.refresh) as refresh:
            panel.set_db_path("C:/tmp/cat_a.db")
            self.assertEqual(refresh.call_count, 1)

    def test_filter_select_and_clear(self):
        from ui.category_tree_panel import CategoryTreePanel

        panel = CategoryTreePanel()
        panel.set_db_path("")
        self.assertGreater(panel.tree.topLevelItemCount(), 0)
        # Required taxonomy parents remain visible.
        labels = [
            panel.tree.topLevelItem(i).text(0)
            for i in range(panel.tree.topLevelItemCount())
        ]
        for required in (
            "Animal",
            "Animal Print",
            "Floral",
            "Camouflage",
            "Geometric",
            "Plaid Check",
            "Stripe",
            "Lace",
        ):
            self.assertIn(required, labels)
        parent = panel.tree.topLevelItem(labels.index("Animal Print"))
        paths: list[str] = []
        panel.category_filter_changed.connect(paths.append)
        panel._on_item_clicked(parent)
        self.assertTrue(paths)
        self.assertEqual(panel.active_path(), paths[-1])
        panel.clear_filter()
        self.assertEqual(paths[-1], "")
        self.assertEqual(panel.active_path(), "")

    def test_hierarchy_preserves_static_and_dynamic(self):
        from core.category_memory import register_category, register_root_category
        from core.category_tree import build_category_tree_hierarchy
        from core.db import Database

        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "patterns.db")
            Database(db_path)
            register_root_category(db_path, "YeniAnaGrup")
            register_category(db_path, "Animal Print", "CustomPrint")
            roots, children = build_category_tree_hierarchy(db_path)
            for p in ("Animal", "Animal Print", "Floral", "Camouflage", "Geometric"):
                self.assertIn(p, roots)
            self.assertIn("YeniAnaGrup", roots)
            self.assertIn("Leopard", children["Animal Print"])
            self.assertIn("CustomPrint", children["Animal Print"])

    def test_one_pass_concepts_call_count(self):
        from core import concept_registry
        from core.category_tree import build_category_tree_hierarchy
        from core.db import Database

        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "patterns.db")
            Database(db_path)
            # Ensure registry exists once via writer.
            concept_registry.concepts(db_path)
            with mock.patch.object(
                concept_registry,
                "concepts_readonly",
                wraps=concept_registry.concepts_readonly,
            ) as concepts_fn:
                build_category_tree_hierarchy(db_path)
                self.assertEqual(concepts_fn.call_count, 1)

    def test_populate_timing_budget(self):
        from core.category_tree import build_category_tree_hierarchy
        from core.db import Database
        from core.settings import AppSettings

        try:
            db_path = str(AppSettings.load().db_path)
            if not Path(db_path).is_file():
                raise FileNotFoundError(db_path)
        except Exception:
            with tempfile.TemporaryDirectory() as td:
                db_path = str(Path(td) / "patterns.db")
                Database(db_path)
                build_category_tree_hierarchy(db_path)
                t0 = time.perf_counter()
                build_category_tree_hierarchy(db_path)
                warm_ms = (time.perf_counter() - t0) * 1000.0
                self.assertLess(warm_ms, 2000.0)
                return

        build_category_tree_hierarchy(db_path)
        t0 = time.perf_counter()
        build_category_tree_hierarchy(db_path)
        warm_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(warm_ms, 2000.0, f"warm hierarchy too slow: {warm_ms:.1f}ms")


if __name__ == "__main__":
    unittest.main()
