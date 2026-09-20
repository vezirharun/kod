"""Unit tests for category selection auto-sync (Teach/Edit Ana/Alt)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# cat_sync root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.category_selection_sync import (
    canonical_path_from_parts,
    merge_tags_preserve_manual,
    path_segment_tags,
    resolve_parts_against_options,
    split_category_selection,
    sync_selection_state,
)


class TestSplitCategorySelection(unittest.TestCase):
    def test_insan_erkek_slash(self):
        p, c = split_category_selection("insan/erkek")
        self.assertEqual(p, "insan")
        self.assertEqual(c, "erkek")
        self.assertEqual(canonical_path_from_parts(p, c), "insan/erkek")

    def test_insan_kadin_spaced(self):
        p, c = split_category_selection("insan / kadın")
        self.assertEqual(p, "insan")
        self.assertEqual(c, "kadın")

    def test_insan_yuz_spaced(self):
        p, c = split_category_selection("insan / yüz")
        self.assertEqual(p, "insan")
        self.assertEqual(c, "yüz")

    def test_multi_level(self):
        p, c = split_category_selection("a/b/c")
        self.assertEqual(p, "a")
        self.assertEqual(c, "b/c")
        self.assertEqual(canonical_path_from_parts(p, c), "a/b/c")

    def test_parent_only(self):
        p, c = split_category_selection("insan")
        self.assertEqual(p, "insan")
        self.assertEqual(c, "")

    def test_empty(self):
        self.assertEqual(split_category_selection(""), ("", ""))
        self.assertEqual(split_category_selection("   "), ("", ""))

    def test_strips_segment_whitespace(self):
        p, c = split_category_selection("  Animal Print  /  Leopard  ")
        self.assertEqual(p, "Animal Print")
        self.assertEqual(c, "Leopard")


class TestPathIdentityAndTags(unittest.TestCase):
    def test_path_identity_preserved(self):
        state = sync_selection_state("insan/erkek")
        self.assertEqual(state["category_path"], "insan/erkek")
        self.assertEqual(state["parent"], "insan")
        self.assertEqual(state["child"], "erkek")

    def test_tags_are_path_segments_only(self):
        state = sync_selection_state("insan/erkek")
        self.assertEqual(state["tags"], ["insan", "erkek"])
        state2 = sync_selection_state("a/b/c")
        self.assertEqual(state2["tags"], ["a", "b", "c"])

    def test_manual_tags_preserved_and_merged(self):
        state = sync_selection_state(
            "insan / erkek",
            existing_tags=["manuel", "özel"],
        )
        self.assertEqual(state["tags"], ["manuel", "özel", "insan", "erkek"])

    def test_duplicate_tags_casefold_not_duplicated(self):
        merged = merge_tags_preserve_manual(
            ["Insan", "extra"],
            ["insan", "erkek", "EXTRA"],
        )
        self.assertEqual(merged, ["Insan", "extra", "erkek"])

    def test_values_style_spaced_parent_empty_child(self):
        """parent 'insan / erkek' + empty child → path insan/erkek not 'insan / erkek/'."""
        state = sync_selection_state("insan / erkek", child_text="")
        self.assertEqual(state["parent"], "insan")
        self.assertEqual(state["child"], "erkek")
        self.assertEqual(state["category_path"], "insan/erkek")
        self.assertFalse(state["category_path"].endswith("/"))
        self.assertNotIn(" / ", state["category_path"])


class TestResolveAgainstOptions(unittest.TestCase):
    def test_prefers_known_casing(self):
        p, c = resolve_parts_against_options(
            "insan",
            "erkek",
            parents=["Insan", "Animal"],
            children_by_parent={"Insan": ["Erkek", "Kadın"]},
        )
        self.assertEqual(p, "Insan")
        self.assertEqual(c, "Erkek")

    def test_unchanged_without_options(self):
        p, c = resolve_parts_against_options("insan", "erkek")
        self.assertEqual((p, c), ("insan", "erkek"))


class TestSyncSelectionState(unittest.TestCase):
    def test_with_child_text_when_parent_plain(self):
        state = sync_selection_state(
            "insan",
            child_text="erkek",
            existing_tags=["x"],
        )
        self.assertEqual(state["parent"], "insan")
        self.assertEqual(state["child"], "erkek")
        self.assertEqual(state["category_path"], "insan/erkek")
        self.assertIn("x", state["tags"])
        self.assertIn("insan", state["tags"])
        self.assertIn("erkek", state["tags"])

    def test_path_qualified_wins_over_child_text(self):
        state = sync_selection_state(
            "insan / kadın",
            child_text="ignored",
        )
        self.assertEqual(state["child"], "kadın")


class TestFormValuesFromRecordPath(unittest.TestCase):
    """Inspector-style path still splits in form_values_from_record."""

    def test_form_values_splits_path(self):
        # form_values_from_record needs textile_terms etc. — exercise split locally
        # matching dialog prefill contract, then sync tags.
        path = "insan/erkek"
        parent, _, child = path.partition("/")
        parent, child = parent.strip(), child.strip()
        self.assertEqual(parent, "insan")
        self.assertEqual(child, "erkek")
        state = sync_selection_state(
            canonical_path_from_parts(parent, child),
            existing_tags=[],
        )
        self.assertEqual(state["category_path"], "insan/erkek")
        self.assertEqual(state["tags"], ["insan", "erkek"])

    def test_import_form_values_if_deps_available(self):
        try:
            from ui.result_metadata_dialog import form_values_from_record
        except Exception as exc:
            self.skipTest(f"result_metadata_dialog deps missing: {exc}")
            return

        class _R:
            debug = {"manual_category_path": "insan/erkek"}
            pattern_family = ""
            color_family = ""

        vals = form_values_from_record(_R(), None)
        # form_values uses overlay/debug path when parent empty —
        # current impl reads ov category_path / dbg manual_category_path
        # debug is on result.debug; manual_category_path is under debug in this stub
        # form_values looks at dbg.get("manual_category_path") — yes
        self.assertEqual(vals["parent"], "insan")
        self.assertEqual(vals["child"], "erkek")


class TestDialogValuesLogic(unittest.TestCase):
    """Simulate ResultMetadataDialog.values() category resolution without Qt."""

    def _values_like(self, parent_raw: str, child_raw: str, tags=None, options=None):
        opts = options or {}
        if "/" in parent_raw:
            parent, child = split_category_selection(parent_raw)
            if not child and child_raw:
                child = child_raw
        else:
            parent, child = parent_raw, child_raw
        parent, child = resolve_parts_against_options(
            parent,
            child,
            parents=list(opts.get("parents") or []),
            children_by_parent=opts.get("children_by_parent") or {},
        )
        path = canonical_path_from_parts(parent, child)
        tags_out = merge_tags_preserve_manual(
            list(tags or []), path_segment_tags(parent, child)
        )
        return {
            "parent": parent,
            "child": child,
            "category_path": path,
            "tags": tags_out,
        }

    def test_path_qualified_parent_empty_child(self):
        v = self._values_like("insan / erkek", "")
        self.assertEqual(v["parent"], "insan")
        self.assertEqual(v["child"], "erkek")
        self.assertEqual(v["category_path"], "insan/erkek")
        self.assertEqual(v["tags"], ["insan", "erkek"])

    def test_path_qualified_no_trailing_slash_mess(self):
        v = self._values_like("insan / erkek", "")
        self.assertNotEqual(v["category_path"], "insan / erkek/")
        self.assertNotEqual(v["category_path"], "insan / erkek")

    def test_plain_parent_and_child(self):
        v = self._values_like("insan", "erkek", tags=["manual"])
        self.assertEqual(v["category_path"], "insan/erkek")
        self.assertEqual(v["tags"], ["manual", "insan", "erkek"])

    def test_with_options_casing(self):
        opts = {
            "parents": ["insan"],
            "children_by_parent": {"insan": ["erkek", "kadın", "yüz"]},
        }
        v = self._values_like("insan / yüz", "", options=opts)
        self.assertEqual(v["parent"], "insan")
        self.assertEqual(v["child"], "yüz")
        self.assertEqual(v["category_path"], "insan/yüz")


@unittest.skipUnless(
    os.environ.get("CAT_SYNC_QT_DIALOG") == "1",
    "Qt dialog tests optional (set CAT_SYNC_QT_DIALOG=1 if PySide6 present)",
)
class TestResultMetadataDialogQt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_values_after_path_qualified_parent(self):
        from ui.result_metadata_dialog import ResultMetadataDialog

        dlg = ResultMetadataDialog(db_path="")
        dlg._options = {
            "parents": ["insan"],
            "children_by_parent": {"insan": ["erkek", "kadın"]},
            "family": [],
            "colors": [],
            "brands": [],
            "tags": [],
        }
        dlg.cmb_parent.set_choices([("insan", "insan")])
        dlg.cmb_parent.set_text("insan / erkek")
        dlg._apply_category_selection_sync(source="parent")
        vals = dlg.values()
        self.assertEqual(vals["parent"], "insan")
        self.assertEqual(vals["child"], "erkek")
        self.assertEqual(vals["category_path"], "insan/erkek")
        self.assertIn("insan", vals["tags"])
        self.assertIn("erkek", vals["tags"])


class TestRegressionImports(unittest.TestCase):
    def test_import_category_selection_sync(self):
        import core.category_selection_sync as m

        self.assertTrue(hasattr(m, "sync_selection_state"))

    def test_import_result_metadata_dialog_or_note(self):
        try:
            import ui.result_metadata_dialog as rmd  # noqa: F401
        except Exception as exc:
            # Expected on box without PySide6 / full core — helpers still tested.
            self.skipTest(f"result_metadata_dialog not importable: {exc}")

    def test_teach_text_search_wire_if_present(self):
        wire = Path("/workspace/vezir_audit/erkek_wire")
        if not (wire / "tests" / "test_teach_text_search_wire.py").exists():
            self.skipTest("test_teach_text_search_wire not on box")
        sys.path.insert(0, str(wire))
        try:
            import tests.test_teach_text_search_wire as tw  # noqa: F401
        except Exception as exc:
            self.skipTest(f"teach wire suite not importable: {exc}")
            return
        loader = unittest.defaultTestLoader.loadTestsFromModule(tw)
        result = unittest.TextTestRunner(verbosity=0).run(loader)
        self.assertTrue(result.wasSuccessful(), f"failures={result.failures} errors={result.errors}")

    def test_customer_memory_if_present(self):
        cm = Path("/workspace/vezir_audit/cm_tests/test_customer_memory.py")
        if not cm.exists():
            self.skipTest("test_customer_memory not on box")
        # May need cm package — try lightly
        cm_root = Path("/workspace/vezir_audit/cm")
        if cm_root.exists():
            sys.path.insert(0, str(cm_root))
        sys.path.insert(0, str(cm.parent))
        try:
            import test_customer_memory as tcm  # noqa: F401
        except Exception as exc:
            self.skipTest(f"customer_memory suite not importable: {exc}")


class TestPathSegmentTags(unittest.TestCase):
    def test_ordered_unique(self):
        self.assertEqual(path_segment_tags("insan", "erkek"), ["insan", "erkek"])
        self.assertEqual(path_segment_tags("a", "b/c"), ["a", "b", "c"])
        self.assertEqual(path_segment_tags("A/B", "b"), ["A", "B"])  # B/b casefold dedupe


if __name__ == "__main__":
    unittest.main()
