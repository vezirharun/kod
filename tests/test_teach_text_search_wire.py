"""Unit tests for teach → text-search wire (minimal, stubbed product tree)."""

from __future__ import annotations

import sqlite3
import sys
import types
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_stubs() -> None:
    """Minimal stubs so text_index.build_text_search_blob imports cleanly."""

    def _ensure(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    def normalize_turkish(s: str) -> str:
        table = {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
            "Ç": "c",
            "Ğ": "g",
            "İ": "i",
            "I": "i",
            "Ö": "o",
            "Ş": "s",
            "Ü": "u",
        }
        out = "".join(table.get(ch, ch) for ch in str(s or ""))
        return out.casefold()

    tt = _ensure("core.textile_terms")
    tt.normalize_turkish = normalize_turkish  # type: ignore[attr-defined]
    tt.tokenize = lambda s: set(normalize_turkish(s).split())  # type: ignore[attr-defined]
    tt.expand_query_terms = lambda q, include_semantic=False: [q]  # type: ignore[attr-defined]
    tt.families_conflict = lambda a, b: False  # type: ignore[attr-defined]
    tt.query_family_hints = lambda q, include_semantic=False: {}  # type: ignore[attr-defined]

    tp = _ensure("core.texture_profile")

    class TextureProfile:
        def __init__(self) -> None:
            self.pattern_family = ""
            self.animal_print_type = ""
            self.texture_family = ""
            self.pattern_subtype = ""
            self.color_family = ""

        @classmethod
        def from_dict(cls, d: dict[str, Any] | None) -> "TextureProfile":
            o = cls()
            src = d or {}
            o.pattern_family = str(src.get("pattern_family") or "")
            o.animal_print_type = str(src.get("animal_print_type") or "")
            o.texture_family = str(src.get("texture_family") or "")
            o.pattern_subtype = str(src.get("pattern_subtype") or "")
            o.color_family = str(src.get("color_family") or "")
            return o

    tp.TextureProfile = TextureProfile  # type: ignore[attr-defined]

    ct = _ensure("core.category_tree")
    ct.is_category_descendant = lambda a, b: False  # type: ignore[attr-defined]

    class _CatMatch:
        category_path = ""
        aliases: list[str] = []
        primary_family = ""
        primary_subtype = ""

    ct.resolve_category_query = lambda q: _CatMatch()  # type: ignore[attr-defined]

    for name, fn in (
        ("core.semantic_tags", "flatten_semantic_tags"),
        ("core.pattern_dna", "flatten_pattern_dna"),
        ("core.color_index", "flatten_color_index"),
        ("core.auto_tags", "flatten_auto_tags"),
    ):
        mod = _ensure(name)
        setattr(mod, fn, lambda *_a, **_k: [])

    ba = _ensure("core.brand_aliases")
    ba.enrich_ocr_text = lambda t: t  # type: ignore[attr-defined]


_install_stubs()

from core.teach_search_wire import (  # noqa: E402
    category_path_search_tokens,
    category_tokens_for_blob,
    keep_taught_evidence_on_gender,
    run_category_path_label_search,
)
from core.text_index import build_text_search_blob  # noqa: E402
from core.textile_terms import normalize_turkish  # noqa: E402


class CategoryPathTokensTests(unittest.TestCase):
    def test_slash_segments(self) -> None:
        toks = category_path_search_tokens("insan/erkek")
        joined = " ".join(toks).casefold()
        self.assertIn("insan", joined)
        self.assertIn("erkek", joined)
        self.assertTrue(any("insan erkek" == t for t in toks) or "insan erkek" in joined)

    def test_empty(self) -> None:
        self.assertEqual(category_path_search_tokens(""), [])
        self.assertEqual(category_path_search_tokens("  /  "), [])


class BuildTextSearchBlobTests(unittest.TestCase):
    def test_tm_path_segments_without_kwargs(self) -> None:
        blob = build_text_search_blob(
            filename="shot.jpg",
            texture_map={"manual_category_path": "insan/erkek"},
        )
        n = normalize_turkish(blob)
        self.assertIn("erkek", n)
        self.assertIn("insan", n)

    def test_category_path_kwarg_contains_token(self) -> None:
        # category_learning-style call: explicit category_path kwarg
        blob = build_text_search_blob(
            filename="a.jpg",
            texture_map={},
            category_path="insan/erkek",
            category_aliases=["insan", "erkek"],
            pattern_type="",
        )
        n = normalize_turkish(blob)
        self.assertIn("erkek", n)
        self.assertTrue(bool(n.strip()), "blob must be non-empty when path set")

    def test_tokens_helper_matches_blob(self) -> None:
        toks = category_tokens_for_blob("foo/bar", ["baz"])
        self.assertIn("bar", toks)
        self.assertIn("baz", toks)


class CategoryPathLabelSearchTests(unittest.TestCase):
    def test_sqlite_finds_manual_category_path(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                status TEXT,
                physical_preview_ready INTEGER DEFAULT 0,
                manual_category_path TEXT DEFAULT '',
                category_path TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            "INSERT INTO files(id, status, physical_preview_ready, "
            "manual_category_path, category_path) VALUES (7, 'indexed', 1, ?, ?)",
            ("insan/erkek", "insan/erkek"),
        )
        conn.execute(
            "INSERT INTO files(id, status, physical_preview_ready, "
            "manual_category_path, category_path) VALUES (8, 'indexed', 1, ?, ?)",
            ("diger/sey", "diger/sey"),
        )
        conn.commit()
        hits = run_category_path_label_search(
            conn, ["erkek"], 50, normalize=normalize_turkish
        )
        self.assertIn(7, hits)
        self.assertNotIn(8, hits)


class GenderKeepHelperTests(unittest.TestCase):
    def test_learned_exact_keeps(self) -> None:
        self.assertTrue(
            keep_taught_evidence_on_gender(
                {"learned_concept_exact": True}, ["erkek"]
            )
        )

    def test_user_taught_keeps(self) -> None:
        self.assertTrue(
            keep_taught_evidence_on_gender(
                {"user_taught_positive": True}, ["anything"]
            )
        )

    def test_manual_category_token_keeps(self) -> None:
        self.assertTrue(
            keep_taught_evidence_on_gender(
                {"manual_category_path": "insan/erkek"},
                ["erkek"],
            )
        )

    def test_no_evidence_demotes(self) -> None:
        self.assertFalse(
            keep_taught_evidence_on_gender(
                {"face_gender_match": False},
                ["erkek"],
            )
        )

    def test_unrelated_path_no_keep(self) -> None:
        self.assertFalse(
            keep_taught_evidence_on_gender(
                {"category_path": "bitki/cicek"},
                ["erkek"],
            )
        )


if __name__ == "__main__":
    unittest.main()
