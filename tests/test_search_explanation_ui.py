"""Tests for core.search_explanation_ui — real-evidence-only why lines."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.search_explanation_ui import format_why_html, format_why_lines


def _result(**kwargs):
    debug = kwargs.pop("debug", {}) or {}
    defaults = dict(
        score=0.42,
        score_percent=42.0,
        same_pattern_family=False,
        same_animal_family=False,
        animal_print_type="",
        match_explanations=[],
        cluster_reason="",
        debug=debug,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class FormatWhyLinesTests(unittest.TestCase):
    def test_query_meaning_customer_concept_colors(self):
        r = _result(
            debug={
                "query_meaning": {
                    "customer": "Ünal Tekstil",
                    "concept_core": "leopar",
                    "colors": ["kırmızı"],
                }
            }
        )
        lines = format_why_lines(r)
        self.assertIn("Kavram: leopar", lines)
        self.assertIn("Renk: kırmızı", lines)
        self.assertIn("Müşteri: Ünal Tekstil", lines)

    def test_search_reason_present_included(self):
        r = _result(debug={"search_reason": "CONCEPT LEOPAR + KIRMIZI"})
        lines = format_why_lines(r)
        self.assertIn("CONCEPT LEOPAR + KIRMIZI", lines)

    def test_search_reason_absent_not_invented(self):
        r = _result(debug={"query_meaning": {"concept_core": "leopar"}})
        lines = format_why_lines(r)
        joined = " | ".join(lines)
        self.assertNotIn("Benzerlik skoruyla", joined)
        self.assertNotIn("VISUAL SIMILAR", joined)
        self.assertEqual(lines, ["Kavram: leopar"])

    def test_fake_fallback_string_suppressed(self):
        r = _result(debug={"search_reason": "Benzerlik skoruyla sonuç geldi"})
        self.assertEqual(format_why_lines(r), [])

    def test_concept_evidence_and_customer_soft_bonus(self):
        r = _result(
            debug={
                "concept_evidence": {
                    "canonical": "leopar",
                    "aligned": True,
                    "applied": True,
                    "delta": 0.04,
                    "parts": {"scale": {"match": True}},
                    "customer_soft_bonus": 0.03,
                }
            }
        )
        lines = format_why_lines(r)
        self.assertTrue(any("Kavram kanıtı" in x for x in lines))
        self.assertTrue(any("Müşteri soft bonus" in x for x in lines))

    def test_color_hits_bullet(self):
        r = _result(
            debug={
                "color_evidence_score": {
                    "hits": 1,
                    "want": ["red"],
                    "have": ["red"],
                }
            }
        )
        lines = format_why_lines(r)
        self.assertTrue(any("Renk kanıtı" in x for x in lines))
        self.assertTrue(any("red" in x for x in lines))

    def test_color_no_hits_no_bullet(self):
        r = _result(
            debug={
                "color_evidence_score": {
                    "hits": 0,
                    "want": ["red"],
                    "misses": 1,
                }
            }
        )
        self.assertEqual(format_why_lines(r), [])

    def test_empty_debug_empty_list(self):
        self.assertEqual(format_why_lines(_result(debug={})), [])
        self.assertEqual(format_why_lines(_result(debug=None)), [])
        self.assertEqual(format_why_lines(None), [])

    def test_visual_verdict_only_when_real(self):
        r = _result(debug={"visual_verdict": "STRONG_MATCH"})
        lines = format_why_lines(r)
        self.assertIn("Görsel eşleşme: STRONG_MATCH", lines)

        r2 = _result(debug={})
        self.assertFalse(any("Görsel eşleşme" in x for x in format_why_lines(r2)))

    def test_same_pattern_family_and_animal(self):
        r = _result(
            same_pattern_family=True,
            same_animal_family=True,
            animal_print_type="leopard",
            debug={},
        )
        lines = format_why_lines(r)
        self.assertIn("Aynı desen ailesi", lines)
        self.assertIn("Hayvan deseni: leopard", lines)

    def test_html_empty_when_no_evidence(self):
        self.assertEqual(format_why_html(_result(debug={})), "")

    def test_html_bullets_when_evidence(self):
        r = _result(debug={"query_meaning": {"concept_core": "leopar"}})
        html = format_why_html(r)
        self.assertIn("• Kavram: leopar", html)

    def test_helper_does_not_mutate_scores(self):
        r = _result(
            score=0.55,
            score_percent=55.0,
            debug={
                "query_meaning": {"concept_core": "leopar", "colors": ["kırmızı"]},
                "search_reason": "CONCEPT LEOPAR",
                "concept_evidence": {
                    "canonical": "leopar",
                    "aligned": True,
                    "customer_soft_bonus": 0.03,
                },
                "color_evidence_score": {"hits": 1, "want": ["red"]},
            },
        )
        before = copy.deepcopy(r.__dict__)
        format_why_lines(r)
        format_why_html(r)
        self.assertEqual(r.score, before["score"])
        self.assertEqual(r.score_percent, before["score_percent"])
        self.assertEqual(r.debug, before["debug"])


class ResultCardChipsRemovedTests(unittest.TestCase):
    def test_result_card_source_has_no_reason_chips_ui(self):
        root = Path(__file__).resolve().parents[1]
        src = (root / "ui" / "result_card.py").read_text(encoding="utf-8")
        self.assertNotIn("reason_chips", src)
        self.assertNotIn("Icon reasons", src)
        self.assertNotIn("chips_row", src)


class InspectorWireSmokeTests(unittest.TestCase):
    def test_inspector_imports_helper_and_collapsible_group(self):
        root = Path(__file__).resolve().parents[1]
        src = (root / "ui" / "inspector_panel.py").read_text(encoding="utf-8")
        self.assertIn("from core.search_explanation_ui import", src)
        self.assertIn("format_why_lines", src)
        self.assertIn("setCheckable(True)", src)
        self.assertIn("setChecked(False)", src)
        # Neden box must use helper, not pattern_explanation text formatter.
        self.assertNotIn(
            "from core.pattern_explanation import build_result_explanation, format_explanation_text",
            src,
        )


if __name__ == "__main__":
    unittest.main()
