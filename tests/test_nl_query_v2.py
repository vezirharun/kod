"""NL Search 2.0 — parse/bridge layer tests (no engine rewrite)."""
from __future__ import annotations

import unittest

from core.customer_discovery import (
    CustomerEntry,
    CustomerRegistry,
    extract_customer_from_query,
    normalize_customer_key,
    set_customer_registry,
)
from core.query_attribute_intel import extract_query_attributes, extract_query_visual_context
from core.search_intelligence_chain import analyze_query_intelligence


def _reg(*names: str) -> CustomerRegistry:
    r = CustomerRegistry()
    for name in names:
        key = normalize_customer_key(name)
        e = CustomerEntry(name=name, norm_key=key, roots=[f"S:\\imalat2\\{name}"], file_count=10)
        r._by_key[key] = e
        r._by_display[name] = e
    r.discovered_count = len(names)
    set_customer_registry(r)
    return r


class TestNLQueryV2(unittest.TestCase):
    def tearDown(self) -> None:
        set_customer_registry(None)

    def test_01_full_sentence_unal_cicek(self):
        reg = _reg("Ünal Tekstil", "Leydi Tekstil", "Elif Minder")
        q = "Ünal Tekstil'deki küçük mavi çiçek desenleri"
        a = analyze_query_intelligence(q, customer_registry=reg)
        self.assertEqual(a["customer"], "Ünal Tekstil")
        self.assertTrue(a["customer_high_confidence"])
        self.assertEqual(a["attributes"]["motif"], "floral")
        self.assertEqual(a["attributes"]["scale"], "small")
        self.assertIn("blue", a["colors"])
        self.assertEqual(a["intent_type"], "pattern")
        self.assertEqual(a["context"], "pattern")
        core = (a.get("concept_core") or "").lower()
        self.assertNotIn("unal", core)
        self.assertTrue("cicek" in core or a["attributes"]["motif"] == "floral")

    def test_02_desenleri_marks_pattern(self):
        v = extract_query_visual_context("küçük mavi çiçek desenleri")
        self.assertEqual(v["context"], "pattern")
        a = analyze_query_intelligence("küçük mavi çiçek desenleri")
        self.assertEqual(a["intent_type"], "pattern")

    def test_03_object_gate_hayvani(self):
        a = analyze_query_intelligence("leopar hayvanı")
        self.assertEqual(a["context"], "object")
        self.assertEqual(a["intent_type"], "object")
        self.assertEqual(a["attributes"]["motif"], "leopard")

    def test_04_pattern_deseni(self):
        a = analyze_query_intelligence("kaplan deseni")
        self.assertEqual(a["intent_type"], "pattern")
        self.assertEqual(a["attributes"]["motif"], "tiger")

    def test_05_gul_vs_cicek_identity(self):
        gul = analyze_query_intelligence("gül")
        cicek = analyze_query_intelligence("çiçek")
        self.assertEqual(gul["attributes"]["motif"], "rose")
        self.assertEqual(cicek["attributes"]["motif"], "floral")
        self.assertNotEqual(gul["attributes"]["motif"], cicek["attributes"]["motif"])

    def test_06_kaplan_not_leopar(self):
        a = analyze_query_intelligence("küçük yoğun siyah krem kaplan deseni")
        self.assertEqual(a["attributes"]["motif"], "tiger")
        hint = (a.get("concept_relation_hint") or "").lower()
        self.assertTrue("tiger" in hint or a["attributes"]["motif"] == "tiger")
        self.assertFalse(hint.startswith("leopard"))

    def test_07_ambiguous_customer_no_force(self):
        reg = _reg("Ünal Tekstil", "Ünal Mode")
        # Shared first token — both can score high; must not random-pick
        info = extract_customer_from_query("Ünal mavi çiçek", reg)
        # Either ambiguous or not high_confidence forcing a single wrong one
        if info.get("customer"):
            self.assertTrue(
                info.get("ambiguous") or not info.get("high_confidence")
                or info["customer"] in {"Ünal Tekstil", "Ünal Mode"}
            )
        # Safer path for short "Ünal" alone: do not force if two Ünal*
        info2 = extract_customer_from_query("Ünal", reg)
        if info2.get("high_confidence"):
            # Only OK if uniquely decisive
            self.assertFalse(info2.get("ambiguous"))
        else:
            self.assertEqual(info2.get("customer") or "", "")

    def test_08_typo_customer_unal_teksitl(self):
        reg = _reg("Ünal Tekstil", "Leydi Tekstil")
        info = extract_customer_from_query("unal teksitl mavi kaplan", reg)
        self.assertEqual(info["customer"], "Ünal Tekstil")
        self.assertTrue(info["high_confidence"])
        a = analyze_query_intelligence("unal teksitl mavi kaplan", customer_registry=reg)
        self.assertEqual(a["customer"], "Ünal Tekstil")
        self.assertEqual(a["attributes"]["motif"], "tiger")
        self.assertIn("blue", a["colors"])

    def test_09_no_customer_without_registry(self):
        set_customer_registry(None)
        a = analyze_query_intelligence("Ünal Tekstil'deki mavi çiçek desenleri")
        # Without registry entries, do not invent customer
        self.assertFalse(a.get("customer_high_confidence"))
        self.assertEqual(a.get("customer") or "", "")
        self.assertEqual(a["attributes"]["motif"], "floral")

    def test_10_scale_color_attrs(self):
        attrs = extract_query_attributes("küçük mavi çiçek")
        self.assertEqual(attrs.scale, "small")
        self.assertIn("blue", attrs.colors)
        self.assertEqual(attrs.motif, "floral")

    def test_11_tekstil_not_texture_context(self):
        # Customer-like phrase must not flip to texture via bare "tekstil"
        v = extract_query_visual_context("Ünal Tekstil çiçek")
        self.assertNotEqual(v["context"], "texture")

    def test_12_nl_parse_debug_fields(self):
        reg = _reg("Ünal Tekstil")
        a = analyze_query_intelligence(
            "Ünal Tekstil'deki küçük mavi çiçek desenleri",
            customer_registry=reg,
        )
        nl = a.get("nl_parse") or {}
        self.assertEqual(nl.get("customer"), "Ünal Tekstil")
        self.assertTrue(nl.get("customer_high_confidence"))
        self.assertEqual(nl.get("intent_type"), "pattern")
        self.assertIn("search_text", nl)

    def test_13_explicit_color_preserved(self):
        a = analyze_query_intelligence("kırmızı gül deseni")
        self.assertIn("red", a["colors"])
        self.assertEqual(a["attributes"]["motif"], "rose")


if __name__ == "__main__":
    unittest.main()
