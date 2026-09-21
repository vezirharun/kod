"""Result/list cards must not render AI style/color/brand chips."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestResultCardNoAiChips(unittest.TestCase):
    def test_result_card_source_omits_ai_style_color_brand_chips(self):
        src = (ROOT / "ui" / "result_card.py").read_text(encoding="utf-8")
        self.assertNotIn("family_badge_for_result", src)
        self.assertNotIn("independent_feature_badge", src)
        self.assertNotIn("color_badge_text", src)
        self.assertNotIn("brand_badge_text", src)
        self.assertNotIn("Mermer", src)
        self.assertNotIn("Krem", src)

    def test_inspector_still_has_style_color_display(self):
        insp = (ROOT / "ui" / "inspector_panel.py").read_text(encoding="utf-8")
        self.assertIn("color_badge_text", insp)
        self.assertIn("family_badge_for_result", insp)


if __name__ == "__main__":
    unittest.main()
