"""Legacy dual-preview suite — see test_detail_preview_second_level_hover.py."""
from __future__ import annotations

import unittest


class TestDualNormalHoverPreviewRetired(unittest.TestCase):
    def test_detail_magnify_overlay_exists(self):
        import ui.fixed_hover_preview as mod

        self.assertTrue(hasattr(mod, "DetailPreviewHoverOverlay"))
