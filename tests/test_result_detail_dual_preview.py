"""Legacy dual overlay suite retired — no floating large overlay.

See tests/test_result_hover_full_detail_preview.py.
"""
from __future__ import annotations

import unittest


class TestDualNormalHoverPreviewRetired(unittest.TestCase):
    def test_floating_overlay_removed(self):
        import ui.fixed_hover_preview as mod

        self.assertFalse(hasattr(mod, "ResultDetailPreviewOverlay"))
        self.assertFalse(hasattr(mod.FixedHoverPreviewPanel, "_show_overlay"))
