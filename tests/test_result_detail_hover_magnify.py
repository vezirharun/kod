"""Legacy overlay suite retired — list hover uses full detail canvas.

See tests/test_result_hover_full_detail_preview.py.
"""
from __future__ import annotations

import unittest


class TestResultDetailHoverOverlayRetired(unittest.TestCase):
    def test_floating_overlay_removed(self):
        import ui.fixed_hover_preview as mod

        self.assertFalse(hasattr(mod, "ResultDetailPreviewOverlay"))
