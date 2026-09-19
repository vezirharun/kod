"""Dosya bazlı index aşamaları — kuyruk durumu."""

from __future__ import annotations

PENDING_LIGHT = "pending_light"
LIGHT_DONE = "light_done"
PENDING_FULL = "pending_full"
FULL_DONE = "full_done"
FAILED = "failed"
SKIPPED_HEAVY_FORMAT = "skipped_heavy_format"
NEEDS_REVIEW = "needs_review"
NEEDS_MEDIUM_PREVIEW = "needs_medium_preview"

ALL_STAGES = frozenset(
    {
        PENDING_LIGHT,
        LIGHT_DONE,
        PENDING_FULL,
        FULL_DONE,
        FAILED,
        SKIPPED_HEAVY_FORMAT,
        NEEDS_REVIEW,
        NEEDS_MEDIUM_PREVIEW,
    }
)

HEAVY_FORMAT_EXTENSIONS = frozenset(
    {
        ".psd",
        ".ai",
        ".eps",
        ".pdf",
        ".cdr",
    }
)
