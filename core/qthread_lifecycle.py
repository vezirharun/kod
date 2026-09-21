"""Small pure helpers for QThread single-flight lifecycle (testable)."""

from __future__ import annotations


def should_defer_new_worker(is_running: bool) -> bool:
    """True when a new worker must wait until the current one finishes."""
    return bool(is_running)
