"""Small pure helpers for QThread single-flight lifecycle (testable)."""

from __future__ import annotations

from typing import Any


def should_defer_new_worker(is_running: bool) -> bool:
    """True when a new worker must wait until the current one finishes."""
    return bool(is_running)


def qobject_is_alive(obj: Any) -> bool:
    """Return False if *obj* is None or its C++ QObject is already deleted.

    Must not log or re-raise libshiboken RuntimeError (zombie wrappers).
    """
    if obj is None:
        return False
    try:
        from shiboken6 import isValid

        return bool(isValid(obj))
    except Exception:
        pass
    try:
        # Touch the C++ object without assuming QThread.
        obj.objectName()
        return True
    except RuntimeError:
        return False
    except Exception:
        return False


def qthread_is_running(obj: Any) -> bool:
    """True only if the QThread wrapper is alive and isRunning()."""
    if not qobject_is_alive(obj):
        return False
    try:
        return bool(obj.isRunning())
    except RuntimeError:
        return False
    except Exception:
        return False
