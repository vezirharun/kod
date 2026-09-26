"""Process-wide OpenCV runtime hardening.

Windows STATUS_HEAP_CORRUPTION (0xC0000374) has been attributed to cv2.pyd
during V3 heavy feature extract after DINOv2/OpenCLIP load (WinDbg:
HEAP_CORRUPTION_ACTIONABLE_BufferOverrun -> cv2!cv::Algorithm::empty /
PyInit_cv2 -> RtlFreeHeap). OpenCL + multi-thread OpenCV fighting torch/native
allocators is the usual trigger.

Call harden_cv2_runtime() before any heavy native work (ideally before the
first cv2 import via OPENCV_OPENCL_DEVICE=disabled). Idempotent and safe if
cv2 is absent.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

_lock = threading.Lock()
_applied = False
_cv2_ok = False
_log = logging.getLogger(__name__)


def _disable_opencl_env() -> None:
    """Prefer env disable before OpenCL context spins up on first cv2 import."""
    # Empty / disabled — OpenCV skips OpenCL device init on many builds.
    os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
    # Do not force a runtime path; leave unset keys alone except the device flag.
    if "OPENCV_OPENCL_RUNTIME" not in os.environ:
        os.environ["OPENCV_OPENCL_RUNTIME"] = ""


def harden_cv2_runtime(*, force: bool = False) -> bool:
    """Set OpenCV threads=1 and disable OpenCL. Idempotent. Returns True if cv2 ok."""
    global _applied, _cv2_ok

    _disable_opencl_env()

    with _lock:
        if _applied and not force:
            return _cv2_ok
        try:
            import cv2
        except Exception as exc:  # ImportError or broken wheel
            _log.debug("cv2 unavailable for harden: %s", exc)
            _applied = True
            _cv2_ok = False
            return False

        try:
            cv2.setNumThreads(1)
        except Exception as exc:
            _log.debug("cv2.setNumThreads failed: %s", exc)

        try:
            ocl = getattr(cv2, "ocl", None)
            if ocl is not None and hasattr(ocl, "setUseOpenCL"):
                ocl.setUseOpenCL(False)
        except Exception as exc:
            _log.debug("cv2.ocl.setUseOpenCL failed: %s", exc)

        _applied = True
        _cv2_ok = True
        return True


def cv2_runtime_status() -> dict[str, Any]:
    """Diagnostic snapshot for health/debug (does not require prior harden)."""
    status: dict[str, Any] = {
        "applied": _applied,
        "cv2_ok": _cv2_ok,
        "OPENCV_OPENCL_DEVICE": os.environ.get("OPENCV_OPENCL_DEVICE"),
    }
    if not _cv2_ok:
        return status
    try:
        import cv2

        status["version"] = getattr(cv2, "__version__", None)
        try:
            status["num_threads"] = int(cv2.getNumThreads())
        except Exception:
            status["num_threads"] = None
        try:
            ocl = getattr(cv2, "ocl", None)
            status["ocl_use"] = bool(ocl.useOpenCL()) if ocl is not None else None
            status["ocl_have"] = bool(ocl.haveOpenCL()) if ocl is not None else None
        except Exception:
            status["ocl_use"] = None
            status["ocl_have"] = None
    except Exception as exc:
        status["error"] = str(exc)
    return status
