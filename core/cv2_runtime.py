"""Process-wide OpenCV policy for Windows heap-corruption avoidance.

Windows STATUS_HEAP_CORRUPTION (0xC0000374) has been attributed to cv2.pyd
during V3 heavy feature extract after DINOv2/OpenCLIP load (WinDbg /
minidump: ntdll fault + cv2.pyd mapped; HEAP_CORRUPTION -> RtlFreeHeap).

Policy (post-d192826):
  * App startup only prepares env (OpenCL disabled) — does NOT import cv2.
  * V3 / feature_extractor / color / texture paths must NOT import cv2;
    they use numpy/PIL helpers in core.safe_image_ops.
  * Face modules may still import cv2 lazily; call harden_cv2_runtime()
    immediately before that import so threads=1 and OpenCL stay off.

Idempotent and safe if cv2 is absent.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

_lock = threading.Lock()
_env_prepared = False
_applied = False
_cv2_ok = False
_log = logging.getLogger(__name__)


def prepare_cv2_env() -> None:
    """Set OPENCV_* env BEFORE any cv2 import. Does not load cv2.pyd."""
    global _env_prepared
    os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
    if "OPENCV_OPENCL_RUNTIME" not in os.environ:
        os.environ["OPENCV_OPENCL_RUNTIME"] = ""
    # Cap internal parallelism even if something imports cv2 later.
    os.environ.setdefault("OPENCV_FORCED_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    _env_prepared = True


def _disable_opencl_env() -> None:
    prepare_cv2_env()


def harden_cv2_runtime(*, force: bool = False, load_cv2: bool = True) -> bool:
    """Disable OpenCL + (optionally) import cv2 and setNumThreads(1).

    Returns True only when cv2 was loaded and configured.
    Call with load_cv2=False from app startup; face paths use default True.
    """
    global _applied, _cv2_ok

    prepare_cv2_env()

    if not load_cv2:
        return False

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
    """Diagnostic snapshot for health/debug (does not force-import cv2)."""
    import sys

    status: dict[str, Any] = {
        "env_prepared": _env_prepared,
        "applied": _applied,
        "cv2_ok": _cv2_ok,
        "OPENCV_OPENCL_DEVICE": os.environ.get("OPENCV_OPENCL_DEVICE"),
        "cv2_in_sys_modules": "cv2" in sys.modules,
    }
    if not _cv2_ok and "cv2" not in sys.modules:
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
