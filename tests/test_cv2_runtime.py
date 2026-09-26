"""OpenCV runtime harden — threads=1, OpenCL off (heap corruption guard)."""

from __future__ import annotations

import os


def test_harden_cv2_runtime_sets_threads_and_disables_opencl():
    from core.cv2_runtime import cv2_runtime_status, harden_cv2_runtime

    ok = harden_cv2_runtime(force=True)
    assert ok is True
    assert os.environ.get("OPENCV_OPENCL_DEVICE") == "disabled"

    import cv2

    assert cv2.getNumThreads() == 1
    if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "useOpenCL"):
        assert cv2.ocl.useOpenCL() is False

    status = cv2_runtime_status()
    assert status["applied"] is True
    assert status["cv2_ok"] is True
    assert status["num_threads"] == 1
    assert status.get("ocl_use") in (False, None)


def test_harden_cv2_runtime_is_idempotent():
    from core.cv2_runtime import harden_cv2_runtime

    assert harden_cv2_runtime(force=True) is True
    assert harden_cv2_runtime() is True
    assert harden_cv2_runtime() is True


def test_feature_extractor_import_applies_harden():
    """Importing FeatureExtractor path must leave cv2 in hardened state."""
    from core.cv2_runtime import harden_cv2_runtime

    harden_cv2_runtime(force=True)

    from core.feature_extractor import HAS_CV2, FeatureExtractor

    assert HAS_CV2 is True
    import cv2

    assert cv2.getNumThreads() == 1
    # Lightweight extract without AI — exercises cv2 color/texture path.
    import numpy as np

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 160)
    fe = FeatureExtractor(use_ai=False, use_gpu=False, fast_hash_only=False)
    feats = fe.extract_from_array(img, deep_analysis=False)
    assert feats.color_hist or feats.texture_features or feats.phash
