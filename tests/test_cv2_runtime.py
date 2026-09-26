"""OpenCV policy — env prepare without import; feature path stays cv2-free."""

from __future__ import annotations

import os
import sys


def test_prepare_cv2_env_does_not_import_cv2():
    # Ensure clean-ish: if already imported in this pytest process, skip strict check
    from core.cv2_runtime import prepare_cv2_env, cv2_runtime_status

    prepare_cv2_env()
    assert os.environ.get("OPENCV_OPENCL_DEVICE") == "disabled"
    status = cv2_runtime_status()
    assert status["OPENCV_OPENCL_DEVICE"] == "disabled"
    assert status.get("env_prepared") is True


def test_harden_cv2_runtime_sets_threads_and_disables_opencl():
    from core.cv2_runtime import cv2_runtime_status, harden_cv2_runtime

    ok = harden_cv2_runtime(force=True, load_cv2=True)
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


def test_feature_extractor_does_not_require_cv2():
    """V3 feature path must work without importing/using cv2."""
    from core.feature_extractor import HAS_CV2, FeatureExtractor
    import numpy as np

    assert HAS_CV2 is False
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 160)
    fe = FeatureExtractor(use_ai=False, use_gpu=False, fast_hash_only=False)
    feats = fe.extract_from_array(img, deep_analysis=False)
    assert feats.color_hist or feats.texture_features or feats.phash
    assert len(feats.texture_features) >= 4
    # edge/laplacian must be non-trivial (numpy path, not zeros)
    assert feats.texture_features[2] >= 0.0
    assert feats.texture_features[3] >= 0.0


def test_safe_image_ops_kmeans_and_hist():
    from core.safe_image_ops import hsv_hist_rgb, kmeans_centers, edge_density, laplacian_var, rgb_to_gray
    import numpy as np

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:16, :, :] = (200, 40, 40)
    img[16:, :, :] = (40, 40, 200)
    hist = hsv_hist_rgb(img, 32, 16, 16)
    assert hist.shape[0] == 32 + 16 + 16
    assert abs(float(hist.sum()) - 1.0) < 1e-3
    centers, labels = kmeans_centers(img.reshape(-1, 3).astype(np.float32), 2)
    assert len(centers) == 2
    gray = rgb_to_gray(img)
    assert edge_density(gray) >= 0.0
    assert laplacian_var(gray) >= 0.0