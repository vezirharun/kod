"""Resource lifecycle: thread/queue leak + PSD safety (index crash root cause)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.index_v3 import safe_decode
from core.index_v3.safe_decode import run_with_timeout
from core.preview_cache import FeaturePreviewCache


def test_run_with_timeout_reuses_pool_not_per_call_executor():
    """Dosya başına ThreadPoolExecutor açılmamalı (thread birikimi)."""
    before = threading.active_count()
    for _ in range(40):
        assert run_with_timeout(lambda: 1, timeout_sec=2.0, retries=0) == 1
    # Küçük salınım OK; 40 yeni thread olmamalı.
    after = threading.active_count()
    assert after - before < 8, f"thread growth {before}→{after}"


def test_timeout_abandon_resets_pool_cap():
    safe_decode._reset_timeout_pool(reason="test")
    with safe_decode._POOL_LOCK:
        safe_decode._ABANDONED = 0

    def _hang():
        time.sleep(5.0)
        return 1

    with pytest.raises(TimeoutError):
        run_with_timeout(_hang, timeout_sec=0.2, retries=0)
    with safe_decode._POOL_LOCK:
        assert safe_decode._ABANDONED >= 1


def test_run_with_timeout_does_not_stack_identical_timeouts():
    """retries=2 must not wait ~3×timeout when hard-timeout (jumbo TIFF lock)."""
    safe_decode._reset_timeout_pool(reason="test_no_stack")
    calls = {"n": 0}

    def _hang():
        calls["n"] += 1
        time.sleep(5.0)
        return 1

    t0 = time.monotonic()
    with pytest.raises(TimeoutError) as ei:
        run_with_timeout(_hang, timeout_sec=0.35, retries=2, retry_on_timeout=False)
    elapsed = time.monotonic() - t0
    assert "v3_decode_timeout" in str(ei.value)
    assert calls["n"] == 1
    assert elapsed < 1.5, f"stacked timeouts elapsed={elapsed:.2f}s"
    safe_decode._reset_timeout_pool(reason="test_no_stack_cleanup")


def test_run_with_timeout_retry_on_timeout_opt_in():
    safe_decode._reset_timeout_pool(reason="test_retry_opt_in")
    calls = {"n": 0}

    def _hang():
        calls["n"] += 1
        time.sleep(5.0)
        return 1

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        run_with_timeout(_hang, timeout_sec=0.25, retries=1, retry_on_timeout=True)
    elapsed = time.monotonic() - t0
    assert calls["n"] == 2
    assert elapsed >= 0.45
    assert elapsed < 2.5
    safe_decode._reset_timeout_pool(reason="test_retry_opt_in_cleanup")


def test_run_in_process_closes_queue_feeder(monkeypatch):
    """mp.Queue close/join_thread çağrılmalı (feeder thread leak önlemi)."""
    closed = {"close": 0, "join": 0}

    class _Q:
        def put(self, *_a, **_k):
            pass

        def get(self, timeout=None):
            return ("ok", 42)

        def close(self):
            closed["close"] += 1

        def join_thread(self):
            closed["join"] += 1

    class _Proc:
        def __init__(self, *a, **k):
            self.exitcode = 0

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

        def terminate(self):
            pass

        def close(self):
            pass

    class _Ctx:
        def Queue(self, *a, **k):
            return _Q()

        def Process(self, *a, **k):
            return _Proc()

    monkeypatch.setattr(
        safe_decode.mp, "get_context", lambda _name: _Ctx()
    )
    assert safe_decode._run_in_process(lambda: 1, timeout_sec=1.0) == 42
    assert closed["close"] == 1
    assert closed["join"] == 1


def test_light_isolated_closes_queue(monkeypatch):
    from core.index_v3 import real_processor as rp
    from core.light_asset_extractor import LightAssetResult
    from core.settings import AppSettings

    closed = {"close": 0, "join": 0, "proc_close": 0}

    class _Q:
        def get(self, timeout=None):
            return (
                "ok",
                {
                    "success": True,
                    "error": "",
                    "thumbnail_path": "",
                    "thumbnail_width": 0,
                    "thumbnail_height": 0,
                    "thumbnail_status": "",
                    "feature_preview_path": "",
                    "metadata": {},
                    "metadata_status": "",
                    "visual_status": "",
                    "visual_renderer": "",
                },
            )

        def close(self):
            closed["close"] += 1

        def join_thread(self):
            closed["join"] += 1

    class _Proc:
        exitcode = 0

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

        def terminate(self):
            pass

        def close(self):
            closed["proc_close"] += 1

    class _Ctx:
        def Queue(self, *a, **k):
            return _Q()

        def Process(self, *a, **k):
            return _Proc()

    monkeypatch.setattr(rp.mp, "get_context", lambda _name: _Ctx())
    settings = AppSettings()
    out = rp._run_light_isolated(settings, "x.jpg", 0.0, timeout_sec=1.0)
    assert isinstance(out, LightAssetResult)
    assert closed["close"] == 1 and closed["join"] == 1
    assert closed["proc_close"] == 1


def test_preview_cache_skips_vips_for_psd(tmp_path: Path, monkeypatch):
    """PSD doğrudan libvips'e gitmemeli (Photoshop Document Data Block OOM)."""
    cache = FeaturePreviewCache(str(tmp_path), max_edge=64)
    psd = tmp_path / "a.psd"
    psd.write_bytes(b"8BPS" + b"\x00" * 64)

    called = {"vips": 0}

    def _boom(*_a, **_k):
        called["vips"] += 1
        raise AssertionError("vips must not open PSD")

    monkeypatch.setattr(cache, "_create_vips", _boom)

    def _pillow(source, out_path):
        from core.preview_cache import FeaturePreviewResult

        return FeaturePreviewResult(
            success=False, error="pillow_skip", unsupported_preview=True
        )

    monkeypatch.setattr(cache, "_create_pillow", _pillow)
    result = cache.create(str(psd))
    assert called["vips"] == 0
    assert result.success is False


def test_bad_decode_marks_failed_continues_not_crash():
    """Timeout/exception process'i öldürmemeli — TimeoutError yükselir."""
    safe_decode._reset_timeout_pool(reason="test_bad_decode")

    def _bad():
        raise RuntimeError("corrupt_psd_sim")

    with pytest.raises(RuntimeError, match="corrupt_psd_sim"):
        run_with_timeout(_bad, timeout_sec=2.0, retries=0)
