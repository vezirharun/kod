"""UI / legacy Preview Pool SSOT — thumb & display bypass guards.

A–H cover Preview→Thumb SSOT without touching V3 Index.
"""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from core.preview_cache import FeaturePreviewCache
from core.preview_renderer import resolve_display_image_path
from core.thumb_resolve import lookup_valid_preview, resolve_thumb_path
from core.thumbnailer import Thumbnailer


def _write_preview(cache: Path, source: Path, size=(96, 64), color=(10, 80, 160)) -> Path:
    fp = FeaturePreviewCache(str(cache), max_edge=96)
    out = fp.preview_path_for(str(source))
    Image.new("RGB", size, color).save(out, "WEBP")
    return Path(out)


def test_a_preview_yes_thumb_no_builds_thumb_from_preview(tmp_path, monkeypatch):
    source = tmp_path / "a.tif"
    source.write_bytes(b"not-a-real-tiff")
    cache = tmp_path / "cache"
    preview = _write_preview(cache, source)
    thumb = Thumbnailer(str(cache), max_edge=48, fmt="webp")

    def boom(*_a, **_k):
        raise AssertionError("NAS/source must not be opened")

    monkeypatch.setattr("core.tif_thumbnail.create_tif_thumbnail", boom)

    opens: list[str] = []
    real_open = Image.open

    def spy_open(path, *a, **k):
        opens.append(str(path))
        if Path(path).resolve() == source.resolve():
            raise AssertionError(f"source opened: {path}")
        return real_open(path, *a, **k)

    monkeypatch.setattr(Image, "open", spy_open)

    r = thumb.create(str(source))
    assert r.success
    assert Path(r.thumbnail_path).is_file()
    assert Path(r.thumbnail_path) == thumb.thumbnail_path_for(str(source))
    assert not any(Path(p).resolve() == source.resolve() for p in opens)
    assert preview.is_file()



def test_b_both_exist_uses_existing_thumb(tmp_path):
    source = tmp_path / "b.jpg"
    Image.new("RGB", (40, 40), (1, 2, 3)).save(source, "JPEG")
    cache = tmp_path / "cache"
    _write_preview(cache, source, color=(9, 9, 9))
    thumb = Thumbnailer(str(cache), max_edge=32, fmt="webp")
    out = thumb.thumbnail_path_for(str(source))
    Image.new("RGB", (32, 32), (200, 0, 0)).save(out, "WEBP")

    with patch.object(thumb, "create_from_existing_preview") as spy:
        r = thumb.create(str(source))
    assert r.success
    assert Path(r.thumbnail_path) == out
    spy.assert_not_called()


def test_c_preview_no_source_yes_safe_fallback(tmp_path):
    source = tmp_path / "c.png"
    Image.new("RGB", (50, 40), (5, 50, 5)).save(source, "PNG")
    cache = tmp_path / "cache"
    thumb = Thumbnailer(str(cache), max_edge=32, fmt="webp")
    r = thumb.create(str(source))
    assert r.success
    assert Path(r.thumbnail_path).is_file()


def test_d_preview_invalid_safe_fallback(tmp_path):
    source = tmp_path / "d.png"
    Image.new("RGB", (40, 40), (8, 8, 80)).save(source, "PNG")
    cache = tmp_path / "cache"
    fp = FeaturePreviewCache(str(cache), max_edge=64)
    bad = fp.preview_path_for(str(source))
    bad.write_bytes(b"not-a-webp")
    thumb = Thumbnailer(str(cache), max_edge=32, fmt="webp")
    r = thumb.create(str(source))
    assert r.success
    assert Path(r.thumbnail_path).is_file()


def test_e_jumbo_tiff_preview_yes_no_nas_reread(tmp_path, monkeypatch):
    source = tmp_path / "jumbo.tif"
    source.write_bytes(b"II*\x00fake-jumbo-tiff")
    cache = tmp_path / "cache"
    _write_preview(cache, source, size=(128, 96))
    thumb = Thumbnailer(str(cache), max_edge=48, fmt="webp")

    calls = {"tif": 0, "vips_src": 0}

    def tif_boom(self, src):
        calls["tif"] += 1
        raise AssertionError(f"create_tif_thumbnail must not run: {src}")

    monkeypatch.setattr("core.tif_thumbnail.create_tif_thumbnail", tif_boom)

    import core.thumbnailer as th_mod

    real_vips = th_mod.vips_new_from_file

    def vips_spy(path, **kw):
        if Path(path).resolve() == source.resolve():
            calls["vips_src"] += 1
            raise AssertionError("vips must not open NAS TIFF")
        return real_vips(path, **kw)

    monkeypatch.setattr(th_mod, "vips_new_from_file", vips_spy)

    r = thumb.create(str(source))
    assert r.success
    assert calls["tif"] == 0
    assert calls["vips_src"] == 0
    assert lookup_valid_preview(str(source), str(cache))


def test_f_cache_miss_preview_yes_scheduler_uses_preview(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from ui.thumbnail_scheduler import _LoadRunnable

    source = tmp_path / "f.tif"
    source.write_bytes(b"fake")
    cache = tmp_path / "cache"
    preview = _write_preview(cache, source)
    thumb = Thumbnailer(str(cache), max_edge=48, fmt="webp")

    done = {}

    def on_done(fid, img, size, reason, name, resolved):
        done["resolved"] = resolved
        done["reason"] = reason
        done["img"] = img

    monkeypatch.setattr(
        "core.tif_thumbnail.create_tif_thumbnail",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no TIFF create")),
    )

    runnable = _LoadRunnable(
        7,
        "",  # thumb path miss
        64,
        str(source),
        "f.tif",
        str(cache),
        thumb,
        on_done,
        mode="thumb",
        feature_preview_path=str(preview),
    )
    with patch.object(thumb, "create", side_effect=AssertionError("create(source) blocked")):
        runnable.run()
    assert done.get("reason") == ""
    assert done.get("resolved")
    assert Path(done["resolved"]).is_file()
    # Prefer thumb produced from preview, or preview path itself
    out = Path(done["resolved"])
    assert out == thumb.thumbnail_path_for(str(source)) or out.resolve() == preview.resolve()


def test_g_detail_preview_prefers_pool(tmp_path):
    pytest.importorskip("PySide6")
    from ui.thumbnail_scheduler import _LoadRunnable

    source = tmp_path / "g.jpg"
    Image.new("RGB", (20, 20), (1, 1, 1)).save(source, "JPEG")
    cache = tmp_path / "cache"
    preview = _write_preview(cache, source, size=(80, 60), color=(0, 255, 0))
    thumb = Thumbnailer(str(cache), max_edge=32, fmt="webp")

    got = {}

    def on_done(fid, img, size, reason, name, resolved):
        got["path"] = resolved
        got["reason"] = reason

    with patch(
        "core.preview_cache.FeaturePreviewCache.create",
        side_effect=AssertionError("must not create preview from source"),
    ):
        runnable = _LoadRunnable(
            3,
            "",
            800,
            str(source),
            "g.jpg",
            str(cache),
            thumb,
            on_done,
            mode="detail",
            feature_preview_path=str(preview),
        )
        runnable.run()
    assert got.get("reason") == ""
    assert Path(got["path"]).resolve() == preview.resolve()


def test_h_ui_cache_miss_does_not_block_ui_thread(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ui.thumbnail_scheduler import ThumbnailScheduler

    app = QApplication.instance() or QApplication([])
    source = tmp_path / "h.jpg"
    Image.new("RGB", (30, 30), (3, 3, 3)).save(source, "JPEG")
    cache = tmp_path / "cache"
    _write_preview(cache, source)
    thumb = Thumbnailer(str(cache), max_edge=32, fmt="webp")

    sched = ThumbnailScheduler()
    sched.configure(cache_dir=str(cache), thumbnailer=thumb)

    create_threads: list[int] = []
    real_create = thumb.create

    def spy_create(path):
        create_threads.append(threading.get_ident())
        return real_create(path)

    main_tid = threading.get_ident()
    with patch.object(thumb, "create", side_effect=spy_create):
        # Cache miss path (empty thumb path) — must queue, not sync-create on UI thread
        sched.request(
            99,
            "",
            64,
            source_path=str(source),
            filename="h.jpg",
        )
    # request() returned without waiting for worker create on this thread
    assert main_tid not in create_threads
    assert sched.queue_depth() >= 0
    # Give pool a moment then drain — worker may run create_from_existing_preview
    # (not create); either way UI thread must stay free of create().
    app.processEvents()
    assert main_tid not in create_threads
    del app


def test_i_request_visible_reprioritizes_over_offscreen(tmp_path):
    """Fast scroll: visible ids must sit ahead; off-screen queued work is dropped."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    import heapq

    from ui.thumbnail_scheduler import ThumbnailScheduler, _ThumbTask

    app = QApplication.instance() or QApplication([])
    sched = ThumbnailScheduler()
    with patch.object(sched, "_pump"):
        with sched._lock:
            for fid in (1, 2, 3, 4, 5):
                sched._seq += 1
                heapq.heappush(
                    sched._heap,
                    _ThumbTask(10, sched._seq, fid, f"{fid}.jpg", 64, "", str(fid)),
                )
                sched._queued.add(fid)
        sched.request_visible(
            [
                (4, "4.jpg", 64, "", "4"),
                (5, "5.jpg", 64, "", "5"),
            ]
        )
        with sched._lock:
            ordered = heapq.nsmallest(len(sched._heap), sched._heap)
            top_ids = [t.file_id for t in ordered[:2]]
            top_pri = [t.priority for t in ordered[:2]]
            leftover = [t.file_id for t in ordered if t.file_id in (1, 2, 3)]
    assert top_ids == [4, 5]
    assert top_pri == [0, 0]
    assert leftover == []
    del app


def test_k_searchresult_thumbnail_request_passes_feature_preview(tmp_path):
    """SearchResult.feature_preview_path must reach ResultCard → scheduler tuple."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from core.search_engine import SearchResult
    from ui.result_card import ResultCard

    app = QApplication.instance() or QApplication([])
    source = tmp_path / "k.jpg"
    Image.new("RGB", (20, 20), (9, 9, 9)).save(source, "JPEG")
    cache = tmp_path / "cache"
    preview = _write_preview(cache, source)
    result = SearchResult(
        file_id=42,
        path=str(source),
        filename="k.jpg",
        customer="",
        thumbnail_path="",
        score=0.9,
        score_percent=90.0,
        feature_preview_path=str(preview),
    )
    card = ResultCard(result)
    req = card.thumbnail_request()
    assert req is not None
    assert len(req) >= 6
    assert int(req[0]) == 42
    assert str(req[5]) == str(preview)
    del app


def test_l_search_existing_preview_no_nas_via_scheduler(tmp_path):
    """Existing FP on thumbnail_request → LoadRunnable must not Thumbnailer.create(source)."""
    pytest.importorskip("PySide6")
    from ui.thumbnail_scheduler import _LoadRunnable

    source = tmp_path / "l.jpg"
    Image.new("RGB", (24, 24), (2, 2, 2)).save(source, "JPEG")
    cache = tmp_path / "cache"
    preview = _write_preview(cache, source)
    thumb = Thumbnailer(str(cache), max_edge=32, fmt="webp")
    done = {}

    def on_done(fid, img, size, reason, name, resolved):
        done["resolved"] = resolved
        done["reason"] = reason

    runnable = _LoadRunnable(
        11,
        "",
        64,
        str(source),
        "l.jpg",
        str(cache),
        thumb,
        on_done,
        mode="thumb",
        feature_preview_path=str(preview),
    )
    with patch.object(thumb, "create", side_effect=AssertionError("NAS create blocked")):
        runnable.run()
    assert done.get("reason") == ""
    assert Path(done["resolved"]).is_file()


def test_m_request_visible_forwards_feature_preview_path(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ui.thumbnail_scheduler import ThumbnailScheduler

    app = QApplication.instance() or QApplication([])
    sched = ThumbnailScheduler()
    captured = {}

    def spy_request(file_id, path, size, **kwargs):
        captured["kwargs"] = dict(kwargs)
        captured["fid"] = int(file_id)

    with patch.object(sched, "request", side_effect=spy_request):
        sched.request_visible(
            [(7, "t.webp", 64, "s.jpg", "s.jpg", "/fp/preview.webp")]
        )
    assert captured["fid"] == 7
    assert captured["kwargs"].get("feature_preview_path") == "/fp/preview.webp"
    assert captured["kwargs"].get("priority") == 0
    del app


def test_j_ui_perf_critical_threshold_is_500ms():
    from core import ui_perf

    assert ui_perf.FREEZE_INTERACT_SEC == 0.10
    assert ui_perf.FREEZE_WARNING_SEC == 0.25
    assert ui_perf.FREEZE_CRITICAL_SEC == 0.50
    assert ui_perf.FREEZE_THRESHOLD_SEC == 0.50
    assert ui_perf.FREEZE_DUMP_COOLDOWN_SEC >= 5.0


def test_resolve_thumb_path_prefers_existing_thumb_over_preview(tmp_path):
    source = tmp_path / "r.jpg"
    Image.new("RGB", (16, 16), (1, 1, 1)).save(source, "JPEG")
    cache = tmp_path / "cache"
    preview = _write_preview(cache, source)
    thumb = Thumbnailer(str(cache), max_edge=16, fmt="webp")
    tpath = thumb.thumbnail_path_for(str(source))
    Image.new("RGB", (16, 16), (255, 0, 0)).save(tpath, "WEBP")
    resolved, status = resolve_thumb_path(
        str(tpath),
        cache_dir=str(cache),
        feature_preview_path=str(preview),
        source_path=str(source),
    )
    assert status == "hit"
    assert Path(resolved).resolve() == tpath.resolve()


def test_resolve_display_prefers_pool_over_source_open(tmp_path, monkeypatch):
    source = tmp_path / "disp.tif"
    source.write_bytes(b"fake-tif")
    cache = tmp_path / "cache"
    preview = _write_preview(cache, source)

    monkeypatch.setattr(
        "core.settings.AppSettings.load",
        lambda: MagicMock(
            cache_dir=str(cache),
            thumbnail_max_edge=48,
            thumbnail_format="webp",
            feature_preview_max_edge=96,
            db_path=str(tmp_path / "missing.db"),
        ),
    )
    with patch(
        "core.thumbnailer.Thumbnailer.create",
        side_effect=AssertionError("display must not Thumbnailer.create(source)"),
    ), patch(
        "core.preview_renderer.render_preview_for_index",
        side_effect=AssertionError("must not re-render"),
    ):
        r = resolve_display_image_path(str(source), cache_dir=str(cache))
    assert r.success
    assert Path(r.image_path).resolve() == preview.resolve()
    assert r.renderer == "feature_cache"
