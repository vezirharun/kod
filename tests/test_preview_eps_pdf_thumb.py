"""EPS/PDF/TIFF preview-thumbnail chain — no search/index changes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.preview_renderer import RenderResult, _try_ghostscript, render_preview_for_index
from core.thumbnailer import Thumbnailer


def test_ghostscript_missing_reports_dependency(tmp_path: Path):
    out = tmp_path / "out.png"
    with patch("shutil.which", return_value=None), patch(
        "pathlib.Path.is_dir", return_value=False
    ):
        # Force empty candidate list via which=None and no Program Files
        r = _try_ghostscript(str(tmp_path / "x.eps"), out)
    assert r.success is False
    assert "dependency" in (r.error or "").lower() or "ghostscript" in (
        r.error or ""
    ).lower()
    assert r.unsupported_preview is True


def test_eps_gs_present_render_fail_not_renderer_missing(tmp_path: Path):
    """GS installed but render fails → keep root cause, never renderer_missing."""
    cache = tmp_path / "cache"
    cache.mkdir()
    src = tmp_path / "bad.eps"
    src.write_text("%!PS\nbogus\n")
    gs_fail = RenderResult(
        success=False,
        error="ghostscript_render_error:gs rc=1: Error: /undefined",
        unsupported_preview=False,
    )
    mu_fail = RenderResult(
        success=False,
        error="pymupdf_render_error:not a PDF",
        unsupported_preview=False,
    )
    with patch(
        "core.preview_renderer._try_pymupdf", return_value=mu_fail
    ), patch(
        "core.preview_renderer._try_ghostscript", return_value=gs_fail
    ):
        r = render_preview_for_index(str(src), str(cache))
    assert r.success is False
    assert r.unsupported_preview is False
    assert "renderer_missing" not in (r.error or "").lower()
    assert "ghostscript_render_error" in (r.error or "")
    assert "gerekli" not in (r.error or "").lower()


def test_eps_both_renderers_missing_is_renderer_missing(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    src = tmp_path / "x.ai"
    src.write_bytes(b"%PDF-1.4 fake")
    with patch(
        "core.preview_renderer._try_pymupdf",
        return_value=RenderResult(
            success=False, error="pymupdf yok", unsupported_preview=True
        ),
    ), patch(
        "core.preview_renderer._try_ghostscript",
        return_value=RenderResult(
            success=False,
            error="dependency eksik: Ghostscript (gswin64c) yok",
            unsupported_preview=True,
        ),
    ):
        r = render_preview_for_index(str(src), str(cache))
    assert r.success is False
    assert r.unsupported_preview is True
    assert "renderer_missing" in (r.error or "")


def test_global_preview_eps_render_fail_not_missing_dep(tmp_path: Path):
    from core.global_preview import STATUS_MISSING_DEP, STATUS_RENDER_FAILED, materialize_visual

    src = tmp_path / "x.eps"
    src.write_text("%!PS\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    with patch(
        "core.preview_renderer.render_preview_for_index",
        return_value=RenderResult(
            success=False,
            error="ghostscript_render_error:gs rc=1: fail",
            unsupported_preview=False,
        ),
    ):
        vis = materialize_visual(str(src), cache_dir=str(cache))
    assert vis.status == STATUS_RENDER_FAILED
    assert vis.status != STATUS_MISSING_DEP
    assert "ghostscript_render_error" in (vis.error or "")


def test_vector_timeout_constant_higher_than_tif():
    """Regression: EPS/PDF must not share 5–8s TIF timeout."""
    import inspect
    from core import thumbnailer as th

    src = inspect.getsource(th.Thumbnailer.create)
    assert "VECTOR_THUMB_TIMEOUT_SEC = 120" in src
    assert "max(8.0, float(TIF_TIMEOUT_SEC))" not in src


def test_render_pdf_eps_dispatch(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    # missing file → graceful fail, no crash
    r = render_preview_for_index(str(tmp_path / "missing.eps"), str(cache))
    assert r.success is False


def test_thumbnailer_jpg_unchanged(tmp_path: Path):
    from PIL import Image

    src = tmp_path / "a.jpg"
    Image.new("RGB", (64, 48), color=(20, 40, 60)).save(src, "JPEG")
    th = Thumbnailer(str(tmp_path / "cache"), max_edge=32, fmt="webp")
    r = th.create(str(src))
    assert r.success
    assert Path(r.thumbnail_path).is_file()


def test_thumbnailer_png_unchanged(tmp_path: Path):
    from PIL import Image

    src = tmp_path / "a.png"
    Image.new("RGB", (40, 40), color=(200, 10, 10)).save(src, "PNG")
    th = Thumbnailer(str(tmp_path / "cache"), max_edge=32, fmt="webp")
    r = th.create(str(src))
    assert r.success


def test_feature_preview_uses_renderer_for_eps(tmp_path: Path):
    from core.preview_cache import FeaturePreviewCache

    src = tmp_path / "x.eps"
    src.write_text("%!PS\n")
    raster = tmp_path / "cache" / "render_previews" / "x_render.png"
    raster.parent.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (20, 20), color=(40, 120, 80)).save(raster, "PNG")

    fp = FeaturePreviewCache(str(tmp_path / "cache"), max_edge=16)
    with patch(
        "core.preview_renderer.render_preview_for_index",
        return_value=RenderResult(
            success=True, image_path=str(raster), renderer="test"
        ),
    ):
        r = fp.create(str(src))
    assert r.success
    assert Path(r.preview_path).is_file()


def test_ghostscript_eps_includes_epscrop():
    """EPS Ghostscript argv must include -dEPSCrop (not PDF path)."""
    import inspect

    from core import preview_renderer as pr

    src = inspect.getsource(pr._try_ghostscript)
    assert "-dEPSCrop" in src
    assert '(".eps", ".ai")' in src or "'.eps', '.ai'" in src


def test_blank_white_png_rejected_by_gate(tmp_path: Path):
    from PIL import Image

    from core.preview_renderer import _eps_ai_preview_gate

    white = tmp_path / "white.png"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(white, "PNG")
    ok, reason = _eps_ai_preview_gate(white)
    assert ok is False
    assert "blank_white_preview" in reason or "invalid_preview" in reason


def test_good_raster_passes_gate(tmp_path: Path):
    from PIL import Image

    from core.preview_renderer import _eps_ai_preview_gate

    good = tmp_path / "good.png"
    Image.new("RGB", (64, 64), (40, 120, 80)).save(good, "PNG")
    ok, reason = _eps_ai_preview_gate(good)
    assert ok is True
    assert reason == "ok"


def test_old_white_feature_cache_rejected(tmp_path: Path):
    """Existing white EPS feature preview must not pass get_existing."""
    from PIL import Image

    from core.preview_cache import FeaturePreviewCache

    src = tmp_path / "old.eps"
    src.write_text("%!PS-Adobe-3.0 EPSF-3.0\n")
    fp = FeaturePreviewCache(str(tmp_path / "cache"), max_edge=32)
    out = fp.preview_path_for(str(src))
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (255, 255, 255)).save(out, "WEBP")
    assert out.is_file()
    ge = fp.get_existing(str(src))
    assert ge.success is False
    assert "blank_white" in (ge.error or "") or "invalid_preview" in (ge.error or "")
    # marked for reopen: artifact removed
    assert not out.is_file()


def test_render_cache_white_eps_reopens(tmp_path: Path):
    """White render_previews artifact must not be accepted as success."""
    from PIL import Image

    from core.utils import file_id_from_path

    cache = tmp_path / "cache"
    src = tmp_path / "w.eps"
    src.write_text("%!PS\n")
    out = cache / "render_previews" / f"{file_id_from_path(str(src))}_render.png"
    out.parent.mkdir(parents=True)
    Image.new("RGB", (80, 80), (255, 255, 255)).save(out, "PNG")

    with patch(
        "core.preview_renderer._try_pymupdf",
        return_value=RenderResult(
            success=False, error="pymupdf yok", unsupported_preview=True
        ),
    ), patch(
        "core.preview_renderer._try_ghostscript",
        return_value=RenderResult(
            success=False,
            error="blank_white_preview:blank_white",
            unsupported_preview=False,
        ),
    ):
        r = render_preview_for_index(str(src), str(cache))
    assert r.success is False
    assert "renderer_missing" not in (r.error or "")
    assert "blank_white_preview" in (r.error or "")
    # old white cache should have been unlinked before re-render attempt
    assert not out.is_file()


def test_scheduler_try_create_returns_error_tuple():
    """API contract: _try_create → (path, err)."""
    import inspect
    from ui import thumbnail_scheduler as ts

    src = inspect.getsource(ts._LoadRunnable._try_create)
    assert "tuple[str, str]" in src or "Return (thumbnail_path" in src
