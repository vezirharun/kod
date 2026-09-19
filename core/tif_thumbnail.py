"""TIF hızlı thumbnail — embedded preview, pyvips, Pillow fallback."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

from PIL import Image

from core.logger import setup_logger
from core.thumbnailer import Thumbnailer, ThumbnailResult
from core.utils import iter_fs_path_candidates, normalize_path

logger = setup_logger(__name__)

TIF_TIMEOUT_SEC = 5.0

_TIF_POOL: ThreadPoolExecutor | None = None
_TIF_POOL_LOCK = threading.Lock()


def _get_tif_pool() -> ThreadPoolExecutor:
    """Executor yalnızca Index Stop / yenilemede kapanır; hata sonrası yeniden doğar."""
    global _TIF_POOL
    with _TIF_POOL_LOCK:
        pool = _TIF_POOL
        if pool is None or getattr(pool, "_shutdown", False):
            _TIF_POOL = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="vezir-tif"
            )
            pool = _TIF_POOL
        return pool


def _reset_tif_pool(*, reason: str = "") -> ThreadPoolExecutor:
    """Shutdown olmuş veya kilitlenmiş pool'u değiştir — worker yaşamaya devam eder."""
    global _TIF_POOL
    with _TIF_POOL_LOCK:
        old = _TIF_POOL
        _TIF_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vezir-tif")
        if old is not None:
            try:
                old.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                old.shutdown(wait=False)
            except Exception:
                pass
        if reason:
            logger.warning("TIF executor yenilendi: %s", reason)
        return _TIF_POOL


def _submit_tif(fn, *args):
    """submit — shutdown hatasında pool yenile ve bir kez daha dene."""
    try:
        return _get_tif_pool().submit(fn, *args)
    except RuntimeError as exc:
        if "shutdown" not in str(exc).lower():
            raise
        return _reset_tif_pool(reason=str(exc)).submit(fn, *args)


def _try_embedded_preview(
    path: str, out_path: Path, max_edge: int, fmt: str, quality: int
) -> ThumbnailResult:
    last_error = ""
    for open_path in iter_fs_path_candidates(path):
        try:
            with Image.open(open_path) as img:
                if not getattr(img, "n_frames", 1) or img.n_frames < 1:
                    return ThumbnailResult(success=False)
                # Pillow bazen embedded thumbnail tag'inde küçük önizleme tutar
                thumb = img.copy()
                thumb = Thumbnailer.normalize_pillow_image(thumb)
                w, h = thumb.size
                scale = max_edge / max(w, h)
                if scale < 1.0:
                    thumb = thumb.resize(
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        Image.Resampling.LANCZOS,
                    )
                save_kw = {"quality": quality}
                if fmt == "webp":
                    thumb.save(out_path, "WEBP", **save_kw)
                else:
                    thumb.save(out_path, "JPEG", **save_kw)
                return ThumbnailResult(
                    success=True,
                    thumbnail_path=str(out_path),
                    width=thumb.width,
                    height=thumb.height,
                )
        except Image.DecompressionBombError as exc:
            return ThumbnailResult(success=False, error=str(exc))
        except Exception as exc:
            last_error = str(exc)
            logger.debug("TIF embedded preview yok %s: %s", open_path, exc)
    return ThumbnailResult(success=False, error=last_error)


def create_tif_thumbnail(
    thumbnailer: Thumbnailer,
    source_path: str,
    *,
    timeout_sec: float = TIF_TIMEOUT_SEC,
) -> ThumbnailResult:
    """TIF için süre sınırlı hızlı yol."""
    source = normalize_path(source_path)
    out_path = thumbnailer.thumbnail_path_for(source)
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with Image.open(out_path) as cached:
                width, height = cached.size
            return ThumbnailResult(
                success=True,
                thumbnail_path=str(out_path),
                width=width,
                height=height,
            )
        except Exception:
            out_path.unlink(missing_ok=True)

    t0 = time.perf_counter()
    try:
        embedded = _submit_tif(
            _try_embedded_preview,
            source,
            out_path,
            thumbnailer.max_edge,
            thumbnailer.fmt,
            thumbnailer.quality,
        ).result(timeout=max(0.5, timeout_sec))
    except FuturesTimeout:
        out_path.unlink(missing_ok=True)
        _reset_tif_pool(reason="tif preview timeout")
        return ThumbnailResult(success=False, error="TIF timeout (preview)")
    except RuntimeError as exc:
        if "shutdown" in str(exc).lower():
            _reset_tif_pool(reason=str(exc))
            return ThumbnailResult(success=False, error=str(exc))
        raise
    except Image.DecompressionBombError as exc:
        return ThumbnailResult(success=False, error=str(exc))
    if embedded.success:
        return embedded
    if embedded.error and (
        "decompression bomb" in embedded.error.lower()
        or "exceeds limit" in embedded.error.lower()
    ):
        return embedded

    if time.perf_counter() - t0 > timeout_sec:
        return ThumbnailResult(success=False, error="TIF timeout (embedded)")

    from core.thumbnailer import HAS_VIPS, pyvips

    if HAS_VIPS and pyvips is not None:
        try:

            def _vips_job() -> ThumbnailResult:
                last_error = ""
                for open_path in iter_fs_path_candidates(source):
                    try:
                        image = pyvips.Image.new_from_file(
                            open_path, access="sequential", page=0
                        )
                        scale = thumbnailer.max_edge / max(image.width, image.height)
                        if scale < 1.0:
                            image = image.resize(scale)
                        if thumbnailer.fmt == "webp":
                            image.webpsave(str(out_path), Q=thumbnailer.quality)
                        else:
                            image.jpegsave(str(out_path), Q=thumbnailer.quality)
                        return ThumbnailResult(
                            success=True,
                            thumbnail_path=str(out_path),
                            width=image.width,
                            height=image.height,
                        )
                    except Exception as exc:
                        last_error = str(exc)
                return ThumbnailResult(success=False, error=last_error)

            remaining = max(0.5, timeout_sec - (time.perf_counter() - t0))
            result = _submit_tif(_vips_job).result(timeout=remaining)
            if result.success and time.perf_counter() - t0 <= timeout_sec:
                return result
            out_path.unlink(missing_ok=True)
            return ThumbnailResult(success=False, error="TIF timeout (pyvips)")
        except FuturesTimeout:
            out_path.unlink(missing_ok=True)
            _reset_tif_pool(reason="tif pyvips timeout")
            return ThumbnailResult(success=False, error="TIF timeout (pyvips)")
        except RuntimeError as exc:
            if "shutdown" in str(exc).lower():
                _reset_tif_pool(reason=str(exc))
                return ThumbnailResult(success=False, error=str(exc))
            raise
        except Exception as exc:
            logger.debug("TIF pyvips %s: %s", source, exc)

    if time.perf_counter() - t0 > timeout_sec:
        return ThumbnailResult(success=False, error="TIF timeout")

    try:
        remaining = max(0.5, timeout_sec - (time.perf_counter() - t0))
        result = _submit_tif(
            thumbnailer._create_pillow,
            source,
            out_path,
        ).result(timeout=remaining)
    except FuturesTimeout:
        out_path.unlink(missing_ok=True)
        _reset_tif_pool(reason="tif pillow timeout")
        return ThumbnailResult(success=False, error="TIF timeout (pillow)")
    except RuntimeError as exc:
        if "shutdown" in str(exc).lower():
            _reset_tif_pool(reason=str(exc))
            return ThumbnailResult(success=False, error=str(exc))
        raise
    except Image.DecompressionBombError as exc:
        return ThumbnailResult(success=False, error=str(exc))
    if not result.success and time.perf_counter() - t0 > timeout_sec:
        return ThumbnailResult(success=False, error="TIF timeout (pillow)")
    return result
