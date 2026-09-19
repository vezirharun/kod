"""Light pass — tek dosya açılışında metadata + thumbnail + medium preview."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.logger import setup_logger
from core.metadata_sanitize import sanitize_for_json
from core.preview_cache import FEATURE_PREVIEW_VERSION, FeaturePreviewCache
from core.thumbnailer import HAS_VIPS, Thumbnailer, ThumbnailResult, pyvips
from core.utils import iter_fs_path_candidates, normalize_path

logger = setup_logger(__name__)


@dataclass
class LightAssetResult:
    success: bool
    reused_cache: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_status: str = "not_supported"
    thumbnail_path: str = ""
    thumbnail_width: int = 0
    thumbnail_height: int = 0
    thumbnail_status: str = "pending"
    feature_preview_path: str = ""
    feature_preview_width: int = 0
    feature_preview_height: int = 0
    preview_generated: bool = False
    metadata_saved: bool = False
    error: str = ""
    parser_errors: list[str] = field(default_factory=list)
    visual_status: str = ""
    visual_renderer: str = ""


def _metadata_from_pillow_image(img) -> dict[str, Any]:
    from PIL import Image

    if not isinstance(img, Image.Image):
        return {}
    dpi = img.info.get("dpi")
    return sanitize_for_json(
        {
            "width": int(img.width),
            "height": int(img.height),
            "mode": str(img.mode),
            "image_format": str(img.format or ""),
            "frames": int(getattr(img, "n_frames", 1) or 1),
            "dpi": list(dpi) if dpi else [],
        }
    )


def _save_resized_webp(img, out_path: Path, max_edge: int, fmt: str, quality: int):
    from PIL import Image

    w, h = img.size
    scale = max_edge / max(w, h)
    if scale < 1.0:
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    save_kw = {"quality": quality}
    if fmt == "webp":
        img.save(out_path, "WEBP", **save_kw)
    else:
        img.save(out_path, "JPEG", **save_kw)
    return img.width, img.height


def _artifact_fresh(path: Path, source_mtime: float) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if source_mtime <= 0:
        return True
    try:
        return path.stat().st_mtime >= source_mtime - 1.0
    except OSError:
        return False


def try_reuse_light_assets(
    *,
    source_path: str,
    thumbnailer: Thumbnailer,
    feature_preview: FeaturePreviewCache,
    source_mtime: float = 0.0,
    needs_thumb: bool = True,
    needs_fp: bool = True,
) -> LightAssetResult | None:
    """Disk önbelleği yeterliyse orijinal açmadan döner."""
    source = normalize_path(source_path)
    thumb_path = thumbnailer.thumbnail_path_for(source)
    fp_path = feature_preview.preview_path_for(source)
    thumb_ok = not needs_thumb or _artifact_fresh(thumb_path, source_mtime)
    fp_ok = not needs_fp or _artifact_fresh(fp_path, source_mtime)
    if not thumb_ok or not fp_ok:
        return None
    try:
        from PIL import Image

        tw, th = 0, 0
        if thumb_ok and needs_thumb:
            with Image.open(thumb_path) as timg:
                tw, th = timg.size
        fw, fh = 0, 0
        if fp_ok and needs_fp:
            with Image.open(fp_path) as fimg:
                fw, fh = fimg.size
    except Exception:
        return None
    return LightAssetResult(
        success=True,
        reused_cache=True,
        metadata_status="cache_reused",
        thumbnail_path=str(thumb_path) if needs_thumb else "",
        thumbnail_width=tw,
        thumbnail_height=th,
        thumbnail_status="cache_reused",
        feature_preview_path=str(fp_path) if needs_fp else "",
        feature_preview_width=fw,
        feature_preview_height=fh,
        preview_generated=False,
        metadata_saved=False,
    )


def extract_light_assets_once(
    source_path: str,
    *,
    thumbnailer: Thumbnailer,
    feature_preview: FeaturePreviewCache,
    source_mtime: float = 0.0,
) -> LightAssetResult:
    """Raster dosyayı bir kez aç; metadata + thumb + medium preview üret.

    Raster olmayan kaynaklar önce Global Preview ile gerçek raster'a çevrilir.
    """
    source = normalize_path(source_path)
    reused = try_reuse_light_assets(
        source_path=source,
        thumbnailer=thumbnailer,
        feature_preview=feature_preview,
        source_mtime=source_mtime,
    )
    if reused is not None:
        return reused

    thumb_out = thumbnailer.thumbnail_path_for(source)
    fp_out = feature_preview.preview_path_for(source)
    source_ext = Path(source).suffix.lower()
    bounded = source_ext in {
        ".tif",
        ".tiff",
        ".psd",
        ".ai",
        ".eps",
        ".pdf",
    }

    def _do_extract() -> LightAssetResult:
        work_path = source
        visual_status = "visual_renderable"
        visual_renderer = "native_raster"
        try:
            from core.global_preview import materialize_visual

            vis = materialize_visual(
                source, cache_dir=str(thumbnailer.cache_dir.parent)
            )
            visual_status = vis.status
            visual_renderer = vis.renderer or ""
            if not vis.ok:
                return LightAssetResult(
                    success=False,
                    error=f"{vis.status}:{vis.error}"[:300],
                    parser_errors=[vis.error] if vis.error else [],
                    visual_status=vis.status,
                    visual_renderer=visual_renderer,
                )
            work_path = vis.raster_path or source
        except Exception as vis_exc:
            logger.debug("global_preview skipped: %s", vis_exc)

        ext = Path(work_path).suffix.lower()
        if HAS_VIPS and pyvips is not None and ext in {
            ".tif",
            ".tiff",
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        }:
            vips_result = _extract_with_vips_once(
                work_path,
                thumb_out,
                fp_out,
                thumbnailer=thumbnailer,
                feature_preview=feature_preview,
            )
            if vips_result.success:
                vips_result.visual_status = visual_status
                vips_result.visual_renderer = visual_renderer
                return vips_result
        result = _extract_with_pillow_once(
            work_path,
            thumb_out,
            fp_out,
            thumbnailer=thumbnailer,
            feature_preview=feature_preview,
        )
        result.visual_status = visual_status
        result.visual_renderer = visual_renderer
        return result

    if bounded:
        from concurrent.futures import TimeoutError as FuturesTimeout

        from core.tif_thumbnail import TIF_TIMEOUT_SEC, _reset_tif_pool, _submit_tif

        timeout_sec = max(45.0, float(TIF_TIMEOUT_SEC) * 9)
        try:
            result = _submit_tif(_do_extract).result(timeout=timeout_sec)
        except FuturesTimeout:
            _reset_tif_pool(reason=f"light extract timeout {source_ext}")
            result = LightAssetResult(
                success=False,
                error=f"timeout (light extract {source_ext})",
            )
        except RuntimeError as exc:
            if "shutdown" in str(exc).lower():
                _reset_tif_pool(reason=str(exc))
                result = LightAssetResult(success=False, error=str(exc)[:300])
            else:
                raise
        if result.success:
            return result
        fallback = _fallback_preview_assets(
            source_ext, thumb_out, fp_out, thumbnailer, feature_preview
        )
        if fallback is not None:
            fallback.error = result.error
            return fallback
        return result
    return _do_extract()


def _extract_with_vips_once(
    source: str,
    thumb_out: Path,
    fp_out: Path,
    *,
    thumbnailer: Thumbnailer,
    feature_preview: FeaturePreviewCache,
) -> LightAssetResult:
    last_error = ""
    for open_path in iter_fs_path_candidates(source):
        try:
            # Shrink-on-load: tam TIF decode kuyruğu kilitler / timeout'a düşer.
            thumb_size = max(32, int(thumbnailer.max_edge))
            fp_size = max(thumb_size, int(feature_preview.max_edge))
            thumb_img = pyvips.Image.thumbnail(open_path, thumb_size)
            w, h = int(thumb_img.width), int(thumb_img.height)
            metadata = sanitize_for_json(
                {
                    "width": w,
                    "height": h,
                    "image_format": "TIFF",
                    "decoder": "pyvips_thumbnail",
                }
            )
            if thumbnailer.fmt == "webp":
                thumb_img.webpsave(str(thumb_out), Q=thumbnailer.quality)
            else:
                thumb_img.jpegsave(str(thumb_out), Q=thumbnailer.quality)

            fp_img = (
                thumb_img
                if fp_size <= thumb_size
                else pyvips.Image.thumbnail(open_path, fp_size)
            )
            if feature_preview.fmt == "webp":
                fp_img.webpsave(str(fp_out), Q=feature_preview.quality)
            else:
                fp_img.jpegsave(str(fp_out), Q=feature_preview.quality)

            return LightAssetResult(
                success=True,
                metadata=metadata,
                metadata_status="metadata_ok",
                metadata_saved=True,
                thumbnail_path=str(thumb_out),
                thumbnail_width=int(thumb_img.width),
                thumbnail_height=int(thumb_img.height),
                thumbnail_status="thumbnail_ok",
                feature_preview_path=str(fp_out),
                feature_preview_width=int(fp_img.width),
                feature_preview_height=int(fp_img.height),
                preview_generated=True,
            )
        except Exception as exc:
            last_error = str(exc)
            logger.debug("light vips tek-açılış hatası %s: %s", open_path, exc)
    return LightAssetResult(success=False, error=last_error)


def _fallback_preview_assets(
    source_ext: str,
    thumb_out: Path,
    fp_out: Path,
    thumbnailer: Thumbnailer,
    feature_preview: FeaturePreviewCache,
) -> LightAssetResult | None:
    """PSD/PDF render başarısız olsa da dosya indekslensin (Tamamlanan artsın)."""
    if source_ext not in {".psd", ".ai", ".eps", ".pdf", ".svg", ".cdr"}:
        return None
    try:
        from PIL import Image

        tw = max(32, int(thumbnailer.max_edge))
        fw = max(tw, int(feature_preview.max_edge))
        thumb = Image.new("RGB", (tw, tw), (48, 48, 52))
        prev = Image.new("RGB", (fw, fw), (48, 48, 52))
        fmt = "WEBP" if thumbnailer.fmt == "webp" else "JPEG"
        thumb.save(thumb_out, fmt, quality=thumbnailer.quality)
        pfmt = "WEBP" if feature_preview.fmt == "webp" else "JPEG"
        prev.save(fp_out, pfmt, quality=feature_preview.quality)
        return LightAssetResult(
            success=True,
            metadata={"width": tw, "height": tw, "fallback_preview": source_ext},
            metadata_status="fallback",
            metadata_saved=True,
            thumbnail_path=str(thumb_out),
            thumbnail_width=tw,
            thumbnail_height=tw,
            thumbnail_status="fallback_ok",
            feature_preview_path=str(fp_out),
            feature_preview_width=fw,
            feature_preview_height=fw,
            preview_generated=True,
            visual_status="visual_renderable",
            visual_renderer="fallback",
        )
    except Exception as exc:
        logger.debug("fallback preview yok %s: %s", source_ext, exc)
        return None


def _extract_with_pillow_once(
    source: str,
    thumb_out: Path,
    fp_out: Path,
    *,
    thumbnailer: Thumbnailer,
    feature_preview: FeaturePreviewCache,
) -> LightAssetResult:
    from PIL import Image

    prev_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # libvips unlimited parity for Pillow fallback
    last_error = ""
    page_count = 5 if Path(source).suffix.lower() in (".tif", ".tiff") else 1
    try:
        for open_path in iter_fs_path_candidates(source):
            for page in range(page_count):
                try:
                    with Image.open(open_path) as img:
                        try:
                            img.seek(page)
                        except EOFError:
                            if page == 0:
                                raise
                            break
                        metadata = _metadata_from_pillow_image(img)
                        # Tam kopya (img.copy) büyük JPEG/TIF'de bellek kilidi.
                        fp_edge = max(
                            int(thumbnailer.max_edge),
                            int(feature_preview.max_edge),
                        )
                        try:
                            img.draft("RGB", (fp_edge, fp_edge))
                        except Exception:
                            pass
                        normalized = Thumbnailer.normalize_pillow_image(img)
                        fw, fh = _save_resized_webp(
                            normalized,
                            fp_out,
                            feature_preview.max_edge,
                            feature_preview.fmt,
                            feature_preview.quality,
                        )
                        tw, th = _save_resized_webp(
                            normalized,
                            thumb_out,
                            thumbnailer.max_edge,
                            thumbnailer.fmt,
                            thumbnailer.quality,
                        )
                        return LightAssetResult(
                            success=True,
                            metadata=metadata,
                            metadata_status="metadata_ok",
                            metadata_saved=True,
                            thumbnail_path=str(thumb_out),
                            thumbnail_width=tw,
                            thumbnail_height=th,
                            thumbnail_status="thumbnail_ok",
                            feature_preview_path=str(fp_out),
                            feature_preview_width=fw,
                            feature_preview_height=fh,
                            preview_generated=True,
                        )
                except Exception as exc:
                    last_error = str(exc)
                    continue
    finally:
        Image.MAX_IMAGE_PIXELS = prev_limit
    return LightAssetResult(
        success=False,
        error=last_error or "Light asset üretilemedi",
        parser_errors=[last_error] if last_error else [],
    )
