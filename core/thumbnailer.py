"""Büyük TIF dosyalarından küçük thumbnail üretimi."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import os

import numpy as np
from PIL import Image, ImageOps

from core.logger import setup_logger
from core.utils import file_id_from_path, fs_access_path, iter_fs_path_candidates, normalize_path

logger = setup_logger(__name__)

# pyvips opsiyonel. Windows'ta uygulama PATH'inde libvips yoksa
# paketlenmiş/runtime libvips klasörlerini otomatik keşfet.
pyvips = None
HAS_VIPS = False


def _configure_vips_runtime() -> None:
    if os.name != "nt":
        return
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.environ.get("VEZIR_LIBVIPS_BIN", ""),
        str(project_root / "vips" / "bin"),
        str(project_root / "libvips" / "bin"),
        str(project_root / "runtime" / "libvips" / "bin"),
        str(project_root / "third_party" / "libvips" / "bin"),
        str(project_root / "third_party" / "vips" / "bin"),
    ]
    for raw in candidates:
        if not raw:
            continue
        bin_dir = Path(raw)
        dll = bin_dir / "libvips-42.dll"
        if not dll.is_file():
            continue
        try:
            os.add_dll_directory(str(bin_dir))
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        return


try:
    _configure_vips_runtime()
    import pyvips as _pyvips

    pyvips = _pyvips
    HAS_VIPS = True
    # libvips varsayılanı CPU×thread havuzu açabilir; index sırasında
    # native thread sayısı şişmesin. Decode zaten lane/pool ile sınırlı.
    try:
        os.environ.setdefault("VIPS_CONCURRENCY", "1")
        if hasattr(pyvips, "concurrency_set"):
            pyvips.concurrency_set(1)
        elif hasattr(pyvips, "cache_set_max"):
            pass
    except Exception:
        pass
    try:
        # Opsiyonel: bellek cache sınırla (PSD resource block OOM riski).
        if hasattr(pyvips, "cache_set_max_mem"):
            pyvips.cache_set_max_mem(64 * 1024 * 1024)
        if hasattr(pyvips, "cache_set_max"):
            pyvips.cache_set_max(100)
    except Exception:
        pass
    logger.info("pyvips/libvips aktif: %s", ".".join(map(str, (
        pyvips.version(0), pyvips.version(1), pyvips.version(2)
    ))))
except Exception as exc:
    logger.info("pyvips/libvips bulunamadı, Pillow fallback kullanılacak: %s", exc)


def vips_new_from_file(source: str, **kwargs):
    """libvips dosya açışı.

    ``\\\\?\\`` öneki ve Unicode Windows yolları GLib/libvips'te
    ``VipsForeignLoad: ... is not a known file format`` üretebilir.
    Önce kanonik yol, sonra OS handle / bellek tamponu.
    """
    if pyvips is None:
        raise RuntimeError("pyvips_unavailable")
    canonical = normalize_path(source)
    ext = Path(canonical).suffix.lower()
    load_kw = dict(kwargs)
    if ext not in {".tif", ".tiff"}:
        load_kw.pop("page", None)
    last_exc: BaseException | None = None
    try:
        return pyvips.Image.new_from_file(canonical, **load_kw)
    except Exception as exc:
        last_exc = exc
    access = fs_access_path(canonical)
    source_cls = getattr(pyvips, "Source", None)
    if source_cls is not None and hasattr(source_cls, "new_from_descriptor"):
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        fd = os.open(access, flags)
        try:
            vs = source_cls.new_from_descriptor(fd)
            return pyvips.Image.new_from_source(vs, "", **load_kw)
        except Exception as exc:
            last_exc = exc
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
    with open(access, "rb") as fh:
        data = fh.read()
    buf_kw = {k: v for k, v in load_kw.items() if k != "access"}
    try:
        return pyvips.Image.new_from_buffer(data, "", **buf_kw)
    except Exception:
        if last_exc is not None:
            raise last_exc
        raise


@dataclass
class ThumbnailResult:
    success: bool
    thumbnail_path: str = ""
    width: int = 0
    height: int = 0
    error: str = ""


class Thumbnailer:
    def __init__(
        self,
        cache_dir: str,
        max_edge: int = 512,
        fmt: str = "webp",
        quality: int = 85,
    ):
        self.cache_dir = Path(cache_dir) / "thumbnails"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_edge = max_edge
        self.fmt = fmt.lower()
        self.quality = quality
        self.ext = ".webp" if self.fmt == "webp" else ".jpg"

    def thumbnail_path_for(self, source_path: str) -> Path:
        fid = file_id_from_path(source_path)
        return self.cache_dir / f"{fid}{self.ext}"

    def create(self, source_path: str) -> ThumbnailResult:
        source = normalize_path(source_path)
        ext = Path(source).suffix.lower()
        out_path = self.thumbnail_path_for(source)
        if out_path.exists() and out_path.stat().st_size > 0:
            eps_ai = ext in {".eps", ".ai"}
            if eps_ai:
                # Do not HIT on tiny/white EPS stubs — unlink and fall through.
                usable = False
                try:
                    from core.thumb_resolve import is_eps_ai_thumb_usable

                    usable = is_eps_ai_thumb_usable(out_path)
                except Exception:
                    try:
                        from core.preview_renderer import _eps_ai_preview_gate

                        usable, _ = _eps_ai_preview_gate(out_path)
                    except Exception:
                        usable = out_path.stat().st_size >= 512
                if not usable:
                    try:
                        out_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                else:
                    try:
                        with Image.open(out_path) as img:
                            w, h = img.size
                        return ThumbnailResult(
                            success=True,
                            thumbnail_path=str(out_path),
                            width=w,
                            height=h,
                        )
                    except Exception:
                        pass
            else:
                try:
                    with Image.open(out_path) as img:
                        w, h = img.size
                    return ThumbnailResult(
                        success=True,
                        thumbnail_path=str(out_path),
                        width=w,
                        height=h,
                    )
                except Exception:
                    pass

        from core.index_freeze import guard_index_write

        guard_index_write("thumbnail.create", "core.thumbnailer")

        # Preview Pool SSOT: valid preview → thumb (no NAS/source re-read).
        # Applies to TIFF/jumbo and all formats before any source open.
        try:
            from core.preview_cache import FeaturePreviewCache

            fp = FeaturePreviewCache(str(self.cache_dir.parent))
            existing = fp.get_existing(source)
            if existing.success and existing.preview_path:
                logger.info(
                    "thumbnail SSOT preview→thumb source=%s preview=%s",
                    source,
                    existing.preview_path,
                )
                return self.create_from_existing_preview(
                    source, existing.preview_path
                )
        except Exception as exc:
            logger.debug("preview_pool thumb reuse: %s", exc)

        if ext in (".tif", ".tiff"):
            from core.tif_thumbnail import create_tif_thumbnail

            return create_tif_thumbnail(self, source)

        if HAS_VIPS and ext not in {".psd", ".ai", ".eps", ".pdf"}:
            result = self._create_vips(source, out_path)
            if result.success:
                return result

        # PSD / AI / PDF / EPS — önce mevcut feature/render cache, sonra rasterize.
        # NOT: Eski kod TIF_TIMEOUT (5–8s) kullanıyordu; Ghostscript/NAS bu süreyi
        # aşınca "Thumbnail oluşturulamadı" oluyordu. Vektör için ayrı süre.
        from core.settings import OPTIONAL_PREVIEW_EXTENSIONS

        VECTOR_THUMB_TIMEOUT_SEC = 120.0

        def _render_or_pillow() -> ThumbnailResult:
            if ext in OPTIONAL_PREVIEW_EXTENSIONS or ext in {".psd", ".ai", ".eps", ".pdf"}:
                try:
                    from core.preview_renderer import render_preview_for_index

                    rendered = render_preview_for_index(
                        source, str(self.cache_dir.parent)
                    )
                    if rendered.success and rendered.image_path:
                        return self._create_pillow(rendered.image_path, out_path)
                    if rendered.error:
                        logger.warning(
                            "preview render failed %s: %s", source, rendered.error
                        )
                        return ThumbnailResult(
                            success=False,
                            error=rendered.error
                            or "önizleme render başarısız",
                        )
                except Exception as exc:
                    logger.warning("preview render thumbnail: %s", exc)
                    return ThumbnailResult(success=False, error=str(exc))
            return self._create_pillow(source, out_path)

        if ext in {".psd", ".ai", ".eps", ".pdf"}:
            from concurrent.futures import TimeoutError as FuturesTimeout

            from core.tif_thumbnail import _reset_tif_pool, _submit_tif

            try:
                return _submit_tif(_render_or_pillow).result(
                    timeout=VECTOR_THUMB_TIMEOUT_SEC
                )
            except FuturesTimeout:
                _reset_tif_pool(reason=f"thumbnail timeout {ext}")
                logger.error(
                    "thumbnail timeout %ss for %s %s",
                    VECTOR_THUMB_TIMEOUT_SEC,
                    ext,
                    source,
                )
                return ThumbnailResult(
                    success=False,
                    error=f"timeout (thumbnail {ext}, {int(VECTOR_THUMB_TIMEOUT_SEC)}s)",
                )
            except RuntimeError as exc:
                if "shutdown" in str(exc).lower():
                    _reset_tif_pool(reason=str(exc))
                    return ThumbnailResult(success=False, error=str(exc))
                raise
        return _render_or_pillow()

    def create_from_existing_preview(self, source_path: str, preview_path: str) -> ThumbnailResult:
        """Mevcut Preview'dan Thumbnail üretir; kaynak dosyayı yeniden okumaz."""
        source = normalize_path(source_path)
        preview = normalize_path(preview_path)
        out_path = self.thumbnail_path_for(source)
        if not Path(preview).is_file() or Path(preview).stat().st_size <= 0:
            return ThumbnailResult(success=False, error="preview_missing")
        if out_path.is_file() and out_path.stat().st_size > 0:
            eps_ai = Path(source).suffix.lower() in {".eps", ".ai"}
            if eps_ai:
                usable = False
                try:
                    from core.thumb_resolve import is_eps_ai_thumb_usable

                    usable = is_eps_ai_thumb_usable(out_path)
                except Exception:
                    try:
                        from core.preview_renderer import _eps_ai_preview_gate

                        usable, _ = _eps_ai_preview_gate(out_path)
                    except Exception:
                        usable = out_path.stat().st_size >= 512
                if not usable:
                    # Invalid tiny/white — overwrite from preview instead of HIT.
                    try:
                        out_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                else:
                    try:
                        with Image.open(out_path) as img:
                            w, h = img.size
                        return ThumbnailResult(
                            success=True,
                            thumbnail_path=str(out_path),
                            width=w,
                            height=h,
                        )
                    except Exception:
                        pass
            else:
                try:
                    with Image.open(out_path) as img:
                        w, h = img.size
                    return ThumbnailResult(
                        success=True,
                        thumbnail_path=str(out_path),
                        width=w,
                        height=h,
                    )
                except Exception:
                    pass
        from core.index_freeze import guard_index_write

        guard_index_write("thumbnail.create", "core.thumbnailer")
        try:
            if HAS_VIPS:
                result = self._create_vips(preview, out_path)
                if result.success:
                    return result
            return self._create_pillow(preview, out_path)
        except Exception as exc:
            return ThumbnailResult(success=False, error=str(exc))

    def _create_vips(self, source: str, out_path: Path) -> ThumbnailResult:
        last_error = ""
        for open_path in (normalize_path(source),):
            image = None
            try:
                image = vips_new_from_file(
                    open_path,
                    access="sequential",
                    page=0,
                )
                w, h = image.width, image.height
                scale = self.max_edge / max(w, h)
                if scale < 1.0:
                    image = image.resize(scale)
                if self.fmt == "webp":
                    image.webpsave(str(out_path), Q=self.quality)
                else:
                    image.jpegsave(str(out_path), Q=self.quality)
                return ThumbnailResult(
                    success=True,
                    thumbnail_path=str(out_path),
                    width=image.width,
                    height=image.height,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.debug("pyvips thumbnail hatası %s: %s", open_path, exc)
            finally:
                image = None
        return ThumbnailResult(success=False, error=last_error)

    def _create_pillow(self, source: str, out_path: Path) -> ThumbnailResult:
        """Çok sayfalı TIFF desteği ile Pillow fallback."""
        last_error = ""
        page_count = 5 if Path(source).suffix.lower() in (".tif", ".tiff") else 1
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
                        img = self.normalize_pillow_image(img)
                        w, h = img.size
                        scale = self.max_edge / max(w, h)
                        if scale < 1.0:
                            new_w = max(1, int(w * scale))
                            new_h = max(1, int(h * scale))
                            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        save_kw = {"quality": self.quality}
                        if self.fmt == "webp":
                            img.save(out_path, "WEBP", **save_kw)
                        else:
                            img.save(out_path, "JPEG", **save_kw)
                        return ThumbnailResult(
                            success=True,
                            thumbnail_path=str(out_path),
                            width=img.width,
                            height=img.height,
                        )
                except Image.DecompressionBombError as exc:
                    return ThumbnailResult(success=False, error=str(exc))
                except Exception as exc:
                    last_error = str(exc)
                    logger.debug(
                        "Pillow sayfa %d hatası %s: %s", page, open_path, exc
                    )
                    continue
        return ThumbnailResult(
            success=False, error=last_error or "Thumbnail üretilemedi"
        )

    @staticmethod
    def normalize_pillow_image(img: Image.Image) -> Image.Image:
        """Create one stable RGB representation for PNG/JPG/TIFF variants."""
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            return bg.convert("RGB")
        if img.mode == "CMYK":
            return img.convert("RGB")
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.array(img).astype(np.float32)
            if arr.size:
                mn = float(arr.min())
                mx = float(arr.max())
                if mx > mn:
                    arr = (arr - mn) * (255.0 / (mx - mn))
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
        return img.convert("RGB")

    @staticmethod
    def load_image(path: str) -> np.ndarray | None:
        from io import BytesIO

        from PIL import Image as _Image

        # Büyük TIFF / archive: Pillow pixel limit'i kapat (libvips unlimited parity)
        prev_limit = _Image.MAX_IMAGE_PIXELS
        _Image.MAX_IMAGE_PIXELS = None
        try:
            ext = Path(normalize_path(path)).suffix.lower()
            if HAS_VIPS and pyvips is not None and ext in (".tif", ".tiff"):
                try:
                    for open_path in (normalize_path(path),):
                        image = vips_new_from_file(
                            open_path, access="sequential", page=0
                        )
                        edge = max(int(image.width), int(image.height))
                        if edge > 2048:
                            image = image.resize(2048.0 / float(edge))
                        buf = image.write_to_buffer(".jpg[Q=90]")
                        with _Image.open(BytesIO(buf)) as img:
                            return np.array(Thumbnailer.normalize_pillow_image(img))
                except Exception as exc:
                    logger.debug("pyvips load_image fallback %s: %s", path, exc)
            with _Image.open(fs_access_path(path)) as img:
                return np.array(Thumbnailer.normalize_pillow_image(img))
        except Exception as exc:
            logger.warning("Görsel yüklenemedi %s: %s", path, exc)
            return None
        finally:
            _Image.MAX_IMAGE_PIXELS = prev_limit

    @staticmethod
    def extract_patches(
        image: np.ndarray,
        grid: int = 3,
    ) -> list[np.ndarray]:
        """Görselden grid parçaları çıkar (3x3 = 9 patch)."""
        h, w = image.shape[:2]
        patches: list[np.ndarray] = []
        step_h = h // grid
        step_w = w // grid
        for row in range(grid):
            for col in range(grid):
                y1 = row * step_h
                x1 = col * step_w
                y2 = h if row == grid - 1 else (row + 1) * step_h
                x2 = w if col == grid - 1 else (col + 1) * step_w
                patch = image[y1:y2, x1:x2]
                if patch.size > 0:
                    patches.append(patch)
        return patches
