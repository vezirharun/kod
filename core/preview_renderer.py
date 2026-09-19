"""AI/EPS/PDF/PSD önizleme render — bağımlılık yoksa güvenli fallback."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.logger import setup_logger
from core.utils import file_id_from_path, fs_access_path, normalize_path

logger = setup_logger(__name__)


@dataclass
class RenderResult:
    success: bool
    image_path: str = ""
    error: str = ""
    unsupported_preview: bool = False
    renderer: str = ""


def _try_pymupdf(source: str, out_path: Path) -> RenderResult:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(source)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pix.save(str(out_path))
        doc.close()
        return RenderResult(success=True, image_path=str(out_path), renderer="pymupdf")
    except ImportError:
        return RenderResult(success=False, error="pymupdf yok", unsupported_preview=True)
    except Exception as exc:
        msg = str(exc)
        low = msg.lower()
        if "permission" in low or isinstance(exc, PermissionError):
            return RenderResult(success=False, error=f"pymupdf_permission:{msg}")
        if any(
            t in low
            for t in ("not a pdf", "no objects", "cannot open", "format error", "damaged")
        ):
            return RenderResult(success=False, error=f"corrupt_eps_ai:{msg}")
        return RenderResult(success=False, error=f"pymupdf_render_error:{msg}")


def _try_psd_tools(source: str, out_path: Path) -> RenderResult:
    try:
        from psd_tools import PSDImage

        psd = PSDImage.open(source)
        img = psd.composite()
        img.save(out_path, "PNG")
        return RenderResult(
            success=True, image_path=str(out_path), renderer="psd-tools"
        )
    except ImportError:
        return RenderResult(success=False, error="psd-tools yok")
    except Exception as exc:
        return RenderResult(success=False, error=str(exc))


def _eps_ai_preview_gate(path: str | Path) -> tuple[bool, str]:
    """Reject pure-white / unusable EPS/AI rasters. Shared via blank.py."""
    from explorer_preview.host.blank import is_valid_preview_image

    ok, reason = is_valid_preview_image(path)
    if ok:
        return True, "ok"
    if reason in ("blank_white", "near_white") or "white" in reason:
        return False, f"blank_white_preview:{reason}"
    return False, f"invalid_preview:{reason}"


def _unlink_bad_preview(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _try_ghostscript(source: str, out_path: Path) -> RenderResult:
    """Ghostscript rasterize — PATH + common Windows install dirs."""
    import shutil

    candidates: list[str] = []
    for name in ("gswin64c", "gswin32c", "gs"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    # PATH dışında kurulu GS (sık Windows durumu)
    for base in (
        Path(r"C:\Program Files\gs"),
        Path(r"C:\Program Files (x86)\gs"),
    ):
        if not base.is_dir():
            continue
        try:
            for sub in sorted(base.iterdir(), reverse=True):
                for exe in ("gswin64c.exe", "gswin32c.exe"):
                    p = sub / "bin" / exe
                    if p.is_file():
                        candidates.append(str(p))
        except OSError:
            pass
    # dedupe preserve order
    seen: set[str] = set()
    gs_bins: list[str] = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            gs_bins.append(c)

    if not gs_bins:
        logger.warning("Ghostscript bulunamadı (gswin64c/gs PATH veya Program Files)")
        return RenderResult(
            success=False,
            error="dependency eksik: Ghostscript (gswin64c) yok",
            unsupported_preview=True,
            renderer="",
        )

    last_err = ""
    access = fs_access_path(source)
    ext = Path(source).suffix.lower()
    is_eps_ai = ext in (".eps", ".ai")
    # EPS/AI: -dEPSCrop fixes negative-Y BoundingBox → white page; FitPage fallback.
    eps_flag_sets: list[list[str]] = (
        [["-dEPSCrop"], ["-dEPSFitPage"]] if is_eps_ai else [[]]
    )

    for gs in gs_bins:
        for extra in eps_flag_sets:
            try:
                if out_path.exists():
                    _unlink_bad_preview(out_path)
                proc = subprocess.run(
                    [
                        gs,
                        "-dSAFER",
                        "-dBATCH",
                        "-dNOPAUSE",
                        "-dFirstPage=1",
                        "-dLastPage=1",
                        "-r72",
                        "-sDEVICE=png16m",
                        *extra,
                        f"-sOutputFile={out_path}",
                        access,
                    ],
                    check=False,
                    capture_output=True,
                    timeout=120,
                )
                if (
                    out_path.exists()
                    and out_path.stat().st_size > 0
                    and proc.returncode == 0
                ):
                    if is_eps_ai:
                        vok, vreason = _eps_ai_preview_gate(out_path)
                        if not vok:
                            _unlink_bad_preview(out_path)
                            last_err = vreason
                            logger.warning(
                                "Ghostscript EPS blank reject %s: %s flags=%s",
                                access,
                                vreason,
                                extra,
                            )
                            continue
                    return RenderResult(
                        success=True,
                        image_path=str(out_path),
                        renderer="ghostscript",
                    )
                err = (proc.stderr or proc.stdout or b"").decode(
                    "utf-8", errors="replace"
                )[:400]
                last_err = (
                    f"ghostscript_render_error:gs rc={proc.returncode}: "
                    f"{err or 'boş çıktı'}"
                )
                logger.warning("Ghostscript render başarısız %s: %s", access, last_err)
            except FileNotFoundError:
                last_err = f"bulunamadı: {gs}"
                break
            except subprocess.TimeoutExpired:
                last_err = "ghostscript_timeout:120s"
                logger.warning("Ghostscript timeout: %s", access)
                continue
            except PermissionError as exc:
                last_err = f"ghostscript_permission:{exc}"
                logger.warning("Ghostscript permission %s: %s", access, exc)
                continue
            except Exception as exc:
                last_err = f"ghostscript_render_error:{exc}"
                logger.warning("Ghostscript exception %s: %s", access, exc)
                continue
    # Executable existed but render failed — not renderer_missing.
    return RenderResult(
        success=False,
        error=last_err or "ghostscript_render_error:unknown",
        unsupported_preview=False,
    )


def resolve_display_image_path(source_path: str, cache_dir: str = "") -> RenderResult:
    """UI önizleme için raster yol üret (PSD/AI/EPS/PDF/SVG).

    Önce mevcut cache/thumbnail/feature_preview, sonra render_preview_for_index.
    """
    source = normalize_path(source_path)
    if not source or not Path(source).is_file():
        return RenderResult(success=False, error="dosya yok")

    ext = Path(source).suffix.lower()
    # Normal raster — doğrudan kullanılabilir
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}:
        return RenderResult(success=True, image_path=source, renderer="native")

    # Mevcut thumbnail / feature preview / render cache (Preview Pool SSOT)
    try:
        from core.settings import AppSettings
        from core.thumbnailer import Thumbnailer

        settings = AppSettings.load()
        cache = cache_dir or settings.cache_dir
        thumb = Thumbnailer(cache, settings.thumbnail_max_edge, settings.thumbnail_format)
        tpath = thumb.thumbnail_path_for(source)
        if tpath.is_file() and tpath.stat().st_size > 0:
            return RenderResult(success=True, image_path=str(tpath), renderer="thumb_cache")

        from core.preview_cache import FeaturePreviewCache
        from core.thumb_resolve import lookup_valid_preview

        # Valid Preview Pool before any source re-open / re-render
        pool = lookup_valid_preview(source, cache)
        if pool:
            return RenderResult(success=True, image_path=pool, renderer="feature_cache")

        fp = FeaturePreviewCache(cache, settings.feature_preview_max_edge)
        # EPS/AI: must pass blank gate (get_existing); other formats keep size>0 hit.
        if ext in {".eps", ".ai"}:
            ge = fp.get_existing(source)
            if ge.success and ge.preview_path:
                return RenderResult(
                    success=True,
                    image_path=ge.preview_path,
                    renderer="feature_cache",
                )
        else:
            fpath = fp.preview_path_for(source)
            if fpath.is_file() and fpath.stat().st_size > 0:
                return RenderResult(
                    success=True, image_path=str(fpath), renderer="feature_cache"
                )

        # DB kayıtlı yollar
        from core.db import Database

        rec = Database(settings.db_path, read_only=True).get_file_by_path(source)
        if rec:
            for key in ("thumbnail_path", "feature_preview_path"):
                p = str(rec.get(key) or "").strip()
                if p and Path(p).is_file() and Path(p).stat().st_size > 0:
                    return RenderResult(success=True, image_path=p, renderer=f"db_{key}")
    except Exception as exc:
        logger.debug("display cache lookup failed: %s", exc)
        cache = cache_dir or str(Path(__file__).resolve().parents[1] / "cache")

    # Özel format render
    rendered = render_preview_for_index(source, cache)
    if rendered.success and rendered.image_path:
        return rendered

    # Thumbnailer ile son deneme — create() itself prefers Preview Pool SSOT
    try:
        from core.settings import AppSettings
        from core.thumbnailer import Thumbnailer

        settings = AppSettings.load()
        thumb = Thumbnailer(
            cache_dir or settings.cache_dir,
            settings.thumbnail_max_edge,
            settings.thumbnail_format,
        )
        tr = thumb.create(source)
        if tr.success and tr.thumbnail_path:
            return RenderResult(
                success=True, image_path=tr.thumbnail_path, renderer="thumbnailer"
            )
    except Exception as exc:
        logger.debug("thumbnailer display fallback: %s", exc)

    return rendered if rendered.error else RenderResult(
        success=False, error="önizleme üretilemedi", unsupported_preview=True
    )


def load_display_qpixmap(source_path: str, cache_dir: str = ""):
    """Qt QPixmap — native veya render edilmiş önizleme."""
    from PySide6.QtGui import QImage, QPixmap

    direct = QPixmap(source_path)
    if not direct.isNull():
        return direct, ""

    resolved = resolve_display_image_path(source_path, cache_dir=cache_dir)
    if not resolved.success or not resolved.image_path:
        return QPixmap(), resolved.error or "önizleme yok"

    pix = QPixmap(resolved.image_path)
    if not pix.isNull():
        return pix, ""

    # numpy/PIL yolu
    try:
        from core.thumbnailer import Thumbnailer
        import numpy as np

        arr = Thumbnailer.load_image(resolved.image_path)
        if arr is None:
            return QPixmap(), resolved.error or "görsel okunamadı"
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            h, w = arr.shape
            qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, c = arr.shape
            if c >= 3:
                rgb = np.ascontiguousarray(arr[:, :, :3])
                bytes_per_line = 3 * w
                qimg = QImage(
                    rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                )
            else:
                return QPixmap(), "desteklenmeyen kanal"
        return QPixmap.fromImage(qimg.copy()), ""
    except Exception as exc:
        return QPixmap(), str(exc)


def render_preview_for_index(source_path: str, cache_dir: str) -> RenderResult:
    """Desteklenmeyen veya vektör formatlar için raster önizleme üret."""
    source = normalize_path(source_path)
    access = fs_access_path(source)
    ext = Path(source).suffix.lower()
    cache = Path(cache_dir) / "render_previews"
    cache.mkdir(parents=True, exist_ok=True)
    # file_id is md5(normalized full path) — distinguishes same stem + different ext/path
    out_path = cache / f"{file_id_from_path(source)}_render.png"

    if out_path.exists() and out_path.stat().st_size > 0:
        if ext in (".eps", ".ai"):
            vok, vreason = _eps_ai_preview_gate(out_path)
            if vok:
                return RenderResult(
                    success=True, image_path=str(out_path), renderer="cache"
                )
            # Old white/unusable artifact — drop and re-render.
            _unlink_bad_preview(out_path)
            logger.info("EPS/AI cache reopen %s: %s", source, vreason)
        else:
            return RenderResult(
                success=True, image_path=str(out_path), renderer="cache"
            )

    if ext == ".pdf":
        r = _try_pymupdf(access, out_path)
        if r.success:
            return r
        return _try_ghostscript(access, out_path)

    if ext in (".eps", ".ai"):
        # Birçok .ai dosyası PDF uyumludur — fallback zinciri korunur.
        r_mu = _try_pymupdf(access, out_path)
        if r_mu.success:
            vok, vreason = _eps_ai_preview_gate(r_mu.image_path or out_path)
            if vok:
                return r_mu
            _unlink_bad_preview(out_path)
            r_mu = RenderResult(
                success=False, error=vreason, unsupported_preview=False
            )
        r_gs = _try_ghostscript(access, out_path)
        if r_gs.success:
            return r_gs

        mu_err = (r_mu.error or "").strip()
        gs_err = (r_gs.error or "").strip()
        mu_missing = mu_err.lower() in ("pymupdf yok",) or mu_err.lower().startswith(
            "pymupdf yok"
        )
        gs_missing = "dependency eksik" in gs_err.lower() or (
            "ghostscript" in gs_err.lower() and "yok" in gs_err.lower()
        )
        # Blank/invalid preview is never renderer_missing.
        # renderer_missing ONLY when executable/library truly absent.
        if mu_missing and gs_missing:
            return RenderResult(
                success=False,
                error="renderer_missing:EPS/AI render için Ghostscript veya PyMuPDF gerekli",
                unsupported_preview=True,
            )
        # Prefer concrete root cause (GS present ⇒ never renderer_missing).
        parts: list[str] = []
        if mu_err and not mu_missing:
            parts.append(mu_err)
        if gs_err and not gs_missing:
            parts.append(gs_err)
        elif gs_missing and not mu_missing:
            parts.append(gs_err)
        elif mu_missing and not gs_missing:
            parts.append(mu_err)
        if not parts:
            parts = [gs_err or mu_err or "eps_ai_render_failed"]
        return RenderResult(
            success=False,
            error="; ".join(parts)[:500],
            unsupported_preview=False,
        )

    if ext == ".psd":
        r = _try_psd_tools(access, out_path)
        if r.success:
            return r
        # Pillow fallback
        try:
            from PIL import Image

            from core.thumbnailer import Thumbnailer

            with Image.open(access) as img:
                img = Thumbnailer.normalize_pillow_image(img)
                img.save(out_path, "PNG")
            return RenderResult(
                success=True, image_path=str(out_path), renderer="pillow_psd"
            )
        except Exception as exc:
            return RenderResult(
                success=False,
                error=str(exc),
                unsupported_preview=True,
            )

    if ext in {".tif", ".tiff"}:
        try:
            from core.thumbnailer import Thumbnailer
            from core.settings import AppSettings

            settings = AppSettings.load()
            thumb = Thumbnailer(cache_dir, settings.thumbnail_max_edge, settings.thumbnail_format)
            tr = thumb.create(source)
            if tr.success and tr.thumbnail_path:
                return RenderResult(
                    success=True, image_path=tr.thumbnail_path, renderer="tif_thumb"
                )
        except Exception as exc:
            return RenderResult(success=False, error=str(exc), unsupported_preview=True)

    if ext == ".cdr":
        return RenderResult(
            success=False,
            error="CDR doğrudan desteklenmiyor — Inkscape/LibreOffice gerekli",
            unsupported_preview=True,
        )

    return RenderResult(
        success=False, error="format desteklenmiyor", unsupported_preview=True
    )
