"""Feature preview cache — arama kalitesi için yüksek çözünürlüklü analiz görüntüsü."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.logger import setup_logger
from core.thumbnailer import HAS_VIPS, Thumbnailer, vips_new_from_file
from core.utils import file_id_from_path, iter_fs_path_candidates, normalize_path

logger = setup_logger(__name__)

FEATURE_PREVIEW_VERSION = 1


@dataclass
class FeaturePreviewResult:
    success: bool
    preview_path: str = ""
    width: int = 0
    height: int = 0
    error: str = ""
    unsupported_preview: bool = False


class FeaturePreviewCache:
    """UI thumbnail (512px) ayrı; feature extraction için 1024px normalize cache."""

    def __init__(
        self,
        cache_dir: str,
        max_edge: int = 1024,
        fmt: str = "webp",
        quality: int = 88,
    ):
        self.cache_dir = Path(cache_dir) / "feature_previews"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_edge = max_edge
        self.fmt = fmt.lower()
        self.quality = quality
        self.ext = ".webp" if self.fmt == "webp" else ".jpg"

    def preview_path_for(self, source_path: str) -> Path:
        fid = file_id_from_path(source_path)
        return self.cache_dir / f"{fid}_fp{self.ext}"

    def get_existing(self, source_path: str) -> FeaturePreviewResult:
        """Diskteki önbelleği oku — orijinal dosyayı açmaz."""
        out_path = self.preview_path_for(source_path)
        if not out_path.is_file() or out_path.stat().st_size <= 0:
            return FeaturePreviewResult(success=False)
        try:
            from PIL import Image

            with Image.open(out_path) as img:
                w, h = img.size
            ext = Path(source_path).suffix.lower()
            if ext in {".eps", ".ai"}:
                # Self-heal: white-on-white raster source → keep; colorful/unreadable
                # EPS with solid white preview → reject (EPSCrop reopen path).
                try:
                    from core.preview_self_heal.validate import (
                        Verdict,
                        ValidationResult,
                        validate_preview,
                    )

                    vr = validate_preview(
                        out_path,
                        source_path,
                        allow_source_compare=True,
                        compare_path=None,
                    )
                    if vr.verdict == Verdict.SUSPICIOUS:
                        # Vector source rarely decodes as a bitmap; solid mono
                        # preview without a matching mono raster source → reopen.
                        if "source_matches_mono" not in vr.reason:
                            vr = ValidationResult(
                                Verdict.INVALID,
                                f"blank_white_preview:{vr.reason}",
                                artifact_path=str(out_path),
                            )
                    if vr.verdict == Verdict.INVALID:
                        # Normalize tiny solid cache for legacy test assertions
                        err = vr.reason
                        if err == "too_small":
                            err = "blank_white_preview:too_small"
                        try:
                            out_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        return FeaturePreviewResult(success=False, error=err)
                except Exception:
                    from core.preview_renderer import _eps_ai_preview_gate

                    vok, vreason = _eps_ai_preview_gate(out_path)
                    if not vok:
                        try:
                            out_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        return FeaturePreviewResult(success=False, error=vreason)
            return FeaturePreviewResult(
                success=True,
                preview_path=str(out_path),
                width=w,
                height=h,
            )
        except Exception:
            return FeaturePreviewResult(success=False)

    def create(
        self, source_path: str, render_path: str | None = None
    ) -> FeaturePreviewResult:
        source = normalize_path(source_path)
        load_from = normalize_path(render_path) if render_path else source
        out_path = self.preview_path_for(source)
        src_ext = Path(source).suffix.lower()

        if out_path.exists() and out_path.stat().st_size > 0:
            try:
                from PIL import Image

                with Image.open(out_path) as img:
                    w, h = img.size
                if src_ext in {".eps", ".ai"}:
                    from core.preview_renderer import _eps_ai_preview_gate

                    vok, vreason = _eps_ai_preview_gate(out_path)
                    if not vok:
                        try:
                            out_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        # fall through to re-create
                    else:
                        return FeaturePreviewResult(
                            success=True,
                            preview_path=str(out_path),
                            width=w,
                            height=h,
                        )
                else:
                    return FeaturePreviewResult(
                        success=True,
                        preview_path=str(out_path),
                        width=w,
                        height=h,
                    )
            except Exception:
                pass

        from core.index_freeze import guard_index_write

        guard_index_write("preview.create", "core.preview_cache")

        # Thumbnailer ile aynı: PSD/AI/EPS/PDF doğrudan libvips'e verilmez.
        # Önce format-specific rasterize (PyMuPDF / Ghostscript / psd-tools).
        ext = Path(load_from).suffix.lower()
        skip_vips = ext in {".psd", ".ai", ".eps", ".pdf"}
        if skip_vips and not render_path:
            try:
                from core.preview_renderer import render_preview_for_index

                rendered = render_preview_for_index(
                    source, str(self.cache_dir.parent)
                )
                if rendered.success and rendered.image_path:
                    load_from = normalize_path(rendered.image_path)
                    skip_vips = False  # raster ara çıktı — vips/pillow OK
                else:
                    err = rendered.error or "vektör önizleme üretilemedi"
                    logger.warning("feature preview render %s: %s", source, err)
                    return FeaturePreviewResult(
                        success=False,
                        error=err,
                        unsupported_preview=bool(rendered.unsupported_preview),
                    )
            except Exception as exc:
                logger.warning("feature preview render exception: %s", exc)
                return FeaturePreviewResult(success=False, error=str(exc))

        # EPS/AI: reject white-page mis-renders; keep true mono-on-mono sources.
        if src_ext in {".eps", ".ai"} and load_from:
            try:
                from core.preview_self_heal.validate import (
                    Verdict,
                    ValidationResult,
                    validate_preview,
                )

                vr = validate_preview(
                    load_from,
                    source,
                    allow_source_compare=True,
                    compare_path=source,
                )
                if vr.verdict == Verdict.SUSPICIOUS and "source_matches_mono" not in vr.reason:
                    vr = ValidationResult(
                        Verdict.INVALID,
                        f"blank_white_preview:{vr.reason}",
                        artifact_path=str(load_from),
                    )
                if vr.verdict == Verdict.INVALID:
                    logger.warning(
                        "feature preview self-heal reject %s: %s", source, vr.reason
                    )
                    return FeaturePreviewResult(success=False, error=vr.reason)
            except Exception:
                from core.preview_renderer import _eps_ai_preview_gate

                vok, vreason = _eps_ai_preview_gate(load_from)
                if not vok:
                    logger.warning(
                        "feature preview blank reject %s: %s", source, vreason
                    )
                    return FeaturePreviewResult(success=False, error=vreason)

        if HAS_VIPS and not skip_vips:
            result = self._create_vips(load_from, out_path)
            if result.success:
                return result

        return self._create_pillow(load_from, out_path)

    def _create_vips(self, source: str, out_path: Path) -> FeaturePreviewResult:
        last_error = ""
        image = None
        try:
            image = vips_new_from_file(
                source, access="sequential", page=0
            )
            w, h = image.width, image.height
            scale = self.max_edge / max(w, h)
            if scale < 1.0:
                image = image.resize(scale)
            if self.fmt == "webp":
                image.webpsave(str(out_path), Q=self.quality)
            else:
                image.jpegsave(str(out_path), Q=self.quality)
            return FeaturePreviewResult(
                success=True,
                preview_path=str(out_path),
                width=image.width,
                height=image.height,
            )
        except Exception as exc:
            last_error = str(exc)
            logger.debug("pyvips feature preview hatası %s: %s", source, exc)
        finally:
            image = None
        return FeaturePreviewResult(success=False, error=str(last_error))

    def _create_pillow(self, source: str, out_path: Path) -> FeaturePreviewResult:
        from PIL import Image

        last_error = ""
        for open_path in iter_fs_path_candidates(source):
            for page in range(5):
                try:
                    with Image.open(open_path) as img:
                        try:
                            img.seek(page)
                        except EOFError:
                            if page == 0:
                                raise
                            break
                        img = Thumbnailer.normalize_pillow_image(img)
                        w, h = img.size
                        scale = self.max_edge / max(w, h)
                        if scale < 1.0:
                            img = img.resize(
                                (max(1, int(w * scale)), max(1, int(h * scale))),
                                Image.Resampling.LANCZOS,
                            )
                        save_kw = {"quality": self.quality}
                        if self.fmt == "webp":
                            img.save(out_path, "WEBP", **save_kw)
                        else:
                            img.save(out_path, "JPEG", **save_kw)
                        return FeaturePreviewResult(
                            success=True,
                            preview_path=str(out_path),
                            width=img.width,
                            height=img.height,
                        )
                except Exception as exc:
                    last_error = str(exc)
                    continue
        return FeaturePreviewResult(
            success=False, error=last_error or "Feature preview üretilemedi"
        )

    def load_array(self, preview_path: str):
        return Thumbnailer.load_image(preview_path)
