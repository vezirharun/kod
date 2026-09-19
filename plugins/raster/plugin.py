"""Low-risk raster formats backed by the existing Thumbnailer/Pillow path."""

from __future__ import annotations

from pathlib import Path

from plugins.base import FormatPlugin, PluginResult
from plugins.registry import RASTER_EXTENSIONS


class RasterPlugin(FormatPlugin):
    plugin_name = "raster_pillow"
    plugin_version = "1.0"
    format_family = "raster"
    capabilities = ("open", "preview", "thumbnail", "metadata", "hash", "semantic")

    def supported_extensions(self) -> frozenset[str]:
        return RASTER_EXTENSIONS

    def can_open(self, path: str) -> bool:
        if not super().can_open(path) or not Path(path).is_file():
            return False
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            return True
        except Exception:
            return False

    def extract_preview(self, path: str) -> PluginResult:
        if not self.can_open(path):
            return PluginResult.unsupported("read_error", "Raster dosyası açılamadı")
        return PluginResult(True, status="source_preview", artifact_path=str(path))

    def extract_thumbnail(self, path: str) -> PluginResult:
        if not self.runtime.cache_dir:
            return PluginResult.unsupported("runtime_not_configured")
        thumbnailer = self.runtime.thumbnailer
        if thumbnailer is None:
            from core.thumbnailer import Thumbnailer

            thumbnailer = Thumbnailer(
                self.runtime.cache_dir,
                self.runtime.thumbnail_max_edge,
                self.runtime.thumbnail_format,
            )
        thumb = thumbnailer.create(path)
        if not thumb.success:
            return PluginResult.unsupported("thumbnail_error", thumb.error)
        return PluginResult(
            True,
            status="thumbnail_ok",
            artifact_path=thumb.thumbnail_path,
            metadata={"width": thumb.width, "height": thumb.height},
        )

    def extract_metadata(self, path: str) -> PluginResult:
        try:
            from PIL import Image

            with Image.open(path) as image:
                from core.metadata_sanitize import sanitize_for_json

                metadata = sanitize_for_json(
                    {
                        "width": int(image.width),
                        "height": int(image.height),
                        "mode": str(image.mode),
                        "image_format": str(image.format or ""),
                        "frames": int(getattr(image, "n_frames", 1) or 1),
                        "dpi": list(image.info.get("dpi", ()))
                        if image.info.get("dpi")
                        else [],
                    }
                )
            return PluginResult(True, status="metadata_ok", metadata=metadata)
        except Exception as exc:
            return PluginResult.unsupported("metadata_error", str(exc))

    def extract_semantic_tags(self, path: str) -> PluginResult:
        return PluginResult(True, status="empty", semantic_tags=[])

    def extract_hash_features(self, path: str) -> PluginResult:
        return PluginResult(True, status="delegated_to_index_pipeline")
