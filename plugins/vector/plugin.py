"""Safe first-page PDF and optional SVG preview adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree

from plugins.base import FormatPlugin, PluginResult
from plugins.registry import VECTOR_PREVIEW_EXTENSIONS


class VectorPreviewPlugin(FormatPlugin):
    plugin_name = "pdf_svg_preview"
    plugin_version = "1.0"
    format_family = "vector"
    capabilities = ("open", "preview", "thumbnail", "metadata", "hash")

    def supported_extensions(self) -> frozenset[str]:
        return VECTOR_PREVIEW_EXTENSIONS

    def extract_preview(self, path: str) -> PluginResult:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            from core.preview_renderer import render_preview_for_index

            result = render_preview_for_index(path, self.runtime.cache_dir)
            if result.success:
                return PluginResult(True, "preview_ok", artifact_path=result.image_path)
            return PluginResult.unsupported(
                "unsupported_preview" if result.unsupported_preview else "preview_error",
                result.error,
            )
        if ext == ".svg":
            return self._svg_preview(path)
        return PluginResult.unsupported("unsupported_preview")

    def _svg_preview(self, path: str) -> PluginResult:
        try:
            root = ElementTree.parse(path).getroot()
            for element in root.iter():
                tag = str(element.tag).lower()
                if tag.endswith("script"):
                    return PluginResult.unsupported("unsafe_content", "SVG script içeriyor")
                for key, value in element.attrib.items():
                    if key.lower().endswith("href") and str(value).lower().startswith(
                        ("http:", "https:", "file:")
                    ):
                        return PluginResult.unsupported("unsafe_content", "SVG dış kaynak içeriyor")
        except Exception as exc:
            return PluginResult.unsupported("metadata_error", str(exc))

        try:
            import cairosvg
        except ImportError:
            return PluginResult.unsupported("missing_dependency", "cairosvg yok")
        cache = Path(self.runtime.cache_dir) / "render_previews"
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()
        target = cache / f"{key}_svg.png"
        try:
            cairosvg.svg2png(
                url=str(path),
                write_to=str(target),
                output_width=min(2048, max(256, self.runtime.thumbnail_max_edge * 2)),
            )
            return PluginResult(True, "preview_ok", artifact_path=str(target))
        except Exception as exc:
            return PluginResult.unsupported("preview_error", str(exc))

    def extract_thumbnail(self, path: str) -> PluginResult:
        preview = self.extract_preview(path)
        if not preview.success:
            return preview
        from core.thumbnailer import Thumbnailer

        thumb = Thumbnailer(
            self.runtime.cache_dir,
            self.runtime.thumbnail_max_edge,
            self.runtime.thumbnail_format,
        ).create(preview.artifact_path)
        if not thumb.success:
            return PluginResult.unsupported("thumbnail_error", thumb.error)
        return PluginResult(
            True, "thumbnail_ok", artifact_path=thumb.thumbnail_path,
            metadata={"width": thumb.width, "height": thumb.height},
        )

    def extract_metadata(self, path: str) -> PluginResult:
        if Path(path).suffix.lower() == ".pdf":
            try:
                import fitz

                doc = fitz.open(path)
                data = {"pages": int(doc.page_count), **dict(doc.metadata or {})}
                doc.close()
                return PluginResult(True, "metadata_ok", metadata=data)
            except ImportError:
                return PluginResult.unsupported("missing_dependency", "pymupdf yok")
            except Exception as exc:
                return PluginResult.unsupported("metadata_error", str(exc))
        try:
            root = ElementTree.parse(path).getroot()
            data = {
                "width": root.attrib.get("width", ""),
                "height": root.attrib.get("height", ""),
                "viewBox": root.attrib.get("viewBox", ""),
            }
            return PluginResult(True, "metadata_ok", metadata=data)
        except Exception as exc:
            return PluginResult.unsupported("metadata_error", str(exc))
