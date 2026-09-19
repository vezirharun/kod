"""Registry-only embroidery support; no stitch parser is loaded in this phase."""

from __future__ import annotations

from pathlib import Path

from plugins.base import FormatPlugin, PluginResult
from plugins.registry import EMBROIDERY_PLACEHOLDER_EXTENSIONS


class EmbroideryPlaceholderPlugin(FormatPlugin):
    plugin_name = "embroidery_placeholder"
    plugin_version = "1.0"
    format_family = "embroidery"
    capabilities = ("metadata_pending", "stitch_supported_later")
    metadata_only = True

    def supported_extensions(self) -> frozenset[str]:
        return EMBROIDERY_PLACEHOLDER_EXTENSIONS

    def can_open(self, path: str) -> bool:
        return super().can_open(path) and Path(path).is_file()

    def extract_metadata(self, path: str) -> PluginResult:
        try:
            stat = Path(path).stat()
            return PluginResult(
                True,
                status="metadata_pending",
                metadata={
                    "file_size": int(stat.st_size),
                    "placeholder": True,
                    "stitch_parser": "planned",
                },
            )
        except OSError as exc:
            return PluginResult.unsupported("metadata_error", str(exc))

    def extract_preview(self, path: str) -> PluginResult:
        return PluginResult.unsupported("pending")

    def extract_thumbnail(self, path: str) -> PluginResult:
        return PluginResult.unsupported("pending")

    def extract_stitch_metadata(self, path: str) -> PluginResult:
        return PluginResult.unsupported("stitch_supported_later")
