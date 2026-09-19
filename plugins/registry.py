"""Format plugin discovery without heavy imports at application startup."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from plugins.base import FormatPlugin, PluginResult, PluginRuntime

RASTER_EXTENSIONS = frozenset(
    {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)
VECTOR_PREVIEW_EXTENSIONS = frozenset({".pdf", ".svg"})
EMBROIDERY_PLACEHOLDER_EXTENSIONS = frozenset(
    {
        ".dst", ".dsb", ".dsz", ".emb", ".pes", ".pec", ".jef",
        ".sew", ".exp", ".vp3", ".hus", ".xxx", ".pcs", ".tap",
        ".ofm", ".ksm", ".art", ".ngs", ".wings",
    }
)
BUILTIN_PLUGIN_EXTENSIONS = (
    RASTER_EXTENSIONS | VECTOR_PREVIEW_EXTENSIONS | EMBROIDERY_PLACEHOLDER_EXTENSIONS
)


class PluginRegistry:
    def __init__(self, runtime: PluginRuntime | None = None):
        self.runtime = runtime or PluginRuntime()
        self._plugins: list[FormatPlugin] = []
        self._by_extension: dict[str, FormatPlugin] = {}
        self._load_builtins()

    def _load_builtins(self) -> None:
        # Modules are lightweight; Pillow/PyMuPDF/CairoSVG remain method-local imports.
        from plugins.embroidery.placeholder import EmbroideryPlaceholderPlugin
        from plugins.raster.plugin import RasterPlugin
        from plugins.vector.plugin import VectorPreviewPlugin

        for plugin_type in (RasterPlugin, VectorPreviewPlugin, EmbroideryPlaceholderPlugin):
            self.register(plugin_type(self.runtime))

    def configure(self, runtime: PluginRuntime) -> None:
        self.runtime = runtime
        for plugin in self._plugins:
            plugin.configure(runtime)

    def register(self, plugin: FormatPlugin) -> None:
        for ext in plugin.supported_extensions():
            normalized = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            if normalized in self._by_extension:
                raise ValueError(f"Uzantı zaten kayıtlı: {normalized}")
            self._by_extension[normalized] = plugin
        self._plugins.append(plugin)

    def resolve(self, path_or_extension: str) -> FormatPlugin | None:
        value = str(path_or_extension or "").lower()
        ext = value if value.startswith(".") and "\\" not in value and "/" not in value else Path(value).suffix
        return self._by_extension.get(ext)

    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._by_extension)

    def describe(self, path: str) -> dict:
        plugin = self.resolve(path)
        return plugin.descriptor(path) if plugin else {}

    @staticmethod
    def safe_call(plugin: FormatPlugin, method: str, path: str) -> PluginResult:
        try:
            result = getattr(plugin, method)(path)
            if not isinstance(result, PluginResult):
                return PluginResult.unsupported(
                    "parser_error", f"{plugin.plugin_name}.{method} geçersiz sonuç döndürdü"
                )
            return result
        except Exception as exc:
            return PluginResult.unsupported("parser_error", str(exc))


_registry: PluginRegistry | None = None
_registry_lock = Lock()


def get_plugin_registry(runtime: PluginRuntime | None = None) -> PluginRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = PluginRegistry(runtime)
        elif runtime is not None:
            _registry.configure(runtime)
        return _registry
