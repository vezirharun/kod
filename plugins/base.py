"""Common contracts for format plugins.

Plugin methods are synchronous by design. Callers must execute them from the
existing index/background workers, never from the Qt UI thread.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PluginRuntime:
    cache_dir: str = ""
    thumbnail_max_edge: int = 512
    thumbnail_format: str = "webp"
    thumbnailer: Any | None = None


@dataclass
class PluginResult:
    success: bool
    status: str
    artifact_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    semantic_tags: list[str] = field(default_factory=list)
    hash_features: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @classmethod
    def unsupported(cls, status: str = "unsupported", error: str = "") -> "PluginResult":
        return cls(False, status=status, error=error)


class FormatPlugin(ABC):
    plugin_name = "base"
    plugin_version = "1.0"
    format_family = "unknown"
    capabilities: tuple[str, ...] = ()
    metadata_only = False

    def __init__(self, runtime: PluginRuntime | None = None):
        self.runtime = runtime or PluginRuntime()

    def configure(self, runtime: PluginRuntime) -> None:
        self.runtime = runtime

    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        raise NotImplementedError

    def can_open(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.supported_extensions()

    def extract_preview(self, path: str) -> PluginResult:
        return PluginResult.unsupported("unsupported_preview")

    def extract_thumbnail(self, path: str) -> PluginResult:
        return PluginResult.unsupported("unsupported_thumbnail")

    def extract_metadata(self, path: str) -> PluginResult:
        return PluginResult.unsupported("unsupported_metadata")

    def extract_semantic_tags(self, path: str) -> PluginResult:
        return PluginResult(True, status="empty", semantic_tags=[])

    def extract_hash_features(self, path: str) -> PluginResult:
        return PluginResult.unsupported("delegated_to_index_pipeline")

    def extract_stitch_metadata(self, path: str) -> PluginResult:
        return PluginResult.unsupported("stitch_parser_not_implemented")

    def descriptor(self, path: str) -> dict[str, Any]:
        return {
            "format_family": self.format_family,
            "format_extension": Path(path).suffix.lower().lstrip("."),
            "format_capabilities": list(self.capabilities),
            "parser_name": self.plugin_name,
            "parser_version": self.plugin_version,
        }
