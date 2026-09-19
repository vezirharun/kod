"""Spatial relation thresholds. Values come from AppSettings, not call-site literals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpatialConfig:
    near_center_ratio: float = 0.28
    near_bbox_ratio: float = 0.12
    far_center_ratio: float = 0.42
    size_k: float = 1.25
    axis_center_min: float = 0.04
    overlap_iou_min: float = 0.02
    inside_ratio: float = 0.85
    center_lo: float = 0.33
    center_hi: float = 0.67

    @classmethod
    def from_settings(cls, settings: Any | None) -> "SpatialConfig":
        if settings is None:
            return cls()
        return cls(
            near_center_ratio=float(getattr(settings, "spatial_near_center_ratio", 0.28)),
            near_bbox_ratio=float(getattr(settings, "spatial_near_bbox_ratio", 0.12)),
            far_center_ratio=float(getattr(settings, "spatial_far_center_ratio", 0.42)),
            size_k=float(getattr(settings, "spatial_size_k", 1.25)),
            axis_center_min=float(getattr(settings, "spatial_axis_center_min", 0.04)),
            overlap_iou_min=float(getattr(settings, "spatial_overlap_iou_min", 0.02)),
            inside_ratio=float(getattr(settings, "spatial_inside_ratio", 0.85)),
            center_lo=float(getattr(settings, "spatial_center_lo", 0.33)),
            center_hi=float(getattr(settings, "spatial_center_hi", 0.67)),
        )
