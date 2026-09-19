"""Normalized boxes and pairwise geometry. Coords are 0–1."""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any


@dataclass(frozen=True)
class NormBox:
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float
    area_ratio: float


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def image_diagonal() -> float:
    return hypot(1.0, 1.0)


def normalize_bbox(
    bbox: Any,
    image_w: float | int | None = None,
    image_h: float | int | None = None,
) -> NormBox | None:
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    max_c = max(abs(x1), abs(y1), abs(x2), abs(y2))
    already_norm = max_c <= 1.0001
    iw = float(image_w or 0)
    ih = float(image_h or 0)
    if already_norm:
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    elif iw > 0 and ih > 0:
        nx1, nx2 = x1 / iw, x2 / iw
        ny1, ny2 = y1 / ih, y2 / ih
    else:
        return None
    nx1, ny1, nx2, ny2 = _clip01(nx1), _clip01(ny1), _clip01(nx2), _clip01(ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    w, h = nx2 - nx1, ny2 - ny1
    area = w * h
    return NormBox(
        x1=nx1,
        y1=ny1,
        x2=nx2,
        y2=ny2,
        center_x=(nx1 + nx2) / 2.0,
        center_y=(ny1 + ny2) / 2.0,
        width=w,
        height=h,
        area_ratio=area,
    )


def center_distance(a: NormBox, b: NormBox) -> float:
    return hypot(a.center_x - b.center_x, a.center_y - b.center_y)


def bbox_distance(a: NormBox, b: NormBox) -> float:
    dx = max(0.0, a.x1 - b.x2, b.x1 - a.x2)
    dy = max(0.0, a.y1 - b.y2, b.y1 - a.y2)
    return hypot(dx, dy)


def intersection_area(a: NormBox, b: NormBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a: NormBox, b: NormBox) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = a.area_ratio + b.area_ratio - inter
    return float(inter / union) if union > 0 else 0.0


def containment_ratio(inner: NormBox, outer: NormBox) -> float:
    """Fraction of inner area that lies inside outer."""
    if inner.area_ratio <= 0:
        return 0.0
    return intersection_area(inner, outer) / inner.area_ratio
