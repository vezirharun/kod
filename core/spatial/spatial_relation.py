"""Instance-level spatial relations. Geometry only — not CLIP."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot

from core.spatial.spatial_config import SpatialConfig
from core.spatial.spatial_geometry import (
    NormBox,
    bbox_distance,
    center_distance,
    containment_ratio,
    image_diagonal,
    iou,
)


class Relation(str, Enum):
    LEFT_OF = "LEFT_OF"
    RIGHT_OF = "RIGHT_OF"
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    CENTER = "CENTER"
    UPPER_LEFT = "UPPER_LEFT"
    UPPER_RIGHT = "UPPER_RIGHT"
    LOWER_LEFT = "LOWER_LEFT"
    LOWER_RIGHT = "LOWER_RIGHT"
    NEAR = "NEAR"
    FAR = "FAR"
    OVERLAPS = "OVERLAPS"
    INSIDE = "INSIDE"
    CONTAINS = "CONTAINS"


@dataclass(frozen=True)
class RelationVerdict:
    ok: bool
    relation: Relation
    score: float
    confidence: float
    reason: str


def _size_scale(a: NormBox, b: NormBox, cfg: SpatialConfig) -> float:
    size = (a.area_ratio + b.area_ratio) ** 0.5
    return 1.0 + float(cfg.size_k) * size


def _axis_ok(delta: float, cfg: SpatialConfig) -> bool:
    return delta >= float(cfg.axis_center_min)


def evaluate_relation(
    located: NormBox,
    anchor: NormBox,
    relation: Relation,
    cfg: SpatialConfig | None = None,
) -> RelationVerdict:
    """True if `located` stands in `relation` relative to `anchor`.

    Query form "A'nın REL B" → B is located, A is anchor.
    """
    cfg = cfg or SpatialConfig()
    diag = image_diagonal()
    scale = _size_scale(located, anchor, cfg)
    cx_d = located.center_x - anchor.center_x
    cy_d = located.center_y - anchor.center_y
    cdist = center_distance(located, anchor)
    bdist = bbox_distance(located, anchor)
    near_c = float(cfg.near_center_ratio) * diag * scale
    near_b = float(cfg.near_bbox_ratio) * diag * scale
    far_c = float(cfg.far_center_ratio) * diag * scale

    def _done(ok: bool, score: float, conf: float, reason: str) -> RelationVerdict:
        return RelationVerdict(ok, relation, max(0.0, min(1.0, score)), max(0.0, min(1.0, conf)), reason)

    if relation is Relation.LEFT_OF:
        ok = cx_d < 0 and _axis_ok(-cx_d, cfg)
        mag = min(1.0, max(0.0, -cx_d) / 0.5)
        return _done(ok, mag, mag if ok else 0.0, "left_of" if ok else "not_left")
    if relation is Relation.RIGHT_OF:
        ok = cx_d > 0 and _axis_ok(cx_d, cfg)
        mag = min(1.0, max(0.0, cx_d) / 0.5)
        return _done(ok, mag, mag if ok else 0.0, "right_of" if ok else "not_right")
    if relation is Relation.ABOVE:
        ok = cy_d < 0 and _axis_ok(-cy_d, cfg)
        mag = min(1.0, max(0.0, -cy_d) / 0.5)
        return _done(ok, mag, mag if ok else 0.0, "above" if ok else "not_above")
    if relation is Relation.BELOW:
        ok = cy_d > 0 and _axis_ok(cy_d, cfg)
        mag = min(1.0, max(0.0, cy_d) / 0.5)
        return _done(ok, mag, mag if ok else 0.0, "below" if ok else "not_below")
    if relation is Relation.UPPER_LEFT:
        ok = cx_d < 0 and cy_d < 0 and _axis_ok(-cx_d, cfg) and _axis_ok(-cy_d, cfg)
        mag = min(1.0, hypot(max(0.0, -cx_d), max(0.0, -cy_d)) / 0.7)
        return _done(ok, mag, mag if ok else 0.0, "upper_left" if ok else "not_upper_left")
    if relation is Relation.UPPER_RIGHT:
        ok = cx_d > 0 and cy_d < 0 and _axis_ok(cx_d, cfg) and _axis_ok(-cy_d, cfg)
        mag = min(1.0, hypot(max(0.0, cx_d), max(0.0, -cy_d)) / 0.7)
        return _done(ok, mag, mag if ok else 0.0, "upper_right" if ok else "not_upper_right")
    if relation is Relation.LOWER_LEFT:
        ok = cx_d < 0 and cy_d > 0 and _axis_ok(-cx_d, cfg) and _axis_ok(cy_d, cfg)
        mag = min(1.0, hypot(max(0.0, -cx_d), max(0.0, cy_d)) / 0.7)
        return _done(ok, mag, mag if ok else 0.0, "lower_left" if ok else "not_lower_left")
    if relation is Relation.LOWER_RIGHT:
        ok = cx_d > 0 and cy_d > 0 and _axis_ok(cx_d, cfg) and _axis_ok(cy_d, cfg)
        mag = min(1.0, hypot(max(0.0, cx_d), max(0.0, cy_d)) / 0.7)
        return _done(ok, mag, mag if ok else 0.0, "lower_right" if ok else "not_lower_right")
    if relation is Relation.CENTER:
        in_anchor = containment_ratio(located, anchor) >= float(cfg.inside_ratio) * 0.5
        near_mid = cdist <= near_c * 0.65
        ax_ok = cfg.center_lo <= located.center_x <= cfg.center_hi
        ay_ok = cfg.center_lo <= located.center_y <= cfg.center_hi
        ok = (in_anchor and near_mid) or (ax_ok and ay_ok and cdist <= near_c)
        mag = max(0.0, 1.0 - cdist / max(near_c, 1e-6))
        return _done(ok, mag, mag if ok else 0.0, "center" if ok else "not_center")
    if relation is Relation.NEAR:
        ok = (cdist <= near_c) or (bdist <= near_b and cdist < far_c)
        mag = max(0.0, 1.0 - cdist / max(near_c, 1e-6))
        return _done(ok, mag, mag if ok else 0.0, "near" if ok else "not_near")
    if relation is Relation.FAR:
        far_unscaled = float(cfg.far_center_ratio) * diag
        ok = cdist >= far_unscaled and bdist > near_b
        mag = min(1.0, cdist / max(diag, 1e-6))
        return _done(ok, mag, mag if ok else 0.0, "far" if ok else "not_far")
    if relation is Relation.OVERLAPS:
        v = iou(located, anchor)
        ok = v >= float(cfg.overlap_iou_min)
        return _done(ok, v, v if ok else 0.0, "overlaps" if ok else "no_overlap")
    if relation is Relation.INSIDE:
        v = containment_ratio(located, anchor)
        ok = v >= float(cfg.inside_ratio)
        return _done(ok, v, v if ok else 0.0, "inside" if ok else "not_inside")
    if relation is Relation.CONTAINS:
        v = containment_ratio(anchor, located)
        ok = v >= float(cfg.inside_ratio)
        return _done(ok, v, v if ok else 0.0, "contains" if ok else "not_contains")
    return _done(False, 0.0, 0.0, "unknown_relation")
