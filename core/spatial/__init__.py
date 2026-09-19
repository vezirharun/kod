"""Spatial Intelligence V1 — read-only geometry over the object index.

Not a detector. Not CLIP. Does not write patterns.db, FAISS, or object_index.db.
"""
from __future__ import annotations

from core.spatial.spatial_engine import (
    SPATIAL_EVIDENCE_AVAILABLE,
    SPATIAL_EVIDENCE_UNAVAILABLE,
    SpatialEngine,
    SpatialHit,
)
from core.spatial.spatial_query import parse_spatial_query
from core.spatial.spatial_relation import Relation

__all__ = [
    "SPATIAL_EVIDENCE_AVAILABLE",
    "SPATIAL_EVIDENCE_UNAVAILABLE",
    "Relation",
    "SpatialEngine",
    "SpatialHit",
    "parse_spatial_query",
]
