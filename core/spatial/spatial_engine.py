"""Read-only spatial search over object_index. Never writes production indexes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.object_index import ObjectIndexStore
from core.spatial.spatial_config import SpatialConfig
from core.spatial.spatial_geometry import normalize_bbox
from core.spatial.spatial_query import parse_spatial_query
from core.spatial.spatial_relation import Relation, evaluate_relation

SPATIAL_EVIDENCE_UNAVAILABLE = "SPATIAL_EVIDENCE_UNAVAILABLE"
SPATIAL_EVIDENCE_AVAILABLE = "SPATIAL_EVIDENCE_AVAILABLE"


@dataclass
class SpatialHit:
    file_id: int
    spatial_score: float
    spatial_confidence: float
    spatial_relation: str
    spatial_evidence: dict[str, Any]
    explanation: str
    path: str = ""


@dataclass
class SpatialSearchResult:
    hits: list[SpatialHit] = field(default_factory=list)
    status: str = "ok"
    spec: dict[str, Any] | None = None
    explanation: str = ""
    availability: str = SPATIAL_EVIDENCE_UNAVAILABLE


def _instance_name(inst: dict[str, Any], counts: dict[str, int]) -> str:
    raw = str(inst.get("instance_id") or "").strip()
    if raw:
        return raw
    lab = str(inst.get("label") or "obj")
    counts[lab] = counts.get(lab, 0) + 1
    return f"{lab}_{counts[lab]:02d}"


def _norm_inst(inst: dict[str, Any], image_w: float, image_h: float):
    box = inst.get("bbox")
    return normalize_bbox(box, image_w, image_h)


class SpatialEngine:
    def __init__(
        self,
        db_path: str,
        *,
        settings: Any | None = None,
        readonly: bool = True,
        textile_db_path: str | None = None,
    ):
        if textile_db_path:
            from core.textile_motif_evidence import TextileMotifEvidenceStore

            self.store = TextileMotifEvidenceStore(textile_db_path, readonly=readonly)
        else:
            self.store = ObjectIndexStore(db_path, readonly=readonly)
        self.cfg = SpatialConfig.from_settings(settings)

    def search(
        self,
        text: str,
        *,
        limit: int = 400,
        image_sizes: dict[int, tuple[int, int]] | None = None,
    ) -> SpatialSearchResult:
        spec = parse_spatial_query(text)
        if not spec:
            return SpatialSearchResult(status="not_spatial", explanation="not a spatial query")
        if self.store.readonly and getattr(self.store, "_missing", False):
            return SpatialSearchResult(
                status=SPATIAL_EVIDENCE_UNAVAILABLE,
                spec=spec,
                explanation=SPATIAL_EVIDENCE_UNAVAILABLE,
                availability=SPATIAL_EVIDENCE_UNAVAILABLE,
            )
        labels_a = spec["object_a"]["labels"]
        labels_b = spec["object_b"]["labels"]
        relation: Relation = spec["relation"]
        ids_a = self.store.search_label(labels_a, limit=max(limit, 4000))
        ids_b = self.store.search_label(labels_b, limit=max(limit, 4000))
        if not ids_a or not ids_b:
            return SpatialSearchResult(
                status=SPATIAL_EVIDENCE_UNAVAILABLE,
                spec=spec,
                explanation=SPATIAL_EVIDENCE_UNAVAILABLE,
                availability=SPATIAL_EVIDENCE_UNAVAILABLE,
            )
        candidate_ids = self.store.search_labels_and(
            [
                {"labels": labels_a, "min_count": 1},
                {"labels": labels_b, "min_count": 1},
            ],
            limit=max(limit, 4000),
        )
        if not candidate_ids:
            return SpatialSearchResult(
                status="ok",
                spec=spec,
                hits=[],
                explanation="✗ relation not verified",
                availability=SPATIAL_EVIDENCE_AVAILABLE,
            )

        sizes = dict(image_sizes or {})
        by_file = self.store.instances_for_files(candidate_ids)
        paths = self.store.file_paths(candidate_ids)
        hits: list[SpatialHit] = []
        for fid in candidate_ids:
            insts = by_file.get(fid) or []
            iw, ih = sizes.get(int(fid), (0, 0))
            if (not iw or not ih) and insts:
                # Pixel boxes without size: skip file rather than invent geometry.
                sample = insts[0].get("bbox") or (0, 0, 0, 0)
                try:
                    if max(float(x) for x in sample) <= 1.0001:
                        iw, ih = 1.0, 1.0
                except (TypeError, ValueError):
                    pass
            anchors = [x for x in insts if str(x.get("label") or "") in labels_a]
            locateds = [x for x in insts if str(x.get("label") or "") in labels_b]
            if not anchors or not locateds:
                continue
            name_counts: dict[str, int] = {}
            named = {id(x): _instance_name(x, name_counts) for x in insts}
            best = None
            for anc in anchors:
                na = _norm_inst(anc, iw, ih)
                if na is None:
                    continue
                for loc in locateds:
                    if loc is anc:
                        continue
                    nb = _norm_inst(loc, iw, ih)
                    if nb is None:
                        continue
                    verdict = evaluate_relation(nb, na, relation, self.cfg)
                    if not verdict.ok:
                        continue
                    conf = min(
                        verdict.confidence,
                        float(anc.get("confidence") or 0),
                        float(loc.get("confidence") or 0),
                    )
                    pack = (verdict.score, conf, anc, loc, verdict, na, nb)
                    if best is None or pack[0] > best[0]:
                        best = pack
            if best is None:
                continue
            score, conf, anc, loc, verdict, na, nb = best
            a_name = named.get(id(anc), str(anc.get("label")))
            b_name = named.get(id(loc), str(loc.get("label")))
            evidence = {
                "anchor_instance": a_name,
                "located_instance": b_name,
                "anchor_label": anc.get("label"),
                "located_label": loc.get("label"),
                "anchor_bbox": [na.x1, na.y1, na.x2, na.y2],
                "located_bbox": [nb.x1, nb.y1, nb.x2, nb.y2],
                "anchor_confidence": float(anc.get("confidence") or 0),
                "located_confidence": float(loc.get("confidence") or 0),
                "file_id": int(fid),
                "source": "object_bbox",
                "clip_as_relation": False,
            }
            hits.append(
                SpatialHit(
                    file_id=int(fid),
                    spatial_score=float(score),
                    spatial_confidence=float(conf),
                    spatial_relation=str(relation.value),
                    spatial_evidence=evidence,
                    explanation=f"✓ found: {b_name} {relation.value} {a_name}",
                    path=paths.get(int(fid), ""),
                )
            )
        hits.sort(key=lambda h: h.spatial_score, reverse=True)
        return SpatialSearchResult(
            hits=hits[: int(limit)],
            status="ok",
            spec=spec,
            explanation="✓ found" if hits else "✗ relation not verified",
            availability=SPATIAL_EVIDENCE_AVAILABLE,
        )
