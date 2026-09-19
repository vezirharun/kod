"""VisualMemoryEngine — similarity clusters, not class labels.

Reads existing embeddings / Visual DNA as evidence. Does not rebuild FAISS,
does not write production DB, does not change ranking, does not fine-tune.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.visual_memory.knowledge import knowledge_pack_info
from core.visual_memory.schema import (
    AUTO_CONFIDENCE_CAP,
    STATUS_DISCOVERED,
    STATUS_UNKNOWN,
)
from core.visual_memory.store import VisualMemoryStore, customer_key, memory_db_path

DEFAULT_SIM_THRESHOLD = 0.82
DEFAULT_MIN_SIZE = 3
DEFAULT_MAX_REPRESENTATIVES = 5


@dataclass
class VisualItem:
    file_id: int
    embedding: np.ndarray
    source_id: int = 0
    visual_dna: dict[str, Any] = field(default_factory=dict)
    object_label: str = ""
    color_label: str = ""
    texture_label: str = ""
    family_label: str = ""
    evidence_sources: list[str] = field(default_factory=list)


@dataclass
class DiscoverResult:
    customer: str
    clusters: list[dict[str, Any]]
    unclustered: int
    knowledge_pack: dict[str, Any]
    embedding_kind: str
    sim_threshold: float


def _norm(vec: np.ndarray) -> np.ndarray:
    x = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n < 1e-8:
        return x
    return x / n


def _parse_texture_map(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _hist(values: Iterable[str]) -> dict[str, int]:
    c = Counter(v for v in values if v)
    return dict(c.most_common())


def _dna_summary(items: list[VisualItem]) -> dict[str, Any]:
    fam = _hist(i.family_label for i in items)
    col = _hist(i.color_label for i in items)
    return {
        "n": len(items),
        "dominant_family": next(iter(fam), ""),
        "dominant_color": next(iter(col), ""),
        "note": "summary of observed DNA/metadata; not a claimed class",
    }


def _auto_status(n: int, min_size: int, cohesion: float) -> tuple[str, float]:
    if n < min_size:
        conf = min(AUTO_CONFIDENCE_CAP * 0.4, max(0.05, cohesion * 0.3))
        return STATUS_UNKNOWN, round(conf, 4)
    conf = min(AUTO_CONFIDENCE_CAP, max(0.15, cohesion * AUTO_CONFIDENCE_CAP))
    return STATUS_DISCOVERED, round(conf, 4)


class VisualMemoryEngine:
    def __init__(self, memory_root: str | Path):
        self.memory_root = Path(memory_root)
        self.memory_root.mkdir(parents=True, exist_ok=True)

    def store_for(self, customer: str) -> VisualMemoryStore:
        path = memory_db_path(self.memory_root, customer)
        return VisualMemoryStore(path, customer=customer)

    def discover_clusters(
        self,
        customer: str,
        items: list[VisualItem],
        *,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        min_size: int = DEFAULT_MIN_SIZE,
        persist: bool = True,
        embedding_kind: str = "clip",
    ) -> DiscoverResult:
        usable = [
            it
            for it in items
            if it.embedding is not None and np.linalg.norm(it.embedding) >= 1e-6
        ]
        clusters = _greedy_clusters(
            usable,
            customer=customer_key(customer),
            sim_threshold=sim_threshold,
            min_size=min_size,
            embedding_kind=embedding_kind,
        )
        if persist:
            self.store_for(customer).replace_clusters(clusters)
        unclustered = sum(
            1 for c in clusters if c["status"] == STATUS_UNKNOWN and c["member_count"] == 1
        )
        return DiscoverResult(
            customer=customer_key(customer),
            clusters=clusters,
            unclustered=unclustered,
            knowledge_pack=knowledge_pack_info(),
            embedding_kind=embedding_kind,
            sim_threshold=sim_threshold,
        )

    def shadow_from_production_db(
        self,
        db_path: str | Path,
        *,
        customer: str = "_shadow",
        per_source: int = 20,
        source_limit: int = 9,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        min_size: int = DEFAULT_MIN_SIZE,
    ) -> dict[str, Any]:
        """Read-only sample of production embeddings. Writes only Visual Memory DB."""
        items, sources = load_shadow_items(
            db_path, per_source=per_source, source_limit=source_limit
        )
        result = self.discover_clusters(
            customer,
            items,
            sim_threshold=sim_threshold,
            min_size=min_size,
            persist=True,
            embedding_kind="clip",
        )
        claimed = [c["claimed_class"] for c in result.clusters if c.get("claimed_class")]
        auto_class = [
            c["status"] for c in result.clusters if c["status"] not in (STATUS_UNKNOWN, STATUS_DISCOVERED)
        ]
        return {
            "customer": result.customer,
            "sources_sampled": sources,
            "items": len(items),
            "clusters": len(result.clusters),
            "discovered": sum(1 for c in result.clusters if c["status"] == STATUS_DISCOVERED),
            "unknown": sum(1 for c in result.clusters if c["status"] == STATUS_UNKNOWN),
            "claimed_class_n": len(claimed),
            "illegal_auto_status_n": len(auto_class),
            "top": [
                {
                    "cluster_id": c["cluster_id"],
                    "member_count": c["member_count"],
                    "status": c["status"],
                    "claimed_class": c["claimed_class"],
                    "confidence": c["confidence"],
                    "cohesion": c["cohesion"],
                    "object_distribution": c["object_distribution"],
                    "pattern_family_distribution": c["pattern_family_distribution"],
                }
                for c in sorted(
                    result.clusters, key=lambda x: x["member_count"], reverse=True
                )[:12]
            ],
            "knowledge_pack": result.knowledge_pack,
        }


def _greedy_clusters(
    items: list[VisualItem],
    *,
    customer: str,
    sim_threshold: float,
    min_size: int,
    embedding_kind: str,
) -> list[dict[str, Any]]:
    if not items:
        return []
    vecs = np.stack([_norm(it.embedding) for it in items]).astype(np.float32)
    assigned = np.full(len(items), -1, dtype=np.int32)
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []

    order = list(range(len(items)))
    for i in order:
        if assigned[i] >= 0:
            continue
        best_k, best_s = -1, -1.0
        for k, cen in enumerate(centroids):
            s = float(vecs[i] @ cen)
            if s > best_s:
                best_k, best_s = k, s
        if best_k >= 0 and best_s >= sim_threshold:
            assigned[i] = best_k
            members[best_k].append(i)
            centroids[best_k] = _norm(np.mean(vecs[members[best_k]], axis=0))
        else:
            assigned[i] = len(centroids)
            centroids.append(vecs[i].copy())
            members.append([i])

    out: list[dict[str, Any]] = []
    for k, idxs in enumerate(members):
        cen = centroids[k]
        sims = [float(vecs[i] @ cen) for i in idxs]
        cohesion = float(np.mean(sims)) if sims else 0.0
        group = [items[i] for i in idxs]
        ranked = sorted(zip(idxs, sims), key=lambda t: t[1], reverse=True)
        reps = [int(items[i].file_id) for i, _ in ranked[:DEFAULT_MAX_REPRESENTATIVES]]
        status, conf = _auto_status(len(idxs), min_size, cohesion)
        sources: list[str] = []
        for it in group:
            for s in it.evidence_sources:
                if s not in sources:
                    sources.append(s)
        if embedding_kind and embedding_kind + "_embedding" not in sources:
            sources.insert(0, f"{embedding_kind}_embedding")
        out.append(
            {
                "cluster_id": f"c-{customer}-{k:04d}",
                "embedding_centroid": cen,
                "embedding_dim": int(cen.size),
                "embedding_kind": embedding_kind,
                "member_count": len(idxs),
                "representative_ids": reps,
                "visual_dna_summary": _dna_summary(group),
                "object_distribution": _hist(it.object_label for it in group),
                "color_distribution": _hist(it.color_label for it in group),
                "texture_distribution": _hist(it.texture_label for it in group),
                "pattern_family_distribution": _hist(it.family_label for it in group),
                "confidence": conf,
                "status": status,
                "claimed_class": "",
                "evidence_sources": sources,
                "cohesion": round(cohesion, 4),
                "members": [
                    {
                        "file_id": int(items[i].file_id),
                        "source_id": int(items[i].source_id),
                        "sim_to_centroid": round(float(vecs[i] @ cen), 4),
                        "evidence": {
                            "object": items[i].object_label,
                            "family": items[i].family_label,
                        },
                    }
                    for i in idxs
                ],
            }
        )
    return out


def load_shadow_items(
    db_path: str | Path,
    *,
    per_source: int = 20,
    source_limit: int = 9,
) -> tuple[list[VisualItem], list[dict[str, Any]]]:
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    sources = [
        dict(r)
        for r in con.execute(
            "SELECT id, name, source_type FROM sources ORDER BY id LIMIT ?",
            (int(source_limit),),
        ).fetchall()
    ]
    items: list[VisualItem] = []
    for src in sources:
        rows = con.execute(
            """
            SELECT f.id, f.source_id, f.pattern_family, f.pattern_type,
                   fe.clip_embedding, fe.dino_embedding, fe.texture_map
            FROM files f
            JOIN features fe ON fe.file_id = f.id
            WHERE f.source_id = ?
              AND f.status NOT IN ('missing','excluded_internal')
              AND fe.clip_embedding IS NOT NULL
              AND length(fe.clip_embedding) >= 64
            ORDER BY f.id
            LIMIT ?
            """,
            (int(src["id"]), int(per_source)),
        ).fetchall()
        src["sampled"] = len(rows)
        for r in rows:
            blob = r["clip_embedding"]
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size == 0 or float(np.linalg.norm(vec)) < 1e-6:
                continue
            tm = _parse_texture_map(r["texture_map"])
            family = str(
                tm.get("pattern_family") or r["pattern_family"] or ""
            ).strip()
            obj = str(
                tm.get("animal_print_type") or r["pattern_type"] or ""
            ).strip()
            color = str(tm.get("color_family") or "").strip()
            texture = str(tm.get("texture_class") or tm.get("pattern_type") or "").strip()
            items.append(
                VisualItem(
                    file_id=int(r["id"]),
                    embedding=vec,
                    source_id=int(r["source_id"] or 0),
                    visual_dna=tm,
                    object_label=obj,
                    color_label=color,
                    texture_label=texture,
                    family_label=family,
                    evidence_sources=["clip_embedding", "visual_dna", "indexed_metadata"],
                )
            )
    con.close()
    return items, sources
