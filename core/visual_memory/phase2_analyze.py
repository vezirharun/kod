"""Phase 2 analysis + split prototype. Does not mutate Phase 1 store or ranking.

candidate_splits are suggestions only.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.visual_memory.engine import (
    DEFAULT_MIN_SIZE,
    DEFAULT_SIM_THRESHOLD,
    VisualItem,
    _greedy_clusters,
    _parse_texture_map,
)
from core.visual_memory.schema import (
    AUTO_CONFIDENCE_CAP,
    STATUS_DISCOVERED,
    STATUS_UNKNOWN,
)

MIN_PART = 2
SPLIT_GAIN_MIN = 0.05


def _norm_rows(m: np.ndarray) -> np.ndarray:
    x = np.asarray(m, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def _pair_mean(vecs: np.ndarray) -> float:
    if vecs is None or len(vecs) < 2:
        return 1.0
    s = _norm_rows(vecs) @ _norm_rows(vecs).T
    iu = np.triu_indices(len(vecs), 1)
    return float(s[iu].mean())


def _cross_mean(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    return float((_norm_rows(a) @ _norm_rows(b).T).mean())


def _centroid_separation(centroids: list[np.ndarray]) -> dict[str, float]:
    if len(centroids) < 2:
        return {"mean": 1.0, "min": 1.0, "n_pairs": 0}
    m = _norm_rows(np.stack(centroids))
    s = m @ m.T
    iu = np.triu_indices(len(m), 1)
    vals = s[iu]
    return {
        "mean": round(float(vals.mean()), 4),
        "min": round(float(vals.min()), 4),
        "n_pairs": int(vals.size),
        "separation_mean": round(float(1.0 - vals.mean()), 4),
        "separation_min": round(float(1.0 - vals.min()), 4),
    }


@dataclass
class Phase2Item:
    file_id: int
    source_id: int = 0
    clip: np.ndarray | None = None
    dino: np.ndarray | None = None
    family: str = ""
    obj: str = ""
    color: str = ""
    texture: str = ""
    scale: str = ""


def _clean(label: str) -> str:
    v = (label or "").strip().lower()
    if v in ("", "unknown", "none"):
        return ""
    return v


def _scale_bucket(tm: dict[str, Any]) -> str:
    raw = tm.get("scale_score", tm.get("scale"))
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return ""
    if x < 0.33:
        return "small"
    if x < 0.66:
        return "medium"
    return "large"


def load_phase2_items(
    db_path: str | Path,
    *,
    per_source: int = 20,
    source_limit: int = 5,
) -> tuple[list[Phase2Item], list[dict[str, Any]]]:
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
    items: list[Phase2Item] = []
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
        src["dino_n"] = 0
        for r in rows:
            clip = np.frombuffer(r["clip_embedding"], dtype=np.float32)
            if clip.size == 0 or float(np.linalg.norm(clip)) < 1e-6:
                continue
            dino = None
            blob = r["dino_embedding"]
            if blob is not None and len(blob) >= 64:
                dvec = np.frombuffer(blob, dtype=np.float32)
                if dvec.size and float(np.linalg.norm(dvec)) >= 1e-6:
                    dino = dvec
                    src["dino_n"] = int(src["dino_n"]) + 1
            tm = _parse_texture_map(r["texture_map"])
            items.append(
                Phase2Item(
                    file_id=int(r["id"]),
                    source_id=int(r["source_id"] or 0),
                    clip=clip,
                    dino=dino,
                    family=_clean(str(tm.get("pattern_family") or r["pattern_family"] or "")),
                    obj=_clean(str(tm.get("animal_print_type") or r["pattern_type"] or "")),
                    color=_clean(str(tm.get("color_family") or "")),
                    texture=_clean(str(tm.get("texture_class") or "")),
                    scale=_scale_bucket(tm),
                )
            )
    con.close()
    return items, sources


def phase1_clusters_from_clip(
    items: list[Phase2Item],
    *,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    min_size: int = DEFAULT_MIN_SIZE,
) -> list[dict[str, Any]]:
    vis = [
        VisualItem(
            file_id=it.file_id,
            embedding=it.clip,
            source_id=it.source_id,
            object_label=it.obj,
            color_label=it.color,
            texture_label=it.texture,
            family_label=it.family,
            evidence_sources=["clip_embedding"],
        )
        for it in items
        if it.clip is not None
    ]
    return _greedy_clusters(
        vis,
        customer="_phase2",
        sim_threshold=sim_threshold,
        min_size=min_size,
        embedding_kind="clip",
    )


def _by_id(items: list[Phase2Item]) -> dict[int, Phase2Item]:
    return {it.file_id: it for it in items}


def _mixed(dist: dict[str, int], *, min_part: int = MIN_PART) -> bool:
    parts = [n for k, n in dist.items() if k and n >= min_part]
    return len(parts) >= 2


def _channel_split_scores(members: list[Phase2Item], key: str) -> dict[str, Any]:
    groups: dict[str, list[Phase2Item]] = defaultdict(list)
    for it in members:
        lab = _clean(getattr(it, key))
        if lab:
            groups[lab].append(it)
    big = {k: v for k, v in groups.items() if len(v) >= MIN_PART}
    if len(big) < 2:
        return {"separable": False, "groups": sorted(big, key=lambda k: -len(big[k]))}
    names = list(big)
    out: dict[str, Any] = {
        "separable": True,
        "groups": {k: len(big[k]) for k in names},
        "clip": {},
        "dino": {},
    }
    for channel in ("clip", "dino"):
        intra = []
        inter = []
        mats = {}
        for k, grp in big.items():
            vecs = [getattr(x, channel) for x in grp if getattr(x, channel) is not None]
            if len(vecs) >= 2:
                mats[k] = np.stack(vecs)
                intra.append(_pair_mean(mats[k]))
        keys = list(mats)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                inter.append(_cross_mean(mats[keys[i]], mats[keys[j]]))
        intra_m = float(np.mean(intra)) if intra else None
        inter_m = float(np.mean(inter)) if inter else None
        gain = None
        if intra_m is not None and inter_m is not None:
            gain = round(intra_m - inter_m, 4)
        out[channel] = {
            "intra": None if intra_m is None else round(intra_m, 4),
            "inter": None if inter_m is None else round(inter_m, 4),
            "gain": gain,
            "n_with_vec": sum(len(v) for v in mats.values()),
        }
    out["visual_dna"] = {
        "note": "family/object labels are the split hypothesis, not an independent embedding",
        "gain": None,
    }
    best = "none"
    best_gain = -1.0
    for ch in ("dino", "clip"):
        g = out[ch].get("gain")
        if g is None:
            continue
        if g > best_gain:
            best_gain = g
            best = ch
    out["best_channel"] = best
    out["best_gain"] = round(best_gain, 4) if best_gain >= 0 else None
    out["recommend_split"] = bool(best != "none" and best_gain >= SPLIT_GAIN_MIN)
    return out


def analyze_clusters(
    items: list[Phase2Item],
    clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    idx = _by_id(items)
    cohesions = [float(c.get("cohesion") or 0) for c in clusters]
    cents = []
    mixed_family = []
    mixed_object = []
    color_variant = []
    scale_variant = []
    texture_variant = []
    candidates = []

    for c in clusters:
        members = [idx[m["file_id"]] for m in (c.get("members") or []) if m["file_id"] in idx]
        if c.get("embedding_centroid") is not None:
            cents.append(np.asarray(c["embedding_centroid"], dtype=np.float32))
        fam = c.get("pattern_family_distribution") or {}
        obj = c.get("object_distribution") or {}
        row = {
            "cluster_id": c["cluster_id"],
            "member_count": c["member_count"],
            "status": c["status"],
            "cohesion": c.get("cohesion"),
            "families": fam,
            "objects": obj,
        }
        if _mixed(fam):
            mixed_family.append(row)
            scores = _channel_split_scores(members, "family")
            candidates.append(
                {
                    "cluster_id": c["cluster_id"],
                    "reason": "mixed_family",
                    "member_count": c["member_count"],
                    "clip_cohesion": c.get("cohesion"),
                    "distribution": fam,
                    "scores": scores,
                    "why": (
                        "CLIP glued different pattern families into one high-cohesion group."
                    ),
                    "split_by": scores.get("best_channel"),
                    "apply": False,
                }
            )
        if _mixed(obj):
            mixed_object.append(row)
            scores = _channel_split_scores(members, "obj")
            candidates.append(
                {
                    "cluster_id": c["cluster_id"],
                    "reason": "mixed_object",
                    "member_count": c["member_count"],
                    "clip_cohesion": c.get("cohesion"),
                    "distribution": obj,
                    "scores": scores,
                    "why": "CLIP mixed competing object labels in one cluster.",
                    "split_by": scores.get("best_channel"),
                    "apply": False,
                }
            )
        colors = {}
        textures = {}
        scales = {}
        for it in members:
            if it.color:
                colors[it.color] = colors.get(it.color, 0) + 1
            if it.texture:
                textures[it.texture] = textures.get(it.texture, 0) + 1
            if it.scale:
                scales[it.scale] = scales.get(it.scale, 0) + 1
        fam_parts = [k for k, n in fam.items() if k and n >= 2]
        if len(fam_parts) <= 1 and _mixed(colors):
            color_variant.append({**row, "colors": colors})
        if len(fam_parts) <= 1 and _mixed(textures):
            texture_variant.append({**row, "textures": textures})
        if len(fam_parts) <= 1 and _mixed(scales):
            scale_variant.append({**row, "scales": scales})

    largest = max(clusters, key=lambda c: c["member_count"]) if clusters else {}
    return {
        "cluster_count": len(clusters),
        "largest_cluster": {
            "cluster_id": largest.get("cluster_id"),
            "member_count": largest.get("member_count"),
            "cohesion": largest.get("cohesion"),
            "families": largest.get("pattern_family_distribution") or {},
            "objects": largest.get("object_distribution") or {},
            "status": largest.get("status"),
        },
        "mean_cohesion": round(float(np.mean(cohesions)), 4) if cohesions else 0.0,
        "inter_cluster_separation": _centroid_separation(cents),
        "mixed_family_clusters": mixed_family,
        "mixed_object_clusters": mixed_object,
        "color_variant_clusters": color_variant,
        "texture_variant_clusters": texture_variant,
        "scale_variant_clusters": scale_variant,
        "candidate_splits": candidates,
        "discovered_n": sum(1 for c in clusters if c["status"] == STATUS_DISCOVERED),
        "unknown_n": sum(1 for c in clusters if c["status"] == STATUS_UNKNOWN),
    }


def _label_hist(members: list[Phase2Item], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for it in members:
        lab = _clean(getattr(it, attr))
        if lab:
            counts[lab] = counts.get(lab, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _clip_cohesion(members: list[Phase2Item]) -> float:
    vecs = [it.clip for it in members if it.clip is not None]
    if len(vecs) < 2:
        return 1.0
    return round(_pair_mean(np.stack(vecs)), 4)


def _cluster_from_members(
    cluster_id: str,
    members: list[Phase2Item],
    *,
    status: str = STATUS_DISCOVERED,
) -> dict[str, Any]:
    clips = [it.clip for it in members if it.clip is not None]
    cen = _norm_rows(np.mean(_norm_rows(np.stack(clips)), axis=0))[0] if clips else None
    cohesion = _clip_cohesion(members)
    conf = min(AUTO_CONFIDENCE_CAP, max(0.15, cohesion * AUTO_CONFIDENCE_CAP))
    return {
        "cluster_id": cluster_id,
        "embedding_centroid": cen,
        "embedding_dim": int(cen.size) if cen is not None else 0,
        "embedding_kind": "clip",
        "member_count": len(members),
        "representative_ids": [int(it.file_id) for it in members[:5]],
        "visual_dna_summary": {"n": len(members), "note": "phase2 shadow split; not a class"},
        "object_distribution": _label_hist(members, "obj"),
        "color_distribution": _label_hist(members, "color"),
        "texture_distribution": _label_hist(members, "texture"),
        "pattern_family_distribution": _label_hist(members, "family"),
        "confidence": round(conf, 4),
        "status": status,
        "claimed_class": "",
        "evidence_sources": ["clip_embedding", "dino_embedding", "pattern_family"],
        "cohesion": cohesion,
        "members": [
            {
                "file_id": int(it.file_id),
                "source_id": int(it.source_id),
                "sim_to_centroid": 0.0,
                "evidence": {"family": it.family, "object": it.obj},
            }
            for it in members
        ],
    }


def apply_dino_gated_family_split(
    items: list[Phase2Item],
    phase1_clusters: list[dict[str, Any]],
    *,
    dino_gain_min: float = SPLIT_GAIN_MIN,
    min_family_n: int = MIN_PART,
) -> dict[str, Any]:
    """Shadow-only. Does not write Phase 1 store, FAISS, or ranking."""
    before = analyze_clusters(items, phase1_clusters)
    idx = _by_id(items)
    after: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for c in phase1_clusters:
        members = [idx[m["file_id"]] for m in (c.get("members") or []) if m["file_id"] in idx]
        status = str(c.get("status") or STATUS_UNKNOWN)
        fam = c.get("pattern_family_distribution") or _label_hist(members, "family")
        gate_ok = status in (STATUS_UNKNOWN, STATUS_DISCOVERED) and _mixed(fam, min_part=min_family_n)
        if not gate_ok:
            if _mixed(fam, min_part=min_family_n):
                skipped.append({"cluster_id": c["cluster_id"], "reason": "status_not_auto"})
            after.append({**c, "parent_cluster_id": "", "split_reason": ""})
            continue
        scores = _channel_split_scores(members, "family")
        dino = scores.get("dino") or {}
        gain = dino.get("gain")
        if not scores.get("recommend_split") or gain is None or gain < dino_gain_min:
            skipped.append(
                {
                    "cluster_id": c["cluster_id"],
                    "reason": "dino_gate_failed",
                    "dino_gain": gain,
                    "clip_gain": (scores.get("clip") or {}).get("gain"),
                }
            )
            after.append({**c, "parent_cluster_id": "", "split_reason": ""})
            continue

        groups: dict[str, list[Phase2Item]] = defaultdict(list)
        remnant: list[Phase2Item] = []
        for it in members:
            lab = _clean(it.family)
            if lab:
                groups[lab].append(it)
            else:
                remnant.append(it)
        eligible = {k: v for k, v in groups.items() if len(v) >= min_family_n}
        if len(eligible) < 2:
            after.append({**c, "parent_cluster_id": "", "split_reason": ""})
            skipped.append({"cluster_id": c["cluster_id"], "reason": "insufficient_family_parts"})
            continue
        for k, v in groups.items():
            if k not in eligible:
                remnant.extend(v)

        child_i = 0
        for family, grp in sorted(eligible.items(), key=lambda kv: -len(kv[1])):
            child_i += 1
            cid = f"{c['cluster_id']}-f{child_i:02d}"
            child = _cluster_from_members(cid, grp, status=STATUS_DISCOVERED)
            child["claimed_class"] = ""
            child["parent_cluster_id"] = c["cluster_id"]
            child["family"] = family
            child["split_gain"] = gain
            child["dino_intra_similarity"] = dino.get("intra")
            child["dino_inter_similarity"] = dino.get("inter")
            child["split_reason"] = "dino_gated_mixed_family"
            after.append(child)
            splits.append(
                {
                    "parent_cluster_id": c["cluster_id"],
                    "child_cluster_id": cid,
                    "member_count": len(grp),
                    "family": family,
                    "clip_cohesion": child["cohesion"],
                    "dino_intra_similarity": dino.get("intra"),
                    "dino_inter_similarity": dino.get("inter"),
                    "split_gain": gain,
                    "reason": "dino_gated_mixed_family",
                    "confidence": child["confidence"],
                    "claimed_class": "",
                    "status": STATUS_DISCOVERED,
                }
            )
        if remnant:
            rem = _cluster_from_members(f"{c['cluster_id']}-remnant", remnant, status=STATUS_DISCOVERED)
            rem["claimed_class"] = ""
            rem["parent_cluster_id"] = c["cluster_id"]
            rem["family"] = ""
            rem["split_reason"] = "unlabeled_or_small_family_remnant"
            after.append(rem)
            continue
        continue

    after_metrics = analyze_clusters(items, after)
    return {
        "persisted": False,
        "production_bound": False,
        "phase1_store_written": False,
        "before": {
            "cluster_count": before["cluster_count"],
            "largest_cluster": before["largest_cluster"],
            "mean_cohesion": before["mean_cohesion"],
            "mixed_family_clusters": len(before["mixed_family_clusters"]),
            "mixed_object_clusters": len(before["mixed_object_clusters"]),
            "candidate_splits": len(before["candidate_splits"]),
            "color_variant_clusters": len(before["color_variant_clusters"]),
        },
        "after": {
            "cluster_count": after_metrics["cluster_count"],
            "largest_cluster": after_metrics["largest_cluster"],
            "mean_cohesion": after_metrics["mean_cohesion"],
            "mixed_family_clusters": len(after_metrics["mixed_family_clusters"]),
            "mixed_object_clusters": len(after_metrics["mixed_object_clusters"]),
            "candidate_splits": len(after_metrics["candidate_splits"]),
            "color_variant_clusters": len(after_metrics["color_variant_clusters"]),
        },
        "splits": splits,
        "skipped": skipped,
        "claimed_class_n": sum(1 for x in after if x.get("claimed_class")),
        "success": {
            "mixed_family_down": len(after_metrics["mixed_family_clusters"])
            < len(before["mixed_family_clusters"])
            or (
                len(before["mixed_family_clusters"]) == 0
                and len(after_metrics["mixed_family_clusters"]) == 0
            ),
            "color_variant_preserved": len(after_metrics["color_variant_clusters"])
            >= len(before["color_variant_clusters"]),
            "no_class_claim": True,
        },
        "after_clusters": after,
    }


def prototype_split_dino_or_family(
    items: list[Phase2Item],
    *,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    min_size: int = DEFAULT_MIN_SIZE,
    dino_gain_min: float = SPLIT_GAIN_MIN,
) -> dict[str, Any]:
    p1 = phase1_clusters_from_clip(items, sim_threshold=sim_threshold, min_size=min_size)
    applied = apply_dino_gated_family_split(items, p1, dino_gain_min=dino_gain_min)
    b, a = applied["before"], applied["after"]
    return {
        "phase1": {
            "cluster_count": b["cluster_count"],
            "mixed_family": b["mixed_family_clusters"],
            "largest": b["largest_cluster"].get("member_count"),
            "mean_cohesion": b["mean_cohesion"],
        },
        "prototype": {
            "cluster_count": a["cluster_count"],
            "mixed_family": a["mixed_family_clusters"],
            "largest": a["largest_cluster"].get("member_count"),
            "mean_cohesion": a["mean_cohesion"],
            "splits_applied": len({s["parent_cluster_id"] for s in applied["splits"]}),
        },
        "improved": a["mixed_family_clusters"] < b["mixed_family_clusters"],
        "note": "prototype only; not persisted, not ranked",
        "splits": applied["splits"],
    }


def run_shadow_split(
    db_path: str | Path,
    *,
    per_source: int = 20,
    source_limit: int = 5,
) -> dict[str, Any]:
    items, sources = load_phase2_items(
        db_path, per_source=per_source, source_limit=source_limit
    )
    phase1 = phase1_clusters_from_clip(items)
    applied = apply_dino_gated_family_split(items, phase1)
    applied["sources_sampled"] = sources
    applied["items"] = len(items)
    applied["dino_items"] = sum(1 for it in items if it.dino is not None)
    applied.pop("after_clusters", None)
    return applied


def persist_phase2_shadow(
    db_path: str | Path,
    shadow_db: str | Path,
    *,
    per_source: int = 20,
    source_limit: int = 5,
) -> dict[str, Any]:
    """Write Phase 2 snapshot to a dedicated shadow DB. Production/Phase 1 untouched."""
    from core.visual_memory.phase2_shadow import Phase2ShadowStore

    items, sources = load_phase2_items(
        db_path, per_source=per_source, source_limit=source_limit
    )
    phase1 = phase1_clusters_from_clip(items)
    applied = apply_dino_gated_family_split(items, phase1)
    store = Phase2ShadowStore(shadow_db)
    first = store.replace_snapshot(
        applied["after_clusters"],
        splits=applied["splits"],
        before=applied["before"],
        after=applied["after"],
    )
    second = store.replace_snapshot(
        applied["after_clusters"],
        splits=applied["splits"],
        before=applied["before"],
        after=applied["after"],
    )
    return {
        "sources_sampled": sources,
        "items": len(items),
        "phase1": applied["before"],
        "phase2": applied["after"],
        "splits": applied["splits"],
        "shadow": first,
        "rerun": second,
        "idempotent": first["ids"] == second["ids"] and first["clusters"] == second["clusters"],
        "claimed_class_n": second["claimed_class_n"],
        "phase1_store_written": False,
        "production_bound": False,
        "persisted": True,
        "shadow_db": str(Path(shadow_db)),
        "success": applied["success"],
    }

