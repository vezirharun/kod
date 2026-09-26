"""Stage 6/7 — Color evidence & palette intelligence.

Uses features.dominant_colors / color_hist / local thumbnail + core.color_index.
Does not import Vezir Variant. Does not rewrite index_v3 / DINO / CLIP / DNA producers.
USER color_source (teach_me/manual_user/user) always wins over AI.
Ratios only from real cluster mass (k-means counts). Never invent %.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from core.color_index import (
    build_color_index,
    classify_rgb,
    color_is_user_locked,
)
from core.textile_terms import normalize_turkish

try:
    import numpy as np

    HAS_NP = True
except ImportError:
    HAS_NP = False

# No cv2 — palette clustering via safe_image_ops (heap-safe with torch).
HAS_CV2 = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Canonical EN names for search / DNA (TR classify_rgb labels mapped here).
_TO_EN: dict[str, str] = {
    "siyah": "black",
    "black": "black",
    "beyaz": "white",
    "white": "white",
    "krem": "cream",
    "cream": "cream",
    "ivory": "cream",
    "bej": "beige",
    "beige": "beige",
    "kahverengi": "brown",
    "brown": "brown",
    "dark_brown": "dark_brown",
    "camel": "brown",
    "deve": "brown",
    "gri": "gray",
    "gray": "gray",
    "grey": "gray",
    "light_gray": "light_gray",
    "kirmizi": "red",
    "kırmızı": "red",
    "red": "red",
    "dark_red": "dark_red",
    "bordo": "burgundy",
    "burgundy": "burgundy",
    "wine": "burgundy",
    "sarap": "burgundy",
    "turuncu": "orange",
    "orange": "orange",
    "sari": "yellow",
    "sarı": "yellow",
    "yellow": "yellow",
    "altin": "gold",
    "altın": "gold",
    "gold": "gold",
    "golden": "gold",
    "yesil": "green",
    "yeşil": "green",
    "green": "green",
    "dark_green": "dark_green",
    "haki": "green",
    "khaki": "green",
    "olive": "green",
    "mavi": "blue",
    "blue": "blue",
    "dark_blue": "dark_blue",
    "lacivert": "navy",
    "navy": "navy",
    "pembe": "pink",
    "pink": "pink",
    "mor": "purple",
    "purple": "purple",
    "turkuaz": "turquoise",
    "turquoise": "turquoise",
}

_DARK = frozenset({"black", "navy", "dark_brown", "dark_red", "dark_green", "dark_blue", "burgundy"})
_WARM = frozenset({"cream", "beige", "brown", "dark_brown", "orange", "gold", "red", "yellow", "pink"})
_COOL = frozenset({"blue", "dark_blue", "navy", "green", "dark_green", "purple", "turquoise"})
_NEUTRAL = frozenset({"white", "gray", "light_gray", "cream", "beige", "black"})

_MAX_COLOR_BONUS = 0.08
_MAX_COLOR_PENALTY = 0.04


def to_en_color(name: str) -> str:
    n = normalize_turkish(str(name or "").strip())
    if not n:
        return ""
    return _TO_EN.get(n, _TO_EN.get(str(name or "").strip().lower(), ""))


def _parse_rgb_clusters(dominant_colors: list[Any]) -> list[list[int]]:
    clusters: list[list[int]] = []
    for c in dominant_colors or []:
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            try:
                clusters.append([int(c[0]), int(c[1]), int(c[2])])
            except (TypeError, ValueError):
                continue
    return clusters


def _mood_tags(named: list[str]) -> list[str]:
    tags: list[str] = []
    s = set(named)
    if s & _DARK:
        tags.append("dark")
    if s & _NEUTRAL:
        tags.append("neutral")
    if s & _WARM:
        tags.append("warm")
    if s & _COOL:
        tags.append("cool")
    return tags


def _name_cluster(rgb: list[int]) -> str:
    labels = classify_rgb(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    for lab in labels:
        en = to_en_color(lab)
        if en:
            return en
    return ""


def extract_palette_clusters(
    image: Any,
    *,
    k: int = 5,
) -> tuple[list[list[int]], list[float]]:
    """K-means dominant RGB clusters + normalized mass (dominance order).

    Single-pixel noise cannot dominate: centers are sorted by sample count.
    """
    if not HAS_NP or image is None:
        return [], []
    try:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return [], []
        small = arr[::8, ::8, :3].reshape(-1, 3).astype(np.float32)
        if len(small) < 2:
            return [], []
        kk = min(int(k), len(small))
        from core.safe_image_ops import kmeans_centers

        centers, labels = kmeans_centers(small, kk)
        if len(centers) == 0:
            return [], []
        counts = np.bincount(labels.flatten(), minlength=len(centers)).astype(np.float64)
        order = np.argsort(-counts)
        total = float(counts.sum()) or 1.0
        clusters = [centers[i].astype(int).tolist() for i in order]
        weights = [float(counts[i] / total) for i in order]
        return clusters, weights
    except Exception:
        return [], []


def load_local_rgb_preview(path: str, *, max_side: int = 256) -> Any | None:
    """Load local thumbnail/preview only (never NAS originals)."""
    if not path or not HAS_PIL:
        return None
    try:
        from pathlib import Path

        p = Path(path)
        if not p.is_file():
            return None
        # Guard: skip obvious network/UNC NAS mounts
        s = str(p)
        if s.startswith("\\\\") or s.startswith("//"):
            return None
        im = Image.open(p).convert("RGB")
        im.thumbnail((max_side, max_side))
        if not HAS_NP:
            return None
        return np.asarray(im)
    except Exception:
        return None


def _evidence_confidence(
    named: list[str],
    *,
    ratio_conf: str,
    n_clusters: int,
) -> float:
    if not named:
        return 0.0
    base = min(0.92, 0.45 + 0.1 * len(named) + 0.05 * min(n_clusters, 5))
    if ratio_conf == "cluster_weight":
        base = min(0.95, base + 0.12)
    return round(base, 3)


def semantic_colors_from_rgb_clusters(
    dominant_colors: list[Any],
    *,
    cluster_weights: list[float] | None = None,
) -> dict[str, Any]:
    """RGB cluster list (order = dominance) → palette evidence.

    color_ratios only when real cluster_weights are provided (same length).
    """
    clusters = _parse_rgb_clusters(dominant_colors)
    empty = {
        "detected_colors": [],
        "dominant_colors": [],
        "color_ratios": {},
        "color_family": "",
        "color_mood": [],
        "ratio_confidence": "unknown",
        "confidence": 0.0,
        "source": "ai",
        "user_verified": False,
        "version": 2,
    }
    if not clusters:
        return empty

    named_ordered: list[str] = []
    for rgb in clusters:
        en = _name_cluster(rgb)
        if en and en not in named_ordered:
            named_ordered.append(en)

    ratios: dict[str, float] = {}
    ratio_conf = "unknown"
    if (
        named_ordered
        and cluster_weights
        and len(cluster_weights) >= len(clusters)
    ):
        agg: dict[str, float] = {}
        for rgb, w in zip(clusters, cluster_weights):
            en = _name_cluster(rgb)
            if not en:
                continue
            try:
                wf = float(w)
            except (TypeError, ValueError):
                continue
            if wf <= 0:
                continue
            agg[en] = agg.get(en, 0.0) + wf
        total = float(sum(agg.values()))
        if total > 0:
            ratios = {k: round(v / total, 4) for k, v in agg.items()}
            ratio_conf = "cluster_weight"

    ci = build_color_index(dominant_colors=clusters)
    fam = str(ci.get("color_family") or "")
    conf = _evidence_confidence(
        named_ordered, ratio_conf=ratio_conf, n_clusters=len(clusters)
    )

    return {
        "detected_colors": named_ordered,
        "dominant_colors": clusters[:5],
        "color_ratios": ratios,
        "color_family": fam,
        "color_mood": _mood_tags(named_ordered),
        "color_index": ci,
        "ratio_confidence": ratio_conf,
        "confidence": conf,
        "source": "ai",
        "user_verified": False,
        "version": 2,
    }


def palette_evidence_from_image(image: Any, *, k: int = 5) -> dict[str, Any]:
    """Full palette evidence from pixels via dominant clusters (not single pixels)."""
    clusters, weights = extract_palette_clusters(image, k=k)
    return semantic_colors_from_rgb_clusters(clusters, cluster_weights=weights)


def rgb_clusters_from_texture_map(tm: dict[str, Any] | None) -> list[list[int]]:
    """Prefer color_index.dominant_colors (RGB), else pattern_dna RGB lists."""
    tm = tm if isinstance(tm, dict) else {}
    ci = tm.get("color_index") if isinstance(tm.get("color_index"), dict) else {}
    dom = ci.get("dominant_colors")
    parsed = _parse_rgb_clusters(list(dom or []))
    if parsed:
        return parsed
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    return _parse_rgb_clusters(list(dna.get("dominant_colors") or []))


def ensure_color_evidence(
    texture_map: dict[str, Any] | None,
    *,
    dominant_colors: list[Any] | None = None,
) -> dict[str, Any]:
    """In-memory ensure color_evidence; never overrides USER lock colors."""
    tm = dict(texture_map or {})
    ev = tm.get("color_evidence") if isinstance(tm.get("color_evidence"), dict) else {}
    if ev.get("detected_colors") and not color_is_user_locked(tm):
        return tm
    rgb = _parse_rgb_clusters(list(dominant_colors or [])) or rgb_clusters_from_texture_map(tm)
    if not rgb:
        return tm
    return attach_color_evidence(tm, rgb, overwrite_ai=not bool(ev.get("detected_colors")))


def attach_color_evidence(
    texture_map: dict[str, Any] | None,
    dominant_colors: list[Any] | None,
    *,
    overwrite_ai: bool = True,
    cluster_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Merge semantic color evidence into texture_map. Never overrides USER lock."""
    tm = dict(texture_map or {})
    if color_is_user_locked(tm):
        if not tm.get("color_evidence") and dominant_colors:
            ev = semantic_colors_from_rgb_clusters(
                list(dominant_colors), cluster_weights=cluster_weights
            )
            tm["color_evidence"] = {**ev, "suppressed_by": "user_verified"}
        return tm

    if not dominant_colors:
        return tm
    existing = tm.get("color_evidence") if isinstance(tm.get("color_evidence"), dict) else {}
    if existing.get("detected_colors") and not overwrite_ai:
        # Upgrade ratios only when missing and real weights provided
        if (
            existing.get("ratio_confidence") != "cluster_weight"
            and cluster_weights
            and existing.get("source") == "ai"
        ):
            ev = semantic_colors_from_rgb_clusters(
                list(dominant_colors), cluster_weights=cluster_weights
            )
            if ev.get("ratio_confidence") == "cluster_weight":
                merged = dict(existing)
                merged["color_ratios"] = ev["color_ratios"]
                merged["ratio_confidence"] = "cluster_weight"
                merged["confidence"] = ev.get("confidence", existing.get("confidence"))
                merged["dominant_colors"] = ev.get("dominant_colors") or existing.get(
                    "dominant_colors"
                )
                tm["color_evidence"] = merged
                dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
                dna = dict(dna)
                dna["color_ratios"] = dict(ev["color_ratios"])
                dna["color_ratio_confidence"] = "cluster_weight"
                tm["pattern_dna"] = dna
        return tm

    ev = semantic_colors_from_rgb_clusters(
        list(dominant_colors), cluster_weights=cluster_weights
    )
    tm["color_evidence"] = ev
    tm["color_index"] = ev.get("color_index") or tm.get("color_index") or {}
    if ev.get("color_family"):
        tm["color_family"] = ev["color_family"]
        tm["color_source"] = "auto_index"
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    dna = dict(dna)
    if ev.get("detected_colors"):
        dna["dominant_colors"] = list(ev["detected_colors"])
    if ev.get("color_family"):
        dna["color_family"] = ev["color_family"]
    if ev.get("color_ratios") and ev.get("ratio_confidence") == "cluster_weight":
        dna["color_ratios"] = dict(ev["color_ratios"])
        dna["color_ratio_confidence"] = "cluster_weight"
    tm["pattern_dna"] = dna
    return tm


def effective_detected_colors(texture_map: dict[str, Any] | None) -> list[str]:
    """USER palette/family labels win; else AI detected_colors."""
    tm = texture_map if isinstance(texture_map, dict) else {}
    if color_is_user_locked(tm):
        user_list = tm.get("user_detected_colors") or tm.get("detected_colors")
        if isinstance(user_list, list) and user_list:
            out: list[str] = []
            for x in user_list:
                en = to_en_color(str(x)) or normalize_turkish(str(x))
                if en and en not in out:
                    out.append(en)
            return out
        fam = normalize_turkish(str(tm.get("color_family") or ""))
        fam_map = {
            "black_white": ["black", "white"],
            "cream": ["cream"],
            "beige": ["beige"],
            "brown_tan": ["brown"],
            "grayscale": ["gray"],
        }
        if fam in fam_map:
            return list(fam_map[fam])
        en = to_en_color(fam)
        return [en] if en else ([fam] if fam else [])

    tm2 = ensure_color_evidence(tm)
    ev = tm2.get("color_evidence") if isinstance(tm2.get("color_evidence"), dict) else {}
    colors = ev.get("detected_colors") or []
    if isinstance(colors, list) and colors:
        return [str(x) for x in colors if x]
    ci = tm2.get("color_index") if isinstance(tm2.get("color_index"), dict) else {}
    out = []
    for lab in ci.get("palette") or []:
        en = to_en_color(str(lab))
        if en and en not in out:
            out.append(en)
    return out


def score_color_evidence_match(
    want: list[str],
    texture_map: dict[str, Any] | None,
    *,
    user_protected: bool = False,
) -> dict[str, Any]:
    """Soft color score. Unknown → no penalty. User-protected → no demote."""
    if not want:
        return {"applied": False, "delta": 0.0, "reason": "no_want"}
    if user_protected:
        return {"applied": False, "delta": 0.0, "skipped": "user_protected"}
    have = {normalize_turkish(c) for c in effective_detected_colors(texture_map)}
    if not have:
        return {"applied": True, "delta": 0.0, "reason": "unknown", "matches": {}}

    want_en: list[str] = []
    for w in want:
        en = to_en_color(w) or normalize_turkish(w)
        if en:
            want_en.append(en)
    if not want_en:
        return {"applied": False, "delta": 0.0}

    hits = sum(1 for w in want_en if w in have)
    misses = len(want_en) - hits
    delta = 0.0
    if hits:
        delta += min(_MAX_COLOR_BONUS, 0.03 * hits + 0.02)
    if misses and hits == 0:
        delta -= min(_MAX_COLOR_PENALTY, 0.02 * misses)
    elif misses and hits > 0:
        delta -= min(0.02, 0.01 * misses)
    delta = max(-_MAX_COLOR_PENALTY, min(_MAX_COLOR_BONUS, delta))
    return {
        "applied": True,
        "delta": round(delta, 4),
        "hits": hits,
        "misses": misses,
        "have": sorted(have),
        "want": want_en,
        "user_verified": color_is_user_locked(texture_map),
    }


def apply_color_evidence_scoring(
    results: list[Any],
    query_text: str,
    *,
    customer_key: str = "",
) -> list[Any]:
    """Soft re-score by query colors vs file color evidence. Concept identity untouched.

    Skips rows already scored by query_attribute_intel color channel to avoid double count.
    Optional customer_key is accepted for chain parity (color evidence itself is global).
    """
    if not results or not (query_text or "").strip():
        return results
    try:
        from core.query_attribute_intel import extract_query_attributes

        attrs = extract_query_attributes(query_text)
    except Exception:
        return results
    if not attrs.colors:
        return results

    for rec in results:
        dbg = getattr(rec, "debug", None) or {}
        if not isinstance(dbg, dict):
            continue
        # Attribute layer already applied color → skip second pass
        qa = dbg.get("query_attribute_intel") if isinstance(dbg.get("query_attribute_intel"), dict) else {}
        matches = qa.get("matches") if isinstance(qa.get("matches"), dict) else {}
        if "color" in matches:
            continue
        protected = bool(
            dbg.get("learned_concept_exact")
            or dbg.get("user_taught_positive")
            or dbg.get("protected_exact")
            or dbg.get("category_source") == "manual_user"
        )
        tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
        tm = ensure_color_evidence(tm)
        dbg["texture_map"] = tm
        meta = score_color_evidence_match(
            list(attrs.colors), tm, user_protected=protected
        )
        if meta.get("skipped") == "user_protected":
            dbg["color_evidence_score"] = meta
            rec.debug = dbg
            continue
        if meta.get("applied") and abs(float(meta.get("delta") or 0)) > 1e-9:
            try:
                old = float(getattr(rec, "score", 0) or 0)
            except (TypeError, ValueError):
                continue
            new = max(0.0, min(1.0, old + float(meta["delta"])))
            rec.score = new
            if hasattr(rec, "score_percent"):
                rec.score_percent = round(new * 100, 1)
            meta["before"] = round(old, 4)
            meta["after"] = round(new, 4)
        if customer_key:
            meta = dict(meta)
            meta["customer_key"] = " ".join(str(customer_key).strip().split())
        dbg["color_evidence_score"] = meta
        rec.debug = dbg
    return results


def backfill_color_evidence_db(
    db_path: str,
    *,
    limit: int = 500,
    only_missing: bool = True,
    use_thumbnails: bool = True,
    batch_size: int = 100,
    upgrade_ratios: bool = True,
) -> dict[str, Any]:
    """Write color_evidence into texture_map for rows with RGB dominant_colors.

    Phase A: DB RGB clusters → named colors (no NAS).
    Phase B: local thumbnail/preview → real k-means ratios when available.
    Does not touch concept_examples / learning. Skips user-locked color.
    Sequential batches only (no ThreadPoolExecutor).
    """
    stats: dict[str, Any] = {
        "scanned": 0,
        "updated": 0,
        "ratios_computed": 0,
        "named_only": 0,
        "skipped_user": 0,
        "skipped_empty": 0,
        "skipped_existing": 0,
        "thumb_used": 0,
        "thumb_missing": 0,
        "batches": 0,
        "elapsed_sec": 0.0,
        "rss_mb_delta": None,
    }
    if not db_path:
        return stats
    t0 = time.perf_counter()
    rss0 = None
    try:
        import os

        import psutil  # type: ignore

        rss0 = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        rss0 = None

    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        fcols = {r[1] for r in conn.execute("PRAGMA table_info(files)")}
        thumb_sel = "f.thumbnail_path" if "thumbnail_path" in fcols else "'' AS thumbnail_path"
        prev_sel = (
            "f.feature_preview_path"
            if "feature_preview_path" in fcols
            else "'' AS feature_preview_path"
        )
        rows = conn.execute(
            f"""
            SELECT f.id, {thumb_sel}, {prev_sel},
                   fe.dominant_colors, fe.texture_map
            FROM files f
            JOIN features fe ON fe.file_id=f.id
            WHERE fe.dominant_colors IS NOT NULL
              AND fe.dominant_colors NOT IN ('', '[]')
            ORDER BY
              CASE
                WHEN fe.texture_map IS NULL OR fe.texture_map='' THEN 0
                WHEN fe.texture_map NOT LIKE '%"detected_colors"%' THEN 0
                WHEN fe.texture_map NOT LIKE '%"cluster_weight"%' THEN 1
                ELSE 2
              END,
              f.id
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()

        pending: list[tuple[int, str]] = []
        bs = max(1, int(batch_size))

        def _flush() -> None:
            if not pending:
                return
            conn.executemany(
                "UPDATE features SET texture_map=? WHERE file_id=?",
                [(tm_json, fid) for fid, tm_json in pending],
            )
            conn.commit()
            stats["batches"] += 1
            pending.clear()

        for row in rows:
            stats["scanned"] += 1
            try:
                dom = json.loads(row["dominant_colors"] or "[]")
            except Exception:
                stats["skipped_empty"] += 1
                continue
            if not isinstance(dom, list) or not dom:
                stats["skipped_empty"] += 1
                continue
            try:
                tm = json.loads(row["texture_map"] or "{}")
            except Exception:
                tm = {}
            if not isinstance(tm, dict):
                tm = {}
            if color_is_user_locked(tm):
                stats["skipped_user"] += 1
                continue
            ev = tm.get("color_evidence") if isinstance(tm.get("color_evidence"), dict) else {}
            has_named = bool(ev.get("detected_colors"))
            has_ratio = ev.get("ratio_confidence") == "cluster_weight" and bool(
                ev.get("color_ratios")
            )
            if only_missing and has_named and (has_ratio or not upgrade_ratios):
                stats["skipped_existing"] += 1
                continue

            weights: list[float] | None = None
            clusters = list(dom)
            if use_thumbnails and (not has_ratio) and upgrade_ratios:
                thumb = str(row["thumbnail_path"] or "").strip()
                prev = str(row["feature_preview_path"] or "").strip()
                img = load_local_rgb_preview(thumb)
                if img is None:
                    img = load_local_rgb_preview(prev)
                if img is not None:
                    clusters2, weights2 = extract_palette_clusters(img, k=5)
                    if clusters2 and weights2:
                        clusters = clusters2
                        weights = weights2
                        stats["thumb_used"] += 1
                    else:
                        stats["thumb_missing"] += 1
                else:
                    stats["thumb_missing"] += 1

            overwrite = not has_named
            new_tm = attach_color_evidence(
                tm,
                clusters,
                overwrite_ai=overwrite,
                cluster_weights=weights,
            )
            new_ev = new_tm.get("color_evidence") if isinstance(new_tm.get("color_evidence"), dict) else {}
            if new_ev.get("ratio_confidence") == "cluster_weight":
                stats["ratios_computed"] += 1
            elif new_ev.get("detected_colors"):
                stats["named_only"] += 1

            pending.append((int(row["id"]), json.dumps(new_tm, ensure_ascii=False)))
            stats["updated"] += 1
            if len(pending) >= bs:
                _flush()
        _flush()
    finally:
        conn.close()

    stats["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    if rss0 is not None:
        try:
            import os

            import psutil  # type: ignore

            rss1 = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            stats["rss_mb_delta"] = round(rss1 - rss0, 2)
        except Exception:
            pass
    return stats
