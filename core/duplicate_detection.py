"""İndeks zamanı duplicate / variant işaretleme — dosya silinmez."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.logger import setup_logger
from core.utils import phash_similarity

logger = setup_logger(__name__)

DUP_EXACT = "exact_duplicate"
DUP_NEAR = "near_duplicate"
DUP_COLOR = "color_variant"
DUP_SCALE = "scale_variant"

DUP_LABELS = {
    DUP_EXACT: "Exact Duplicate",
    DUP_NEAR: "Near Duplicate",
    DUP_COLOR: "Color Variant",
    DUP_SCALE: "Scale Variant",
}


def classify_duplicate_relation(
    a: dict[str, Any],
    b: dict[str, Any],
) -> tuple[str, float] | None:
    """
    İki kayıt arasındaki ilişkiyi sınıflandır.
    Dönüş: (relation_key, score) veya None.
    """
    ph = phash_similarity(str(a.get("phash") or ""), str(b.get("phash") or ""))
    dh = phash_similarity(str(a.get("dhash") or ""), str(b.get("dhash") or ""))
    wh = phash_similarity(str(a.get("whash") or ""), str(b.get("whash") or ""))
    struct = max(ph, (ph + dh) / 2.0)

    partial_a = str(a.get("partial_hash") or "")
    partial_b = str(b.get("partial_hash") or "")
    same_partial = bool(partial_a and partial_a == partial_b)

    color_a = _color_key(a)
    color_b = _color_key(b)
    same_color = bool(color_a and color_b and color_a == color_b)
    color_diff = bool(color_a and color_b and color_a != color_b)

    scale_a = _scale_key(a)
    scale_b = _scale_key(b)
    scale_diff = bool(scale_a and scale_b and scale_a != scale_b)

    if same_partial or (ph >= 0.98 and dh >= 0.96):
        return DUP_EXACT, max(ph, 0.99 if same_partial else ph)

    if ph >= 0.88 and dh >= 0.82 and color_diff:
        return DUP_COLOR, struct

    if ph >= 0.82 and dh >= 0.75 and scale_diff and not color_diff:
        return DUP_SCALE, struct

    if ph >= 0.90 or (ph >= 0.85 and dh >= 0.80) or (wh >= 0.90 and ph >= 0.80):
        return DUP_NEAR, struct

    if ph >= 0.78 and dh >= 0.72 and (same_color or scale_diff or color_diff):
        if color_diff:
            return DUP_COLOR, struct
        if scale_diff:
            return DUP_SCALE, struct
        return DUP_NEAR, struct

    return None


def _color_key(rec: dict[str, Any]) -> str:
    tm = rec.get("texture_map") or {}
    if isinstance(tm, str):
        return ""
    ci = tm.get("color_index") if isinstance(tm, dict) else {}
    if isinstance(ci, dict) and ci.get("color_family"):
        return str(ci.get("color_family"))
    return str(
        (tm.get("color_family") if isinstance(tm, dict) else "")
        or rec.get("color_family")
        or ""
    )


def _scale_key(rec: dict[str, Any]) -> str:
    tm = rec.get("texture_map") or {}
    if not isinstance(tm, dict):
        return ""
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    scale = str(dna.get("scale") or "")
    if scale:
        return scale
    try:
        sc = float(tm.get("scale_pattern_score", 0) or 0)
    except (TypeError, ValueError):
        sc = 0.0
    if sc >= 0.45:
        return "large"
    if sc > 0 and sc < 0.30:
        return "small"
    return ""


def find_duplicate_candidates(db, *, phash: str, partial_hash: str = "", exclude_id: int = 0, limit: int = 12) -> list[dict[str, Any]]:
    """Aynı/yakın hash adayları — silme yok, sadece okuma."""
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    with db.connect() as conn:
        if partial_hash:
            rows = conn.execute(
                """
                SELECT f.id, f.path, f.filename, f.partial_hash,
                       fe.phash, fe.dhash, fe.whash, fe.texture_map
                FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                WHERE f.partial_hash=? AND f.status='indexed' AND f.id!=?
                LIMIT ?
                """,
                (partial_hash, exclude_id, limit),
            ).fetchall()
            for row in rows:
                d = dict(row)
                _parse_tm(d)
                seen.add(int(d["id"]))
                out.append(d)
        if phash and len(phash) >= 8:
            # Tam phash eşleşmesi + prefix bucket
            rows = conn.execute(
                """
                SELECT f.id, f.path, f.filename, f.partial_hash,
                       fe.phash, fe.dhash, fe.whash, fe.texture_map
                FROM files f
                JOIN features fe ON fe.file_id=f.id
                WHERE f.status='indexed' AND f.id!=?
                  AND (fe.phash=? OR fe.phash LIKE ?)
                LIMIT ?
                """,
                (exclude_id, phash, phash[:8] + "%", limit),
            ).fetchall()
            for row in rows:
                d = dict(row)
                fid = int(d["id"])
                if fid in seen:
                    continue
                _parse_tm(d)
                seen.add(fid)
                out.append(d)
    return out[:limit]


def _parse_tm(d: dict[str, Any]) -> None:
    import json

    tm = d.get("texture_map")
    if isinstance(tm, str) and tm:
        try:
            d["texture_map"] = json.loads(tm)
        except json.JSONDecodeError:
            d["texture_map"] = {}
    elif not isinstance(tm, dict):
        d["texture_map"] = {}


def apply_duplicate_marks(
    db,
    *,
    file_id: int,
    features: dict[str, Any],
    texture_map: dict[str, Any],
    partial_hash: str = "",
) -> dict[str, Any]:
    """
    Benzer kayıtları bul, texture_map'e işaret yaz, pattern_group'a bağla.
    Dosya silinmez.
    """
    phash = str(features.get("phash") or "")
    if not phash and not partial_hash:
        return texture_map

    self_rec = {
        "phash": phash,
        "dhash": features.get("dhash", ""),
        "whash": features.get("whash", ""),
        "partial_hash": partial_hash,
        "texture_map": texture_map,
        "color_family": texture_map.get("color_family", ""),
    }
    candidates = find_duplicate_candidates(
        db, phash=phash, partial_hash=partial_hash, exclude_id=file_id
    )
    best: tuple[str, float, dict[str, Any]] | None = None
    peers: list[dict[str, Any]] = []
    for cand in candidates:
        rel = classify_duplicate_relation(self_rec, cand)
        if not rel:
            continue
        kind, score = rel
        peers.append({"file_id": int(cand["id"]), "type": kind, "score": round(score, 4)})
        if best is None or score > best[1]:
            best = (kind, score, cand)

    if not best:
        return texture_map

    kind, score, peer = best
    texture_map = dict(texture_map)
    texture_map["duplicate_info"] = {
        "type": kind,
        "label": DUP_LABELS.get(kind, kind),
        "peer_file_id": int(peer["id"]),
        "peer_path": peer.get("path", ""),
        "score": round(score, 4),
        "peers": peers[:8],
    }

    try:
        _link_pattern_group(db, file_id, peer, kind, score, texture_map)
    except Exception as exc:
        logger.debug("pattern group link failed: %s", exc)

    return texture_map


def _link_pattern_group(
    db,
    file_id: int,
    peer: dict[str, Any],
    relation: str,
    score: float,
    texture_map: dict[str, Any],
) -> None:
    from core.pattern_family_tree import family_root_label

    peer_id = int(peer["id"])
    now = datetime.now(timezone.utc).isoformat()
    existing = db.get_pattern_group_for_file(peer_id) or db.get_pattern_group_for_file(file_id)
    label = family_root_label(texture_map) or "Pattern Family"
    if existing:
        gid = int(existing["id"])
    else:
        gid = db.create_pattern_group(
            {
                "representative_file_id": peer_id,
                "group_type": "duplicate_family",
                "label": label,
                "pattern_family": str(texture_map.get("pattern_family") or ""),
                "animal_print_type": str(texture_map.get("animal_print_type") or ""),
                "color_family": str(texture_map.get("color_family") or ""),
            }
        )
        db.add_pattern_group_member(gid, peer_id, "representative", 1.0, now)
    db.add_pattern_group_member(gid, file_id, relation, score, now)
