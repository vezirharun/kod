"""Bana Öğret — doğrulanmış görsel kavram havuzu. Model eğitmez; indeksi ezmez."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from core.concept_registry import (
    dismissed_file_ids,
    dismiss_file,
    learn,
    positive_example_vectors,
    taught_file_ids,
)
from core.manual_label_guard import is_manual_labeled, parse_texture_map

_UNKNOWN = frozenset({"", "unknown", "none", "null", "belirsiz"})
MATCH_OK = 0.82
POOL_SCAN_LIMIT = 400
POOL_SHOW_LIMIT = 80


def resolve_active_customer_key(explicit: str = "") -> str:
    """Stamp for concept_examples. Empty = global. Does not rename concepts."""
    raw = " ".join(str(explicit or "").strip().split())
    if not raw:
        try:
            from core.settings import AppSettings

            settings = AppSettings.load() if hasattr(AppSettings, "load") else None
            if settings is None:
                settings = AppSettings()
            raw = " ".join(str(getattr(settings, "customer_filter", "") or "").strip().split())
        except Exception:
            raw = ""
    if not raw:
        return ""
    try:
        from core.concept_registry import _norm_customer_key

        return _norm_customer_key(raw)
    except Exception:
        try:
            from core.customer_discovery import normalize_customer_key

            return normalize_customer_key(raw) or raw.casefold()
        except Exception:
            return raw.casefold()



@dataclass
class TeachMeCard:
    file_id: int
    filename: str
    path: str
    preview_path: str
    pool: str
    reason: str
    guess: str
    confidence: float
    category: str = ""
    clip_embedding: bytes | None = None
    feature_preview_path: str = ""
    color_family: str = ""
    pattern_family: str = ""
    brand: str = ""
    tags: list[str] | None = None
    rivals: list | None = None
    cluster_size: int = 1
    suggested: str = ""
    member_ids: list | None = None
    family_id: str = ""


def _norm_fam(value: Any) -> str:
    return str(value or "").strip().lower()


def _conf(tm: dict[str, Any], file_row: dict[str, Any]) -> float:
    try:
        c = float(tm.get("classification_confidence") or 0)
    except (TypeError, ValueError):
        c = 0.0
    if c <= 0:
        try:
            c = float(file_row.get("pattern_confidence") or 0)
        except (TypeError, ValueError):
            c = 0.0
    return max(0.0, min(1.0, c))


def _guess_fields(tm: dict[str, Any], file_row: dict[str, Any]) -> tuple[str, str, list[str]]:
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    sem = tm.get("semantic_tags") if isinstance(tm.get("semantic_tags"), dict) else {}
    fams = [
        _norm_fam(file_row.get("pattern_family")),
        _norm_fam(tm.get("pattern_family")),
        _norm_fam(dna.get("family") or dna.get("main_family")),
        _norm_fam(sem.get("family")),
    ]
    known = [f for f in fams if f not in _UNKNOWN]
    display = known[0] if known else ""
    cat = str(file_row.get("category_path") or tm.get("category_path") or "")
    return display, cat, known


def _auto_settled(tm: dict[str, Any]) -> bool:
    try:
        auto_conf = float(tm.get("autonomous_confidence") or 0)
    except (TypeError, ValueError):
        auto_conf = 0.0
    return bool(str(tm.get("autonomous_label") or "").strip() and auto_conf >= MATCH_OK)


def assign_pool(file_row: dict[str, Any], texture_map: dict[str, Any] | None) -> tuple[str, str] | None:
    """None = havuza girmez. undefined | suspicious."""
    tm = parse_texture_map(texture_map)
    if is_manual_labeled(tm) or _auto_settled(tm):
        return None
    _display, _cat, known = _guess_fields(tm, file_row)
    conf = _conf(tm, file_row)
    unique = list(dict.fromkeys(known))
    if len(unique) >= 2:
        return "suspicious", "Motorlar farklı sınıf diyor"
    if not unique:
        return "undefined", "Anlamlı bir sınıf yok"
    if conf < 0.55:
        return "suspicious", "Güven düşük"
    return None


_UNDECIDED_STRONG = 0.45


def competing_strong_guesses(
    file_row: dict[str, Any], texture_map: dict[str, Any] | None
) -> list[tuple[str, float]]:
    """Distinct labels that are each a strong prediction. Does not train models."""
    tm = parse_texture_map(texture_map)
    scored: dict[str, float] = {}

    def _add(label: Any, conf: Any) -> None:
        lab = _norm_fam(label)
        if lab in _UNKNOWN:
            return
        try:
            c = float(conf or 0)
        except (TypeError, ValueError):
            c = 0.0
        if c < _UNDECIDED_STRONG:
            return
        scored[lab] = max(scored.get(lab, 0.0), min(1.0, c))

    file_conf = _conf(tm, file_row)
    _add(file_row.get("pattern_family"), file_row.get("pattern_confidence") or file_conf)
    _add(tm.get("pattern_family"), tm.get("classification_confidence") or file_conf)
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    _add(dna.get("family") or dna.get("main_family"), dna.get("confidence") or 0.72)
    sem = tm.get("semantic_tags") if isinstance(tm.get("semantic_tags"), dict) else {}
    _add(sem.get("family"), sem.get("confidence") or 0.72)
    goi = tm.get("global_object_intelligence")
    if isinstance(goi, dict):
        for item in goi.get("objects") or []:
            if not isinstance(item, dict):
                continue
            _add(item.get("label_tr") or item.get("label"), item.get("confidence"))
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) < 2:
        return []
    return ranked


def assign_undecided_pool(
    file_row: dict[str, Any], texture_map: dict[str, Any] | None
) -> tuple[str, str] | None:
    """Multiple strong predictions. Independent of undefined/suspicious."""
    tm = parse_texture_map(texture_map)
    if is_manual_labeled(tm) or _auto_settled(tm):
        return None
    guesses = competing_strong_guesses(file_row, tm)
    if len(guesses) < 2:
        return None
    detail = ", ".join(f"{name} %{int(round(conf * 100))}" for name, conf in guesses[:3])
    return "undecided", f"Birden fazla güçlü tahmin: {detail}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def match_clip_to_concepts(db_path: str, clip_blob: bytes | None) -> list[dict[str, Any]]:
    """Learned CLIP examples vs one file. Does not train weights."""
    if not clip_blob or len(clip_blob) < 16:
        return []
    try:
        vec = np.frombuffer(clip_blob, dtype=np.float32)
    except (ValueError, TypeError):
        return []
    if vec.size == 0 or not np.isfinite(vec).all():
        return []
    best: dict[int, dict[str, Any]] = {}
    for ex in positive_example_vectors(db_path):
        raw = ex.get("embedding") or b""
        try:
            other = np.frombuffer(raw, dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if other.size != vec.size:
            continue
        score = _cosine(vec, other)
        cid = int(ex["concept_id"])
        prev = best.get(cid)
        if prev is None or score > float(prev["score"]):
            best[cid] = {
                "concept_id": cid,
                "canonical": ex["canonical"],
                "score": score,
                "strong": score >= MATCH_OK,
            }
    return sorted(best.values(), key=lambda x: -float(x["score"]))


_INBOX_SELECT = (
    "SELECT f.id, f.filename, f.path, f.thumbnail_path, f.feature_preview_path, "
    "f.pattern_family, f.pattern_confidence, f.needs_review, f.category_path, "
    "fe.texture_map "
    "FROM files f LEFT JOIN features fe ON fe.file_id=f.id "
)

# Önizlemesi olan belirsiz dosyalar önce gelsin (yeni thumb'suz keşifler tüm havuzu kilitlemesin).
_INBOX_ORDER_PREFER_MEDIA = (
    "ORDER BY CASE WHEN coalesce(f.thumbnail_path,'')!='' "
    "OR coalesce(f.feature_preview_path,'')!='' THEN 0 ELSE 1 END, f.id DESC LIMIT ?"
)


def enrich_teach_card_paths(
    card: "TeachMeCard",
    *,
    db=None,
    cache_dir: str = "",
) -> "TeachMeCard":
    """Ortak preview sözleşmesi: DB alanları + mevcut cache. Yeni thumb üretmez."""
    from pathlib import Path

    from core.thumb_resolve import resolve_thumb_path

    thumb = str(card.preview_path or "").strip()
    feat = str(card.feature_preview_path or "").strip()
    path = str(card.path or "").strip()
    name = str(card.filename or "").strip()

    # Eksik metadata: files satırından tek seferlik hydrate.
    if db is not None and int(card.file_id or 0) > 0 and (not thumb or not feat or not path or not name):
        try:
            with db.connect() as conn:
                row = conn.execute(
                    "SELECT filename, path, thumbnail_path, feature_preview_path "
                    "FROM files WHERE id=?",
                    (int(card.file_id),),
                ).fetchone()
            if row:
                name = name or str(row["filename"] or "")
                path = path or str(row["path"] or "")
                thumb = thumb or str(row["thumbnail_path"] or "")
                feat = feat or str(row["feature_preview_path"] or "")
        except Exception:
            pass

    resolved, status = resolve_thumb_path(
        thumb, cache_dir=str(cache_dir or ""), feature_preview_path=feat
    )
    if status == "hit" and resolved:
        thumb = thumb or resolved
        if not feat and "feature_preview" in resolved.replace("\\", "/").lower():
            feat = resolved
        elif not thumb:
            thumb = resolved

    # DB boş ama cache'te dosya varsa (üretim pipeline'ına yazmadan) bağla.
    if path and (not thumb or not Path(thumb).is_file()) and cache_dir:
        try:
            from core.thumbnailer import Thumbnailer

            cand = Thumbnailer(str(cache_dir)).thumbnail_path_for(path)
            if cand.is_file() and cand.stat().st_size > 0:
                thumb = str(cand)
        except Exception:
            pass
    if path and (not feat or not Path(feat).is_file()) and cache_dir:
        try:
            from core.preview_cache import FeaturePreviewCache

            ge = FeaturePreviewCache(str(cache_dir)).get_existing(path)
            if ge.success and ge.preview_path:
                feat = str(ge.preview_path)
        except Exception:
            pass

    card.filename = name or card.filename
    card.path = path or card.path
    card.preview_path = thumb or feat
    card.feature_preview_path = feat
    return card


def _inbox_card(rec: dict[str, Any], tm: dict[str, Any], kind: str, reason: str) -> TeachMeCard:
    display, cat, _known = _guess_fields(tm, rec)
    if kind == "undecided":
        rivals = competing_strong_guesses(rec, tm)
        if rivals:
            display = " / ".join(name for name, _c in rivals[:3])
    thumb = str(rec.get("thumbnail_path") or "")
    feat_prev = str(rec.get("feature_preview_path") or "")
    tags = tm.get("user_tags") or []
    if not isinstance(tags, list):
        tags = []
    pool_reason = {
        "undefined": reason or "Bu görsel için henüz anlamlı bir sınıf yok",
        "suspicious": reason or "Sistem bir kavramdan şüpheleniyor",
        "undecided": reason or "Birden fazla olası kavram / düşük ayrım var",
        "new_concept": reason or "Mevcut kavramlara uymayan yeni görsel kümesi",
    }.get(kind, reason)
    return TeachMeCard(
        file_id=int(rec.get("id") or 0),
        filename=str(rec.get("filename") or ""),
        path=str(rec.get("path") or ""),
        preview_path=thumb or feat_prev,
        pool=kind,
        reason=pool_reason,
        guess=display or "Belirsiz",
        confidence=_conf(tm, rec),
        category=cat,
        clip_embedding=None,
        feature_preview_path=feat_prev,
        color_family=str(tm.get("color_family") or ""),
        pattern_family=str(rec.get("pattern_family") or tm.get("pattern_family") or ""),
        brand=str(tm.get("brand_name") or ""),
        tags=[str(t).strip() for t in tags if str(t).strip()],
    )


def _fetch_inbox_rows(db, sql: str) -> list[dict[str, Any]]:
    try:
        with db.connect() as conn:
            rows = conn.execute(sql, (int(POOL_SCAN_LIMIT),)).fetchall()
    except Exception:
        return []
    return [dict(row) for row in rows]


def list_inbox_pools(db, db_path: str, *, limit: int = POOL_SHOW_LIMIT) -> dict[str, list[TeachMeCard]]:
    """Tek tarama: CLIP/DINO blob okumadan üç havuzu doldur."""
    dismissed = dismissed_file_ids(db_path)
    taught = taught_file_ids(db_path)
    skip = dismissed | taught
    pools: dict[str, list[TeachMeCard]] = {
        "undefined": [],
        "suspicious": [],
        "undecided": [],
        "new_concept": [],
    }
    cap = int(limit)
    uncertain_sql = (
        _INBOX_SELECT
        + "WHERE IFNULL(f.pattern_family,'') IN ('','unknown') "
        + "OR IFNULL(f.pattern_confidence,1) < 0.55 "
        + "OR IFNULL(f.needs_review,0)=1 "
        + _INBOX_ORDER_PREFER_MEDIA
    )
    # Undecided: recent by id — NOT media-first. Prefer-media SQL burns the scan
    # LIMIT on media-rich rows that fail assign_undecided_pool; real undecided
    # then arrive without thumb/fp. Prefer media among true candidates below.
    recent_sql = _INBOX_SELECT + "ORDER BY f.id DESC LIMIT ?"
    cache_dir = ""
    try:
        from core.settings import AppSettings

        cache_dir = str(AppSettings.load().cache_dir or "")
    except Exception:
        cache_dir = ""
    if not cache_dir:
        try:
            from pathlib import Path

            root = Path(str(db_path)).resolve().parent
            for cand in (root / "cache", root.parent / "cache"):
                if cand.is_dir():
                    cache_dir = str(cand)
                    break
        except Exception:
            cache_dir = ""
    for rec in _fetch_inbox_rows(db, uncertain_sql):
        fid = int(rec.get("id") or 0)
        if fid <= 0 or fid in skip:
            continue
        tm = parse_texture_map(rec.get("texture_map"))
        assigned = assign_pool(rec, tm)
        if not assigned:
            continue
        kind, reason = assigned
        bucket = pools.get(kind)
        if bucket is None or len(bucket) >= cap:
            continue
        bucket.append(enrich_teach_card_paths(_inbox_card(rec, tm, kind, reason), db=db, cache_dir=cache_dir))
    undecided_hits: list[tuple[dict[str, Any], dict[str, Any], str, str]] = []
    for rec in _fetch_inbox_rows(db, recent_sql):
        fid = int(rec.get("id") or 0)
        if fid <= 0 or fid in skip:
            continue
        tm = parse_texture_map(rec.get("texture_map"))
        assigned = assign_undecided_pool(rec, tm)
        if not assigned:
            continue
        kind, reason = assigned
        if kind != "undecided":
            continue
        undecided_hits.append((rec, tm, kind, reason))

    def _undecided_media_rank(rec: dict[str, Any]) -> tuple[int, int]:
        has_media = bool(
            str(rec.get("thumbnail_path") or "").strip()
            or str(rec.get("feature_preview_path") or "").strip()
        )
        return (0 if has_media else 1, -int(rec.get("id") or 0))

    undecided_hits.sort(key=lambda item: _undecided_media_rank(item[0]))
    for rec, tm, kind, reason in undecided_hits:
        if len(pools["undecided"]) >= cap:
            break
        pools["undecided"].append(
            enrich_teach_card_paths(
                _inbox_card(rec, tm, kind, reason), db=db, cache_dir=cache_dir
            )
        )
    try:
        from core.autonomous_learn import pending_review_cards

        extra = pending_review_cards(db, db_path)
    except Exception:
        extra = {}
    extra_ids = {
        int(c.file_id)
        for cards in extra.values()
        for c in cards
        if int(c.file_id) > 0 and int(c.file_id) not in skip
    }
    for kind in list(pools.keys()):
        pools[kind] = [c for c in pools[kind] if int(c.file_id) not in extra_ids]
    for kind, cards in extra.items():
        dest = pools.setdefault(kind, [])
        for card in cards:
            fid = int(card.file_id)
            if fid in skip or fid not in extra_ids:
                continue
            if len(dest) >= cap:
                continue
            dest.append(enrich_teach_card_paths(card, db=db, cache_dir=cache_dir))
            extra_ids.discard(fid)
    try:
        from core.archive_intelligence.hooks import merge_archive_candidates_into_pools

        present = {
            int(c.file_id)
            for cards in pools.values()
            for c in cards
            if int(c.file_id) > 0
        }
        merge_archive_candidates_into_pools(
            pools,
            db,
            db_path,
            skip_ids=skip | present,
            limit=cap,
            cache_dir=cache_dir,
        )
    except Exception:
        pass
    try:
        from core.visual_family_candidates import collapse_pools_with_families

        pools = collapse_pools_with_families(pools, db_path=db_path)
    except Exception:
        pass
    return pools


def list_inbox(db, db_path: str, *, pool: str, limit: int = POOL_SHOW_LIMIT) -> list[TeachMeCard]:
    """Read-only scan of files that need teaching. Does not write the index."""
    want = str(pool or "")
    return list(list_inbox_pools(db, db_path, limit=limit).get(want) or [])


_OVERLAY_KEYS = (
    "parent",
    "child",
    "category_path",
    "pattern_family",
    "animal_print_type",
    "color_family",
    "brand",
    "tags",
)
_CONCEPT_KEYS = (
    "parent",
    "child",
    "category_path",
    "pattern_family",
    "animal_print_type",
)


def nonempty_overlay(overlay: dict[str, Any] | None) -> dict[str, Any]:
    """Only filled fields — empty values must not wipe existing metadata."""
    out: dict[str, Any] = {}
    for key in _OVERLAY_KEYS:
        val = (overlay or {}).get(key)
        if val in (None, "", []):
            continue
        if isinstance(val, list):
            tags = [str(t).strip() for t in val if str(t).strip()]
            if tags:
                out[key] = tags
            continue
        text = str(val).strip()
        if text:
            out[key] = text
    return out


def merge_overlay_keep_existing(
    existing: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(existing or {})
    merged.update(nonempty_overlay(overlay))
    parent = str(merged.get("parent") or "").strip()
    child = str(merged.get("child") or "").strip()
    path = str(merged.get("category_path") or "").strip()
    if not path and parent:
        path = f"{parent}/{child}".rstrip("/") if child else parent
        merged["category_path"] = path
    return merged


def _lite_file_row(db, file_id: int) -> dict[str, Any]:
    try:
        with db.connect() as conn:
            row = conn.execute(
                "SELECT id, filename, path, category_path, pattern_family "
                "FROM files WHERE id=?",
                (int(file_id),),
            ).fetchone()
        return dict(row) if row else {}
    except Exception:
        return {}


def _lite_texture_map(db, file_id: int) -> dict[str, Any]:
    try:
        with db.connect() as conn:
            row = conn.execute(
                "SELECT texture_map FROM features WHERE file_id=?",
                (int(file_id),),
            ).fetchone()
        raw = row["texture_map"] if row is not None else None
        return parse_texture_map(raw)
    except Exception:
        return {}


def snapshot_file_classification(db, file_id: int) -> dict[str, Any]:
    rec = _lite_file_row(db, int(file_id)) or {}
    tm = _lite_texture_map(db, int(file_id))
    path = str(rec.get("category_path") or tm.get("category_path") or "").strip()
    parent, child = "", ""
    if path:
        parent, _, child = path.partition("/")
        parent, child = parent.strip(), child.strip()
    tags = tm.get("user_tags") or []
    if not isinstance(tags, list):
        tags = []
    return {
        "file_id": int(file_id),
        "filename": str(rec.get("filename") or ""),
        "parent": parent,
        "child": child,
        "category_path": path,
        "pattern_family": str(rec.get("pattern_family") or tm.get("pattern_family") or ""),
        "color_family": str(rec.get("color_family") or tm.get("color_family") or ""),
        "brand": str(tm.get("brand_name") or rec.get("brand") or ""),
        "tags": [str(t).strip() for t in tags if str(t).strip()],
    }


def common_overlay_from_snapshots(snaps: list[dict[str, Any]]) -> dict[str, Any]:
    if not snaps:
        return {}
    common: dict[str, Any] = {}
    keys = ("parent", "child", "category_path", "pattern_family", "color_family", "brand", "tags")
    for key in keys:
        values = [snaps[0].get(key)]
        same = True
        for row in snaps[1:]:
            if row.get(key) != values[0]:
                same = False
                break
        if same and values[0] not in (None, "", []):
            common[key] = values[0]
    return common


def summarize_classification_changes(
    snapshots: list[dict[str, Any]],
    overlay: dict[str, Any] | None,
) -> str:
    applied = nonempty_overlay(overlay)
    if not applied:
        return "Boş alanlar mevcut dosya bilgilerini korur."
    lines = ["Uygulanacak değişiklikler (boş bırakılan alanlar korunur):"]
    labels = {
        "parent": "Ana kategori",
        "child": "Alt kategori",
        "category_path": "Kategori yolu",
        "pattern_family": "Desen ailesi",
        "color_family": "Renk",
        "brand": "Marka",
        "tags": "Etiketler",
    }
    for key, title in labels.items():
        if key not in applied:
            continue
        val = applied[key]
        show = ", ".join(val) if isinstance(val, list) else str(val)
        lines.append(f"• {title}: {show}")
    diffs: list[str] = []
    for snap in snapshots:
        bits: list[str] = []
        for key in applied:
            old = snap.get(key)
            new = applied[key]
            if old in (None, "", []):
                continue
            if old != new:
                old_s = ", ".join(old) if isinstance(old, list) else str(old)
                bits.append(f"{labels.get(key, key)}: {old_s}")
        if bits:
            diffs.append(f"{snap.get('filename') or snap.get('file_id')}: {'; '.join(bits)}")
    if diffs:
        lines.append("Mevcut farklı değerler üzerine yazılacak:")
        lines.extend(f"• {d}" for d in diffs[:12])
        extra = len(diffs) - 12
        if extra > 0:
            lines.append(f"• … ve {extra} dosya daha")
    return "\n".join(lines)


def best_preview_path(card: TeachMeCard) -> str:
    """EPS/TIFF dahil — orijinali değiştirmeden cache/preview tercih et."""
    from pathlib import Path

    original = str(card.path or "")
    suffix = Path(original).suffix.lower()
    raster_ok = suffix in {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
    }
    candidates = [
        str(card.feature_preview_path or ""),
        str(card.preview_path or ""),
    ]
    if raster_ok:
        candidates.append(original)
    else:
        candidates.append(original)
    for p in candidates:
        if p and Path(p).is_file():
            return p
    return str(card.preview_path or card.feature_preview_path or original)


def _clip_blob_for_file(db, file_id: int) -> bytes | None:
    blobs = _clip_blobs_for_files(db, [int(file_id)])
    return blobs.get(int(file_id))


def _clip_blobs_for_files(db, file_ids: list[int]) -> dict[int, bytes]:
    ids = [int(x) for x in file_ids if int(x) > 0]
    if not ids:
        return {}
    out: dict[int, bytes] = {}
    ph = ",".join("?" * len(ids))
    try:
        with db.connect() as conn:
            rows = conn.execute(
                f"SELECT file_id, clip_embedding FROM features WHERE file_id IN ({ph})",
                ids,
            ).fetchall()
    except Exception:
        return {}
    for row in rows:
        fid = int(row["file_id"] if hasattr(row, "keys") else row[0])
        raw = row["clip_embedding"] if hasattr(row, "keys") else row[1]
        if raw:
            out[fid] = bytes(raw)
    return out


def _paths_for_files(db, file_ids: list[int]) -> dict[int, str]:
    ids = [int(x) for x in file_ids if int(x) > 0]
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    out: dict[int, str] = {}
    try:
        with db.connect() as conn:
            rows = conn.execute(
                f"SELECT id, path FROM files WHERE id IN ({ph})",
                ids,
            ).fetchall()
    except Exception:
        return {}
    for row in rows:
        out[int(row["id"])] = str(row["path"] or "")
    return out


def _write_file_classification(db, file_id: int, overlay: dict[str, Any]) -> None:
    """Normal sınıflandırma alanlarına yaz; Pattern DNA gövdesini silme."""
    applied = nonempty_overlay(overlay)
    if not applied:
        tm = _lite_texture_map(db, file_id)
        tm["user_labeled"] = True
        tm["user_label_source"] = "teach_me"
        tm["classification_confidence"] = 1.0
        db.upsert_texture_map(file_id, tm)
        return
    tm = _lite_texture_map(db, file_id)
    path = str(applied.get("category_path") or "").strip()
    parent = str(applied.get("parent") or "").strip()
    child = str(applied.get("child") or "").strip()
    if not path and parent:
        path = f"{parent}/{child}".rstrip("/") if child else parent
    family = str(applied.get("pattern_family") or "").strip()
    color = str(applied.get("color_family") or "").strip()
    brand = str(applied.get("brand") or "").strip()
    tags = applied.get("tags") if isinstance(applied.get("tags"), list) else None
    animal = str(applied.get("animal_print_type") or "").strip()
    # Kategori yolu taksonomi alanlarını dayatsın; serbest "Leopard / Animal"
    # metni eski alt türü sonuç kartında canlı tutmasın.
    if path:
        try:
            from core.category_tree import pattern_fields_for_path

            pf = pattern_fields_for_path(path) or {}
            tax_family = str(pf.get("pattern_family") or "").strip()
            tax_animal = str(pf.get("animal_print_type") or "").strip()
            if tax_family:
                family = tax_family
            if tax_animal:
                animal = tax_animal
        except Exception:
            pass
    tm["user_labeled"] = True
    tm["user_label_source"] = "teach_me"
    tm["classification_confidence"] = 1.0
    tm["category_source"] = "manual_user"
    tm["category_confidence"] = 1.0
    if path:
        tm["category_path"] = path
        tm["manual_category_path"] = path
    if family:
        tm["pattern_family"] = family
    if animal:
        tm["animal_print_type"] = animal
    if color:
        tm["color_family"] = color
        tm["color_source"] = "teach_me"
    if brand:
        tm["brand_name"] = brand
    if tags is not None:
        tm["user_tags"] = [str(t).strip() for t in tags if str(t).strip()]
    db.upsert_texture_map(file_id, tm)
    try:
        db.update_file_category(
            file_id,
            category_path=path or str(tm.get("category_path") or ""),
            manual_category_path=path or str(tm.get("manual_category_path") or ""),
            category_confidence=1.0,
            category_source="manual_user",
            pattern_family=family,
            pattern_type=animal,
        )
    except Exception:
        pass
    extras: dict[str, Any] = {}
    if color:
        extras["color_family"] = color
    if family:
        extras["pattern_family"] = family
    extras["pattern_confidence"] = 1.0
    extras["needs_review"] = 0
    if extras:
        try:
            with db.connect() as conn:
                sets = ", ".join(f"{k}=?" for k in extras)
                conn.execute(
                    f"UPDATE files SET {sets} WHERE id=?",
                    (*extras.values(), file_id),
                )
        except Exception:
            pass


def concept_label_from_overlay(overlay: dict[str, Any] | None) -> str:
    applied = nonempty_overlay(overlay)
    child = str(applied.get("child") or "").strip()
    if child:
        return child
    path = str(applied.get("category_path") or "").strip()
    if path:
        return path.rsplit("/", 1)[-1].strip()
    parent = str(applied.get("parent") or "").strip()
    if parent:
        return parent
    animal = str(applied.get("animal_print_type") or "").strip()
    if animal:
        return animal
    return str(applied.get("pattern_family") or "").strip()


def concept_correction_fields(
    previous: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> dict[str, Any]:
    """Kategori/desen kavram alanlarında gerçek değişen dolu değerler."""
    after = nonempty_overlay(overlay)
    before = nonempty_overlay(previous)
    changed: dict[str, Any] = {}
    for key in _CONCEPT_KEYS:
        new = after.get(key)
        old = before.get(key)
        if new not in (None, "", []) and new != old:
            changed[key] = new
    return changed


def is_meaningful_concept_correction(
    previous: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> bool:
    """Renk/marka/etiket-only değişiklikler kavram örneği değildir."""
    return bool(concept_correction_fields(previous, overlay))


def record_verified_concept_examples(
    db,
    db_path: str,
    file_ids: Iterable[int],
    label: str,
    *,
    parent: str = "",
    customer_key: str = "",
) -> dict[str, int]:
    """concept_registry + concept_examples + CLIP. Dosya/indeks yazmaz.

    Stage 3: TR/EN leaf translation aliases + category_memory bridge (registry-first).
    Optional customer_key stamps examples only — never renames global canonical.
    """
    name = " ".join(str(label or "").strip().split())
    ids = [int(x) for x in file_ids if int(x) > 0]
    if not name or not ids:
        return {"taught": 0, "concept_id": 0}
    ck = resolve_active_customer_key(customer_key)
    concept_id = 0
    n = 0
    paths = _paths_for_files(db, ids)
    blobs = _clip_blobs_for_files(db, ids)
    # Registry-first aliases — never creates a new hard-coded engine.
    alias_list: list[str] = [name]
    try:
        from core.concept_query_normalize import leaf_translation_keys

        for k in sorted(leaf_translation_keys(name)):
            if k and k not in {a.casefold() for a in alias_list}:
                alias_list.append(k)
    except Exception:
        pass
    parent_name = str(parent or "").strip()
    if parent_name:
        try:
            from core.category_memory import register_category, register_root_category

            register_root_category(db_path, parent_name)
            register_category(db_path, parent_name, name)
        except Exception:
            pass
    for fid in ids:
        blob = blobs.get(fid)
        concept_id = learn(
            db_path,
            name,
            file_id=fid,
            file_path=str(paths.get(fid) or ""),
            parent=parent_name,
            concept_type="visual_concept",
            embedding=blob,
            embedding_backend="clip" if blob else "",
            aliases=alias_list,
            customer_key=ck,
        )
        n += 1
    try:
        from core.pattern_relations import invalidate_pattern_relations_cache

        invalidate_pattern_relations_cache()
    except Exception:
        pass
    return {"taught": n, "concept_id": int(concept_id or 0)}


def learn_from_metadata_edit(
    db,
    db_path: str,
    file_id: int,
    overlay: dict[str, Any] | None,
    previous: dict[str, Any] | None = None,
    *,
    customer_key: str = "",
) -> dict[str, int]:
    """Düzenle Kaydet: overlay zaten yazılır; burada yalnızca anlamlı kavram örneği.

    Aynı dosyada yeni kavram (Tiger) öğretilince eski user-positive (Leopard)
    demote edilir. Diğer dosyaların Leopard örneklerine dokunulmaz.
    """
    from core.concept_registry import demote_file_rivals

    fid = int(file_id or 0)
    if fid <= 0:
        return {"taught": 0, "concept_id": 0, "demoted": 0}
    snap = snapshot_file_classification(db, fid)
    before = merge_overlay_keep_existing(snap, previous)
    fields = concept_correction_fields(before, overlay)
    if not fields:
        return {"taught": 0, "concept_id": 0, "demoted": 0}
    merged = {**nonempty_overlay(overlay), **fields}
    label = concept_label_from_overlay(merged)
    if not label:
        return {"taught": 0, "concept_id": 0, "demoted": 0}
    stats = record_verified_concept_examples(
        db,
        db_path,
        [fid],
        label,
        parent=str(merged.get("parent") or ""),
        customer_key=customer_key,
    )
    demoted = 0
    try:
        demoted = int(
            demote_file_rivals(
                db_path,
                fid,
                keep_canonical=label,
                keep_as_negative=True,
            )
            or 0
        )
    except Exception:
        demoted = 0
    stats["demoted"] = demoted
    return stats


def correct_concept_membership(
    db,
    db_path: str,
    file_ids: Iterable[int],
    label: str,
    *,
    overlay: dict[str, Any] | None = None,
    rival_labels: Iterable[str] | None = None,
) -> dict[str, int]:
    """Cluster correction: demote selected files from rival concepts, teach new label as anchors.

    Does not expand to a whole autonomous cluster. Other concept positives stay intact.
    """
    from core.concept_registry import demote_file_rivals, demote_positive, upsert

    applied = nonempty_overlay(overlay)
    name = " ".join(str(label or "").strip().split())
    if not name:
        name = concept_label_from_overlay(applied)
    ids = [int(x) for x in file_ids if int(x) > 0]
    if not name or not ids:
        return {"taught": 0, "concept_id": 0, "demoted": 0}
    rivals = [
        " ".join(str(x or "").strip().split())
        for x in (rival_labels or ())
        if str(x or "").strip()
    ]
    demoted = 0
    for fid in ids:
        blob = None
        try:
            blobs = _clip_blobs_for_files(db, [fid])
            blob = blobs.get(fid)
        except Exception:
            blob = None
        for rival in rivals:
            if not rival or rival.casefold() == name.casefold():
                continue
            cid = upsert(db_path, rival, concept_type="visual_concept", source="user")
            if not cid:
                continue
            if demote_positive(db_path, cid, file_id=fid, keep_as_negative=True):
                demoted += 1
            else:
                # Positive yoksa bile boundary negative yaz (cluster adayı düzeltmesi).
                from core.concept_registry import add_example

                add_example(
                    db_path,
                    cid,
                    file_id=fid,
                    role="negative",
                    source="user",
                    embedding=blob,
                    embedding_backend="clip" if blob else "",
                )
                demoted += 1
        demoted += int(
            demote_file_rivals(db_path, fid, keep_canonical=name, keep_as_negative=True)
            or 0
        )
    stats = teach_files(db, db_path, ids, name, overlay=applied or overlay)
    stats["demoted"] = int(demoted)
    return stats


def teach_files(
    db,
    db_path: str,
    file_ids: Iterable[int],
    label: str,
    overlay: dict[str, Any] | None = None,
    *,
    customer_key: str = "",
) -> dict[str, int]:
    """Verified examples + optional file classification. Does not retrain models."""
    applied = nonempty_overlay(overlay)
    name = " ".join(str(label or "").strip().split())
    if not name:
        name = concept_label_from_overlay(applied)
    ids = [int(x) for x in file_ids if int(x) > 0]
    if not name or not ids:
        return {"taught": 0, "concept_id": 0}
    stats = record_verified_concept_examples(
        db,
        db_path,
        ids,
        name,
        parent=str(applied.get("parent") or ""),
        customer_key=customer_key,
    )
    try:
        from core.user_feedback import UserFeedbackStore

        store = UserFeedbackStore(db)
    except Exception:
        store = None
    for fid in ids:
        existing = {}
        if store is not None:
            try:
                existing = store.metadata_overlay_for_file(fid) or {}
            except Exception:
                existing = {}
        merged = merge_overlay_keep_existing(existing, applied)
        if store is not None and nonempty_overlay(merged):
            try:
                store.save_metadata_overlay(fid, merged)
            except Exception:
                pass
        try:
            _write_file_classification(db, fid, merged)
        except Exception:
            pass
    try:
        from core.archive_intelligence.hooks import resolve_candidates_for_files

        resolve_candidates_for_files(db_path, ids, status="resolved")
    except Exception:
        pass
    return stats


def build_common_edit_overlay(db, file_ids: Iterable[int]) -> dict[str, Any]:
    """Seçilen dosyalarda ortak dolu sınıflandırma; karışık alanlar boş kalır."""
    try:
        from core.user_feedback import UserFeedbackStore

        store = UserFeedbackStore(db)
    except Exception:
        store = None
    snaps: list[dict[str, Any]] = []
    for raw in file_ids:
        try:
            fid = int(raw)
        except (TypeError, ValueError):
            continue
        if fid <= 0:
            continue
        snap = snapshot_file_classification(db, fid)
        ov = {}
        if store is not None:
            try:
                ov = store.metadata_overlay_for_file(fid) or {}
            except Exception:
                ov = {}
        snaps.append(merge_overlay_keep_existing(snap, ov))
    return common_overlay_from_snapshots(snaps)


def apply_metadata_edit_to_files(
    db,
    db_path: str,
    file_ids: Iterable[int],
    overlay: dict[str, Any] | None,
    *,
    query_path: str = "",
) -> dict[str, int]:
    """Düzenle toplu kaydı: overlay + anlamlı kavram örneği. İndeks/FAISS yazmaz."""
    ids = []
    seen: set[int] = set()
    for raw in file_ids:
        try:
            fid = int(raw)
        except (TypeError, ValueError):
            continue
        if fid <= 0 or fid in seen:
            continue
        seen.add(fid)
        ids.append(fid)
    applied = nonempty_overlay(overlay)
    if not ids:
        return {"updated": 0, "taught": 0}
    try:
        from core.user_feedback import UserFeedbackStore

        store = UserFeedbackStore(db)
    except Exception:
        store = None
    updated = 0
    taught = 0
    demoted = 0
    concept_id = 0
    for fid in ids:
        previous = {}
        if store is not None:
            try:
                previous = store.metadata_overlay_for_file(fid) or {}
            except Exception:
                previous = {}
        merged = merge_overlay_keep_existing(previous, applied)
        if store is not None:
            try:
                store.save_metadata_overlay(fid, merged, query_path=query_path)
            except Exception:
                pass
        try:
            _write_file_classification(db, fid, merged)
        except Exception:
            pass
        stats = learn_from_metadata_edit(
            db, db_path, fid, merged, previous=previous
        )
        taught += int(stats.get("taught") or 0)
        demoted += int(stats.get("demoted") or 0)
        if int(stats.get("concept_id") or 0) > 0:
            concept_id = int(stats["concept_id"])
        updated += 1
    return {
        "updated": updated,
        "taught": taught,
        "concept_id": concept_id,
        "demoted": demoted,
    }


def dismiss_files(db_path: str, file_ids: Iterable[int]) -> int:
    n = 0
    ids = [int(fid) for fid in file_ids]
    for fid in ids:
        if dismiss_file(db_path, int(fid), note="user_dismiss"):
            n += 1
    try:
        from core.archive_intelligence.hooks import resolve_candidates_for_files

        resolve_candidates_for_files(db_path, ids, status="dismissed")
    except Exception:
        pass
    return n
