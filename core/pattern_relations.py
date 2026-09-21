"""BU DESENİN DİĞERLERİ — detail-view relation layer (read-only).

Reuses Exact/duplicate marks, pattern_groups, taught concept_examples,
visual_family rejects, and family-graph vetoes. Does NOT change search ranking,
index pipeline, or Preview Pool writes.

SIMILARITY ≠ SAME PATTERN: embedding/CLIP alone never lands in AYNI DESEN.
Tiger≠Leopard and animal≠textile are hard vetoes (existing infra).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.duplicate_detection import (
    DUP_COLOR,
    DUP_EXACT,
    DUP_NEAR,
    DUP_SCALE,
    classify_duplicate_relation,
    find_duplicate_candidates,
)
from core.visual_family_candidates import (
    FamilyMember,
    hydrate_member_clips,
    is_rejected_family,
    normalize_variant_stem,
    pair_family_signals,
    path_affinity,
    design_scope_dirs,
    design_scope_affinity,
)

SECTION_SAME = "ayni_desen"
SECTION_VARIANT = "varyantlar"
SECTION_FAMILY = "gorsel_aile"
SECTION_OTHER = "diger_klasorler"
SECTION_SIMILAR = "benzer_desenler"

SECTION_LABELS = {
    SECTION_SAME: "AYNI DESEN",
    SECTION_VARIANT: "VARYANTLAR",
    SECTION_FAMILY: "GÖRSEL AİLE",
    SECTION_OTHER: "DİĞER KLASÖRLER / MÜŞTERİLER",
    SECTION_SIMILAR: "BENZER DESENLER",
}

# Claim order — first wins; no double-listing
_CLAIM_ORDER = (
    SECTION_SAME,
    SECTION_VARIANT,
    SECTION_FAMILY,
    SECTION_OTHER,
    SECTION_SIMILAR,
)

_AYNI_RELS = frozenset({DUP_EXACT, "representative", "exact_same"})
_VARIANT_RELS = frozenset(
    {DUP_COLOR, DUP_SCALE, "format_variant", "crop_variant", "color_variant", "scale_variant"}
)
_NEAR_AYNI_SCORE = 0.95
_NEAR_BENZER_MIN = 0.78

_cache_lock = threading.RLock()
_CACHE: dict[int, tuple[float, "RelationBundle"]] = {}
_CACHE_GEN = 0
_CACHE_TTL_SEC = 120.0
_CACHE_MAX = 64


@dataclass
class RelationItem:
    file_id: int
    path: str = ""
    filename: str = ""
    customer: str = ""
    folder: str = ""
    section: str = ""
    relation: str = ""
    score: float = 0.0
    reason: str = ""
    preview_path: str = ""


@dataclass
class RelationSection:
    key: str
    label: str
    items: list[RelationItem] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)


@dataclass
class RelationBundle:
    source_file_id: int
    sections: list[RelationSection] = field(default_factory=list)
    loaded_stage: str = "empty"  # empty|meta|index|full
    cache_gen: int = 0
    error: str = ""

    def non_empty(self) -> list[RelationSection]:
        return [s for s in self.sections if s.items]

    def total_count(self) -> int:
        return sum(s.count for s in self.sections)


def invalidate_pattern_relations_cache(file_id: int | None = None) -> None:
    """Call after family learn / reject so detail relations refresh."""
    global _CACHE_GEN
    with _cache_lock:
        _CACHE_GEN += 1
        if file_id is None:
            _CACHE.clear()
            return
        _CACHE.pop(int(file_id), None)


def get_pattern_relations(
    db: Any,
    file_id: int,
    *,
    db_path: str = "",
    include_similar: bool = True,
    stage: str = "full",
    use_cache: bool = True,
) -> RelationBundle:
    """Build relation sections for an open detail file.

    stage: 'meta' (group + taught only) | 'index' (+ hash peers) | 'full'
    Never blocks on embeddings for AYNI; BENZER uses hash/near only (no CLIP-alone).
    """
    fid = int(file_id or 0)
    if fid <= 0 or db is None:
        return RelationBundle(source_file_id=fid, error="invalid_file")

    if use_cache:
        cached = _cache_get(fid)
        if cached is not None:
            want = {"meta": 1, "index": 2, "full": 3}.get(stage, 3)
            have = {"meta": 1, "index": 2, "full": 3}.get(cached.loaded_stage, 0)
            if have >= want:
                return cached

    bundle = _build_relations(
        db,
        fid,
        db_path=db_path or str(getattr(db, "db_path", "") or ""),
        include_similar=include_similar,
        stage=stage,
    )
    if use_cache and not bundle.error:
        _cache_put(fid, bundle)
    return bundle


def _cache_get(file_id: int) -> RelationBundle | None:
    with _cache_lock:
        hit = _CACHE.get(int(file_id))
        if not hit:
            return None
        ts, bundle = hit
        if bundle.cache_gen != _CACHE_GEN:
            _CACHE.pop(int(file_id), None)
            return None
        if (time.monotonic() - ts) > _CACHE_TTL_SEC:
            _CACHE.pop(int(file_id), None)
            return None
        return bundle


def _cache_put(file_id: int, bundle: RelationBundle) -> None:
    with _cache_lock:
        bundle.cache_gen = _CACHE_GEN
        _CACHE[int(file_id)] = (time.monotonic(), bundle)
        if len(_CACHE) > _CACHE_MAX:
            oldest = sorted(_CACHE.items(), key=lambda kv: kv[1][0])[: len(_CACHE) - _CACHE_MAX]
            for k, _ in oldest:
                _CACHE.pop(k, None)


def _build_relations(
    db: Any,
    file_id: int,
    *,
    db_path: str,
    include_similar: bool,
    stage: str,
) -> RelationBundle:
    src = _load_record(db, file_id)
    if not src:
        return RelationBundle(source_file_id=file_id, error="not_found")

    src_folder = _folder_key(str(src.get("path") or ""))
    src_customer = _customer_label(src, db_path=db_path)
    src_member = _member_from_rec(src)
    claimed: set[int] = {file_id}
    buckets: dict[str, list[RelationItem]] = {k: [] for k in _CLAIM_ORDER}

    # --- Stage meta: pattern_group + taught family ---
    for item in _from_pattern_group(db, src, src_member):
        _claim(buckets, claimed, item, src_folder, src_customer)

    for item in _from_taught_family(db, db_path, src, src_member):
        _claim(buckets, claimed, item, src_folder, src_customer)

    loaded = "meta"

    if stage in ("index", "full"):
        for item in _from_hash_peers(db, src, src_member):
            _claim(buckets, claimed, item, src_folder, src_customer)
        _retag_other_folders(buckets, src_folder, src_customer)
        loaded = "index"

    if stage == "full":
        # Additive body/part completion via existing PATH+CLIP / FAMILY+CLIP consensus
        for item in _from_part_completion(db, db_path, src, src_member, claimed):
            _claim(buckets, claimed, item, src_folder, src_customer)
        if include_similar:
            for item in _from_near_similar(db, src, src_member, claimed):
                _claim(buckets, claimed, item, src_folder, src_customer)
        loaded = "full"

    # Preview paths (SSOT read-only)
    all_items = [it for sec in buckets.values() for it in sec]
    _attach_previews(all_items)

    sections = [
        RelationSection(key=k, label=SECTION_LABELS[k], items=buckets[k])
        for k in _CLAIM_ORDER
        if buckets[k]
    ]
    return RelationBundle(source_file_id=file_id, sections=sections, loaded_stage=loaded)


def _claim(
    buckets: dict[str, list[RelationItem]],
    claimed: set[int],
    item: RelationItem,
    src_folder: str,
    src_customer: str,
) -> None:
    fid = int(item.file_id)
    if fid <= 0 or fid in claimed:
        return
    # Weak near-dup in another folder/customer → DİĞER (exact stays AYNI)
    if (
        item.section == SECTION_SAME
        and item.relation == DUP_NEAR
        and float(item.score or 0) < _NEAR_AYNI_SCORE
        and _is_other_place(item, src_folder, src_customer)
    ):
        item.section = SECTION_OTHER
        item.reason = item.reason or "other_folder"
    claimed.add(fid)
    buckets.setdefault(item.section, []).append(item)


def _retag_other_folders(
    buckets: dict[str, list[RelationItem]],
    src_folder: str,
    src_customer: str,
) -> list[RelationItem]:
    """Move non-exact same-stem items in other folders to DİĞER if still in AYNI with weak reason."""
    moved: list[RelationItem] = []
    keep_same: list[RelationItem] = []
    for it in buckets.get(SECTION_SAME, []):
        if (
            it.relation == DUP_NEAR
            and float(it.score or 0) < _NEAR_AYNI_SCORE
            and _is_other_place(it, src_folder, src_customer)
        ):
            it.section = SECTION_OTHER
            it.reason = it.reason or "other_folder"
            buckets[SECTION_OTHER].append(it)
            moved.append(it)
        else:
            keep_same.append(it)
    buckets[SECTION_SAME] = keep_same
    return moved


def _is_other_place(item: RelationItem, src_folder: str, src_customer: str) -> bool:
    if src_customer and item.customer and item.customer != src_customer:
        return True
    if src_folder and item.folder and item.folder != src_folder:
        return True
    return False


def _load_record(db: Any, file_id: int) -> dict[str, Any] | None:
    try:
        rows = db.get_indexed_files_by_ids(
            [int(file_id)],
            include_pending=True,
            include_inactive_sources=True,
            lightweight=True,
        )
        if rows:
            return rows[0]
    except Exception:
        pass
    try:
        base = db.get_file_by_id(int(file_id))
        if not base:
            return None
        feat = db.get_features(int(file_id)) or {}
        out = dict(base)
        out.update(
            {
                "phash": feat.get("phash", ""),
                "dhash": feat.get("dhash", ""),
                "whash": feat.get("whash", ""),
                "texture_map": feat.get("texture_map") or {},
            }
        )
        return out
    except Exception:
        return None


def _member_from_rec(rec: dict[str, Any]) -> FamilyMember:
    tm = rec.get("texture_map") if isinstance(rec.get("texture_map"), dict) else {}
    from core.visual_family_candidates import decode_clip_blob

    raw = rec.get("clip_embedding")
    if raw is None and isinstance(tm, dict):
        raw = tm.get("clip_embedding")
    return FamilyMember(
        file_id=int(rec.get("id") or rec.get("file_id") or 0),
        filename=str(rec.get("filename") or ""),
        path=str(rec.get("path") or ""),
        guess=str(tm.get("guess") or tm.get("predicted_label") or ""),
        pattern_family=str(
            rec.get("pattern_family")
            or tm.get("pattern_family")
            or (tm.get("pattern_dna") or {}).get("motif_family")
            or ""
        ),
        category=str(tm.get("category") or rec.get("category") or ""),
        color_family=str(
            rec.get("color_family")
            or tm.get("color_family")
            or ((tm.get("color_index") or {}) if isinstance(tm.get("color_index"), dict) else {}).get(
                "color_family"
            )
            or ""
        ),
        clip=decode_clip_blob(raw),
    )


def _item_from_rec(
    rec: dict[str, Any],
    *,
    section: str,
    relation: str,
    score: float,
    reason: str,
    db_path: str = "",
) -> RelationItem:
    path = str(rec.get("path") or "")
    return RelationItem(
        file_id=int(rec.get("id") or rec.get("file_id") or 0),
        path=path,
        filename=str(rec.get("filename") or Path(path).name),
        customer=_customer_label(rec, db_path=db_path),
        folder=_folder_key(path),
        section=section,
        relation=relation,
        score=float(score or 0),
        reason=reason,
    )


def _folder_key(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().parent).lower()
    except Exception:
        return str(Path(path).parent).lower()


def _customer_label(rec: dict[str, Any], *, db_path: str = "") -> str:
    """Known customer only — never invent names from random path tokens."""
    cust = str(rec.get("customer") or "").strip()
    if cust:
        return cust
    path = str(rec.get("path") or "")
    if not path:
        return ""
    try:
        from core.customer_discovery import get_customer_registry, path_under_any_prefix

        reg = get_customer_registry(db=None)
        for entry in reg.entries():
            prefs = entry.all_scope_prefixes()
            if prefs and path_under_any_prefix(path, prefs):
                return str(entry.name or "")
    except Exception:
        pass
    return ""


def _compatible(src: FamilyMember, other: FamilyMember) -> bool:
    """Hard veto: distinct concepts / domain mismatch ⇒ not same pattern."""
    for left, right in (
        (src.guess, other.guess),
        (src.pattern_family, other.pattern_family),
        (src.category, other.category),
    ):
        try:
            from core.canonical_correction import are_distinct_concepts

            a = " ".join(str(left or "").split())
            b = " ".join(str(right or "").split())
            if a and b and are_distinct_concepts(a, b):
                return False
        except Exception:
            pass
    try:
        from core.visual_family_candidates import _domain_bucket

        da, dbucket = _domain_bucket(src), _domain_bucket(other)
        if da and dbucket and da != dbucket:
            return False
    except Exception:
        pass
    return True


def _section_for_dup_rel(rel: str, score: float) -> str:
    if rel in _AYNI_RELS or (rel == DUP_EXACT):
        return SECTION_SAME
    if rel in _VARIANT_RELS:
        return SECTION_VARIANT
    if rel == DUP_NEAR:
        if score >= _NEAR_AYNI_SCORE:
            return SECTION_SAME
        return SECTION_SIMILAR
    return SECTION_SIMILAR


def _from_pattern_group(db: Any, src: dict[str, Any], src_member: FamilyMember) -> list[RelationItem]:
    out: list[RelationItem] = []
    fid = int(src.get("id") or 0)
    try:
        group = db.get_pattern_group_for_file(fid)
    except Exception:
        group = None
    if not group:
        return out
    try:
        members = db.list_pattern_group_members(int(group["id"]))
    except Exception:
        return out
    peer_ids = [int(m["file_id"]) for m in members if int(m.get("file_id") or 0) != fid]
    if not peer_ids:
        return out
    try:
        rows = db.get_indexed_files_by_ids(
            peer_ids, include_pending=True, include_inactive_sources=True, lightweight=True
        )
    except Exception:
        rows = []
    by_id = {int(r["id"]): r for r in rows}
    for m in members:
        mid = int(m.get("file_id") or 0)
        if mid == fid or mid not in by_id:
            continue
        rec = by_id[mid]
        other = _member_from_rec(rec)
        if not _compatible(src_member, other):
            continue
        rel = str(m.get("relation") or DUP_NEAR)
        score = float(m.get("score") or 0)
        section = _section_for_dup_rel(rel, score)
        # Stem + path affinity soft-upgrade to variant when group says near
        if section == SECTION_SIMILAR:
            stem_same = normalize_variant_stem(src_member.filename) == normalize_variant_stem(
                other.filename
            )
            if stem_same and normalize_variant_stem(src_member.filename):
                section = SECTION_VARIANT
                rel = rel or "stem_variant"
        out.append(
            _item_from_rec(
                rec,
                section=section,
                relation=rel,
                score=score,
                reason="pattern_group",
            )
        )
    return out


def _from_taught_family(
    db: Any, db_path: str, src: dict[str, Any], src_member: FamilyMember
) -> list[RelationItem]:
    out: list[RelationItem] = []
    if not db_path:
        return out
    fid = int(src.get("id") or 0)
    try:
        from core.concept_registry import positives_for_file

        concepts = positives_for_file(db_path, fid)
    except Exception:
        concepts = []
    # Skip candidate-only auto noise; keep user/auto taught
    concept_ids = [
        int(c["concept_id"])
        for c in concepts
        if str(c.get("source") or "") not in ("candidate",)
    ]
    if not concept_ids:
        return out

    sibling_ids: set[int] = set()
    try:
        import sqlite3
        from core.concept_registry import _write_path

        c = sqlite3.connect(str(_write_path(db_path)), timeout=10)
        c.row_factory = sqlite3.Row
        try:
            for cid in concept_ids:
                rows = c.execute(
                    """SELECT DISTINCT file_id FROM concept_examples
                       WHERE concept_id=? AND role='positive' AND file_id>0
                         AND IFNULL(source,'user') NOT IN ('candidate')""",
                    (cid,),
                ).fetchall()
                for r in rows:
                    mid = int(r["file_id"])
                    if mid != fid:
                        sibling_ids.add(mid)
        finally:
            c.close()
    except Exception:
        return out

    if not sibling_ids:
        return out

    # Banned fingerprints: hide if {src}+siblings fingerprint rejected
    try:
        all_ids = sorted({fid, *sibling_ids})
        if is_rejected_family(db_path, all_ids):
            return out
    except Exception:
        pass

    try:
        rows = db.get_indexed_files_by_ids(
            list(sibling_ids),
            include_pending=True,
            include_inactive_sources=True,
            lightweight=True,
        )
    except Exception:
        rows = []

    for rec in rows:
        other = _member_from_rec(rec)
        if not _compatible(src_member, other):
            continue
        mid = int(rec.get("id") or 0)
        # Pair reject with source
        try:
            if is_rejected_family(db_path, [fid, mid]):
                continue
        except Exception:
            pass
        out.append(
            _item_from_rec(
                rec,
                section=SECTION_FAMILY,
                relation="taught_family",
                score=1.0,
                reason="concept_examples",
            )
        )
    return out


def _from_hash_peers(
    db: Any, src: dict[str, Any], src_member: FamilyMember
) -> list[RelationItem]:
    out: list[RelationItem] = []
    fid = int(src.get("id") or 0)
    phash = str(src.get("phash") or "")
    partial = str(src.get("partial_hash") or "")
    tm = src.get("texture_map") if isinstance(src.get("texture_map"), dict) else {}
    # Also harvest duplicate_info.peers metadata
    dup = tm.get("duplicate_info") if isinstance(tm.get("duplicate_info"), dict) else {}
    peer_meta = {int(p.get("file_id") or 0): p for p in (dup.get("peers") or []) if p}

    try:
        cands = find_duplicate_candidates(
            db, phash=phash, partial_hash=partial, exclude_id=fid, limit=24
        )
    except Exception:
        cands = []

    self_rec = {
        "phash": phash,
        "dhash": src.get("dhash", ""),
        "whash": src.get("whash", ""),
        "partial_hash": partial,
        "texture_map": tm,
        "color_family": src_member.color_family,
    }
    for cand in cands:
        cid = int(cand.get("id") or 0)
        if cid <= 0:
            continue
        other = _member_from_rec(cand)
        if not _compatible(src_member, other):
            continue
        classified = classify_duplicate_relation(self_rec, cand)
        if not classified:
            meta = peer_meta.get(cid)
            if meta:
                classified = (str(meta.get("type") or DUP_NEAR), float(meta.get("score") or 0))
            else:
                continue
        rel, score = classified
        # Exact / variant only from hash — never CLIP
        section = _section_for_dup_rel(rel, float(score))
        if section == SECTION_SIMILAR:
            # Hash near but not ayni — leave for benzer stage
            continue
        out.append(
            _item_from_rec(
                cand,
                section=section,
                relation=rel,
                score=float(score),
                reason="hash",
            )
        )
    return out



def _design_candidate_records(
    db: Any,
    db_path: str,
    src: dict[str, Any],
    *,
    limit: int = 60,
) -> list[dict[str, Any]]:
    """Harvest part-completion candidates under design-scope dirs (not exact parent only).

    Includes sibling size/part subfolders. Also merges Tanımsızlar cards that share
    a design-scope directory. Does not invent embeddings.
    """
    path = str(src.get("path") or "")
    if not path:
        return []
    dirs = design_scope_dirs(path, max_up=2)
    if not dirs:
        try:
            dirs = [str(Path(path).parent)]
        except Exception:
            return []
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    fid = int(src.get("id") or 0)
    for d in dirs:
        rows: list[dict[str, Any]] = []
        try:
            if hasattr(db, "list_files_in_directory"):
                rows = list(db.list_files_in_directory(d, limit=min(80, limit)) or [])
        except Exception:
            rows = []
        for rec in rows:
            mid = int(rec.get("id") or rec.get("file_id") or 0)
            if mid <= 0 or mid == fid or mid in seen:
                continue
            seen.add(mid)
            out.append(rec)
            if len(out) >= limit:
                return out
    # Extra path-prefix scan already covers Tanımsız siblings under the same
    # design-scope directories via list_files_in_directory (includes unclassified).
    # Avoid list_inbox_pools here — too heavy for relation builds.
    return out


def _same_customer_or_unknown(src: dict[str, Any], other: dict[str, Any], db_path: str) -> bool:
    """Reject when both resolve to different known customers."""
    a = _customer_label(src, db_path=db_path)
    b = _customer_label(other, db_path=db_path)
    if a and b and a != b:
        return False
    return True


def _from_part_completion(
    db: Any,
    db_path: str,
    src: dict[str, Any],
    src_member: FamilyMember,
    claimed: set[int],
) -> list[RelationItem]:
    """Design-scope siblings that pass path/visual consensus → VARYANTLAR.

    Additive on Bu Desenin Diğerleri. Not taught_family.
    Subfolders OK. Tanımsızlar OK when consensus holds.
    Requires visual (CLIP/DINO) + path/design-scope, OR stem+path size variants.
    Never filename-only. Never cross-customer.
    """
    out: list[RelationItem] = []
    path = str(src.get("path") or "")
    if not path:
        return out
    siblings = _design_candidate_records(db, db_path, src, limit=60)
    if not siblings:
        return out

    fid = int(src.get("id") or 0)
    members: list[FamilyMember] = [src_member]
    id_to_rec: dict[int, dict[str, Any]] = {}
    for rec in siblings:
        mid = int(rec.get("id") or rec.get("file_id") or 0)
        if mid <= 0 or mid == fid or mid in claimed:
            continue
        if not _same_customer_or_unknown(src, rec, db_path):
            continue
        other = _member_from_rec(rec)
        if not _compatible(src_member, other):
            continue
        # Must share design scope (sibling folders allowed)
        if not design_scope_affinity(path, str(rec.get("path") or other.path or "")):
            continue
        members.append(other)
        id_to_rec[mid] = rec

    if len(members) < 2:
        return out

    try:
        hydrate_member_clips(members, db_path=db_path or str(getattr(db, "db_path", "") or ""))
    except Exception:
        pass

    src_m = members[0]
    for other in members[1:]:
        mid = int(other.file_id)
        if mid in claimed or mid not in id_to_rec:
            continue
        try:
            if db_path and is_rejected_family(db_path, [fid, mid]):
                continue
        except Exception:
            pass
        # Force path signal via design-scope when pair_family_signals path_affinity is narrow
        if design_scope_affinity(src_m.path, other.path) and not path_affinity(src_m.path, other.path):
            # temporarily share parent-key for signal — use pair after copying path? better: check signals then augment
            pass
        signals, conf = pair_family_signals(src_m, other)
        # Augment: design-scope counts as path for completion when visual present
        sigs = list(signals or [])
        if design_scope_affinity(src_m.path, other.path) and "path" not in sigs:
            # Only inject path when visual evidence exists (CLIP/DINO) or stem match
            has_visual = "clip" in sigs or "dino" in sigs
            has_stem = "stem" in sigs
            if has_visual or has_stem:
                sigs = ["path"] + sigs
                conf = max(float(conf or 0), 0.55)
        if not sigs or conf <= 0:
            continue
        visual = "clip" in sigs or "dino" in sigs
        hard = [s for s in sigs if s not in ("clip", "dino")]
        # Require: (path/design + visual) OR (stem + path) — never visual alone, never family alone
        ok = False
        if visual and "path" in hard:
            ok = True
        elif visual and "family" in hard:
            ok = True
        elif "stem" in hard and "path" in hard:
            ok = True
        if not ok:
            continue
        # Different garment parts / no shared stem → demand stronger visual align
        try:
            from core.visual_family_candidates import (
                garment_part_token,
                normalize_variant_stem,
                _cosine,
                CLIP_ALIGN,
            )
            stem_a = normalize_variant_stem(src_m.filename)
            stem_b = normalize_variant_stem(other.filename)
            part_a = garment_part_token(src_m.filename)
            part_b = garment_part_token(other.filename)
            stems_differ = bool(stem_a and stem_b and stem_a != stem_b)
            parts_differ = bool(part_a and part_b and part_a != part_b)
            if stems_differ or parts_differ:
                clip_v = _cosine(getattr(src_m, "clip", None), getattr(other, "clip", None))
                dino_v = _cosine(getattr(src_m, "dino", None), getattr(other, "dino", None))
                # Stricter than base CLIP_ALIGN when joining dissimilar names in same folder
                if max(clip_v, dino_v) < max(0.90, float(CLIP_ALIGN) + 0.05):
                    continue
        except Exception:
            pass
        reason = "multi_signal"
        if "clip" in sigs and "path" in hard:
            reason = "path_clip_consensus"
        elif "dino" in sigs and "path" in hard:
            reason = "path_dino_consensus"
        elif "clip" in sigs and "family" in hard:
            reason = "family_clip_consensus"
        elif "dino" in sigs and "family" in hard:
            reason = "family_dino_consensus"
        elif "stem" in hard and "path" in hard:
            reason = "stem_path_size_variant"
        section = SECTION_VARIANT
        if "family" in hard and "stem" not in hard:
            section = SECTION_FAMILY
        out.append(
            _item_from_rec(
                id_to_rec[mid],
                section=section,
                relation="part_completion",
                score=float(conf),
                reason=reason,
            )
        )
    return out



def _from_near_similar(
    db: Any,
    src: dict[str, Any],
    src_member: FamilyMember,
    claimed: set[int],
) -> list[RelationItem]:
    """Close-but-not-same. Hash near only — embedding-alone forbidden."""
    out: list[RelationItem] = []
    fid = int(src.get("id") or 0)
    phash = str(src.get("phash") or "")
    partial = str(src.get("partial_hash") or "")
    tm = src.get("texture_map") if isinstance(src.get("texture_map"), dict) else {}
    self_rec = {
        "phash": phash,
        "dhash": src.get("dhash", ""),
        "whash": src.get("whash", ""),
        "partial_hash": partial,
        "texture_map": tm,
        "color_family": src_member.color_family,
    }
    try:
        cands = find_duplicate_candidates(
            db, phash=phash, partial_hash=partial, exclude_id=fid, limit=24
        )
    except Exception:
        cands = []
    for cand in cands:
        cid = int(cand.get("id") or 0)
        if cid in claimed or cid <= 0:
            continue
        other = _member_from_rec(cand)
        if not _compatible(src_member, other):
            continue
        classified = classify_duplicate_relation(self_rec, cand)
        if not classified:
            continue
        rel, score = classified
        if rel != DUP_NEAR:
            continue
        if float(score) < _NEAR_BENZER_MIN or float(score) >= _NEAR_AYNI_SCORE:
            continue
        # Extra gate: if stem+path strongly agree, prefer variant not benzer
        stem_same = (
            normalize_variant_stem(src_member.filename)
            and normalize_variant_stem(src_member.filename)
            == normalize_variant_stem(other.filename)
        )
        if stem_same and path_affinity(src_member.path, other.path):
            continue
        out.append(
            _item_from_rec(
                cand,
                section=SECTION_SIMILAR,
                relation=rel,
                score=float(score),
                reason="near_hash",
            )
        )
    return out


def _attach_previews(items: Iterable[RelationItem]) -> None:
    try:
        from core.preview_cache import FeaturePreviewCache

        cache = FeaturePreviewCache()
    except Exception:
        return
    for it in items:
        if not it.path:
            continue
        try:
            existing = cache.get_existing(it.path)
            if existing.success and existing.preview_path:
                it.preview_path = str(existing.preview_path)
        except Exception:
            continue
