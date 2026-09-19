"""Exact-first pipeline — korumalı adaylar asla kaybolmaz."""

from __future__ import annotations

from typing import Any

from core.utils import normalize_path, phash_similarity

# Hash eşikleri — protected katman
PHASH_PROTECTED = 0.85
DHASH_PROTECTED = 0.88
WHASH_PROTECTED = 0.85
PHASH_STRONG = 0.92
DHASH_STRONG = 0.94


def is_protected_exact_match(
    *,
    is_self: bool = False,
    phash_sim: float = 0.0,
    dhash_sim: float = 0.0,
    whash_sim: float = 0.0,
    score: float = 0.0,
    query_partial_hash: str = "",
    rec_partial_hash: str = "",
    query_full_hash: str = "",
    rec_full_hash: str = "",
) -> bool:
    """Aday exact koruma katmanında mı?"""
    if is_self:
        return True
    if query_full_hash and rec_full_hash and query_full_hash == rec_full_hash:
        return True
    # Kısmi hash tek başına yeterli değil — aynı dosya içeriği için perceptual hash de uyuşmalı.
    if (
        query_partial_hash
        and rec_partial_hash
        and query_partial_hash == rec_partial_hash
        and phash_sim >= 0.92
        and dhash_sim >= 0.88
    ):
        return True
    if phash_sim >= 0.95 and dhash_sim >= 0.90:
        return True
    if dhash_sim >= 0.98 and whash_sim >= 0.90 and phash_sim >= 0.88:
        return True
    if whash_sim >= 0.95 and phash_sim >= 0.88 and dhash_sim >= 0.88:
        return True
    return False


def protected_score_floor(
    *,
    is_self: bool,
    phash_sim: float,
    dhash_sim: float,
    whash_sim: float,
    different_extension: bool,
) -> float:
    """Korumalı adaylar için minimum skor — exact-first sıralama."""
    if is_self:
        return 1.0
    if phash_sim >= PHASH_STRONG and dhash_sim >= DHASH_STRONG:
        return 0.99
    if phash_sim >= 0.95 and dhash_sim >= 0.90:
        if different_extension:
            return 0.90
        return 0.95
    if dhash_sim >= 0.98 and whash_sim >= 0.90 and phash_sim >= 0.88:
        return 0.95
    if phash_sim >= 0.88 and dhash_sim >= 0.88 and whash_sim >= 0.88:
        return 0.88
    return 0.0


def collect_protected_exact_ids(
    indexed: list[dict[str, Any]],
    *,
    query_phash: str = "",
    query_dhash: str = "",
    query_whash: str = "",
    query_partial_hash: str = "",
    query_full_hash: str = "",
    query_path: str = "",
    query_file_id: int | None = None,
    pattern_group_ids: set[int] | None = None,
) -> set[int]:
    """Prefilter öncesi/sonrası zorunlu aday kümesi."""
    protected: set[int] = set()
    norm_query = normalize_path(query_path) if query_path else ""
    group_ids = pattern_group_ids or set()
    # Path scan is O(n) normalize — skip when file_id already identifies the query.
    need_path_scan = bool(norm_query) and query_file_id is None

    for rec in indexed:
        fid = int(rec["id"])

        if query_file_id and fid == query_file_id:
            protected.add(fid)
            continue
        if fid in group_ids:
            protected.add(fid)
            continue
        if need_path_scan:
            rec_path = normalize_path(rec.get("path", ""))
            if rec_path == norm_query:
                protected.add(fid)
                continue

        if query_full_hash and rec.get("full_hash") == query_full_hash:
            protected.add(fid)
            continue
        if (
            query_partial_hash
            and rec.get("partial_hash") == query_partial_hash
            and query_phash
            and rec.get("phash")
        ):
            ph = phash_similarity(query_phash, rec.get("phash", ""))
            dh = (
                phash_similarity(query_dhash, rec.get("dhash", ""))
                if query_dhash
                else 0.0
            )
            if ph >= 0.92 and dh >= 0.88:
                protected.add(fid)
            continue

        # Prefix gate: full hamming over 100k+ rows alone exceeds the 500ms
        # progressive first-result budget.
        c_phash = rec.get("phash", "") or ""
        if not query_phash or not c_phash or c_phash[:4] != query_phash[:4]:
            continue

        ph = phash_similarity(query_phash, c_phash)
        dh = phash_similarity(query_dhash, rec.get("dhash", "")) if query_dhash else 0.0
        wh = phash_similarity(query_whash, rec.get("whash", "")) if query_whash else 0.0

        if is_protected_exact_match(
            phash_sim=ph,
            dhash_sim=dh,
            whash_sim=wh,
            query_partial_hash=query_partial_hash,
            rec_partial_hash=rec.get("partial_hash", ""),
            query_full_hash=query_full_hash,
            rec_full_hash=rec.get("full_hash", ""),
        ):
            protected.add(fid)

    return protected


def merge_candidate_records(
    narrowed: list[dict[str, Any]],
    indexed: list[dict[str, Any]],
    protected_ids: set[int],
) -> list[dict[str, Any]]:
    """Protected adayları final candidate set'e zorunlu ekle."""
    if not protected_ids:
        return narrowed
    present = {int(r["id"]) for r in narrowed}
    id_map = {int(r["id"]): r for r in indexed}
    out = list(narrowed)
    for fid in protected_ids:
        if fid not in present and fid in id_map:
            out.append(id_map[fid])
    return out
