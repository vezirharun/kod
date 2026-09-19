"""Stage 5 — Real visual evidence consolidation (read-only).

Joins user-taught concept_examples with index files/features Pattern DNA /
Visual Concept DNA / metadata. Never writes learning, never demotes rivals,
never creates concepts, never changes aliases. Profiles are NOT user-verified.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.textile_terms import normalize_turkish

_CONF_ORDER = {"strong": 3, "medium": 2, "weak": 1, "unknown": 0}
_MAX_EVIDENCE_BONUS = 0.06  # soft only; below concept exact gap


@dataclass
class AttributeConfidence:
    value: str = ""
    confidence: str = "unknown"  # strong|medium|weak|unknown
    support: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConceptEvidenceProfile:
    canonical: str = ""
    concept_id: int = 0
    parent: str = ""
    positive_count: int = 0
    dna_available: int = 0
    dna_missing: int = 0
    textile_evidence: int = 0
    category_path_count: int = 0
    motif: str = ""
    likely_pattern_type: str = ""
    structure: str = ""
    scale: AttributeConfidence = field(default_factory=AttributeConfidence)
    density: AttributeConfidence = field(default_factory=AttributeConfidence)
    colors: list[AttributeConfidence] = field(default_factory=list)
    attribute_coverage: float = 0.0
    rival_hints: dict[str, str] = field(default_factory=dict)
    user_verified: bool = False  # always False for auto profiles
    file_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _parse_tm(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _band(v: str) -> str:
    n = normalize_turkish(str(v or ""))
    if n in {"high", "yuksek", "large", "buyuk", "dense", "sik"}:
        return "high"
    if n in {"low", "dusuk", "small", "kucuk", "sparse", "seyrek", "fine", "mini"}:
        return "low"
    if n in {"medium", "med", "orta"}:
        return "medium"
    return n


def _conf(support: int, total: int) -> str:
    if total <= 0 or support <= 0:
        return "unknown"
    ratio = support / max(total, 1)
    if support >= 3 and ratio >= 0.55:
        return "strong"
    if support >= 2 and ratio >= 0.35:
        return "medium"
    if support >= 1:
        return "weak"
    return "unknown"


def _user_positive_rows(memory_db: str, concept_id: int) -> list[int]:
    if not memory_db or int(concept_id or 0) <= 0:
        return []
    try:
        conn = sqlite3.connect(memory_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT DISTINCT file_id FROM concept_examples
            WHERE concept_id=? AND role='positive' AND file_id>0
              AND IFNULL(source,'user') NOT IN ('auto','autonomous','candidate')
            ORDER BY file_id
            """,
            (int(concept_id),),
        ).fetchall()
        conn.close()
        return [int(r["file_id"]) for r in rows]
    except Exception:
        return []


def _resolve_paths(index_db: str) -> tuple[str, str]:
    """Return (memory_db, patterns_db)."""
    from core.concept_registry import _write_path

    mem = str(_write_path(index_db))
    pat = str(index_db)
    if not Path(pat).is_file() and Path(mem).is_file():
        # fall back: sibling patterns.db next to memory
        sibling = Path(mem).with_name("patterns.db")
        if sibling.is_file():
            pat = str(sibling)
    return mem, pat


def load_file_evidence(
    patterns_db: str,
    file_ids: list[int],
) -> list[dict[str, Any]]:
    """Read-only join of files + features for given ids."""
    if not patterns_db or not file_ids:
        return []
    out: list[dict[str, Any]] = []
    try:
        conn = sqlite3.connect(patterns_db)
        conn.row_factory = sqlite3.Row
        # chunk
        for i in range(0, len(file_ids), 400):
            chunk = file_ids[i : i + 400]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"""
                SELECT f.id, f.path, f.filename, f.status, f.category_path,
                       f.manual_category_path, f.pattern_family, f.pattern_subtype,
                       f.pattern_type, fe.texture_map, fe.phash,
                       length(fe.dino_embedding) AS dino_len,
                       length(fe.clip_embedding) AS clip_len
                FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                WHERE f.id IN ({ph})
                """,
                chunk,
            ).fetchall()
            for r in rows:
                tm = _parse_tm(r["texture_map"])
                dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
                vcdna = (
                    tm.get("visual_concept_dna")
                    if isinstance(tm.get("visual_concept_dna"), dict)
                    else {}
                )
                out.append(
                    {
                        "file_id": int(r["id"]),
                        "path": r["path"] or "",
                        "filename": r["filename"] or "",
                        "status": r["status"] or "",
                        "category_path": r["category_path"] or "",
                        "manual_category_path": r["manual_category_path"] or "",
                        "pattern_family": r["pattern_family"]
                        or tm.get("pattern_family")
                        or "",
                        "pattern_subtype": r["pattern_subtype"]
                        or tm.get("animal_print_type")
                        or tm.get("pattern_subtype")
                        or "",
                        "texture_map": tm,
                        "pattern_dna": dna,
                        "visual_concept_dna": vcdna,
                        "has_dna": bool(
                            dna
                            and (
                                dna.get("scale")
                                or dna.get("density")
                                or dna.get("dominant_colors")
                                or dna.get("confidence")
                                or dna.get("repeat_type")
                            )
                        ),
                        "textile_hint": float(dna.get("confidence") or 0) >= 0.45
                        or str(r["pattern_family"] or tm.get("pattern_family") or "")
                        in {
                            "animal_print",
                            "floral",
                            "stripe",
                            "geometric",
                        },
                    }
                )
        conn.close()
    except Exception:
        return out
    return out


def build_concept_evidence_profile(
    index_db: str,
    canonical: str,
    *,
    concept_id: int = 0,
    parent: str = "",
) -> ConceptEvidenceProfile:
    """Aggregate user-positive file evidence into a non-authoritative profile."""
    mem, pat = _resolve_paths(index_db)
    name = " ".join(str(canonical or "").strip().split())
    prof = ConceptEvidenceProfile(
        canonical=name,
        concept_id=int(concept_id or 0),
        parent=str(parent or ""),
        user_verified=False,
    )
    if not name:
        return prof
    cid = int(concept_id or 0)
    if cid <= 0:
        try:
            from core.concept_registry import concepts

            for row in concepts(index_db):
                if str(row.get("canonical") or "").casefold() == name.casefold():
                    cid = int(row["id"])
                    prof.parent = str(row.get("parent") or parent or "")
                    break
        except Exception:
            cid = 0
    prof.concept_id = cid
    fids = _user_positive_rows(mem, cid)
    prof.file_ids = list(fids)
    prof.positive_count = len(fids)
    evidence = load_file_evidence(pat, fids)
    if not evidence:
        return prof

    scale_c: Counter[str] = Counter()
    dens_c: Counter[str] = Counter()
    color_c: Counter[str] = Counter()
    fam_c: Counter[str] = Counter()
    struct_c: Counter[str] = Counter()
    dna_ok = 0
    textile = 0
    cat = 0
    for ev in evidence:
        if ev.get("has_dna"):
            dna_ok += 1
        else:
            pass
        if ev.get("textile_hint"):
            textile += 1
        if ev.get("category_path") or ev.get("manual_category_path"):
            cat += 1
        fam = normalize_turkish(str(ev.get("pattern_family") or ""))
        if fam:
            fam_c[fam] += 1
        dna = ev.get("pattern_dna") or {}
        sb = _band(str(dna.get("scale") or ""))
        if sb:
            scale_c[sb] += 1
        db = _band(str(dna.get("density") or ""))
        if db:
            dens_c[db] += 1
        for col in dna.get("dominant_colors") or []:
            cn = normalize_turkish(str(col))
            if cn:
                color_c[cn] += 1
        # structure cues from DNA / subtype
        rt = normalize_turkish(str(dna.get("repeat_type") or dna.get("geometry") or ""))
        sub = normalize_turkish(str(ev.get("pattern_subtype") or ""))
        if "stripe" in rt or "stripe" in sub or "tiger" in sub:
            struct_c["stripe"] += 1
        if "spot" in rt or "leopard" in sub or "cheetah" in sub:
            struct_c["spot"] += 1
        if "snake" in sub or "scale" in rt:
            struct_c["scale_skin"] += 1

    n = len(evidence)
    prof.dna_available = dna_ok
    prof.dna_missing = max(0, n - dna_ok)
    prof.textile_evidence = textile
    prof.category_path_count = cat
    prof.motif = normalize_turkish(name)
    if fam_c:
        prof.likely_pattern_type = fam_c.most_common(1)[0][0]
    if struct_c:
        prof.structure = struct_c.most_common(1)[0][0]

    if scale_c:
        val, support = scale_c.most_common(1)[0]
        prof.scale = AttributeConfidence(
            value=val, confidence=_conf(support, dna_ok or n), support=support, total=dna_ok or n
        )
    if dens_c:
        val, support = dens_c.most_common(1)[0]
        prof.density = AttributeConfidence(
            value=val, confidence=_conf(support, dna_ok or n), support=support, total=dna_ok or n
        )
    for col, support in color_c.most_common(6):
        prof.colors.append(
            AttributeConfidence(
                value=col,
                confidence=_conf(support, dna_ok or n),
                support=support,
                total=dna_ok or n,
            )
        )

    known_bits = 0
    total_bits = 3  # scale, density, color
    if prof.scale.confidence != "unknown":
        known_bits += 1
    if prof.density.confidence != "unknown":
        known_bits += 1
    if prof.colors:
        known_bits += 1
    prof.attribute_coverage = round(known_bits / total_bits, 3) if n else 0.0

    # Rival structure hints (descriptive only — not authority).
    # Only for animal-print siblings; never imply Rose↔Tiger rivalry.
    if prof.likely_pattern_type == "animal_print" or normalize_turkish(
        prof.canonical
    ) in {"tiger", "kaplan", "leopard", "leopar", "zebra", "snake", "yilan"}:
        if prof.structure == "stripe":
            prof.rival_hints["vs_leopard"] = "stripe_vs_spot"
        elif prof.structure == "spot":
            prof.rival_hints["vs_tiger"] = "spot_vs_stripe"
    return prof


def build_rival_profile(
    index_db: str,
    left_canonical: str,
    right_canonical: str,
) -> dict[str, Any]:
    """Compare two sibling concepts' evidence (read-only, non-authoritative)."""
    a = build_concept_evidence_profile(index_db, left_canonical)
    b = build_concept_evidence_profile(index_db, right_canonical)
    return {
        "left": a.to_dict(),
        "right": b.to_dict(),
        "discriminative": {
            "structure": {
                left_canonical: a.structure,
                right_canonical: b.structure,
            },
            "scale": {
                left_canonical: a.scale.to_dict(),
                right_canonical: b.scale.to_dict(),
            },
            "user_verified": False,
        },
    }


def concept_data_quality_report(
    index_db: str,
    canonicals: list[str] | None = None,
) -> list[dict[str, Any]]:
    names = canonicals or ["Tiger", "Leopard", "Snake Skin", "Rose", "Floral"]
    out: list[dict[str, Any]] = []
    for name in names:
        p = build_concept_evidence_profile(index_db, name)
        out.append(
            {
                "concept": p.canonical,
                "positive_count": p.positive_count,
                "DNA_available": p.dna_available,
                "DNA_missing": p.dna_missing,
                "attribute_coverage": p.attribute_coverage,
                "textile_evidence": p.textile_evidence,
                "rival_evidence": dict(p.rival_hints),
                "scale": p.scale.to_dict(),
                "density": p.density.to_dict(),
                "colors": [c.to_dict() for c in p.colors],
                "user_verified": False,
            }
        )
    return out


def _want_band_from_attr(scale: str, density: str) -> tuple[str, str]:
    s = ""
    d = ""
    if scale == "small":
        s = "low"
    elif scale == "large":
        s = "high"
    if density == "dense":
        d = "high"
    elif density == "sparse":
        d = "low"
    return s, d


def score_result_against_concept_evidence(
    result: Any,
    profile: ConceptEvidenceProfile,
    *,
    query_scale: str = "",
    query_density: str = "",
    query_colors: list[str] | None = None,
) -> dict[str, Any]:
    """Soft evidence match. Unknown → no penalty. Never demotes user_protected."""
    dbg = getattr(result, "debug", None) or {}
    if isinstance(dbg, dict):
        if (
            dbg.get("learned_concept_exact")
            or dbg.get("user_taught_positive")
            or dbg.get("protected_exact")
            or dbg.get("category_source") == "manual_user"
        ):
            return {"applied": False, "skipped": "user_protected", "delta": 0.0}

    tm = {}
    if isinstance(dbg, dict):
        tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}

    delta = 0.0
    parts: dict[str, Any] = {}
    want_s, want_d = _want_band_from_attr(query_scale, query_density)

    # Scale: only score when profile confidence known AND query asks AND dna present
    if want_s and profile.scale.confidence != "unknown":
        have = _band(str(dna.get("scale") or ""))
        if not have:
            parts["scale"] = {"confidence": "unknown", "delta": 0.0}
        elif have == want_s:
            w = 0.025 if profile.scale.confidence == "strong" else 0.015
            delta += w
            parts["scale"] = {"match": True, "confidence": profile.scale.confidence, "delta": w}
        else:
            # mismatch only if profile strong/medium — weak → ignore
            if profile.scale.confidence in {"strong", "medium"}:
                w = -0.02 if profile.scale.confidence == "strong" else -0.01
                delta += w
                parts["scale"] = {"match": False, "confidence": profile.scale.confidence, "delta": w}

    if want_d and profile.density.confidence != "unknown":
        have = _band(str(dna.get("density") or ""))
        if not have:
            parts["density"] = {"confidence": "unknown", "delta": 0.0}
        elif have == want_d:
            w = 0.025 if profile.density.confidence == "strong" else 0.015
            delta += w
            parts["density"] = {"match": True, "confidence": profile.density.confidence, "delta": w}
        elif profile.density.confidence in {"strong", "medium"}:
            w = -0.02 if profile.density.confidence == "strong" else -0.01
            delta += w
            parts["density"] = {"match": False, "confidence": profile.density.confidence, "delta": w}

    qcols = [normalize_turkish(c) for c in (query_colors or []) if c]
    if qcols and profile.colors:
        known_cols = {
            normalize_turkish(c.value)
            for c in profile.colors
            if c.confidence != "unknown"
        }
        if known_cols:
            hit = len(set(qcols) & known_cols)
            if hit:
                w = min(0.02, 0.01 * hit)
                delta += w
                parts["color"] = {"match": True, "hits": hit, "delta": w}
            else:
                # DNA file colors empty → unknown path (no penalty)
                have_colors = [
                    normalize_turkish(str(x))
                    for x in (dna.get("dominant_colors") or [])
                ]
                if not have_colors:
                    parts["color"] = {"confidence": "unknown", "delta": 0.0}

    # Structure discriminative soft nudge when aligned concept
    label = normalize_turkish(
        str(
            (dbg.get("learned_canonical") if isinstance(dbg, dict) else "")
            or getattr(result, "animal_print_type", "")
            or ""
        )
    )
    motif = normalize_turkish(profile.motif or profile.canonical)
    aligned = bool(label and motif and (motif in label or label in motif or label == motif))
    if not aligned and isinstance(dbg, dict) and dbg.get("learned_concept"):
        try:
            from core.concept_query_normalize import leaf_translation_keys

            aligned = bool(
                leaf_translation_keys(motif) & leaf_translation_keys(label)
            )
        except Exception:
            aligned = False

    if not aligned:
        # Wrong concept: no evidence bonus (and no demote — rivals handled in 2D)
        return {
            "applied": True,
            "aligned": False,
            "delta": 0.0,
            "parts": parts,
            "reason": "not_aligned",
            "user_verified": False,
        }

    delta = max(-0.04, min(_MAX_EVIDENCE_BONUS, delta))
    return {
        "applied": True,
        "aligned": True,
        "delta": round(delta, 4),
        "parts": parts,
        "profile_coverage": profile.attribute_coverage,
        "user_verified": False,
    }


_profile_cache: dict[str, ConceptEvidenceProfile] = {}


def get_cached_profile(index_db: str, canonical: str) -> ConceptEvidenceProfile:
    key = f"{index_db}::{canonical.casefold()}"
    if key not in _profile_cache:
        _profile_cache[key] = build_concept_evidence_profile(index_db, canonical)
    return _profile_cache[key]


def clear_evidence_cache() -> None:
    _profile_cache.clear()


def apply_concept_evidence_scoring(
    results: list[Any],
    query_text: str,
    *,
    index_db: str = "",
    canonical: str = "",
) -> list[Any]:
    """Soft re-score using concept evidence profile. Safe no-op if no DB/profile."""
    if not results or not (query_text or "").strip():
        return results
    from core.query_attribute_intel import extract_query_attributes

    attrs = extract_query_attributes(query_text)
    name = canonical
    if not name:
        # Prefer learned primary from first result debug / analysis motif
        try:
            from core.query_attribute_intel import concept_core_text
            from core.concept_query_normalize import relation_to_concept

            core = concept_core_text(query_text) or query_text
            for can, aliases in (
                ("Tiger", ["kaplan", "Tiger"]),
                ("Leopard", ["leopar", "Leopard"]),
                ("Snake Skin", ["snake skin", "Snake Skin"]),
                ("Snake", ["yılan", "snake", "yilan"]),
            ):
                if relation_to_concept(core, can, aliases=aliases):
                    name = can
                    break
            if not name and attrs.motif:
                name = str(attrs.motif)
        except Exception:
            name = attrs.motif or ""
    if not name or not index_db:
        return results

    try:
        profile = get_cached_profile(index_db, name)
    except Exception:
        return results
    if profile.positive_count <= 0:
        return results

    for rec in results:
        try:
            old = float(getattr(rec, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        meta = score_result_against_concept_evidence(
            rec,
            profile,
            query_scale=attrs.scale,
            query_density=attrs.density,
            query_colors=list(attrs.colors),
        )
        if meta.get("skipped") == "user_protected":
            dbg = dict(getattr(rec, "debug", None) or {})
            dbg["concept_evidence"] = meta
            rec.debug = dbg
            continue
        if meta.get("applied") and abs(float(meta.get("delta") or 0)) > 1e-9:
            new = max(0.0, min(1.0, old + float(meta["delta"])))
            rec.score = new
            if hasattr(rec, "score_percent"):
                rec.score_percent = round(new * 100, 1)
            meta["before"] = round(old, 4)
            meta["after"] = round(new, 4)
        dbg = dict(getattr(rec, "debug", None) or {})
        dbg["concept_evidence"] = {
            **meta,
            "canonical": profile.canonical,
            "dna_available": profile.dna_available,
            "dna_missing": profile.dna_missing,
        }
        rec.debug = dbg
    return results
