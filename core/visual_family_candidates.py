"""Visual family candidates for Teach Me — multi-signal, never similarity-only.

Groups size/color/path variants of the SAME design (Mickey_S/M/L + Mickey_final)
so the user teaches once. Distinct concepts (Kaplan≠Leopar, Çiçek≠Gül,
animal leopard ≠ textile leopard) must not auto-merge.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.archive_intelligence.family_graph import (
    family_snapshot,
    may_merge_as_same_family,
)

_UNKNOWN = frozenset({"", "unknown", "none", "null", "belirsiz", "other", "diger", "diğer"})
_SIZE_TOKENS = frozenset(
    {
        "s",
        "m",
        "l",
        "xl",
        "xxl",
        "xxxl",
        "2xl",
        "3xl",
        "4xl",
        "xs",
        "sm",
        "md",
        "lg",
        "small",
        "medium",
        "large",
        "kucuk",
        "küçük",
        "orta",
        "buyuk",
        "büyük",
    }
)

# Garment production parts — metadata only; NEVER stripped for stem equality alone.
_GARMENT_PART_TOKENS = frozenset(
    {
        "ön",
        "on",
        "arka",
        "kol",
        "manşet",
        "manset",
        "yaka",
        "metraj",
        "ay",
        "parça",
        "parca",
        "parçası",
        "parcasi",
        "parçasi",
    }
)

# Hard signals that may pair with CLIP for consensus (never CLIP alone).
_CLIP_CONSENSUS_HARD = frozenset({"stem", "path", "family"})
_VARIANT_TOKENS = frozenset(
    {
        "final",
        "fin",
        "v1",
        "v2",
        "v3",
        "copy",
        "kopya",
        "new",
        "yeni",
        "old",
        "eski",
        "rev",
        "rev1",
        "rev2",
        "a",
        "b",
        "c",
        "bw",
        "color",
        "renk",
        "renkli",
        "mono",
    }
)
_GENERIC_FAMILIES = frozenset(
    {
        "floral",
        "flower",
        "cicek",
        "çiçek",
        "animal",
        "animal_print",
        "animal print",
        "geometric",
        "geo",
        "abstract",
        "textile",
        "pattern",
        "desen",
    }
)
_ANIMAL_HINTS = frozenset(
    {"animal", "animal_print", "hayvan", "wildlife", "zoo", "foto", "photo", "real"}
)
_TEXTILE_HINTS = frozenset(
    {"textile", "fabric", "kumas", "kumaş", "print", "desen", "dokuma", "repeat"}
)

MIN_MEMBERS = 2
MIN_HARD_SIGNALS = 2
CLIP_ALIGN = 0.82
MIN_FAMILY_CONF = 0.55
MAX_CANDIDATES = 40


@dataclass
class FamilyMember:
    file_id: int
    filename: str = ""
    path: str = ""
    pool: str = ""
    guess: str = ""
    pattern_family: str = ""
    category: str = ""
    color_family: str = ""
    confidence: float = 0.0
    clip: Any = None  # optional np.ndarray
    dino: Any = None  # optional np.ndarray (read-only hydrate)
    card: Any = None


@dataclass
class FamilyCandidate:
    family_id: str
    label: str
    member_ids: list[int]
    confidence: float
    signals: list[str] = field(default_factory=list)
    reason: str = ""
    sample_filenames: list[str] = field(default_factory=list)
    pool: str = ""


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _write_path(db_path: str) -> str:
    try:
        from core.concept_registry import _write_path as wp

        return str(wp(db_path))
    except Exception:
        return str(db_path)


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(str(_write_path(db_path)), timeout=10)
    c.row_factory = sqlite3.Row
    c.execute(
        """CREATE TABLE IF NOT EXISTS visual_family_rejects(
            fingerprint TEXT PRIMARY KEY,
            member_ids TEXT DEFAULT '[]',
            label TEXT DEFAULT '',
            created_at TEXT DEFAULT '')"""
    )
    return c


def family_fingerprint(member_ids: Iterable[int]) -> str:
    ids = sorted({int(x) for x in member_ids if int(x) > 0})
    raw = ",".join(str(i) for i in ids)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def record_rejected_family(
    db_path: str,
    member_ids: Iterable[int],
    *,
    label: str = "",
) -> str:
    """Ban this exact member set from reforming. Does not dismiss files or overwrite concepts."""
    ids = sorted({int(x) for x in member_ids if int(x) > 0})
    if len(ids) < MIN_MEMBERS or not db_path:
        return ""
    fp = family_fingerprint(ids)
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    c = _conn(db_path)
    try:
        c.execute(
            """INSERT OR REPLACE INTO visual_family_rejects
               (fingerprint, member_ids, label, created_at) VALUES(?,?,?,?)""",
            (fp, json.dumps(ids), str(label or ""), now),
        )
        c.commit()
    finally:
        c.close()
    try:
        from core.pattern_relations import invalidate_pattern_relations_cache

        invalidate_pattern_relations_cache()
    except Exception:
        pass
    return fp


def is_rejected_family(db_path: str, member_ids: Iterable[int]) -> bool:
    ids = sorted({int(x) for x in member_ids if int(x) > 0})
    if len(ids) < MIN_MEMBERS or not db_path:
        return False
    fp = family_fingerprint(ids)
    c = _conn(db_path)
    try:
        row = c.execute(
            "SELECT 1 FROM visual_family_rejects WHERE fingerprint=?", (fp,)
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        c.close()
    return row is not None


def rejected_family_fingerprints(db_path: str) -> set[str]:
    if not db_path:
        return set()
    c = _conn(db_path)
    try:
        rows = c.execute("SELECT fingerprint FROM visual_family_rejects").fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        c.close()
    return {str(r["fingerprint"]) for r in rows}


def normalize_variant_stem(filename: str) -> str:
    """Strip extension, size/repeat/color suffix tokens → shared design stem."""
    stem = Path(str(filename or "")).stem
    stem = stem.replace("-", "_").replace(" ", "_")
    parts = [p for p in re.split(r"[_\.]+", stem) if p]
    kept: list[str] = []
    for p in parts:
        low = p.lower()
        if low in _SIZE_TOKENS or low in _VARIANT_TOKENS:
            continue
        if re.fullmatch(r"v?\d{1,3}", low):
            continue
        if re.fullmatch(r"(?:size|ölçek|olcek|renk|color).+", low):
            continue
        kept.append(low)
    if not kept:
        return _norm(stem)
    return "_".join(kept)


def _parent_key(path: str) -> str:
    try:
        return str(Path(path).resolve().parent).lower()
    except Exception:
        p = Path(str(path or ""))
        return str(p.parent).lower() if str(p.parent) else ""


def _meaningful_dir_parts(path: str) -> list[str]:
    """Directory components only — skip filename, roots, and drive letters."""
    try:
        parts = list(Path(path).parts)
    except Exception:
        return []
    if parts:
        parts = parts[:-1]  # drop filename
    out: list[str] = []
    root_tokens = {".", "/", chr(92)}
    for x in parts:
        xl = str(x or "").strip().lower().rstrip("/" + chr(92))
        if not xl or xl in root_tokens:
            continue
        # Windows drive root e.g. "c:"
        if len(xl) == 2 and xl[1] == ":":
            continue
        out.append(xl)
    return out[-2:] if out else []



def path_affinity(path_a: str, path_b: str) -> bool:
    if not path_a or not path_b:
        return False
    pa, pb = _parent_key(path_a), _parent_key(path_b)
    if pa and pb and pa == pb:
        return True
    # Shared meaningful folder name (not OS root / drive). Prevents Windows
    # Path.parts root token intersecting across unrelated "/x" vs "/y" paths.
    da, db = _meaningful_dir_parts(path_a), _meaningful_dir_parts(path_b)
    if not da or not db:
        return False
    return bool(set(da) & set(db))


def design_scope_dirs(path: str, *, max_up: int = 2) -> list[str]:
    """Parent + ancestors (size/part folder names walk up). Not customer-root unlimited."""
    out: list[str] = []
    try:
        cur = Path(path).parent
    except Exception:
        return out

    def _is_root(p: Path) -> bool:
        try:
            s = str(p).strip().lower().rstrip("\\/")
            if not s or s in (".", "/", "\\"):
                return True
            # Windows drive root: "c:" or "c:\"
            if len(s) == 2 and s[1] == ":":
                return True
            if len(s) == 3 and s[1] == ":" and s[2] in ("\\", "/"):
                return True
            name = p.name
            if not name or name in ("\\", "/", ""):
                return True
            return False
        except Exception:
            return True

    size_part = _SIZE_TOKENS | _GARMENT_PART_TOKENS | {
        "mlxl",
        "m-l-xl",
        "xxl",
        "3xl",
        "4xl",
        "on",
        "ön",
        "arka",
        "kol",
        "yaka",
        "manset",
        "manşet",
    }
    for _ in range(max(1, int(max_up) + 1)):
        if _is_root(cur):
            break
        s = str(cur)
        if not s or s in out:
            break
        out.append(s)
        name = re.sub(r"[^a-z0-9]+", "", cur.name.lower())
        tokenish = any(
            str(t).replace("-", "") in name or str(t) in cur.name.lower() for t in size_part
        )
        if tokenish or re.fullmatch(
            r"(m-?l-?xl|xxl.?3xl.?4xl|s-?m-?l)", cur.name.lower().replace(" ", "")
        ):
            cur = cur.parent
            continue
        # one extra ancestor for job folder when we started in a size subfolder
        if len(out) == 1:
            cur = cur.parent
            continue
        break
    # Drop any accidental roots
    return [d for d in out if d and not _is_root(Path(d))]

    size_part = _SIZE_TOKENS | _GARMENT_PART_TOKENS | {
        "mlxl", "m-l-xl", "xxl", "3xl", "4xl", "on", "ön", "arka", "kol", "yaka", "manset", "manşet",
    }
    for _ in range(max(1, int(max_up) + 1)):
        s = str(cur)
        if not s or s in out:
            break
        out.append(s)
        name = re.sub(r"[^a-z0-9]+", "", cur.name.lower())
        # walk up when folder looks like a size/part bucket
        tokenish = any(t.replace("-", "") in name or t in cur.name.lower() for t in size_part)
        if tokenish or re.fullmatch(r"(m-?l-?xl|xxl.?3xl.?4xl|s-?m-?l)", cur.name.lower().replace(" ", "")):
            cur = cur.parent
            continue
        # one extra ancestor for job folder (gömlek1) when we started in size subfolder
        if len(out) == 1:
            cur = cur.parent
            continue
        break
    return out


def design_scope_affinity(path_a: str, path_b: str) -> bool:
    """True when both paths share a design-scope directory (sibling subfolders OK)."""
    if path_affinity(path_a, path_b):
        return True
    da = {d.lower() for d in design_scope_dirs(path_a)}
    db = {d.lower() for d in design_scope_dirs(path_b)}
    if not da or not db:
        return False
    return bool(da & db)


def _leaf(text: str) -> str:
    t = str(text or "").strip()
    if "/" in t:
        t = t.rsplit("/", 1)[-1]
    return _norm(t)


# TR/EN display → taxonomy leaf for distinct-concept veto (similarity cannot override).
_LEAF_ALIASES: dict[str, str] = {
    "kaplan": "tiger",
    "tiger": "tiger",
    "leopar": "leopard",
    "leopard": "leopard",
    "leo": "leopard",
    "gul": "rose",
    "gül": "rose",
    "rose": "rose",
    "cicek": "floral",
    "çiçek": "floral",
    "flower": "floral",
    "floral": "floral",
}


def _tax_leaf(label: str) -> str:
    leaf = _leaf(label)
    if not leaf or leaf in _UNKNOWN:
        return ""
    return _LEAF_ALIASES.get(leaf, leaf)


def _labels_conflict(a: str, b: str) -> bool:
    la, lb = _tax_leaf(a), _tax_leaf(b)
    if not la or not lb:
        return False
    if may_merge_as_same_family(la, lb):
        return False
    try:
        from core.canonical_correction import are_distinct_concepts

        if are_distinct_concepts(la, lb):
            return True
    except Exception:
        pass
    # Alias-normalized siblings still conflict when keys differ
    if la != lb and la in _LEAF_ALIASES.values() and lb in _LEAF_ALIASES.values():
        return True
    return False


def _domain_bucket(member: FamilyMember) -> str:
    blob = " ".join(
        [
            member.category,
            member.pattern_family,
            member.guess,
            member.path,
            member.filename,
        ]
    ).lower()
    animal = any(h in blob for h in _ANIMAL_HINTS)
    textile = any(h in blob for h in _TEXTILE_HINTS)
    if animal and not textile:
        return "animal"
    if textile and not animal:
        return "textile"
    # Explicit animal_print_type vs fabric category leaves
    cat = _leaf(member.category)
    if cat in ("animal", "hayvan", "wildlife"):
        return "animal"
    if cat in ("textile", "kumas", "kumaş", "fabric", "dokuma"):
        return "textile"
    return ""


def _cosine(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    try:
        import numpy as np

        va = np.asarray(a, dtype=np.float32).ravel()
        vb = np.asarray(b, dtype=np.float32).ravel()
        if va.size == 0 or vb.size != va.size:
            return 0.0
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    except Exception:
        return 0.0



def garment_part_token(filename: str) -> str:
    """Return known garment-part token if present (metadata; not a merge key alone)."""
    stem = Path(str(filename or "")).stem
    stem = stem.replace("-", "_").replace(" ", "_")
    parts = [p.lower() for p in re.split(r"[_\.]+", stem) if p]
    for p in parts:
        if p in _GARMENT_PART_TOKENS:
            return p
        # multi-word like "ay_parçası" already split
    joined = "_".join(parts)
    for tok in ("ay_parçası", "ay_parcasi", "ay_parçasi", "kol_manşet", "kol_manset"):
        if tok in joined:
            return tok
    return ""


def decode_clip_blob(raw: Any) -> Any:
    """Decode stored CLIP bytes/array → float32 vector. No new embedding."""
    if raw is None:
        return None
    try:
        import numpy as np

        if isinstance(raw, np.ndarray):
            vec = np.asarray(raw, dtype=np.float32).ravel()
        elif isinstance(raw, (bytes, bytearray, memoryview)):
            b = bytes(raw)
            if len(b) < 16 or (len(b) % 4) != 0:
                return None
            vec = np.frombuffer(b, dtype=np.float32).copy()
        elif isinstance(raw, (list, tuple)):
            vec = np.asarray(raw, dtype=np.float32).ravel()
        else:
            return None
        if vec.size == 0 or not np.isfinite(vec).all():
            return None
        return vec
    except Exception:
        return None


def _feature_db_paths(db_path: str) -> list[str]:
    """Prefer live patterns DB for features; also try write-path sibling if distinct."""
    paths: list[str] = []
    raw = str(db_path or "").strip()
    if raw:
        paths.append(raw)
    try:
        alt = str(_write_path(raw)) if raw else ""
    except Exception:
        alt = ""
    if alt and alt not in paths:
        paths.append(alt)
    # Common layout: patterns.db next to search_memory.db
    for base in list(paths):
        try:
            parent = Path(base).parent
            for name in ("patterns.db", "pattern_search.db", "vezir.db"):
                cand = str(parent / name)
                if cand not in paths and Path(cand).is_file():
                    paths.append(cand)
        except Exception:
            pass
    return paths


def load_clip_vectors_for_ids(db_path: str, file_ids: Iterable[int]) -> dict[int, Any]:
    """Read-only hydrate of existing CLIP blobs from features (+ concept_examples).

    Uses the live DB path (not search_memory redirect) so features.clip_embedding
    is visible. No new embedding / Indexer / DINO.
    """
    ids = sorted({int(x) for x in file_ids if int(x) > 0})
    if not ids or not db_path:
        return {}
    out: dict[int, Any] = {}
    ph = ",".join("?" * len(ids))
    for db_file in _feature_db_paths(db_path):
        try:
            c = sqlite3.connect(str(db_file), timeout=10)
            c.row_factory = sqlite3.Row
        except Exception:
            continue
        try:
            try:
                rows = c.execute(
                    f"""SELECT file_id, clip_embedding FROM features
                        WHERE file_id IN ({ph})
                          AND clip_embedding IS NOT NULL
                          AND length(clip_embedding) >= 16""",
                    ids,
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for r in rows:
                fid = int(r["file_id"])
                if fid in out:
                    continue
                vec = decode_clip_blob(r["clip_embedding"])
                if vec is not None:
                    out[fid] = vec
            missing = [i for i in ids if i not in out]
            if missing:
                ph2 = ",".join("?" * len(missing))
                for sql in (
                    f"""SELECT file_id, embedding FROM concept_examples
                        WHERE file_id IN ({ph2})
                          AND role IN ('positive','pos')
                          AND IFNULL(source,'user') NOT IN ('candidate')
                          AND embedding IS NOT NULL AND length(embedding) >= 16""",
                    f"""SELECT file_id, embedding FROM concept_examples
                        WHERE file_id IN ({ph2})
                          AND embedding IS NOT NULL AND length(embedding) >= 16""",
                ):
                    try:
                        erows = c.execute(sql, missing).fetchall()
                    except sqlite3.OperationalError:
                        erows = []
                    for r in erows:
                        fid = int(r["file_id"] or 0)
                        if fid <= 0 or fid in out:
                            continue
                        vec = decode_clip_blob(r["embedding"])
                        if vec is not None:
                            out[fid] = vec
                    missing = [i for i in missing if i not in out]
                    if not missing:
                        break
        finally:
            c.close()
        if len(out) >= len(ids):
            break
    return out



def load_dino_vectors_for_ids(db_path: str, file_ids: Iterable[int]) -> dict[int, Any]:
    """Read-only hydrate of existing DINO blobs from features. No new embedding."""
    out: dict[int, Any] = {}
    ids = [int(x) for x in file_ids if int(x or 0) > 0]
    if not ids or not str(db_path or "").strip():
        return out
    ph = ",".join("?" * len(ids))
    for path in _feature_db_paths(db_path):
        try:
            import sqlite3

            c = sqlite3.connect(str(path), timeout=10)
            try:
                rows = c.execute(
                    f"SELECT file_id, dino_embedding FROM features WHERE file_id IN ({ph})",
                    ids,
                ).fetchall()
            finally:
                c.close()
        except Exception:
            continue
        for fid, blob in rows:
            vec = decode_clip_blob(blob)
            if vec is not None:
                out[int(fid)] = vec
        if len(out) >= len(ids):
            break
    return out


def hydrate_member_clips(
    members: Sequence[FamilyMember],
    *,
    db_path: str = "",
) -> None:
    """Fill member.clip from card bytes / DB features / taught embeddings (in-place)."""
    if not members:
        return
    for m in members:
        if m.clip is not None:
            continue
        card = m.card
        if card is None:
            continue
        raw = getattr(card, "clip_embedding", None)
        vec = decode_clip_blob(raw)
        if vec is not None:
            m.clip = vec
    need = [int(m.file_id) for m in members if m.clip is None and int(m.file_id) > 0]
    if not need or not db_path:
        return
    loaded = load_clip_vectors_for_ids(db_path, need)
    for m in members:
        if m.clip is None:
            vec = loaded.get(int(m.file_id))
            if vec is not None:
                m.clip = vec
    need_d = [int(m.file_id) for m in members if getattr(m, "dino", None) is None and int(m.file_id) > 0]
    if need_d and db_path:
        loaded_d = load_dino_vectors_for_ids(db_path, need_d)
        for m in members:
            if getattr(m, "dino", None) is None and int(m.file_id) in loaded_d:
                m.dino = loaded_d[int(m.file_id)]


def pair_family_signals(
    a: FamilyMember,
    b: FamilyMember,
) -> tuple[list[str], float]:
    """Return (signals, confidence). Empty signals ⇒ do not merge."""
    if int(a.file_id) == int(b.file_id):
        return [], 0.0

    # Hard vetoes — similarity cannot override
    for left, right in (
        (a.guess, b.guess),
        (a.pattern_family, b.pattern_family),
        (a.category, b.category),
    ):
        if _labels_conflict(left, right):
            return [], 0.0
    da, db = _domain_bucket(a), _domain_bucket(b)
    if da and db and da != db:
        return [], 0.0

    signals: list[str] = []
    stem_a = normalize_variant_stem(a.filename)
    stem_b = normalize_variant_stem(b.filename)
    if stem_a and stem_b and stem_a == stem_b and len(stem_a) >= 3:
        signals.append("stem")

    if path_affinity(a.path, b.path) or design_scope_affinity(a.path, b.path):
        signals.append("path")

    pf_a = _leaf(a.pattern_family) or _leaf(a.guess)
    pf_b = _leaf(b.pattern_family) or _leaf(b.guess)
    if (
        pf_a
        and pf_b
        and pf_a not in _UNKNOWN
        and pf_b not in _UNKNOWN
        and may_merge_as_same_family(pf_a, pf_b)
    ):
        if pf_a not in _GENERIC_FAMILIES:
            signals.append("family")
        elif "stem" in signals or "path" in signals:
            signals.append("family")

    # Color / size variants under same concept (from snapshot) — soft
    try:
        sa = family_snapshot(
            {"filename": a.filename, "pattern_family": a.pattern_family, "path": a.path},
            {"pattern_family": a.pattern_family, "color_family": a.color_family},
        )
        sb = family_snapshot(
            {"filename": b.filename, "pattern_family": b.pattern_family, "path": b.path},
            {"pattern_family": b.pattern_family, "color_family": b.color_family},
        )
        fa = _leaf(sa.get("family") or "")
        fb = _leaf(sb.get("family") or "")
        if fa and fb and may_merge_as_same_family(fa, fb) and fa not in _UNKNOWN:
            if fa not in _GENERIC_FAMILIES or "stem" in signals:
                if "family" not in signals:
                    signals.append("variant")
    except Exception:
        pass

    clip = _cosine(a.clip, b.clip)
    if clip >= CLIP_ALIGN:
        signals.append("clip")
    dino = _cosine(getattr(a, "dino", None), getattr(b, "dino", None))
    if dino >= CLIP_ALIGN:
        signals.append("dino")

    hard = [s for s in signals if s not in ("clip", "dino")]
    # Consensus: never visual-alone; never filename/stem-alone; never path-alone;
    # allow PATH+CLIP/DINO or FAMILY+CLIP/DINO (additive). Do not strip garment parts
    # for stem equality — size tokens remain helper-only metadata.
    if not hard:
        return [], 0.0
    visual = "clip" in signals or "dino" in signals
    visual_assisted = visual and any(s in _CLIP_CONSENSUS_HARD for s in hard)
    if len(hard) < MIN_HARD_SIGNALS and not visual_assisted:
        return [], 0.0
    # Without visual: allow stem+path or stem+named-family size variants only.
    # Path+family (e.g. generic Marka) without stem/visual is NOT enough.
    if not visual:
        if "stem" in hard and ("path" in hard or "family" in hard):
            pass
        else:
            return [], 0.0
    if hard == ["variant"]:
        return [], 0.0
    if hard == ["family"] and "clip" not in signals:
        return [], 0.0

    conf = 0.35 + 0.18 * len(hard)
    if "stem" in hard:
        conf += 0.12
    if "clip" in signals:
        conf += 0.08 * min(1.0, (clip - CLIP_ALIGN) / 0.1 + 1.0)
    if "dino" in signals:
        conf += 0.08 * min(1.0, (dino - CLIP_ALIGN) / 0.1 + 1.0)
    conf = max(0.0, min(0.97, conf))
    return signals, conf


def _suggest_label(members: Sequence[FamilyMember]) -> str:
    stems = [normalize_variant_stem(m.filename) for m in members]
    stems = [s for s in stems if s]
    if stems:
        # most common stem, title-ish
        best = max(set(stems), key=stems.count)
        pretty = best.replace("_", " ").strip()
        if pretty:
            return pretty.title()
    for m in members:
        g = _leaf(m.guess) or _leaf(m.pattern_family)
        if g and g not in _UNKNOWN:
            return g
    return "Görsel aile"


def build_family_candidates(
    members: Sequence[FamilyMember],
    *,
    db_path: str = "",
    min_confidence: float = MIN_FAMILY_CONF,
) -> list[FamilyCandidate]:
    """Multi-signal union of members. Low confidence → no candidate (stay undecided)."""
    items = [m for m in members if int(m.file_id) > 0]
    if len(items) < MIN_MEMBERS:
        return []
    if db_path and any(m.clip is None for m in items):
        hydrate_member_clips(items, db_path=db_path)
    banned = rejected_family_fingerprints(db_path) if db_path else set()
    parent: dict[int, int] = {int(m.file_id): int(m.file_id) for m in items}
    by_id = {int(m.file_id): m for m in items}
    edge_conf: dict[tuple[int, int], float] = {}
    edge_signals: dict[tuple[int, int], list[str]] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            sigs, conf = pair_family_signals(a, b)
            if not sigs or conf < min_confidence:
                continue
            ia, ib = int(a.file_id), int(b.file_id)
            union(ia, ib)
            key = (min(ia, ib), max(ia, ib))
            edge_conf[key] = conf
            edge_signals[key] = sigs

    groups: dict[int, list[FamilyMember]] = {}
    for m in items:
        groups.setdefault(find(int(m.file_id)), []).append(m)

    out: list[FamilyCandidate] = []
    for group in groups.values():
        if len(group) < MIN_MEMBERS:
            continue
        ids = sorted(int(m.file_id) for m in group)
        fp = family_fingerprint(ids)
        if fp in banned:
            continue
        # Mean edge confidence among connected pairs in group
        pair_vals: list[float] = []
        sig_set: list[str] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = (ids[i], ids[j])
                if key in edge_conf:
                    pair_vals.append(edge_conf[key])
                    for s in edge_signals.get(key) or []:
                        if s not in sig_set:
                            sig_set.append(s)
        if not pair_vals:
            continue
        conf = sum(pair_vals) / len(pair_vals)
        if conf < min_confidence:
            continue
        label = _suggest_label(group)
        pool = group[0].pool
        reason = (
            f"Aynı görsel aile adayı «{label}» · {len(ids)} dosya · "
            f"sinyaller: {', '.join(sig_set)} · onaylanmadan birleştirilmez"
        )
        out.append(
            FamilyCandidate(
                family_id=fp,
                label=label,
                member_ids=ids,
                confidence=float(conf),
                signals=list(sig_set),
                reason=reason,
                sample_filenames=[m.filename for m in group[:4]],
                pool=pool,
            )
        )
        if len(out) >= MAX_CANDIDATES:
            break
    out.sort(key=lambda c: (-c.confidence, -len(c.member_ids), c.label))
    return out


def member_from_card(card: Any) -> FamilyMember:
    raw = getattr(card, "clip_embedding", None) if card is not None else None
    return FamilyMember(
        file_id=int(getattr(card, "file_id", 0) or 0),
        filename=str(getattr(card, "filename", "") or ""),
        path=str(getattr(card, "path", "") or ""),
        pool=str(getattr(card, "pool", "") or ""),
        guess=str(getattr(card, "guess", "") or ""),
        pattern_family=str(getattr(card, "pattern_family", "") or ""),
        category=str(getattr(card, "category", "") or ""),
        color_family=str(getattr(card, "color_family", "") or ""),
        confidence=float(getattr(card, "confidence", 0) or 0),
        clip=decode_clip_blob(raw),
        card=card,
    )


def collapse_pools_with_families(
    pools: dict[str, list],
    *,
    db_path: str = "",
    min_confidence: float = MIN_FAMILY_CONF,
) -> dict[str, list]:
    """Attach family candidates: one representative card per group; no duplicate members."""
    if not pools:
        return pools
    flat: list[FamilyMember] = []
    for _pool, cards in pools.items():
        for card in cards or []:
            m = member_from_card(card)
            if m.file_id > 0:
                flat.append(m)
    # Read-only CLIP hydrate from DB/blob — no new embedding / Indexer / DINO.
    hydrate_member_clips(flat, db_path=db_path)
    candidates = build_family_candidates(
        flat, db_path=db_path, min_confidence=min_confidence
    )
    if not candidates:
        return pools

    id_to_cand: dict[int, FamilyCandidate] = {}
    for cand in candidates:
        for mid in cand.member_ids:
            # Prefer larger / higher-conf group if overlap
            prev = id_to_cand.get(mid)
            if prev is None or (
                len(cand.member_ids),
                cand.confidence,
            ) > (len(prev.member_ids), prev.confidence):
                id_to_cand[mid] = cand

    consumed: set[int] = set()
    shown_fp: set[str] = set()
    new_pools: dict[str, list] = {k: [] for k in pools}
    # Preserve pool order keys
    for pool_name, cards in pools.items():
        dest = new_pools.setdefault(pool_name, [])
        for card in cards or []:
            fid = int(getattr(card, "file_id", 0) or 0)
            if fid <= 0:
                continue
            if fid in consumed:
                continue
            cand = id_to_cand.get(fid)
            if cand is None or len(cand.member_ids) < MIN_MEMBERS:
                dest.append(card)
                continue
            if cand.family_id in shown_fp:
                consumed.add(fid)
                continue
            shown_fp.add(cand.family_id)
            for mid in cand.member_ids:
                consumed.add(mid)
            # Representative = first member still in this pool's cards, else card
            rep = card
            members_map = {
                int(getattr(c, "file_id", 0) or 0): c
                for plist in pools.values()
                for c in (plist or [])
            }
            for mid in cand.member_ids:
                if mid in members_map:
                    rep = members_map[mid]
                    break
            try:
                rep.cluster_size = len(cand.member_ids)
                rep.member_ids = list(cand.member_ids)
                rep.family_id = cand.family_id
                rep.suggested = cand.label
                rep.guess = cand.label
                rep.confidence = max(float(getattr(rep, "confidence", 0) or 0), cand.confidence)
                rep.reason = cand.reason
            except Exception:
                pass
            dest.append(rep)
    return new_pools


def expand_member_ids(card: Any, selected_ids: Iterable[int] | None = None) -> list[int]:
    """Selected file ids expanded by family member_ids on the card(s)."""
    out: list[int] = []
    seen: set[int] = set()
    seed = list(selected_ids) if selected_ids is not None else []
    if not seed and card is not None:
        seed = [int(getattr(card, "file_id", 0) or 0)]
    members = list(getattr(card, "member_ids", None) or []) if card is not None else []
    if members and any(int(s) in {int(m) for m in members} for s in seed):
        for mid in members:
            i = int(mid)
            if i > 0 and i not in seen:
                seen.add(i)
                out.append(i)
        return out
    for s in seed:
        i = int(s or 0)
        if i > 0 and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def reject_and_split_family(db_path: str, card: Any) -> list[int]:
    """Record reject fingerprint; return member ids (still teachable individually)."""
    members = list(getattr(card, "member_ids", None) or [])
    if len(members) < MIN_MEMBERS:
        fid = int(getattr(card, "file_id", 0) or 0)
        return [fid] if fid > 0 else []
    label = str(getattr(card, "suggested", "") or getattr(card, "guess", "") or "")
    record_rejected_family(db_path, members, label=label)
    return [int(x) for x in members if int(x) > 0]
