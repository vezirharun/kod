"""Concept query normalization — case / TR / typo / alias / translation.

Search & learning layer only. Does not write index / DNA / CLIP.
Typo correction is QUERY CORRECTION — never creates user learning by itself.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable

from core.textile_terms import TERM_SYNONYMS, normalize_turkish

# Inverted leaf synonym index — built once; leaf_translation_keys must stay O(1).
_LEAF_SYN_LOCK = threading.RLock()
_LEAF_SYN_INDEX: dict[str, frozenset[str]] | None = None


def invalidate_leaf_translation_cache() -> None:
    """Drop synonym→keys index (thread-safe). Call after taxonomy/synonym edits."""
    global _LEAF_SYN_INDEX
    with _LEAF_SYN_LOCK:
        _LEAF_SYN_INDEX = None


def _build_leaf_synonym_index() -> dict[str, frozenset[str]]:
    """Map each leaf match-key → full TR↔EN synonym key set (no family bags)."""
    accum: dict[str, set[str]] = {}
    for src, alts in TERM_SYNONYMS.items():
        sk = concept_match_key(src)
        if not sk or sk in _FAMILY_SKIP:
            continue
        group: set[str] = {sk}
        for a in alts:
            ak = concept_match_key(a)
            if ak and ak not in _FAMILY_SKIP:
                group.add(ak)
        for k in group:
            accum.setdefault(k, set()).update(group)
    return {k: frozenset(v) for k, v in accum.items()}


def _leaf_synonym_index() -> dict[str, frozenset[str]]:
    global _LEAF_SYN_INDEX
    with _LEAF_SYN_LOCK:
        if _LEAF_SYN_INDEX is None:
            _LEAF_SYN_INDEX = _build_leaf_synonym_index()
        return _LEAF_SYN_INDEX

# Family / parent labels — never treat as leaf translation targets for merge/match.
# Note: flower/çiçek are LEAF language variants of each other — not skipped.
# "floral" stays a parent/family hub and must not absorb Rose/Daisy leaves.
_FAMILY_SKIP = frozenset(
    {
        "animal",
        "animal print",
        "hayvan",
        "hayvan deseni",
        "floral",
        "desen",
        "pattern",
        "print",
        "textile",
        "kumas",
        "kumaş",
        "logo",
        "monogram",
        "symbol",
        "symbol pattern",
    }
)

# Never auto-merge these leaf pairs (related/parent-child, not translation).
_TRANSLATION_MERGE_BLOCK = frozenset(
    {
        frozenset({"snake", "snake skin"}),
        frozenset({"yilan", "snake skin"}),
        frozenset({"snake", "yilan dokusu"}),
        frozenset({"yilan", "snake skin"}),
    }
)

_PUNCT_STRIP = re.compile(r"[!?,.;:]+")
_SPACE = re.compile(r"\s+")

# Confidence tiers for typo auto-apply
TYPO_AUTO = 0.88
TYPO_SOFT = 0.78


@dataclass
class NormalizedQuery:
    raw: str = ""
    cleaned: str = ""
    casefolded: str = ""
    match_key: str = ""
    corrected: str = ""
    corrected_key: str = ""
    was_typo_corrected: bool = False
    typo_confidence: float = 1.0
    typo_tier: str = "none"  # auto | soft | none
    concept_hints: list[str] = field(default_factory=list)
    translation_keys: list[str] = field(default_factory=list)
    kind_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def effective(self) -> str:
        """Best text for downstream concept matching."""
        if self.was_typo_corrected and self.typo_tier == "auto" and self.corrected:
            return self.corrected
        return self.cleaned or self.raw


def turkish_casefold(text: str) -> str:
    """Locale-aware-ish Turkish fold: İ→i, I→ı before lower."""
    if not text:
        return ""
    out: list[str] = []
    for ch in str(text):
        if ch == "İ":
            out.append("i")
        elif ch == "I":
            out.append("ı")
        elif ch == "Ş":
            out.append("ş")
        elif ch == "Ğ":
            out.append("ğ")
        elif ch == "Ü":
            out.append("ü")
        elif ch == "Ö":
            out.append("ö")
        elif ch == "Ç":
            out.append("ç")
        else:
            out.append(ch)
    return "".join(out).lower()


# Deterministic Turkish alphabet order for UI lists (case-insensitive).
# A B C Ç D E F G Ğ H I İ J K L M N O Ö P Q R S Ş T U Ü V W X Y Z
_TR_LETTER_RANK: dict[str, int] = {
    "a": 0,
    "b": 1,
    "c": 2,
    "ç": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "ğ": 8,
    "h": 9,
    "ı": 10,
    "i": 11,
    "j": 12,
    "k": 13,
    "l": 14,
    "m": 15,
    "n": 16,
    "o": 17,
    "ö": 18,
    "p": 19,
    "q": 20,
    "r": 21,
    "s": 22,
    "ş": 23,
    "t": 24,
    "u": 25,
    "ü": 26,
    "v": 27,
    "w": 28,
    "x": 29,
    "y": 30,
    "z": 31,
}


def turkish_sort_key(text: str) -> tuple:
    """Sort key for displayed labels: Turkish letter order, case-insensitive."""
    folded = turkish_casefold(str(text or "").strip())
    parts: list[tuple[int, int]] = []
    for ch in folded:
        rank = _TR_LETTER_RANK.get(ch)
        if rank is not None:
            parts.append((0, rank))
        elif ch.isdigit():
            parts.append((1, ord(ch)))
        elif ch.isspace():
            parts.append((-1, 0))
        else:
            parts.append((2, ord(ch)))
    return tuple(parts)


def sorted_turkish(items: Iterable[str]) -> list[str]:
    """Stable TR-alphabetical unique display list (first spelling kept)."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        s = str(raw or "").strip()
        if not s:
            continue
        fold = turkish_casefold(s)
        if fold in seen:
            continue
        seen.add(fold)
        out.append(s)
    out.sort(key=turkish_sort_key)
    return out


def clean_query_text(text: str) -> str:
    """Strip trailing punctuation / collapse spaces; keep original meaning tokens."""
    t = str(text or "").strip()
    if not t:
        return ""
    t = t.replace("–", "-").replace("—", "-")
    t = _PUNCT_STRIP.sub(" ", t)
    t = t.replace("_", " ")
    # hyphenated motif: tiger-print → tiger print
    t = t.replace("-", " ")
    t = _SPACE.sub(" ", t).strip()
    return t


def concept_match_key(text: str) -> str:
    """Stable match key: TR casefold + ASCII fold + optional print/desen suffix strip."""
    cleaned = clean_query_text(text)
    folded = turkish_casefold(cleaned)
    key = normalize_turkish(folded)
    for suf in (
        " print",
        " desen",
        " deseni",
        " pattern",
        " baski",
        " baskı",
    ):
        sk = normalize_turkish(suf)
        if key.endswith(sk) and len(key) > len(sk) + 2:
            key = key[: -len(sk)].strip()
    return _SPACE.sub(" ", key).strip()


def leaf_translation_keys(text: str) -> set[str]:
    """TR↔EN leaf synonyms only.

    Expands only from *leaf* TERM_SYNONYMS keys. Family bags like
    ``animal print`` (leopard+zebra+snake) are never expanded — that would
    falsely link siblings.

    Uses a module-level inverted synonym index (built once) — never rescans
    TERM_SYNONYMS on the autocomplete hot path.
    """
    root = concept_match_key(text)
    if not root:
        return set()
    # Parent/family labels are not leaf translation hubs.
    if root in _FAMILY_SKIP:
        return {root}
    group = _leaf_synonym_index().get(root)
    if group:
        return set(group)
    return {root}


def normalize_concept_query(query: str) -> NormalizedQuery:
    """Full normalization pipeline for concept resolution (not a new DNA engine)."""
    raw = str(query or "")
    cleaned = clean_query_text(raw)
    casefolded = turkish_casefold(cleaned)
    match_key = concept_match_key(cleaned)
    nq = NormalizedQuery(
        raw=raw,
        cleaned=cleaned,
        casefolded=casefolded,
        match_key=match_key,
        corrected=cleaned,
        corrected_key=match_key,
        translation_keys=sorted(leaf_translation_keys(cleaned)),
    )
    # Typo correction — QUERY CORRECTION only
    try:
        from core.fuzzy_correct import correct_query

        cr = correct_query(cleaned)
        if cr.was_corrected and cr.corrected:
            conf = float(cr.confidence or 0)
            nq.corrected = cr.corrected
            nq.corrected_key = concept_match_key(cr.corrected)
            nq.was_typo_corrected = True
            nq.typo_confidence = conf
            if conf >= TYPO_AUTO:
                nq.typo_tier = "auto"
                nq.kind_notes.append("typo_auto")
            elif conf >= TYPO_SOFT:
                nq.typo_tier = "soft"
                nq.kind_notes.append("typo_soft")
            else:
                nq.typo_tier = "none"
                nq.was_typo_corrected = False
                nq.kind_notes.append("typo_rejected_low_conf")
            # Refresh translations from corrected form when auto
            if nq.typo_tier == "auto":
                nq.translation_keys = sorted(
                    set(nq.translation_keys) | leaf_translation_keys(nq.corrected)
                )
    except Exception:
        pass
    return nq


def _relation_from_normalized(
    nq: NormalizedQuery,
    names: list[str],
    parent: str = "",
) -> str | None:
    """Match normalized query keys against concept names (no attribute logic)."""
    if not names:
        return None
    q_keys = {nq.match_key, nq.corrected_key} | set(nq.translation_keys)
    q_keys = {k for k in q_keys if k}

    # exact / case / TR-char
    if nq.match_key in names or nq.corrected_key in names:
        if nq.was_typo_corrected and nq.typo_tier == "auto" and nq.match_key not in names:
            return "typo"
        return "exact"

    # translation / alias via leaf synonym keys
    name_set = set(names)
    concept_keys = set(name_set)
    for n in list(name_set):
        concept_keys |= leaf_translation_keys(n)
    overlap = q_keys & concept_keys
    if overlap:
        if nq.match_key not in name_set and nq.corrected_key not in name_set:
            return "translation"
        return "alias"

    # typo against concept names (bounded)
    probe = nq.corrected_key or nq.match_key
    if probe and len(probe) >= 4:
        best = 0.0
        for n in names:
            if abs(len(probe) - len(n)) > max(2, len(n) // 3):
                continue
            best = max(best, SequenceMatcher(None, probe, n).ratio())
        if best >= TYPO_AUTO:
            return "typo"
        if best >= TYPO_SOFT:
            return None

    pk = concept_match_key(parent)
    if pk and (nq.match_key == pk or nq.match_key in leaf_translation_keys(pk)):
        return "child"
    if pk and pk in q_keys and nq.match_key == pk:
        return "child"

    return None


def relation_to_concept(
    query: str,
    canonical: str,
    aliases: Iterable[str] | None = None,
    parent: str = "",
) -> str | None:
    """exact | typo | alias | translation | child | parent | None.

    Never returns a sibling collapse (Tiger↛Leopard).

    Stage 2C: attribute tokens (küçük/yoğun/siyah/…) must not demote an
    otherwise exact/translation leaf match. Concept identity is resolved on
    the attribute-stripped core (motif), then attributes only re-rank.
    """
    names = []
    for raw in (canonical, *(aliases or ())):
        k = concept_match_key(str(raw or ""))
        if k:
            names.append(k)
    if not names:
        return None

    nq = normalize_concept_query(query)
    hit = _relation_from_normalized(nq, names, parent)
    if hit:
        return hit

    # Compound query: strip attributes → match concept core only.
    # Only when real attributes present — never collapse "snake skin" → "snake".
    try:
        from core.query_attribute_intel import (
            concept_core_text,
            extract_query_attributes,
        )

        attrs = extract_query_attributes(query)
        core = concept_core_text(query) if attrs.has_attributes else ""
    except Exception:
        core = ""
    core_key = concept_match_key(core) if core else ""
    if core_key and core_key != nq.match_key:
        core_nq = normalize_concept_query(core)
        core_hit = _relation_from_normalized(core_nq, names, parent)
        # Attributes must not invent parent/child from a stripped motif alone
        # when the full query already failed — only identity-preserving tiers.
        if core_hit in ("exact", "typo", "translation", "alias"):
            return core_hit

    return None


# ---------------------------------------------------------------------------
# Migration (search_memory / concept_registry only)
# ---------------------------------------------------------------------------


def _parse_aliases(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except Exception:
            return []
    return []


def audit_concept_registry(db_path: str) -> dict[str, Any]:
    """Pre-migration report (read-only)."""
    from core.concept_registry import _write_path, concepts

    path = str(_write_path(db_path))
    rows = concepts(path)
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(concept_match_key(r.get("canonical") or ""), []).append(r)
    dups = {k: v for k, v in by_key.items() if k and len(v) > 1}
    alias_map: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        cid = int(r["id"])
        can = str(r.get("canonical") or "")
        for a in [can, *_parse_aliases(r.get("aliases"))]:
            k = concept_match_key(a)
            if not k:
                continue
            alias_map.setdefault(k, []).append((cid, can))
    ambiguous = {
        k: list(dict.fromkeys(v))
        for k, v in alias_map.items()
        if len({x[0] for x in v}) > 1
    }
    # translation candidates (leaf only)
    pairs: list[tuple[str, str, int, int]] = []
    seen: set[tuple[int, int]] = set()
    id_by_key = {
        concept_match_key(r["canonical"]): int(r["id"])
        for r in rows
        if concept_match_key(r.get("canonical") or "")
    }
    can_by_id = {int(r["id"]): str(r["canonical"]) for r in rows}
    for r in rows:
        k = concept_match_key(r["canonical"])
        for t in leaf_translation_keys(k):
            other = id_by_key.get(t)
            if not other or other == int(r["id"]):
                continue
            a, b = sorted((int(r["id"]), other))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((can_by_id[a], can_by_id[b], a, b))
    return {
        "db_path": path,
        "total_concepts": len(rows),
        "duplicate_canonical_keys": len(dups),
        "duplicates": {
            k: [(int(x["id"]), x["canonical"], x.get("parent")) for x in v]
            for k, v in dups.items()
        },
        "ambiguous_alias_keys": len(ambiguous),
        "ambiguous": {k: v for k, v in list(ambiguous.items())[:50]},
        "possible_leaf_translations": pairs,
        "conflicts_preserved": [],
    }


def _example_count(c: sqlite3.Connection, concept_id: int) -> int:
    row = c.execute(
        "SELECT COUNT(*) n FROM concept_examples WHERE concept_id=? AND role='positive'",
        (int(concept_id),),
    ).fetchone()
    return int(row[0] if not isinstance(row, sqlite3.Row) else row["n"])


def _merge_concept_into(
    c: sqlite3.Connection,
    *,
    keep_id: int,
    drop_id: int,
    extra_aliases: Iterable[str] = (),
) -> dict[str, int]:
    """Move examples/events/aliases from drop → keep. keep canonical unchanged."""
    stats = {"examples_moved": 0, "events_retargeted": 0, "aliases_added": 0}
    keep = c.execute("SELECT * FROM concept_registry WHERE id=?", (keep_id,)).fetchone()
    drop = c.execute("SELECT * FROM concept_registry WHERE id=?", (drop_id,)).fetchone()
    if not keep or not drop:
        return stats
    aliases = _parse_aliases(keep["aliases"])
    before = len(aliases)
    for a in (
        drop["canonical"],
        *_parse_aliases(drop["aliases"]),
        *extra_aliases,
    ):
        s = str(a or "").strip()
        if s and s not in aliases:
            aliases.append(s)
    stats["aliases_added"] = max(0, len(aliases) - before)
    c.execute(
        "UPDATE concept_registry SET aliases=?, updated_at=datetime('now') WHERE id=?",
        (json.dumps(aliases, ensure_ascii=False), keep_id),
    )
    # Move examples — respect UNIQUE(concept_id,file_id,role,file_path)
    rows = c.execute(
        "SELECT * FROM concept_examples WHERE concept_id=?", (drop_id,)
    ).fetchall()
    for r in rows:
        exists = c.execute(
            """SELECT id, source FROM concept_examples
               WHERE concept_id=? AND file_id=? AND role=? AND file_path=?""",
            (keep_id, r["file_id"], r["role"], r["file_path"]),
        ).fetchone()
        if exists:
            # Prefer user source
            if str(r["source"] or "") == "user" and str(exists["source"] or "") != "user":
                c.execute(
                    "UPDATE concept_examples SET source='user' WHERE id=?",
                    (exists["id"],),
                )
            c.execute("DELETE FROM concept_examples WHERE id=?", (r["id"],))
        else:
            c.execute(
                "UPDATE concept_examples SET concept_id=? WHERE id=?",
                (keep_id, r["id"]),
            )
            stats["examples_moved"] += 1
    # learning_events
    try:
        ev = c.execute(
            "SELECT id FROM learning_events WHERE concept_id=?", (drop_id,)
        ).fetchall()
        for e in ev:
            c.execute(
                "UPDATE learning_events SET concept_id=? WHERE id=?",
                (keep_id, e["id"]),
            )
            stats["events_retargeted"] += 1
        ev2 = c.execute(
            "SELECT id FROM learning_events WHERE rival_concept_id=?", (drop_id,)
        ).fetchall()
        for e in ev2:
            c.execute(
                "UPDATE learning_events SET rival_concept_id=? WHERE id=?",
                (keep_id, e["id"]),
            )
            stats["events_retargeted"] += 1
    except sqlite3.OperationalError:
        pass
    # feedback
    try:
        c.execute(
            "UPDATE concept_feedback SET concept_id=? WHERE concept_id=?",
            (keep_id, drop_id),
        )
    except sqlite3.OperationalError:
        pass
    c.execute("DELETE FROM concept_registry WHERE id=?", (drop_id,))
    return stats


def migrate_concept_normalization(
    db_path: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Safe merge of case duplicates + leaf TR/EN pairs. Preserves user evidence."""
    from core.concept_registry import _now, _write_path, ensure

    ensure(db_path)
    path = str(_write_path(db_path))
    report = audit_concept_registry(db_path)
    report["dry_run"] = dry_run
    report["merged"] = 0
    report["aliases_added"] = 0
    report["examples_moved"] = 0
    report["events_retargeted"] = 0
    report["skipped_conflicts"] = []
    report["merged_pairs"] = []

    c = sqlite3.connect(path, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        c.execute("BEGIN")
        rows = c.execute("SELECT * FROM concept_registry").fetchall()
        # 1) case / TR-key duplicates
        by_key: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            by_key.setdefault(concept_match_key(r["canonical"]), []).append(r)
        for key, group in by_key.items():
            if not key or len(group) < 2:
                continue
            # Prefer more positives, then user source, then older id
            ranked = sorted(
                group,
                key=lambda r: (
                    -_example_count(c, int(r["id"])),
                    0 if str(r["source"] or "") == "user" else 1,
                    int(r["id"]),
                ),
            )
            keep = ranked[0]
            for drop in ranked[1:]:
                if dry_run:
                    report["merged_pairs"].append(
                        (keep["canonical"], drop["canonical"], "case_duplicate")
                    )
                    report["merged"] += 1
                    continue
                st = _merge_concept_into(
                    c, keep_id=int(keep["id"]), drop_id=int(drop["id"])
                )
                report["merged"] += 1
                report["aliases_added"] += st["aliases_added"]
                report["examples_moved"] += st["examples_moved"]
                report["events_retargeted"] += st["events_retargeted"]
                report["merged_pairs"].append(
                    (keep["canonical"], drop["canonical"], "case_duplicate")
                )

        # refresh rows after case merges
        rows = c.execute("SELECT * FROM concept_registry").fetchall()
        id_by_key = {
            concept_match_key(r["canonical"]): int(r["id"]) for r in rows
        }
        row_by_id = {int(r["id"]): r for r in rows}

        # 2) leaf translation pairs (tiger↔kaplan, snake↔yılan, …)
        seen: set[tuple[int, int]] = set()
        for r in rows:
            k = concept_match_key(r["canonical"])
            for t in leaf_translation_keys(k):
                other_id = id_by_key.get(t)
                if not other_id or other_id == int(r["id"]):
                    continue
                a, b = sorted((int(r["id"]), other_id))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                if a not in row_by_id or b not in row_by_id:
                    continue
                ra, rb = row_by_id[a], row_by_id[b]
                pair_keys = frozenset(
                    {
                        concept_match_key(ra["canonical"]),
                        concept_match_key(rb["canonical"]),
                    }
                )
                if pair_keys in _TRANSLATION_MERGE_BLOCK:
                    report["skipped_conflicts"].append(
                        (ra["canonical"], rb["canonical"], "related_not_translation")
                    )
                    continue
                # Extra guard: *skin* compound vs bare animal lemma
                ka, kb = concept_match_key(ra["canonical"]), concept_match_key(
                    rb["canonical"]
                )
                if ("skin" in ka and kb in {"snake", "yilan"}) or (
                    "skin" in kb and ka in {"snake", "yilan"}
                ):
                    report["skipped_conflicts"].append(
                        (ra["canonical"], rb["canonical"], "skin_vs_animal")
                    )
                    continue
                # Skip if either looks like parent family concept
                if concept_match_key(ra["canonical"]) in _FAMILY_SKIP:
                    report["skipped_conflicts"].append(
                        (ra["canonical"], rb["canonical"], "family_skip")
                    )
                    continue
                if concept_match_key(rb["canonical"]) in _FAMILY_SKIP:
                    report["skipped_conflicts"].append(
                        (ra["canonical"], rb["canonical"], "family_skip")
                    )
                    continue
                # Prefer richer / Animal Print parent / English title-ish
                def rank(row: sqlite3.Row) -> tuple:
                    parent = str(row["parent"] or "")
                    can = str(row["canonical"] or "")
                    return (
                        -_example_count(c, int(row["id"])),
                        0 if "print" in parent.lower() or parent else 1,
                        0 if can[:1].isupper() else 1,
                        int(row["id"]),
                    )

                keep, drop = (ra, rb) if rank(ra) <= rank(rb) else (rb, ra)
                # Guard: do not merge siblings like Tiger/Leopard (not in leaf_translation)
                if dry_run:
                    report["merged_pairs"].append(
                        (keep["canonical"], drop["canonical"], "translation")
                    )
                    report["merged"] += 1
                    continue
                st = _merge_concept_into(
                    c,
                    keep_id=int(keep["id"]),
                    drop_id=int(drop["id"]),
                    extra_aliases=[drop["canonical"], keep["canonical"]],
                )
                report["merged"] += 1
                report["aliases_added"] += st["aliases_added"]
                report["examples_moved"] += st["examples_moved"]
                report["events_retargeted"] += st["events_retargeted"]
                report["merged_pairs"].append(
                    (keep["canonical"], drop["canonical"], "translation")
                )
                # update maps
                row_by_id.pop(int(drop["id"]), None)
                id_by_key = {
                    concept_match_key(x["canonical"]): int(x["id"])
                    for x in c.execute("SELECT id, canonical FROM concept_registry")
                }
                row_by_id = {
                    int(x["id"]): x
                    for x in c.execute("SELECT * FROM concept_registry")
                }

        # 3) Ensure leaf synonyms present as aliases on surviving concepts
        rows = c.execute("SELECT * FROM concept_registry").fetchall()
        for r in rows:
            k = concept_match_key(r["canonical"])
            if k in _FAMILY_SKIP:
                continue
            aliases = _parse_aliases(r["aliases"])
            before = len(aliases)
            for t in leaf_translation_keys(k):
                # store a readable alias: prefer synonym spelling from TERM_SYNONYMS
                candidate = t
                for src, alts in TERM_SYNONYMS.items():
                    if concept_match_key(src) == t:
                        candidate = src
                        break
                    for a in alts:
                        if concept_match_key(a) == t:
                            candidate = a
                            break
                if candidate and all(
                    concept_match_key(x) != concept_match_key(candidate) for x in aliases
                ):
                    aliases.append(candidate)
            # also add case variants of canonical for search (stored once)
            can = str(r["canonical"] or "")
            for variant in {can.lower(), turkish_casefold(can), can.upper()}:
                if variant and all(
                    concept_match_key(x) != concept_match_key(variant) for x in aliases
                ):
                    # don't flood with UPPER if same key — skip pure case dupes in storage
                    pass
            if not dry_run and len(aliases) != before:
                c.execute(
                    "UPDATE concept_registry SET aliases=?, updated_at=? WHERE id=?",
                    (json.dumps(aliases, ensure_ascii=False), _now(), int(r["id"])),
                )
                report["aliases_added"] += len(aliases) - before

        if dry_run:
            c.execute("ROLLBACK")
        else:
            c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise
    finally:
        c.close()
    report["post"] = audit_concept_registry(db_path)
    return report


def ensure_query_vocab_typos() -> None:
    """Register common concept typos into fuzzy map (non-destructive)."""
    try:
        from core import fuzzy_correct as fc

        extra = {
            "kaplaan": "kaplan",
            "kaplann": "kaplan",
            "kaplanı": "kaplan",
            "kaplani": "kaplan",
            "tigr": "tiger",
            "tigerr": "tiger",
            "tigre": "tiger",
            "tiiger": "tiger",
            "yilann": "yilan",
            "yılaan": "yılan",
            "snak": "snake",
            "snakes": "snake",
        }
        for k, v in extra.items():
            fc._TYPO_MAP.setdefault(k, v)
            # also ascii-folded keys
            fk = normalize_turkish(k)
            fv = v
            if fk and fk not in fc._TYPO_MAP:
                fc._TYPO_MAP[fk] = fv
    except Exception:
        pass


# warm typo map on import
ensure_query_vocab_typos()
