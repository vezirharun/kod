"""Semantic category autocomplete — concept vs context from learned/canonical data.

Edit / Teach typeahead only. Does not invent meanings, write index, or change Search scoring.
Reuses concept_query_normalize + NL/taxonomy tokens for parse & match.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from core.textile_terms import normalize_turkish

# Rank tiers (lower = better) — product order.
RANK_FULL_CONCEPT_CONTEXT = 1
RANK_CONCEPT_STRONG_CONTEXT = 2
RANK_CONCEPT_CATEGORY = 3
RANK_CONCEPT_TAG = 4
RANK_CONCEPT_ONLY = 5
RANK_WEAK = 9

# Catalog key-index generation — bumped on invalidate so stale catalogs rebuild.
_CATALOG_LOCK = threading.RLock()
_CATALOG_GEN = 0


def invalidate_autocomplete_caches() -> None:
    """Thread-safe: drop synonym index + bump catalog generation for rebuild."""
    global _CATALOG_GEN
    try:
        from core.concept_query_normalize import invalidate_leaf_translation_cache

        invalidate_leaf_translation_cache()
    except Exception:
        pass
    with _CATALOG_LOCK:
        _CATALOG_GEN += 1


def _catalog_gen() -> int:
    with _CATALOG_LOCK:
        return _CATALOG_GEN

# Context / type tokens drawn from existing taxonomy + NL skip sets (not free invention).
_CONTEXT_TOKEN_CANON: dict[str, str] = {
    # Texture / textile (PARENT_ALIASES Texture Ground / Textile + NL)
    "doku": "texture",
    "dokusu": "texture",
    "dokular": "texture",
    "dokuyu": "texture",
    "texture": "texture",
    "textile": "texture",
    "tekstil": "texture",
    "kumas": "texture",
    "kumaş": "texture",
    "fabric": "texture",
    "deri": "texture",
    "leather": "texture",
    # Pattern / print (GENERIC_SKIP + concept_match_key suffixes)
    "desen": "pattern",
    "deseni": "pattern",
    "desenler": "pattern",
    "pattern": "pattern",
    "print": "pattern",
    "baski": "pattern",
    "baskı": "pattern",
    "motif": "pattern",
    # Animal / object / head (Animal, Object, Global Object parents)
    "hayvan": "animal",
    "animal": "animal",
    "kafa": "animal",
    "kafasi": "animal",
    "kafası": "animal",
    "head": "animal",
    "nesne": "object",
    "object": "object",
    "obje": "object",
}

# Parent → context kinds they satisfy (canonical CATEGORY_TREE parents only).
_PARENT_CONTEXT_KINDS: dict[str, frozenset[str]] = {
    "Texture Ground": frozenset({"texture"}),
    "Textile": frozenset({"texture", "pattern"}),
    "Animal Print": frozenset({"pattern", "animal"}),
    "Animal": frozenset({"animal"}),
    "Global Object": frozenset({"animal", "object"}),
    "Object": frozenset({"object"}),
    "Hayvan Detayı": frozenset({"animal"}),
    "Floral": frozenset({"pattern"}),
    "Camouflage": frozenset({"pattern"}),
    "Geometric": frozenset({"pattern"}),
    "Marble Abstract": frozenset({"texture", "pattern"}),
}


@dataclass(frozen=True)
class ParsedAutocompleteQuery:
    raw: str = ""
    concept: str = ""
    concept_key: str = ""
    concept_keys: frozenset[str] = field(default_factory=frozenset)
    context_tokens: tuple[str, ...] = ()
    context_kinds: frozenset[str] = field(default_factory=frozenset)
    prefix: str = ""


@dataclass(frozen=True)
class AutocompleteCandidate:
    display: str
    data: str
    parent: str = ""
    path: str = ""
    aliases: tuple[str, ...] = ()
    kind: str = "category"  # category | tag | family | other
    # Precomputed leaf/concept keys — set at catalog build; hot path must not rebuild.
    match_keys: frozenset[str] | None = None
    keys_gen: int = -1


@dataclass(frozen=True)
class AutocompleteHit:
    display: str
    data: str
    rank: int
    reason: str = ""
    parent: str = ""
    path: str = ""
    is_new: bool = False


def _norm(text: str) -> str:
    return normalize_turkish(str(text or "").strip())


def _concept_key(text: str) -> str:
    try:
        from core.concept_query_normalize import concept_match_key

        return concept_match_key(text)
    except Exception:
        return _norm(text)


def _leaf_keys(text: str) -> set[str]:
    try:
        from core.concept_query_normalize import leaf_translation_keys

        return set(leaf_translation_keys(text) or ())
    except Exception:
        k = _concept_key(text)
        return {k} if k else set()


def _compute_candidate_keys(c: AutocompleteCandidate) -> frozenset[str]:
    """One-shot key expansion for catalog build (not per keypress)."""
    keys: set[str] = set()
    parts = [c.data]
    if c.path and "/" in c.path:
        parts.append(c.path.rsplit("/", 1)[-1])
    parts.extend(c.aliases or ())
    for part in parts:
        if not part:
            continue
        keys |= _leaf_keys(str(part))
        ck = _concept_key(str(part))
        if ck:
            keys.add(ck)
    return frozenset(k for k in keys if k)


def index_candidate(c: AutocompleteCandidate) -> AutocompleteCandidate:
    """Attach precomputed match_keys for the current catalog generation."""
    gen = _catalog_gen()
    if c.match_keys is not None and c.keys_gen == gen:
        return c
    return replace(c, match_keys=_compute_candidate_keys(c), keys_gen=gen)


def ensure_candidates_indexed(
    candidates: Sequence[AutocompleteCandidate] | None,
) -> list[AutocompleteCandidate]:
    """Prebuild leaf keys for a catalog (catalog load / set_semantic_catalog)."""
    if not candidates:
        return []
    return [index_candidate(c) for c in candidates]


def context_token_map() -> dict[str, str]:
    """Normalized token → context kind (from existing taxonomy/NL vocabulary)."""
    out: dict[str, str] = {}
    for tok, kind in _CONTEXT_TOKEN_CANON.items():
        nk = _norm(tok)
        if nk:
            out[nk] = kind
    try:
        from core.natural_language_query import GENERIC_SKIP, TEXTURE_PHRASES

        for tok in GENERIC_SKIP:
            nk = _norm(tok)
            if nk and nk not in out:
                out[nk] = "pattern"
        for phrase in TEXTURE_PHRASES:
            for part in _norm(phrase).split():
                if len(part) >= 3 and part not in out:
                    out[part] = "texture"
    except Exception:
        pass
    try:
        from core.category_tree import PARENT_ALIASES

        for parent, aliases in PARENT_ALIASES.items():
            kinds = _PARENT_CONTEXT_KINDS.get(parent)
            if not kinds:
                continue
            kind = next(iter(kinds))
            for a in aliases:
                for part in _norm(a).split():
                    if len(part) >= 3 and part not in out:
                        out[part] = kind
    except Exception:
        pass
    return out


def split_concept_context(query: str) -> ParsedAutocompleteQuery:
    """Separate concept identity tokens from context/type tokens."""
    raw = " ".join(str(query or "").strip().split())
    if not raw:
        return ParsedAutocompleteQuery()

    try:
        from core.concept_query_normalize import clean_query_text

        cleaned = clean_query_text(raw)
    except Exception:
        cleaned = raw

    cmap = context_token_map()
    tokens = [t for t in _norm(cleaned).replace("-", " ").split() if t]
    concept_parts: list[str] = []
    ctx_tokens: list[str] = []
    ctx_kinds: set[str] = set()

    def _kind_for(tok: str) -> str | None:
        if tok in cmap:
            return cmap[tok]
        if len(tok) < 2:
            return None
        matched = {kind for ct, kind in cmap.items() if ct.startswith(tok)}
        if len(matched) == 1:
            return next(iter(matched))
        if len(tok) >= 3 and matched:
            if "texture" in matched and tok.startswith("dok"):
                return "texture"
            return sorted(matched)[0]
        return None

    for tok in tokens:
        kind = _kind_for(tok)
        if kind:
            ctx_tokens.append(tok)
            ctx_kinds.add(kind)
        else:
            concept_parts.append(tok)

    if (
        concept_parts
        and len(concept_parts[-1]) == 1
        and any(ct.startswith(concept_parts[-1]) for ct in cmap)
    ):
        stub = concept_parts.pop()
        ctx_tokens.append(stub)
        matched = {kind for ct, kind in cmap.items() if ct.startswith(stub)}
        if "texture" in matched:
            ctx_kinds.add("texture")
        elif matched:
            ctx_kinds.add(next(iter(matched)))

    concept = " ".join(concept_parts).strip()
    if not concept:
        try:
            from core.query_attribute_intel import concept_core_text

            concept = concept_core_text(raw) or ""
        except Exception:
            concept = ""

    # Normalize CONCEPT only — never fold context tokens into leaf keys.
    nq = None
    if concept:
        try:
            from core.concept_query_normalize import normalize_concept_query

            nq = normalize_concept_query(concept)
            concept = (nq.effective or concept).strip()
        except Exception:
            nq = None

    ckey = _concept_key(concept) if concept else ""
    keys = _leaf_keys(concept) if concept else set()
    if ckey:
        keys.add(ckey)
    if nq is not None and getattr(nq, "corrected_key", ""):
        keys.add(str(nq.corrected_key))

    prefix = ckey[:4] if len(ckey) >= 3 else ckey
    return ParsedAutocompleteQuery(
        raw=raw,
        concept=concept,
        concept_key=ckey,
        concept_keys=frozenset(k for k in keys if k),
        context_tokens=tuple(ctx_tokens),
        context_kinds=frozenset(ctx_kinds),
        prefix=prefix,
    )


def _candidate_text_blob(c: AutocompleteCandidate) -> str:
    parts = [c.display, c.data, c.parent, c.path, *c.aliases]
    return " ".join(str(p) for p in parts if p)


def _candidate_keys(c: AutocompleteCandidate) -> set[str]:
    # Hot path: use precomputed keys only — never rebuild leaf_translation_keys here.
    if c.match_keys is not None and c.keys_gen == _catalog_gen():
        return set(c.match_keys)
    # Lazy index once if catalog was built without keys (still avoids per-call synonym scan
    # once synonym index exists; prefer ensure_candidates_indexed at catalog load).
    indexed = index_candidate(c)
    return set(indexed.match_keys or ())


def _parent_kinds(parent: str) -> frozenset[str]:
    p = str(parent or "").strip()
    if p in _PARENT_CONTEXT_KINDS:
        return _PARENT_CONTEXT_KINDS[p]
    # Soft: parent name tokens mapped via context map
    kinds: set[str] = set()
    cmap = context_token_map()
    for tok in _norm(p).split():
        k = cmap.get(tok)
        if k:
            kinds.add(k)
    return frozenset(kinds)


def _blob_has_marker(blob: str, markers: tuple[str, ...]) -> bool:
    padded = f" {blob} "
    return any(f" {m} " in padded for m in markers if m)


def _context_strength(parsed: ParsedAutocompleteQuery, c: AutocompleteCandidate) -> str:
    """none | weak | strong | full — based on existing parent/alias text only."""
    if not parsed.context_kinds and not parsed.context_tokens:
        return "none"
    blob = _norm(_candidate_text_blob(c))
    parent_kinds = _parent_kinds(c.parent)
    hit_kinds = parsed.context_kinds & parent_kinds
    token_hits = 0
    padded = f" {blob} "
    for tok in parsed.context_tokens:
        if tok and (f" {tok} " in padded or blob.startswith(tok) or blob.endswith(tok)):
            token_hits += 1
    if hit_kinds and token_hits:
        return "full"
    if hit_kinds:
        return "strong"
    if token_hits:
        return "strong"
    for kind in parsed.context_kinds:
        markers = {
            "texture": ("doku", "texture", "textile", "tekstil", "deri", "leather", "fabric"),
            "pattern": ("desen", "pattern", "print", "baski", "baski"),
            "animal": ("hayvan", "animal", "kafa", "head"),
            "object": ("nesne", "object", "obje"),
        }.get(kind, ())
        if _blob_has_marker(blob, markers):
            return "weak"
    return "none"


def _concept_match_tier(parsed: ParsedAutocompleteQuery, c: AutocompleteCandidate) -> str | None:
    """exact | prefix | contains | None. Never collapses Tiger↔Leopard."""
    if not parsed.concept_key and not parsed.concept_keys:
        return None
    ckeys = _candidate_keys(c)
    if not ckeys:
        return None
    # Exact / translation leaf overlap
    if parsed.concept_keys & ckeys:
        return "exact"
    # Prefix / typo stem (leop → leopard, leoprad under leop)
    q = parsed.concept_key
    if len(q) >= 3:
        for ck in ckeys:
            if ck.startswith(q) or q.startswith(ck[: max(3, min(len(ck), len(q)))]):
                # Guard sibling: if leaf keys are disjoint known rivals, skip
                if _are_rival_leaves(parsed.concept_keys, ckeys):
                    return None
                return "prefix"
    # Soft contains on display only (short queries)
    blob = _norm(c.display + " " + c.data)
    if q and len(q) >= 3 and q in blob:
        if _are_rival_leaves(parsed.concept_keys, ckeys):
            return None
        return "contains"
    return None


def _are_rival_leaves(q_keys: Iterable[str], c_keys: Iterable[str]) -> bool:
    try:
        from core.canonical_correction import are_distinct_concepts
    except Exception:
        return False
    q_list = [k for k in q_keys if k]
    c_list = [k for k in c_keys if k]
    if not q_list or not c_list:
        return False
    # If any pair is distinct sibling concepts, treat as rival when no shared key
    if set(q_list) & set(c_list):
        return False
    for a in q_list[:4]:
        for b in c_list[:4]:
            try:
                if are_distinct_concepts(a, b):
                    return True
            except Exception:
                continue
    return False


def _rank_for(
    concept_tier: str,
    ctx: str,
    *,
    kind: str,
) -> tuple[int, str]:
    if concept_tier == "exact" and ctx == "full":
        return RANK_FULL_CONCEPT_CONTEXT, "concept+context"
    if concept_tier == "exact" and ctx == "strong":
        return RANK_CONCEPT_STRONG_CONTEXT, "concept+strong_context"
    if concept_tier == "exact" and ctx == "weak":
        return RANK_CONCEPT_STRONG_CONTEXT, "concept+context_weak"
    if concept_tier == "exact" and kind == "category" and ctx == "none":
        return RANK_CONCEPT_CATEGORY, "concept+category"
    if concept_tier == "exact" and kind == "tag":
        return RANK_CONCEPT_TAG, "concept+tag"
    if concept_tier == "exact":
        return RANK_CONCEPT_ONLY, "concept"
    if concept_tier in {"prefix", "contains"} and ctx in {"full", "strong"}:
        return RANK_CONCEPT_STRONG_CONTEXT, f"{concept_tier}+context"
    if concept_tier in {"prefix", "contains"}:
        return RANK_CONCEPT_ONLY, concept_tier
    return RANK_WEAK, "weak"


def candidates_from_choices(
    choices: Sequence[tuple[str, str]],
    *,
    parent: str = "",
    kind: str = "category",
) -> list[AutocompleteCandidate]:
    out: list[AutocompleteCandidate] = []
    for display, data in choices:
        d, v = str(display or "").strip(), str(data or "").strip()
        if not d:
            continue
        path = f"{parent}/{d}" if parent else d
        aliases: tuple[str, ...] = ()
        if parent:
            try:
                from core.category_tree import aliases_for_path

                aliases = tuple(aliases_for_path(path))
            except Exception:
                aliases = ()
        out.append(
            index_candidate(
                AutocompleteCandidate(
                    display=d,
                    data=v or d,
                    parent=parent,
                    path=path,
                    aliases=aliases,
                    kind=kind,
                )
            )
        )
    return out


def flatten_children_catalog(
    children_by_parent: dict[str, list[str]] | None,
) -> list[AutocompleteCandidate]:
    """All learned/canonical parent→child paths (no invention)."""
    out: list[AutocompleteCandidate] = []
    for parent, kids in (children_by_parent or {}).items():
        p = str(parent or "").strip()
        for child in kids or []:
            c = str(child or "").strip()
            if not c:
                continue
            path = f"{p}/{c}" if p else c
            aliases: tuple[str, ...] = ()
            try:
                from core.category_tree import aliases_for_path

                aliases = tuple(aliases_for_path(path))
            except Exception:
                aliases = ()
            display = f"{p} / {c}" if p else c
            out.append(
                index_candidate(
                    AutocompleteCandidate(
                        display=display,
                        data=c,
                        parent=p,
                        path=path,
                        aliases=aliases,
                        kind="category",
                    )
                )
            )
    return out


def suggest_semantic(
    query: str,
    candidates: Sequence[AutocompleteCandidate],
    *,
    allow_new: bool = True,
    limit: int = 40,
) -> list[AutocompleteHit]:
    """Rank candidates for typeahead. Empty query → []. No match → optional Yeni ekle."""
    q = " ".join(str(query or "").strip().split())
    if not q:
        return []

    parsed = split_concept_context(q)
    # Need at least a concept stem or raw prefix to search
    stem = parsed.concept_key or _concept_key(q)
    if not stem and not parsed.context_tokens:
        return []

    # Ensure catalog keys are ready before the loop (one pass, not per keypress rebuild).
    gen = _catalog_gen()
    indexed = [
        c if (c.match_keys is not None and c.keys_gen == gen) else index_candidate(c)
        for c in candidates
    ]

    hits: list[AutocompleteHit] = []
    seen: set[tuple[str, str]] = set()
    for c in indexed:
        tier = _concept_match_tier(parsed, c)
        if tier is None:
            # Context-only typing should not dump unrelated rows
            if parsed.concept_key:
                continue
            # Allow pure context filter only when concept empty and blob matches tokens
            if parsed.context_tokens and any(
                t in _norm(_candidate_text_blob(c)) for t in parsed.context_tokens
            ):
                tier = "contains"
            else:
                continue
        ctx = _context_strength(parsed, c)
        # When user supplied context, demote concept-only mismatches hard
        if parsed.context_kinds and ctx == "none" and tier in {"exact", "prefix", "contains"}:
            rank, reason = RANK_WEAK, "concept_no_context"
        else:
            rank, reason = _rank_for(tier, ctx, kind=c.kind)
            # Prefer parent whose kinds cover query context
            if parsed.context_kinds and ctx in {"full", "strong"}:
                cover = len(parsed.context_kinds & _parent_kinds(c.parent))
                rank = max(1, rank - min(cover, 1))
        key = (_norm(c.display), _norm(c.data))
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            AutocompleteHit(
                display=c.display,
                data=c.data,
                rank=rank,
                reason=reason,
                parent=c.parent,
                path=c.path,
            )
        )

    hits.sort(key=lambda h: (h.rank, len(h.display), h.display.lower()))
    hits = hits[: max(1, int(limit))]
    if not hits and allow_new and q:
        hits.append(
            AutocompleteHit(
                display="Yeni ekle",
                data=q,
                rank=99,
                reason="new",
                is_new=True,
            )
        )
    return hits


def filter_choice_tuples(
    query: str,
    choices: Sequence[tuple[str, str]],
    *,
    parent: str = "",
    kind: str = "category",
    allow_new: bool = False,
    catalog: Sequence[AutocompleteCandidate] | None = None,
) -> list[tuple[str, str]]:
    """UI helper: return (display, data) ranked; empty query → []."""
    if not " ".join(str(query or "").strip().split()):
        return []
    cands = (
        ensure_candidates_indexed(catalog)
        if catalog is not None
        else candidates_from_choices(choices, parent=parent, kind=kind)
    )
    hits = suggest_semantic(query, cands, allow_new=allow_new)
    out: list[tuple[str, str]] = []
    for h in hits:
        if h.is_new:
            continue
        out.append((h.display, h.data))
    return out


def has_learned_match(query: str, candidates: Sequence[AutocompleteCandidate]) -> bool:
    hits = suggest_semantic(query, candidates, allow_new=False)
    return any(not h.is_new for h in hits)
