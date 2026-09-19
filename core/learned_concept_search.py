"""User-taught concepts as a search signal. Does not train models or write FAISS."""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Iterable

import numpy as np

from core.concept_registry import find, negative_example_vectors, negative_file_ids, positive_example_vectors
from core.teach_me import _cosine
from core.textile_terms import normalize_turkish

LEARNED_EXACT_SCORE = 0.96
LEARNED_CHILD_SCORE = 0.93
LEARNED_RELATED_SCORE = 0.88
LEARNED_PARENT_SCORE = 0.80
LEARNED_REASON = "Öğrenilmiş kavram"
# Same floor already used for user-taught query overlay injection.
_NEIGHBOR_FLOOR = 0.72
# Boundary: negative CLIP benzerliği bu eşiğin üstündeyse neighbor skorunu düşür.
_BOUNDARY_NEG_FLOOR = 0.72
_BOUNDARY_PENALTY = 0.18
_REL_RANK = {
    "exact": 0,
    "translation": 0,
    "alias": 0,
    "typo": 1,
    "child": 2,
    "related": 3,
    "parent": 4,
}


def _norm(text: str) -> str:
    try:
        from core.concept_query_normalize import concept_match_key

        return concept_match_key(text)
    except Exception:
        return normalize_turkish(text or "")


def _word_tokens(text: str) -> set[str]:
    q = _norm(text)
    return {t for t in q.replace("-", " ").replace("/", " ").split() if len(t) >= 2}


def _stem_tok(tok: str) -> str:
    t = _norm(tok)
    for suf in ("lari", "leri", "lar", "ler", "nin", "si", "i", "u"):
        if len(t) >= len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _tok_eq(a: str, b: str) -> bool:
    if a == b or _stem_tok(a) == _stem_tok(b) or _stem_tok(a) == b or a == _stem_tok(b):
        return True
    if min(len(a), len(b)) < 5:
        return False
    return _typo_ratio(a, b) >= 0.84


def _covered(need: set[str], have: set[str]) -> bool:
    if not need:
        return False
    for qt in need:
        if any(_tok_eq(qt, ht) for ht in have):
            continue
        return False
    return True


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


def _names(canonical: str, aliases: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in (canonical, *(aliases or ())):
        n = _norm(str(raw))
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _typo_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _is_typo(query: str, name: str) -> bool:
    """General edit-distance typo; no per-word tables."""
    q, n = _norm(query), _norm(name)
    if not q or not n or q == n:
        return False
    if min(len(q), len(n)) < 5:
        return False
    if abs(len(q) - len(n)) > max(2, len(n) // 4):
        return False
    ratio = _typo_ratio(q, n)
    if ratio >= 0.84:
        return True
    qt, nt = _word_tokens(q), _word_tokens(n)
    if qt and nt and len(qt) == len(nt):
        used: set[str] = set()
        ok = True
        for t in qt:
            hit = ""
            best = 0.0
            for u in nt:
                if u in used:
                    continue
                r = _typo_ratio(t, u)
                if r > best:
                    best, hit = r, u
            if best >= 0.84 and hit:
                used.add(hit)
            else:
                ok = False
                break
        if ok:
            return True
    return False


def _uvi_descendants(node_id: str) -> set[str]:
    from core.universal_visual_intel import children

    out: set[str] = set()
    stack = list(children(node_id))
    while stack:
        nid = stack.pop()
        if not nid or nid in out or nid == "entity":
            continue
        out.add(nid)
        stack.extend(children(nid))
    return out


def _uvi_scope(query: str) -> set[str]:
    try:
        from core.universal_visual_intel import ALIAS_TO_NODE, ONTOLOGY
    except Exception:
        return set()
    seeds: set[str] = set()
    mapped = ALIAS_TO_NODE.get(_norm(query)) or ""
    if mapped:
        seeds.add(mapped)
    for tok in _word_tokens(query):
        hit = ALIAS_TO_NODE.get(tok) or ""
        if hit:
            seeds.add(hit)
    scope: set[str] = set()
    for nid in seeds:
        if not nid or nid == "entity" or nid not in ONTOLOGY:
            continue
        level = str(ONTOLOGY[nid].get("level") or "")
        if level in ("species", "breed", "brand", "model"):
            scope.add(nid)
            continue
        scope.add(nid)
        scope |= _uvi_descendants(nid)
    return scope


def _concept_uvi_nodes(canonical: str, aliases: Iterable[str] | None) -> set[str]:
    try:
        from core.universal_visual_intel import ALIAS_TO_NODE
    except Exception:
        return set()
    nodes: set[str] = set()
    for name in (canonical, *(aliases or ())):
        text = str(name or "").strip()
        if not text:
            continue
        mapped = ALIAS_TO_NODE.get(_norm(text)) or ""
        if mapped and mapped != "entity":
            nodes.add(mapped)
        for tok in _word_tokens(text):
            hit = ALIAS_TO_NODE.get(tok) or ""
            if hit and hit != "entity":
                nodes.add(hit)
    return nodes


def _parent_label_keys(text: str) -> set[str]:
    """Tight synonym keys for parent↔query linking (no aggressive expand)."""
    try:
        from core.textile_terms import parent_synonym_keys

        return parent_synonym_keys(text)
    except Exception:
        n = _norm(text)
        return {n} if len(n) >= 2 else set()


def _parent_links_query(query: str, parent: str) -> bool:
    p = str(parent or "").strip()
    if not p:
        return False
    return bool(_parent_label_keys(query) & _parent_label_keys(p))


def concept_query_relation(
    query: str,
    canonical: str,
    aliases: Iterable[str] | None = None,
    parent: str = "",
    scope: set[str] | None = None,
) -> str | None:
    """exact | typo | alias | translation | child | related | parent | None."""
    # Shared TR/EN / case / typo normalization (does not create learning).
    try:
        from core.concept_query_normalize import relation_to_concept

        rel = relation_to_concept(query, canonical, aliases, parent)
        if rel:
            return rel
    except Exception:
        pass
    q = _norm(query)
    if not q or len(q) < 2:
        return None
    names = _names(canonical, aliases)
    if not names:
        return None
    q_tokens = _word_tokens(query)
    if q in names or any(_tok_eq(q, n) for n in names):
        return "exact"
    if any(_is_typo(query, n) for n in names):
        return "typo"
    c_words: set[str] = set()
    for n in names:
        c_words |= _word_tokens(n)
    if q_tokens and c_words:
        if q_tokens != c_words and _covered(q_tokens, c_words) and len(c_words) >= len(q_tokens):
            if len(c_words) > len(q_tokens) or any(len(n) > len(q) for n in names):
                return "child"
        if q_tokens != c_words and _covered(c_words, q_tokens) and len(q_tokens) > len(c_words):
            return "parent"
    if _parent_links_query(query, parent):
        return "child"
    if scope is None:
        scope = _uvi_scope(query)
    if scope:
        c_nodes = _concept_uvi_nodes(canonical, aliases)
        if c_nodes & scope:
            # UVI = soft related; parent→child yalnızca registry parent / name hyponym
            return "related"
    return None


def match_taught_concepts(db_path: str, query: str) -> list[dict[str, Any]]:
    """All taught concepts related to this query. New labels join automatically."""
    q = _norm(query)
    if not db_path or len(q) < 2:
        return []
    try:
        from core.concept_registry import concepts

        rows = concepts(db_path)
    except Exception:
        return []
    scope = _uvi_scope(query)
    out: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "active")
        if status in ("inactive", "retired"):
            continue
        can = str(row.get("canonical") or "")
        aliases = _parse_aliases(row.get("aliases"))
        rel = concept_query_relation(
            query,
            can,
            aliases,
            str(row.get("parent") or ""),
            scope=scope,
        )
        if not rel:
            continue
        out.append(
            {
                "id": int(row["id"]),
                "canonical": can,
                "concept_type": row.get("concept_type") or "visual_concept",
                "parent": row.get("parent") or "",
                "confidence": float(row.get("confidence") or 0),
                "score": {"exact": 1.0, "typo": 0.96, "child": 0.9, "related": 0.86, "parent": 0.8}.get(
                    rel, 0.7
                ),
                "relation": rel,
            }
        )
    out.sort(key=lambda x: (_REL_RANK.get(str(x.get("relation") or ""), 9), -float(x["score"])))
    return out


def query_matches_verified(
    query: str,
    canonical: str,
    aliases: Iterable[str] | None = None,
    parent: str = "",
) -> bool:
    """Taught label matches this text query (exact, typo, hypernym/hyponym, UVI)."""
    return concept_query_relation(query, canonical, aliases, parent) is not None


def verified_concepts_by_file(db_path: str) -> dict[int, list[tuple[str, list[str]]]]:
    """file_id → [(canonical, aliases), ...] from concept_examples. No index writes."""
    out: dict[int, list[tuple[str, list[str]]]] = {}
    if not db_path:
        return out
    try:
        from core.concept_registry import _conn

        c = _conn(db_path)
        rows = c.execute(
            """
            SELECT e.file_id, r.canonical, r.aliases, r.status
            FROM concept_examples e
            JOIN concept_registry r ON r.id = e.concept_id
            WHERE e.role='positive' AND e.file_id>0
              AND IFNULL(r.status,'active') NOT IN ('inactive','retired')
            """
        ).fetchall()
        c.close()
    except Exception:
        return out
    for row in rows:
        fid = int(row["file_id"] if hasattr(row, "keys") else row[0] or 0)
        if fid <= 0:
            continue
        can = str(row["canonical"] if hasattr(row, "keys") else row[1] or "")
        raw = row["aliases"] if hasattr(row, "keys") else row[2]
        aliases: list[str] = []
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    aliases = [str(x) for x in parsed if str(x).strip()]
            except Exception:
                aliases = []
        out.setdefault(fid, []).append((can, aliases))
    return out


def conflicting_taught_ids(db_path: str, query: str) -> set[int]:
    """Verified files whose taught concept does not match this text query."""
    q = _norm(query)
    if not db_path or len(q) < 2:
        return set()
    blocked: set[int] = set()
    for fid, labels in verified_concepts_by_file(db_path).items():
        if any(query_matches_verified(query, can, aliases) for can, aliases in labels):
            continue
        blocked.add(int(fid))
    return blocked


def filter_taught_conflicts(results: list[Any], db_path: str, query: str) -> list[Any]:
    blocked = conflicting_taught_ids(db_path, query)
    if not blocked:
        return results
    kept: list[Any] = []
    for row in results or []:
        if isinstance(row, dict):
            fid = int(row.get("id") or row.get("file_id") or 0)
        else:
            fid = int(getattr(row, "file_id", 0) or 0)
        if fid in blocked:
            continue
        kept.append(row)
    return kept


def resolve_learned_concept(db_path: str, query: str) -> dict[str, Any] | None:
    """Best taught concept for this query, including typo and hypernym hits."""
    hits = match_taught_concepts(db_path, query)
    if hits:
        return dict(hits[0])
    raw = str(query or "").strip()
    q = _norm(raw)
    if not db_path or len(q) < 2:
        return None
    try:
        hits = find(db_path, raw, limit=12)
    except Exception:
        return None
    tokens = [q] + [t for t in q.split() if len(t) >= 2]

    def _ok(hit: dict[str, Any], needle: str) -> bool:
        can = _norm(str(hit.get("canonical") or ""))
        score = float(hit.get("score") or 0)
        if not can or score < 0.92:
            return False
        if can == needle or needle == can:
            return True
        if needle in can or can in needle:
            return score >= 0.96
        return False

    for needle in tokens:
        for hit in hits:
            if _ok(hit, needle):
                return dict(hit)
    try:
        from core.concept_registry import concepts

        for row in concepts(db_path):
            status = str(row.get("status") or "active")
            if status in ("inactive", "retired"):
                continue
            can = _norm(str(row.get("canonical") or ""))
            if can and can in tokens:
                return {
                    "id": int(row["id"]),
                    "canonical": row["canonical"],
                    "concept_type": row.get("concept_type") or "visual_concept",
                    "parent": row.get("parent") or "",
                    "confidence": float(row.get("confidence") or 0),
                    "score": 1.0,
                }
    except Exception:
        pass
    return None


def example_file_ids(
    db_path: str,
    concept_id: int,
    *,
    user_only: bool = False,
    customer_key: str = "",
) -> list[int]:
    """Positive example file ids. With customer_key: prefer scoped, then global.

    Never includes another customer's examples. Global identity stays intact.
    """
    out: list[int] = []
    seen: set[int] = set()
    ck = ""
    try:
        from core.concept_registry import _norm_customer_key, example_rows_for_concept

        ck = _norm_customer_key(customer_key)
        rows = example_rows_for_concept(
            db_path,
            int(concept_id),
            role="positive",
            user_only=user_only,
            customer_key=ck,
        )
        for r in rows:
            fid = int(r.get("file_id") or 0)
            if fid > 0 and fid not in seen:
                seen.add(fid)
                out.append(fid)
    except Exception:
        # Legacy fallback (no customer column / helper)
        try:
            from core.concept_registry import _conn

            c = _conn(db_path)
            if user_only:
                rows = c.execute(
                    """SELECT DISTINCT file_id FROM concept_examples
                       WHERE concept_id=? AND file_id>0 AND role='positive'
                         AND IFNULL(source,'user') NOT IN ('auto','autonomous','candidate')""",
                    (int(concept_id),),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT DISTINCT file_id FROM concept_examples "
                    "WHERE concept_id=? AND file_id>0 AND role='positive'",
                    (int(concept_id),),
                ).fetchall()
            c.close()
            for r in rows:
                fid = int(r["file_id"] or 0)
                if fid > 0 and fid not in seen:
                    seen.add(fid)
                    out.append(fid)
        except Exception:
            pass
    if out:
        return out
    if user_only:
        return out
    for ex in positive_example_vectors(db_path):
        if int(ex.get("concept_id") or 0) != int(concept_id):
            continue
        fid = int(ex.get("file_id") or 0)
        if fid > 0 and fid not in seen:
            seen.add(fid)
            out.append(fid)
    return out


def max_clip_to_examples(clip_blob: bytes | None, example_blobs: list[bytes]) -> float:
    if not clip_blob or not example_blobs:
        return 0.0
    try:
        vec = np.frombuffer(clip_blob, dtype=np.float32)
    except (ValueError, TypeError):
        return 0.0
    if vec.size == 0 or not np.isfinite(vec).all():
        return 0.0
    best = 0.0
    for raw in example_blobs:
        try:
            other = np.frombuffer(raw, dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if other.size != vec.size:
            continue
        best = max(best, _cosine(vec, other))
    return float(best)


def _example_blobs(db_path: str, concept_id: int) -> list[bytes]:
    out: list[bytes] = []
    for ex in positive_example_vectors(db_path):
        if int(ex.get("concept_id") or 0) != int(concept_id):
            continue
        blob = ex.get("embedding") or b""
        if blob:
            out.append(bytes(blob))
    return out


def _negative_blobs(db_path: str, concept_id: int) -> list[bytes]:
    out: list[bytes] = []
    for ex in negative_example_vectors(db_path, concept_id):
        blob = ex.get("embedding") or b""
        if blob:
            out.append(bytes(blob))
    return out


def apply_boundary_penalty(
    scores: dict[int, float],
    db_path: str,
    concept_id: int,
    *,
    db=None,
) -> dict[int, float]:
    """Negative/boundary evidence: hard-filter'ı kaldırmaz; skor cezası uygular."""
    if not scores or not db_path or int(concept_id or 0) <= 0:
        return scores
    blocked = negative_file_ids(db_path, concept_id)
    neg_blobs = _negative_blobs(db_path, concept_id)
    out: dict[int, float] = {}
    clip_cache: dict[int, bytes] = {}
    if neg_blobs and db is not None and scores:
        need = [fid for fid in scores if fid not in blocked]
        if need:
            try:
                ph = ",".join("?" * len(need))
                with db.connect() as conn:
                    rows = conn.execute(
                        f"SELECT file_id, clip_embedding FROM features "
                        f"WHERE file_id IN ({ph}) AND clip_embedding IS NOT NULL",
                        need,
                    ).fetchall()
                for row in rows:
                    fid = int(row["file_id"] if hasattr(row, "keys") else row[0] or 0)
                    raw = row["clip_embedding"] if hasattr(row, "keys") else row[1]
                    if fid > 0 and raw:
                        clip_cache[fid] = bytes(raw)
            except Exception:
                clip_cache = {}
    for fid, sc in scores.items():
        if fid in blocked:
            continue
        adj = float(sc)
        raw = clip_cache.get(fid)
        if raw and neg_blobs:
            neg_sim = max_clip_to_examples(raw, neg_blobs)
            if neg_sim >= _BOUNDARY_NEG_FLOOR:
                adj = max(0.0, adj - _BOUNDARY_PENALTY * float(neg_sim))
        if adj > 0:
            out[fid] = adj
    return out


def clip_neighbor_scores(
    db_path: str,
    concept_id: int,
    *,
    db=None,
    faiss_store=None,
    exclude: Iterable[int] = (),
    limit: int = 400,
    query: str = "",
) -> dict[int, float]:
    """Other files similar to taught CLIP examples. Read-only FAISS / features."""
    skip = {int(x) for x in exclude if int(x) > 0}
    if query:
        skip |= conflicting_taught_ids(db_path, query)
    skip |= negative_file_ids(db_path, concept_id)
    blobs = _example_blobs(db_path, concept_id)
    scores: dict[int, float] = {}
    if not blobs:
        return scores

    if faiss_store is not None and int(getattr(faiss_store, "clip_count", 0) or 0) > 0:
        try:
            for raw in blobs[:12]:
                vec = np.frombuffer(raw, dtype=np.float32)
                if vec.size == 0:
                    continue
                hits = faiss_store.search_clip(vec, k=min(200, max(32, int(limit))))
                for fid, sc in hits:
                    i = int(fid)
                    if i <= 0 or i in skip:
                        continue
                    scores[i] = max(float(scores.get(i, 0) or 0), float(sc or 0))
        except Exception:
            pass

    if db is not None and len(scores) < 8:
        try:
            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT file_id, clip_embedding FROM features "
                    "WHERE clip_embedding IS NOT NULL AND length(clip_embedding)>=16 "
                    "LIMIT 2500"
                ).fetchall()
            for row in rows:
                fid = int(row["file_id"] if hasattr(row, "keys") else row[0])
                if fid <= 0 or fid in skip:
                    continue
                raw = row["clip_embedding"] if hasattr(row, "keys") else row[1]
                if not raw:
                    continue
                sc = max_clip_to_examples(bytes(raw), blobs)
                if sc > float(scores.get(fid, 0) or 0):
                    scores[fid] = sc
        except Exception:
            pass

    scores = apply_boundary_penalty(scores, db_path, concept_id, db=db)
    kept = {fid: sc for fid, sc in scores.items() if sc >= _NEIGHBOR_FLOOR}
    ranked = sorted(kept.items(), key=lambda x: -x[1])[: max(1, int(limit))]
    return dict(ranked)


def _is_hierarchical_parent_query(db_path: str, query: str, matches: list[dict[str, Any]]) -> bool:
    """True when query targets a parent label (incl. synonyms), not a leaf concept."""
    exact_ms = [
        m
        for m in matches
        if str(m.get("relation") or "") in ("exact", "typo", "translation", "alias")
    ]
    if any(str(m.get("concept_type") or "") == "parent_group" for m in exact_ms):
        return True
    parent_linked = [
        m
        for m in matches
        if _parent_links_query(query, str(m.get("parent") or ""))
    ]
    if not parent_linked:
        return False
    for m in exact_ms:
        if str(m.get("concept_type") or "") == "parent_group":
            return True
        # Exact hit is a synonym-parent concept that itself sits under the family
        # (e.g. taught "çizgi film" with parent=cartoon) → still aggregate siblings.
        if _parent_links_query(query, str(m.get("parent") or "")):
            return True
        # Exact leaf (Rose / Miki Mouse): do not aggregate siblings.
        return False
    return True


def collect_learned_hits(
    db_path: str,
    query: str,
    *,
    db=None,
    faiss_store=None,
    customer_key: str = "",
) -> dict[str, Any]:
    try:
        from core.concept_registry import _norm_customer_key

        _ck_out = _norm_customer_key(customer_key)
    except Exception:
        _ck_out = " ".join(str(customer_key or "").strip().split())
    customer_key = _ck_out
    matches = match_taught_concepts(db_path, query)
    if not matches:
        hit = resolve_learned_concept(db_path, query)
        matches = [hit] if hit else []
    if not matches:
        return {}
    hierarchical = _is_hierarchical_parent_query(db_path, query, matches)
    primary = matches[0]
    if hierarchical:
        # Prefer parent_group shell as display primary when present.
        for m in matches:
            if str(m.get("concept_type") or "") == "parent_group" and str(
                m.get("relation") or ""
            ) in ("exact", "typo", "translation", "alias"):
                primary = m
                break
        else:
            # Keep first child match but label from query parent keys
            for m in matches:
                if str(m.get("relation") or "") == "child":
                    primary = {
                        **m,
                        "canonical": str(m.get("parent") or query).strip() or m.get("canonical"),
                        "concept_type": "parent_group",
                    }
                    break
    leaf_exact = any(
        str(m.get("relation") or "") in ("exact", "typo", "translation", "alias")
        and str(m.get("concept_type") or "") != "parent_group"
        for m in matches
    )
    exact: list[int] = []
    seen_exact: set[int] = set()
    neighbors: dict[int, float] = {}
    file_scores: dict[int, float] = {}
    rel_score = {
        "exact": LEARNED_EXACT_SCORE,
        "translation": LEARNED_EXACT_SCORE,
        "alias": LEARNED_EXACT_SCORE,
        "typo": LEARNED_EXACT_SCORE,
        "child": LEARNED_CHILD_SCORE,
        "related": LEARNED_RELATED_SCORE,
        "parent": LEARNED_PARENT_SCORE,
    }
    child_concept_ids: list[int] = []
    for m in matches:
        cid = int(m.get("id") or 0)
        if cid <= 0:
            continue
        rel = str(m.get("relation") or "related")
        sc = float(rel_score.get(rel, LEARNED_RELATED_SCORE))
        # Parent aramada verified child evidence = user positives only.
        use_user = hierarchical and rel in ("child", "exact", "typo")
        if hierarchical and rel == "child":
            child_concept_ids.append(cid)
            sc = LEARNED_CHILD_SCORE
        if hierarchical and str(m.get("concept_type") or "") == "parent_group":
            # Parent shell örnekleri yoksa atla; child user evidence esas.
            ids = example_file_ids(db_path, cid, user_only=True, customer_key=customer_key)
        else:
            ids = example_file_ids(db_path, cid, user_only=use_user, customer_key=customer_key)
        soft_only = (not hierarchical) and leaf_exact and rel in ("related", "parent")
        if soft_only or (rel == "parent" and not hierarchical):
            for fid in ids:
                neighbors[fid] = max(float(neighbors.get(fid, 0) or 0), sc)
                file_scores[fid] = max(float(file_scores.get(fid, 0) or 0), sc)
            continue
        if rel == "related" and not hierarchical and not leaf_exact:
            # Hypernym / UVI: exact listesine al (eski davranış)
            pass
        for fid in ids:
            if fid not in seen_exact:
                seen_exact.add(fid)
                exact.append(fid)
            file_scores[fid] = max(float(file_scores.get(fid, 0) or 0), sc)

    clip_cap = LEARNED_CHILD_SCORE - 0.05 if hierarchical else 1.0
    clip_sources = child_concept_ids if hierarchical and child_concept_ids else [
        int(primary.get("id") or 0)
    ]
    for src_cid in clip_sources:
        if src_cid <= 0:
            continue
        clip = clip_neighbor_scores(
            db_path,
            src_cid,
            db=db,
            faiss_store=faiss_store,
            exclude=exact,
            query=query,
        )
        for fid, sc in clip.items():
            if fid in seen_exact:
                continue
            capped = min(float(sc), clip_cap) if hierarchical else float(sc)
            neighbors[fid] = max(float(neighbors.get(fid, 0) or 0), capped)

    # Exact listeden boundary negative'leri çıkar (demote sonrası dual-positive kalmasın).
    neg_ids: set[int] = set()
    for src_cid in clip_sources:
        neg_ids |= negative_file_ids(db_path, int(src_cid or 0))
    if hierarchical:
        for cid in child_concept_ids:
            neg_ids |= negative_file_ids(db_path, cid)
    else:
        neg_ids |= negative_file_ids(db_path, int(primary.get("id") or 0))
    if neg_ids:
        exact = [fid for fid in exact if fid not in neg_ids]
        for fid in list(neighbors):
            if fid in neg_ids:
                neighbors.pop(fid, None)
        for fid in list(file_scores):
            if fid in neg_ids:
                file_scores.pop(fid, None)
    for fid, sc in neighbors.items():
        if fid not in file_scores:
            floor = _NEIGHBOR_FLOOR if not hierarchical else min(_NEIGHBOR_FLOOR, clip_cap)
            file_scores[fid] = max(float(sc), floor) if not hierarchical else float(sc)
    return {
        "concept_id": int(primary.get("id") or 0),
        "canonical": str(primary.get("canonical") or ""),
        "exact_ids": exact,
        "neighbor_scores": neighbors,
        "file_scores": file_scores,
        "relations": [(m.get("canonical"), m.get("relation")) for m in matches],
        "boundary_negative_ids": sorted(neg_ids),
        "hierarchical_parent": hierarchical,
        "customer_key": _ck_out,
    }


def tag_candidate_records(records: list[dict[str, Any]], pack: dict[str, Any] | None) -> None:
    if not pack:
        return
    exact = {int(x) for x in (pack.get("exact_ids") or [])}
    neigh = {int(k): float(v) for k, v in (pack.get("neighbor_scores") or {}).items()}
    for rec in records:
        try:
            fid = int(rec.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if fid in exact:
            rec["_learned_concept_exact"] = True
            rec["_learned_concept_clip"] = LEARNED_EXACT_SCORE
        elif fid in neigh:
            rec["_learned_concept_clip"] = neigh[fid]


def stamp_learned_result(
    result: Any, *, exact: bool, clip: float = 0.0, canonical: str = ""
) -> None:
    dbg = dict(getattr(result, "debug", {}) or {})
    dbg["learned_concept"] = True
    dbg["learned_concept_exact"] = bool(exact)
    label = str(canonical or dbg.get("learned_canonical") or "").strip()
    if label:
        dbg["learned_canonical"] = label
    if exact:
        dbg["user_taught_positive"] = True
        result.score = max(float(getattr(result, "score", 0) or 0), LEARNED_EXACT_SCORE)
    elif clip > 0:
        result.score = max(float(getattr(result, "score", 0) or 0), float(clip))
    result.score_percent = round(float(result.score) * 100, 1)
    result.debug = dbg
    reasons = list(getattr(result, "match_explanations", None) or [])
    if LEARNED_REASON not in reasons:
        reasons.insert(0, LEARNED_REASON)
    result.match_explanations = reasons[:4]
    result.text_match_reason = " · ".join(f"✓ {x}" for x in result.match_explanations[:4])


def apply_learned_to_results(
    results: list[Any],
    pack: dict[str, Any] | None,
    extra_rows: list[Any] | None = None,
) -> list[Any]:
    """Keep learned hits in the list; inject missing exact examples."""
    if not pack:
        return results
    exact = {int(x) for x in (pack.get("exact_ids") or [])}
    neigh = {int(k): float(v) for k, v in (pack.get("neighbor_scores") or {}).items()}
    canonical = str(pack.get("canonical") or "").strip()
    seen: set[int] = set()
    out: list[Any] = []
    for row in results or []:
        fid = int(getattr(row, "file_id", 0) or 0)
        if fid in exact:
            stamp_learned_result(row, exact=True, canonical=canonical)
        elif fid in neigh:
            stamp_learned_result(row, exact=False, clip=neigh[fid], canonical=canonical)
        out.append(row)
        if fid:
            seen.add(fid)
    for row in extra_rows or []:
        fid = int(getattr(row, "file_id", 0) or 0)
        if not fid or fid in seen:
            continue
        if fid in exact:
            stamp_learned_result(row, exact=True, canonical=canonical)
        elif fid in neigh:
            stamp_learned_result(row, exact=False, clip=neigh[fid], canonical=canonical)
        else:
            continue
        out.append(row)
        seen.add(fid)
    out.sort(
        key=lambda row: (
            0
            if int(getattr(row, "file_id", 0) or 0) in exact
            else 1
            if int(getattr(row, "file_id", 0) or 0) in neigh
            else 2,
            -float(getattr(row, "score", 0) or 0),
        )
    )
    return out
