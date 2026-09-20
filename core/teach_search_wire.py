"""Minimal teach → text-search wire helpers (generic; no gender hardcode)."""

from __future__ import annotations

from typing import Any


def category_path_search_tokens(path: str) -> list[str]:
    """Slash-split a category path into searchable tokens for text blobs / matching.

    Example: ``parent/leaf`` → ``["parent/leaf", "parent leaf", "parent", "leaf"]``.
    """
    raw = str(path or "").strip().strip("/").replace("\\", "/")
    if not raw:
        return []
    segments = [p.strip() for p in raw.split("/") if p.strip()]
    out: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        v = str(value or "").strip()
        if not v:
            return
        key = v.casefold()
        if key in seen:
            return
        seen.add(key)
        out.append(v)

    _add(raw)
    if len(segments) > 1:
        _add(" ".join(segments))
    for seg in segments:
        _add(seg)
    return out


def resolve_category_blob_fields(
    texture_map: dict[str, Any] | None = None,
    *,
    category_path: str = "",
    category_aliases: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Fill category_path / aliases from kwargs or texture_map fallbacks."""
    tm = texture_map if isinstance(texture_map, dict) else {}
    path = str(category_path or "").strip()
    if not path:
        path = str(
            tm.get("manual_category_path") or tm.get("category_path") or ""
        ).strip()
    aliases: list[str] = []
    if category_aliases:
        aliases = [str(x).strip() for x in category_aliases if str(x).strip()]
    elif tm.get("category_aliases"):
        raw = tm.get("category_aliases")
        if isinstance(raw, list):
            aliases = [str(x).strip() for x in raw if str(x).strip()]
        elif raw:
            aliases = [str(raw).strip()]
    return path, aliases


def category_tokens_for_blob(
    category_path: str = "",
    category_aliases: list[str] | None = None,
) -> list[str]:
    """Tokens that must appear in text_search_blob for category teach recall."""
    out: list[str] = []
    seen: set[str] = set()

    def _extend(values: list[str]) -> None:
        for v in values:
            key = v.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(v)

    _extend(category_path_search_tokens(category_path))
    for alias in category_aliases or []:
        text = str(alias or "").strip()
        if not text:
            continue
        if "/" in text or "\\" in text:
            _extend(category_path_search_tokens(text))
        else:
            _extend([text])
    return out


def keep_taught_evidence_on_gender(
    dbg: dict[str, Any] | None,
    query_tokens: list[str] | None,
) -> bool:
    """Skip gender-only demotion when result carries taught / manual category evidence.

    Checks learned_concept_exact, user_taught_positive, learned_concept, or a
    manual/category_path whose slash-split tokens intersect query_tokens.
    """
    d = dbg if isinstance(dbg, dict) else {}
    if d.get("learned_concept_exact") or d.get("user_taught_positive") or d.get(
        "learned_concept"
    ):
        return True
    tokens = [
        str(t).strip().casefold()
        for t in (query_tokens or [])
        if str(t).strip()
    ]
    if not tokens:
        return False
    path = str(
        d.get("manual_category_path") or d.get("category_path") or ""
    ).strip()
    if not path:
        return False
    path_tokens = [
        t.casefold() for t in category_path_search_tokens(path)
    ]
    path_l = path.casefold().replace("\\", "/")
    for tok in tokens:
        if tok in path_l:
            return True
        if tok in path_tokens:
            return True
        for pt in path_tokens:
            if tok and (tok in pt or pt in tok):
                return True
    return False


def run_category_path_label_search(
    conn: Any,
    terms: list[str],
    limit: int,
    *,
    normalize=None,
) -> dict[int, dict[str, Any]]:
    """SQLite search: manual_category_path / category_path contains each term.

    Generic path matching (equals, substring, slash-bounded). No gender words.
    """
    hits: dict[int, dict[str, Any]] = {}
    if normalize is None:
        try:
            from core.textile_terms import normalize_turkish as normalize
        except Exception:

            def normalize(s: str) -> str:  # type: ignore[misc]
                return str(s or "").casefold()

    needles: list[str] = []
    for raw in terms or []:
        text = str(raw or "").strip()
        if len(text) < 2:
            continue
        for candidate in (normalize(text), text.casefold(), text.lower()):
            c = str(candidate or "").strip()
            if len(c) < 2:
                continue
            if c not in needles:
                needles.append(c)
    if not needles:
        return hits

    cap = max(1, int(limit or 800))
    clauses: list[str] = []
    params: list[Any] = []
    for term in needles[:12]:
        like = f"%{term}%"
        for col in ("f.manual_category_path", "f.category_path"):
            clauses.append(f"lower(coalesce({col}, '')) = ?")
            params.append(term)
            clauses.append(f"lower(coalesce({col}, '')) LIKE ?")
            params.append(like)
            clauses.append(f"lower(coalesce({col}, '')) LIKE ?")
            params.append(f"%/{term}")
            clauses.append(f"lower(coalesce({col}, '')) LIKE ?")
            params.append(f"{term}/%")
            clauses.append(f"lower(coalesce({col}, '')) LIKE ?")
            params.append(f"%/{term}/%")

    sql = (
        "SELECT f.id FROM files f "
        "WHERE f.status NOT IN ('missing','excluded_internal') "
        "AND (f.status='indexed' OR COALESCE(f.physical_preview_ready,0)=1) "
        "AND (" + " OR ".join(clauses) + ") LIMIT ?"
    )
    params.append(cap)
    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        return hits
    for row in rows:
        try:
            fid = int(row["id"] if hasattr(row, "keys") else row[0])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if fid <= 0:
            continue
        hits[fid] = {"id": fid, "_category_path_label": True}
        if len(hits) >= cap:
            break
    return hits
