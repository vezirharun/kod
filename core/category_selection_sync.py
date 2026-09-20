"""Pure helpers: split path-qualified category UI selections into parent/child + tags.

Used by ResultMetadataDialog (Teach/Edit) so semantic catalog displays like
``insan / erkek`` do not corrupt identity when values() builds category_path.
No synonym invention; no hardcodes for specific labels.
"""
from __future__ import annotations


def _segments(text: str) -> list[str]:
    """Split on '/' and strip each segment; drop empty segments."""
    raw = str(text or "")
    if not raw.strip():
        return []
    return [p.strip() for p in raw.split("/") if p.strip()]


def split_category_selection(text: str) -> tuple[str, str]:
    """Split UI selection into (parent, child).

    Accepts ``insan/erkek``, ``insan / erkek``, multi-level ``a/b/c`` →
    parent=``a``, child=``b/c``. Parent-only → (parent, '').
    Strip whitespace around segments. Do not invent synonyms.
    """
    parts = _segments(text)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], "/".join(parts[1:])


def canonical_path_from_parts(parent: str, child: str) -> str:
    """parent/child if child else parent. Preserve segment text (no forced titlecase)."""
    p = str(parent or "").strip()
    c = str(child or "").strip()
    if not p:
        return c
    if not c:
        return p
    return f"{p}/{c}"


def path_segment_tags(parent: str, child: str) -> list[str]:
    """Slash-split parent+child into ordered unique segment tags.

    Preserve casing of segments. child may itself contain '/'. No synonyms.
    """
    seen: set[str] = set()
    out: list[str] = []
    for seg in _segments(parent) + _segments(child):
        key = seg.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
    return out


def merge_tags_preserve_manual(existing: list[str], path_tags: list[str]) -> list[str]:
    """Append path_tags not already present (casefold match). Never remove existing."""
    out: list[str] = []
    seen: set[str] = set()
    for t in list(existing or []):
        s = str(t or "").strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    for t in list(path_tags or []):
        s = str(t or "").strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _lookup_known(name: str, options: list[str] | None) -> str | None:
    """Return option with matching casefold key, else None."""
    needle = str(name or "").strip()
    if not needle or not options:
        return None
    nk = needle.casefold()
    for opt in options:
        o = str(opt or "").strip()
        if o and o.casefold() == nk:
            return o
    return None


def resolve_parts_against_options(
    parent: str,
    child: str,
    *,
    parents: list[str] | None = None,
    children_by_parent: dict[str, list[str]] | None = None,
) -> tuple[str, str]:
    """If options provided, prefer known casing from parents / children_by_parent.

    Key-tolerant (casefold). Else return parent, child unchanged. Still no hardcodes.
    """
    p = str(parent or "").strip()
    c = str(child or "").strip()
    known_p = _lookup_known(p, parents)
    if known_p is not None:
        p = known_p

    by_parent = children_by_parent or {}
    kids: list[str] | None = None
    if p and p in by_parent:
        kids = list(by_parent.get(p) or [])
    elif p and by_parent:
        pk = p.casefold()
        for key, vals in by_parent.items():
            if str(key or "").strip().casefold() == pk:
                kids = list(vals or [])
                # Also prefer this parent's casing if we didn't already
                if known_p is None:
                    p = str(key).strip()
                break

    if c and kids is not None:
        # child may be multi-segment (b/c); resolve leaf-by-leaf when possible,
        # but prefer exact full-string match in kids first.
        known_c = _lookup_known(c, kids)
        if known_c is not None:
            c = known_c
        elif "/" in c:
            # Resolve each segment independently against kids only for first segment;
            # remaining path kept as-is (no invention).
            first, _, rest = c.partition("/")
            known_first = _lookup_known(first.strip(), kids)
            if known_first is not None:
                c = f"{known_first}/{rest.strip()}" if rest.strip() else known_first

    return p, c


def sync_selection_state(
    selection_text: str,
    *,
    existing_tags: list[str] | None = None,
    parents: list[str] | None = None,
    children_by_parent: dict | None = None,
    child_text: str | None = None,
) -> dict:
    """Returns {parent, child, category_path, tags} ready for UI/values.

    ``selection_text`` is typically cmb_parent current text (may be path-qualified).
    If ``child_text`` is provided and selection has no path sep, use it as child.
    When selection is path-qualified, it wins over child_text for the split.
    """
    text = str(selection_text or "").strip()
    # Detect path-qualified parent (slash with optional spaces).
    has_path = "/" in text
    if has_path:
        parent, child = split_category_selection(text)
    else:
        parent = text
        child = str(child_text or "").strip()
        # Child itself may be path-like leftover; leave as-is (no invent).

    parent, child = resolve_parts_against_options(
        parent,
        child,
        parents=parents,
        children_by_parent=children_by_parent,
    )
    path = canonical_path_from_parts(parent, child)
    path_tags = path_segment_tags(parent, child)
    tags = merge_tags_preserve_manual(list(existing_tags or []), path_tags)
    return {
        "parent": parent,
        "child": child,
        "category_path": path,
        "tags": tags,
    }
