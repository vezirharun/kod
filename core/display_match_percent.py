"""Display-only card match percent.

Never used as a ranking input. Does not write result.score / score_percent.
"""

from __future__ import annotations

from typing import Any

DISPLAY_MATCH_TOOLTIP = "Sıralama kalitesi (bu liste), CLIP benzerliği değil."
_DISPLAY_LO = 70.0
_DISPLAY_HI = 99.0


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def display_match_percent(result: Any) -> float:
    dbg = getattr(result, "debug", None) or {}
    raw = dbg.get("display_match_percent")
    if raw is not None:
        return float(raw)
    return float(getattr(result, "score_percent", 0.0) or 0.0)


def format_display_match_label(result: Any, rank: int = 0) -> str:
    text = f"Eşleşme %{display_match_percent(result):.0f}"
    if int(rank or 0) > 0:
        return f"#{int(rank)} · {text}"
    return text


def display_sort_tuple(result: Any) -> tuple:
    """DNA-first tuple for display compression only (not a sort)."""
    dbg = getattr(result, "debug", None) or {}
    bd = getattr(result, "breakdown", None) or {}
    sig = dbg.get("pattern_visual_signals") or {}
    visual_only = 0 if (dbg.get("pattern_visual_positive") or sig.get("positive")) else 1
    negative = 1 if (dbg.get("pattern_visual_negative") or sig.get("negative")) else 0
    dna = _f(bd.get("dna_score") or bd.get("pattern_dna_score") or sig.get("dna"))
    family = _f(bd.get("family_score") or sig.get("family"))
    texture = _f(bd.get("texture_score") or sig.get("texture"))
    dino = _f(dbg.get("dino_score") or bd.get("dino") or sig.get("dino"))
    clip = _f(dbg.get("clip_score") or bd.get("clip") or sig.get("clip"))
    return (visual_only, negative, -dna, -family, -texture, -dino, -clip)


def _tuple_to_scalar(key: tuple) -> float:
    visual_only, negative, neg_dna, neg_family, neg_tex, neg_dino, neg_clip = key
    return (
        (1 - int(visual_only)) * 1_000_000.0
        + (1 - int(negative)) * 100_000.0
        + (-float(neg_dna)) * 10_000.0
        + (-float(neg_family)) * 1_000.0
        + (-float(neg_tex)) * 100.0
        + (-float(neg_dino)) * 10.0
        + (-float(neg_clip))
    )


def merge_results_in_engine_order(full: list[Any], incoming: list[Any]) -> list[Any]:
    """Keep incoming engine file_id sequence, then leftover rows. Never sort by percent."""
    if not incoming:
        return list(full or [])
    seen: set[int] = set()
    out: list[Any] = []
    for row in incoming:
        fid = int(getattr(row, "file_id", 0) or 0)
        if fid in seen:
            continue
        out.append(row)
        seen.add(fid)
    for row in full or []:
        fid = int(getattr(row, "file_id", 0) or 0)
        if fid in seen:
            continue
        out.append(row)
        seen.add(fid)
    return out


def stamp_display_match_percent(results: list[Any]) -> list[Any]:
    """Write debug display_match_percent from current list order. Does not reorder."""
    rows = list(results or [])
    n = len(rows)
    if n == 0:
        return rows
    scalars = [_tuple_to_scalar(display_sort_tuple(r)) for r in rows]
    lo = min(scalars)
    hi = max(scalars)
    span = hi - lo
    if span <= 1e-12:
        scaled = [_DISPLAY_HI] * n
    else:
        scaled = [
            _DISPLAY_LO + (s - lo) / span * (_DISPLAY_HI - _DISPLAY_LO) for s in scalars
        ]
    isotonic: list[float] = []
    for i, raw in enumerate(scaled):
        pct = raw if i == 0 else min(raw, isotonic[-1])
        isotonic.append(round(pct, 1))
    for row, pct in zip(rows, isotonic):
        dbg = dict(getattr(row, "debug", None) or {})
        dbg["display_match_percent"] = pct
        row.debug = dbg
    return rows
