"""Thin UI formatter: real-evidence-only lines for the inspector "Neden" box.

Not a scoring/ranking engine. Reads existing SearchResult / debug fields and
returns short bullet strings. Never invents fallbacks (e.g. no
"Benzerlik skoruyla sonuç geldi"). Does not mutate the result.
"""

from __future__ import annotations

from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _debug(result: Any) -> dict[str, Any]:
    dbg = getattr(result, "debug", None)
    return dbg if isinstance(dbg, dict) else {}


def format_why_lines(result: Any) -> list[str]:
    """Return real-evidence-only explanation lines for the Neden section.

    Empty / missing evidence → empty list (caller hides or keeps collapsed).
    Never mutates ``result`` or score fields.
    """
    if result is None:
        return []

    dbg = _debug(result)
    lines: list[str] = []
    seen: set[str] = set()

    def add(line: str) -> None:
        text = " ".join(str(line or "").split()).strip()
        if not text:
            return
        key = text.casefold()
        if key in seen:
            return
        seen.add(key)
        lines.append(text)

    qm = _as_dict(dbg.get("query_meaning"))

    concept = str(qm.get("concept_core") or "").strip()
    if concept:
        add(f"Kavram: {concept}")

    colors = qm.get("colors")
    if isinstance(colors, (list, tuple)):
        color_txt = ", ".join(
            str(c).strip() for c in colors if str(c or "").strip()
        )
        if color_txt:
            add(f"Renk: {color_txt}")

    customer = str(qm.get("customer") or "").strip()
    if customer:
        add(f"Müşteri: {customer}")

    search_reason = str(dbg.get("search_reason") or "").strip()
    # Never surface the pattern_explanation fake fallback if it leaked in.
    _FAKE = "Benzerlik skoruyla sonuç geldi"
    if search_reason == _FAKE:
        search_reason = ""
    elif search_reason.endswith(_FAKE):
        search_reason = search_reason[: -len(_FAKE)].rstrip(" .")
    if search_reason:
        add(search_reason)

    ce = _as_dict(dbg.get("concept_evidence"))
    if ce:
        canonical = str(ce.get("canonical") or "").strip()
        parts = _as_dict(ce.get("parts"))
        matched_parts = [
            name
            for name, meta in parts.items()
            if isinstance(meta, dict) and meta.get("match")
        ]
        if ce.get("aligned") or matched_parts or (
            ce.get("applied") and abs(float(ce.get("delta") or 0)) > 1e-9
        ):
            if canonical and matched_parts:
                add(f"Kavram kanıtı: {canonical} ({', '.join(matched_parts)})")
            elif canonical:
                add(f"Kavram kanıtı: {canonical}")
            elif matched_parts:
                add(f"Kavram kanıtı: {', '.join(matched_parts)}")
            else:
                add("Kavram kanıtı")

        soft = ce.get("customer_soft_bonus")
        try:
            soft_f = float(soft) if soft is not None else 0.0
        except (TypeError, ValueError):
            soft_f = 0.0
        if soft is not None and soft_f > 0:
            add(f"Müşteri soft bonus: +{soft_f:.2f}")

    ces = _as_dict(dbg.get("color_evidence_score"))
    try:
        hits = int(ces.get("hits") or 0)
    except (TypeError, ValueError):
        hits = 0
    if hits > 0:
        want = ces.get("want") or []
        if isinstance(want, (list, tuple)) and want:
            add(f"Renk kanıtı: {', '.join(str(w) for w in want[:4])} (hit)")
        else:
            add("Renk kanıtı: eşleşme")

    if getattr(result, "same_pattern_family", False) or dbg.get("same_family"):
        add("Aynı desen ailesi")

    animal = str(
        getattr(result, "animal_print_type", None)
        or dbg.get("animal_print_type")
        or ""
    ).strip()
    if animal and animal.lower() not in {"", "unknown", "none", "—", "-"}:
        if getattr(result, "same_animal_family", False) or dbg.get(
            "same_animal_family"
        ):
            add(f"Hayvan deseni: {animal}")
        elif concept and animal.casefold() in concept.casefold():
            add(f"Hayvan deseni: {animal}")
        elif getattr(result, "same_pattern_family", False):
            add(f"Hayvan deseni: {animal}")

    qev = _as_dict(dbg.get("query_evidence_report"))
    visual_verdict = str(
        dbg.get("visual_verdict") or qev.get("visual_verdict") or ""
    ).strip()
    visual_grade = str(
        dbg.get("visual_grade") or qev.get("visual_grade") or ""
    ).strip()
    if visual_verdict:
        add(f"Görsel eşleşme: {visual_verdict}")
    elif visual_grade:
        add(f"Görsel eşleşme: {visual_grade}")

    return lines


def format_why_html(result: Any) -> str:
    """HTML for the Neden label; empty string when no real evidence."""
    lines = format_why_lines(result)
    if not lines:
        return ""
    return "<br>".join(f"• {line}" for line in lines)
