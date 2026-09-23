"""Intent-aware visual evidence routing — soft layer on existing search.

Query text answers: \"which evidence in this image am I looking for?\"
Does not replace DINO/CLIP/FAISS/Pattern DNA. Additive only.
Visual-only searches (no text) are left untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.query_intent_router import QueryIntent, classify_query
from core.textile_terms import normalize_turkish

# Map router channels → evidence classes used for routing weights.
_CHANNEL_TO_EVIDENCE = {
    "person": "person",
    "pattern": "pattern",
    "object": "object",
    "motif_object": "motif",
    "material_color": "color",
    "category": "product",
    "texture_style": "texture",
    "brand": "brand",
    "learned": "learned",
}

# Soft deltas — below Exact/OCR; never wipe other channels.
W_PRIMARY = 0.10
W_SECONDARY = 0.05
W_MULTI = 0.04
W_PATTERN_ONLY_WHEN_PERSON = -0.06  # soft; fabric-only vs person intent
W_PERSON_ONLY_WHEN_PATTERN = -0.05


@dataclass(frozen=True)
class IntentEvidencePlan:
    raw: str = ""
    wanted: frozenset[str] = field(default_factory=frozenset)
    values: dict[str, str] = field(default_factory=dict)
    compound: bool = False
    intent: QueryIntent | None = None

    def wants(self, *kinds: str) -> bool:
        return bool(self.wanted.intersection(kinds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "wanted": sorted(self.wanted),
            "values": dict(self.values),
            "compound": self.compound,
            "intent": self.intent.to_dict() if self.intent else None,
        }


def _garment_product_tokens() -> set[str]:
    """Reuse existing filename/family stems — not a new product engine."""
    out: set[str] = set()
    try:
        from core.texture_profile import FAMILY_FILENAME_HINTS

        for tok in FAMILY_FILENAME_HINTS.get("garment_photo", ()):
            out.add(normalize_turkish(str(tok)))
    except Exception:
        pass
    # Turkish garment forms commonly left as residual by the router.
    for tok in (
        "gomlek",
        "gömlek",
        "shirt",
        "elbise",
        "dress",
        "etek",
        "pantolon",
        "ceket",
        "jacket",
        "tisort",
        "tişört",
        "tshirt",
    ):
        out.add(normalize_turkish(tok))
    return out


_GARMENT_TOKENS = _garment_product_tokens()


def _strip_adj_suffix(tok: str) -> str:
    """Strip Turkish -lı/-li/-lu/-lü adjective endings (keep diacritics when present)."""
    t = str(tok or "").strip().lower()
    for suf in ("lü", "lı", "lu", "li"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[: -len(suf)]
    n = normalize_turkish(t)
    for suf in ("lu", "li"):
        if n.endswith(suf) and len(n) > len(suf) + 2:
            # Prefer returning ASCII stem only when input had no diacritics.
            if n == t:
                return n[: -len(suf)]
    return t


def _hint_object_from_token(tok: str) -> str:
    """Reuse textile/KB family hints — empty if unknown."""
    try:
        from core.textile_terms import query_family_hints

        for candidate in (tok, _strip_adj_suffix(tok), normalize_turkish(tok)):
            if not candidate:
                continue
            h = query_family_hints(candidate) or {}
            ptype = normalize_turkish(str(h.get("pattern_type") or ""))
            if ptype in {"sunglasses", "gozluk", "glasses"}:
                return "sunglasses"
    except Exception:
        pass
    return ""


def build_intent_evidence_plan(
    text: str, db_path: str | None = None
) -> IntentEvidencePlan:
    raw = " ".join((text or "").strip().split())
    if not raw:
        return IntentEvidencePlan()
    intent = classify_query(raw, db_path=db_path)
    wanted: set[str] = set()
    values: dict[str, str] = {}

    if intent.kind == "gender" or intent.channel == "person":
        wanted.add("person")
        values["person"] = str(intent.value or "person")

    for ch in intent.channels:
        ev = _CHANNEL_TO_EVIDENCE.get(ch.channel)
        if not ev:
            continue
        wanted.add(ev)
        values.setdefault(ev, str(ch.value or ch.token or ""))

    # Enrich from existing NL / textile hints (no new synonym tables).
    try:
        from core.natural_language_query import parse_natural_query

        pq = parse_natural_query(raw)
        if getattr(pq, "color", None):
            wanted.add("color")
            values.setdefault("color", str(pq.color))
        if getattr(pq, "motif", None):
            motif = normalize_turkish(str(pq.motif))
            if motif in {"leopard", "zebra", "snake", "tiger", "floral", "geometric"}:
                wanted.add("pattern")
                values.setdefault("pattern", motif)
            else:
                wanted.add("motif")
                values.setdefault("motif", motif)
    except Exception:
        pass

    try:
        from core.textile_terms import query_family_hints

        hints = query_family_hints(raw) or {}
        fam = str(hints.get("pattern_family") or "")
        apt = str(hints.get("animal_print_type") or "")
        ptype = normalize_turkish(str(hints.get("pattern_type") or ""))
        if fam == "animal_print" or apt:
            wanted.add("pattern")
            values.setdefault("pattern", apt or "animal_print")
        if fam == "floral":
            wanted.add("pattern")
            values.setdefault("pattern", "floral")
        if ptype in {"sunglasses", "gozluk", "glasses"}:
            wanted.add("object")
            values.setdefault("object", "sunglasses")
    except Exception:
        pass

    # Residual tokens → product (garment stems) or re-classify after adj strip.
    import re as _re

    raw_toks = [
        x
        for x in _re.findall(r"[a-zA-Z0-9ğüşöçıİĞÜŞÖÇ]+", raw.lower())
        if len(x) >= 2
    ]
    residual_pool = list(intent.unknown_tokens) + [
        t for t in raw_toks if normalize_turkish(t) in {
            normalize_turkish(u) for u in intent.unknown_tokens
        }
    ]
    seen_forms: set[str] = set()
    for unk in residual_pool:
        form_key = str(unk).strip().lower()
        if not form_key or form_key in seen_forms:
            continue
        seen_forms.add(form_key)
        u = normalize_turkish(form_key)
        stem = normalize_turkish(_strip_adj_suffix(str(unk)))
        stem_raw = _strip_adj_suffix(str(unk))
        if u in _GARMENT_TOKENS or stem in _GARMENT_TOKENS:
            wanted.add("product")
            values.setdefault("product", u)
            continue
        obj = _hint_object_from_token(str(unk)) or _hint_object_from_token(stem_raw)
        if obj:
            wanted.add("object")
            values.setdefault("object", obj)
            continue
        if stem != u or stem_raw != str(unk):
            sub = classify_query(stem_raw or stem, db_path=db_path)
            for ch in sub.channels:
                ev = _CHANNEL_TO_EVIDENCE.get(ch.channel)
                if ev:
                    wanted.add(ev)
                    values.setdefault(ev, str(ch.value or ch.token or ""))
            if sub.kind == "gender" or sub.channel == "person":
                wanted.add("person")
                values.setdefault("person", str(sub.value or "person"))

    return IntentEvidencePlan(
        raw=raw,
        wanted=frozenset(wanted),
        values=values,
        compound=len(wanted) >= 2,
        intent=intent,
    )


def collect_candidate_evidence(result: Any) -> dict[str, Any]:
    """Read-only evidence from existing result fields. No invented confidence."""
    dbg = getattr(result, "debug", None) or {}
    if not isinstance(dbg, dict):
        dbg = {}
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    if not tm:
        bd_tm = getattr(result, "breakdown", None) or {}
        if isinstance(bd_tm, dict) and isinstance(bd_tm.get("texture_map"), dict):
            tm = bd_tm["texture_map"]
    fam = normalize_turkish(
        str(
            getattr(result, "pattern_family", "")
            or dbg.get("pattern_family")
            or tm.get("pattern_family")
            or ""
        )
    )
    apt = normalize_turkish(
        str(
            getattr(result, "animal_print_type", "")
            or dbg.get("animal_print_type")
            or tm.get("animal_print_type")
            or ""
        )
    )
    path = normalize_turkish(
        str(getattr(result, "path", "") or "")
        + " "
        + str(getattr(result, "filename", "") or "")
        + " "
        + str(getattr(result, "category_path", "") or dbg.get("category_path") or "")
    )
    ev: dict[str, Any] = {}

    if (
        dbg.get("face_match")
        or dbg.get("face_gender_match")
        or dbg.get("human_semantic_only")
        or dbg.get("human_semantic_mode")
        or float(dbg.get("gender_visual_score") or 0) >= 0.18
        or float(dbg.get("human_semantic_score") or 0) >= 0.18
    ):
        ev["person"] = True
        gv = str(dbg.get("face_gender_value") or "")
        if gv:
            ev["person_value"] = gv
    if fam == "garment_photo":
        ev["person"] = True  # worn garment photo often implies person present
        ev["product"] = True

    if fam == "animal_print" or apt:
        ev["pattern"] = True
        if apt:
            ev["pattern_value"] = apt
    if fam == "floral":
        ev["pattern"] = True
        ev["pattern_value"] = "floral"
    if fam and fam not in {
        "",
        "unknown",
        "plain",
        "texture_ground",
        "garment_photo",
        "document",
        "icon_logo_non_textile",
    }:
        ev.setdefault("pattern", True)
        ev.setdefault("pattern_value", fam)

    # Object / bag / glasses — filename/category/path only when explicit.
    if any(t in path for t in ("canta", "bag", "handbag", "clutch")):
        ev["object"] = True
        ev["object_value"] = "handbag"
    if any(t in path for t in ("gozluk", "sunglass", "glasses", "eyewear")):
        ev["object"] = True
        ev["object_value"] = "sunglasses"
    if any(t in path for t in _GARMENT_TOKENS):
        ev["product"] = True

    # Learned / concept channel hits already on the result.
    if dbg.get("concept_exact") or dbg.get("learned_exact") or dbg.get("visual_concept_hit"):
        ev["learned"] = True

    return ev


def apply_intent_evidence_routing(
    results: list[Any],
    query_text: str,
    *,
    has_image: bool = True,
    db_path: str | None = None,
) -> list[Any]:
    """Soft reweight by query intent. No-op when text empty (visual-only)."""
    text = " ".join((query_text or "").strip().split())
    if not text or not results:
        return results
    plan = build_intent_evidence_plan(text, db_path=db_path)
    if not plan.wanted:
        return results

    person_q = plan.wants("person")
    pattern_q = plan.wants("pattern", "motif")
    object_q = plan.wants("object")
    product_q = plan.wants("product")
    color_q = plan.wants("color")

    for r in results:
        if getattr(r, "is_self_match", False):
            continue
        dbg = getattr(r, "debug", None)
        if not isinstance(dbg, dict):
            dbg = {}
            r.debug = dbg
        if dbg.get("protected_exact"):
            continue

        ev = collect_candidate_evidence(r)
        path_n = normalize_turkish(
            str(getattr(r, "path", "") or "")
            + " "
            + str(getattr(r, "filename", "") or "")
            + " "
            + str(getattr(r, "category_path", "") or dbg.get("category_path") or "")
        )
        matched: list[str] = []
        delta = 0.0

        def _hit(kind: str) -> bool:
            return bool(ev.get(kind))

        primary_kinds = []
        if person_q:
            primary_kinds.append("person")
        if pattern_q:
            primary_kinds.append("pattern")
        if object_q:
            primary_kinds.append("object")
        if product_q:
            primary_kinds.append("product")
        if color_q:
            primary_kinds.append("color")
        if plan.wants("learned"):
            primary_kinds.append("learned")

        # Primary matches.
        for i, kind in enumerate(primary_kinds):
            ok = _hit(kind)
            if kind == "pattern" and not ok:
                ok = _hit("motif")
            if kind == "learned":
                lv = normalize_turkish(str(plan.values.get("learned") or ""))
                ok = bool(lv and lv in path_n) or _hit("learned") or _hit("product")
            if not ok:
                continue
            # Value alignment when both sides have a value.
            qv = normalize_turkish(str(plan.values.get(kind) or ""))
            cv = normalize_turkish(
                str(
                    ev.get(f"{kind}_value")
                    or ev.get("pattern_value")
                    or ""
                )
            )
            if qv and cv and qv != cv and kind in {"pattern", "object"}:
                # Related family still counts as secondary.
                delta += W_SECONDARY * 0.5
                matched.append(f"~{kind}")
                continue
            w = W_PRIMARY if i == 0 and not plan.compound else W_SECONDARY
            if plan.compound:
                w = W_SECONDARY
            delta += w
            matched.append(kind)

        if plan.compound and len(matched) >= 2:
            delta += W_MULTI

        # Soft conflict: person intent + fabric-only animal print siblings.
        if person_q and not pattern_q and ev.get("pattern") and not ev.get("person"):
            delta += W_PATTERN_ONLY_WHEN_PERSON
            dbg["intent_conflict"] = "pattern_without_person"
        if pattern_q and not person_q and ev.get("person") and not ev.get("pattern"):
            delta += W_PERSON_ONLY_WHEN_PATTERN
            dbg["intent_conflict"] = "person_without_pattern"
        if object_q and not pattern_q and ev.get("pattern") and not ev.get("object"):
            delta += W_PATTERN_ONLY_WHEN_PERSON * 0.8
            dbg["intent_conflict"] = "pattern_without_object"

        if abs(delta) < 1e-9 and not matched:
            dbg["intent_evidence_plan"] = plan.to_dict()
            dbg["candidate_evidence"] = {k: v for k, v in ev.items() if v}
            continue

        old = float(getattr(r, "score", 0) or 0)
        new = max(0.0, min(1.0, old + delta))
        r.score = new
        if hasattr(r, "score_percent"):
            r.score_percent = round(new * 100, 1)
        bd = getattr(r, "breakdown", None)
        if isinstance(bd, dict):
            bd["intent_evidence_delta"] = round(delta, 4)
        dbg["intent_evidence_plan"] = plan.to_dict()
        dbg["candidate_evidence"] = {k: v for k, v in ev.items() if v}
        dbg["intent_evidence_matched"] = matched
        dbg["intent_evidence_delta"] = round(delta, 4)
        dbg["intent_routing"] = True

    return results


def should_suppress_pattern_first(plan: IntentEvidencePlan) -> bool:
    """When user asks for person/object only, do not lift animal-print fabric peers."""
    if not plan.wanted:
        return False
    if plan.wants("pattern", "motif"):
        return False
    return plan.wants("person", "object", "product")
