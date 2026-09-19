"""Search intelligence unification — orchestrate existing modules (no new engine).

Single decision surface for query meaning → soft re-rank layers → result reasons.
Does not rewrite index/DINO/CLIP/DNA/learning. Avoids double scoring.
"""
from __future__ import annotations

from typing import Any

from core.query_attribute_intel import extract_query_attributes
from core.textile_terms import normalize_turkish


def normalize_query_text(query_text: str) -> str:
    """Case / TR character / spacing normalize (existing textile_terms)."""
    return normalize_turkish(str(query_text or "").strip())


def analyze_query_intelligence(
    query_text: str,
    *,
    customer_registry: Any = None,
) -> dict[str, Any]:
    """Unified QUERY MEANING snapshot (read-only). Concept ≠ attributes ≠ context.

    NL 2.0: optional customer/folder scope via existing fuzzy discovery; concept
    parse runs on customer-stripped search_text so scope tokens do not pollute
    concept identity. Ambiguous customer → no forced filter.
    """
    raw = str(query_text or "").strip()
    norm = normalize_query_text(raw)
    out: dict[str, Any] = {
        "raw": raw,
        "normalized": norm,
        "search_text": raw,
        "concept_core": "",
        "concept_relation_hint": "",
        "intent_type": "pattern",
        "context": "none",
        "visual_type": "general",
        "attributes": {},
        "colors": [],
        "customer": "",
        "customer_score": 0.0,
        "customer_high_confidence": False,
        "customer_ambiguous": False,
        "scopes": [],
        "confidence": 0.0,
        "nl_parse": {},
        "authority": "query",
    }
    if not raw:
        return out

    # --- Customer / folder scope (existing discovery; no invention) ---
    cust_info: dict[str, Any] = {}
    try:
        from core.customer_discovery import extract_customer_from_query

        reg = customer_registry
        if reg is None:
            try:
                from core.customer_discovery import get_customer_registry

                reg = get_customer_registry()
            except Exception:
                reg = None
        cust_info = extract_customer_from_query(raw, reg)
    except Exception:
        cust_info = {}

    search_text = str(cust_info.get("search_text") or raw).strip() or raw
    out["search_text"] = search_text
    out["customer"] = str(cust_info.get("customer") or "")
    out["customer_score"] = float(cust_info.get("score") or 0.0)
    out["customer_high_confidence"] = bool(cust_info.get("high_confidence"))
    out["customer_ambiguous"] = bool(cust_info.get("ambiguous"))
    out["scopes"] = list(cust_info.get("scope_prefixes") or [])

    # Attribute / concept parse on search_text (customer tokens removed when confident)
    attrs = extract_query_attributes(search_text)
    out["attributes"] = {
        "motif": attrs.motif,
        "scale": attrs.scale,
        "density": attrs.density,
        "colors": list(attrs.colors),
        "repeat": attrs.repeat,
        "orientation": attrs.orientation,
        "style": attrs.style,
        "pattern_type": attrs.pattern_type,
    }
    out["colors"] = list(attrs.colors)

    try:
        from core.query_attribute_intel import (
            concept_core_text,
            extract_query_visual_context,
        )

        vctx = extract_query_visual_context(search_text)
        out["context"] = str(vctx.get("context") or "none")
        out["visual_type"] = str(vctx.get("visual_type") or "general")
        out["concept_core"] = (
            str(vctx.get("concept_core") or "")
            or concept_core_text(search_text)
            or attrs.motif
            or search_text
        )
    except Exception:
        out["concept_core"] = attrs.motif or search_text

    # Intent from generic CONTEXT (not concept-specific rules).
    # Bare concept (context=none) stays "general" even if NL fills motif.
    ctx = out["context"]
    if ctx == "photo":
        out["intent_type"] = "photo"
    elif ctx == "object":
        out["intent_type"] = "object"
    elif ctx == "texture":
        out["intent_type"] = "texture"
    elif ctx == "pattern":
        out["intent_type"] = "pattern"
    elif attrs.style or attrs.pattern_type:
        out["intent_type"] = "pattern"
    else:
        out["intent_type"] = "general"

    try:
        from core.concept_query_normalize import relation_to_concept

        for can, aliases in (
            ("Tiger", ["kaplan", "tiger", "Tiger"]),
            ("Leopard", ["leopar", "leopard", "Leopard"]),
            ("Zebra", ["zebra", "Zebra"]),
            ("Snake Skin", ["snake skin", "yilan derisi", "yılan derisi"]),
            ("Snake", ["yilan", "yılan", "snake"]),
            ("Floral", ["cicek", "çiçek", "flower", "floral"]),
            ("Rose", ["gul", "gül", "rose"]),
        ):
            rel = relation_to_concept(search_text, can, aliases=aliases)
            if rel:
                out["concept_relation_hint"] = f"{can}:{rel}"
                break
    except Exception:
        pass

    # Compact NL debug (existing debug surface — no heavy UI)
    try:
        from core.natural_language_query import parse_natural_query

        parsed = parse_natural_query(search_text)
        out["nl_parse"] = {
            **(parsed.to_dict() if parsed else {}),
            "customer": out["customer"],
            "customer_score": out["customer_score"],
            "customer_high_confidence": out["customer_high_confidence"],
            "customer_ambiguous": out["customer_ambiguous"],
            "matched_span": cust_info.get("matched_span") or "",
            "search_text": search_text,
            "intent_type": out["intent_type"],
            "context": out["context"],
            "concept_core": out["concept_core"],
        }
    except Exception:
        out["nl_parse"] = {
            "customer": out["customer"],
            "search_text": search_text,
            "intent_type": out["intent_type"],
            "concept_core": out["concept_core"],
        }

    # Confidence: mean of available strong signals (customer / motif / context)
    conf_bits: list[float] = []
    if out["customer_high_confidence"]:
        conf_bits.append(min(1.0, float(out["customer_score"])))
    elif out["customer_ambiguous"]:
        conf_bits.append(0.35)
    if attrs.motif:
        conf_bits.append(0.85)
    if out["context"] != "none":
        conf_bits.append(0.8)
    if attrs.colors:
        conf_bits.append(0.75)
    if attrs.scale:
        conf_bits.append(0.7)
    out["confidence"] = round(sum(conf_bits) / len(conf_bits), 4) if conf_bits else 0.4
    return out


def build_result_reason(
    result: Any,
    analysis: dict[str, Any] | None = None,
) -> str:
    """Compact internal reason label (debug only; UI optional)."""
    dbg = getattr(result, "debug", None) or {}
    if not isinstance(dbg, dict):
        dbg = {}
    analysis = analysis or {}

    if getattr(result, "is_self_match", False) or dbg.get("is_self_match"):
        return "EXACT"

    parts: list[str] = []
    can = str(
        dbg.get("learned_canonical")
        or analysis.get("concept_core")
        or getattr(result, "animal_print_type", "")
        or ""
    ).strip()
    can_u = can.upper().replace("_", " ") if can else ""

    if dbg.get("learned_concept_exact"):
        parts.append(f"LEARNED {can_u or 'CONCEPT'}")
    elif dbg.get("user_taught_positive"):
        parts.append(f"USER {can_u or 'TAUGHT'}")
    elif dbg.get("learned_concept"):
        parts.append(f"CONCEPT {can_u or 'MATCH'}")

    # Attribute / color hits (same concept only — already enforced in layers)
    qa = dbg.get("query_attribute_intel") if isinstance(dbg.get("query_attribute_intel"), dict) else {}
    matches = qa.get("matches") if isinstance(qa.get("matches"), dict) else {}
    colors = list((analysis or {}).get("colors") or [])
    if not colors:
        attrs = qa.get("attrs") if isinstance(qa.get("attrs"), dict) else {}
        colors = list(attrs.get("colors") or [])
    ces = dbg.get("color_evidence_score") if isinstance(dbg.get("color_evidence_score"), dict) else {}
    color_hits = list(ces.get("want") or []) if float(ces.get("hits") or 0) > 0 else []
    if matches.get("color", 0) and matches.get("color", 0) > 0 and colors:
        for c in colors[:3]:
            parts.append(f"{can_u or 'CONCEPT'} + {str(c).upper()}" if can_u else str(c).upper())
            break
        if len(colors) > 1:
            parts.append(f"{str(colors[1]).upper()}")
    elif color_hits:
        parts.append(f"{can_u or 'COLOR'} + {str(color_hits[0]).upper()}")

    if matches.get("scale", 0) and matches.get("scale", 0) > 0:
        sc = (qa.get("attrs") or {}).get("scale") or ""
        if sc:
            parts.append(str(sc).upper())
    if matches.get("density", 0) and matches.get("density", 0) > 0:
        dens = (qa.get("attrs") or {}).get("density") or ""
        if dens:
            parts.append(str(dens).upper())

    vv = dbg.get("visual_variant_intel") if isinstance(dbg.get("visual_variant_intel"), dict) else {}
    if vv.get("is_same_concept_variant"):
        parts.append("VARIANT")
    elif dbg.get("same_family") or dbg.get("result_layer") == "same_family":
        parts.append("SAME FAMILY")
    elif dbg.get("result_layer") in {"similar", "visual_similar"}:
        parts.append("VISUAL SIMILAR")

    og = dbg.get("object_pattern_gate") if isinstance(dbg.get("object_pattern_gate"), dict) else {}
    if og.get("reason") == "object_photo_over_pattern" or float(og.get("delta") or 0) < -0.01:
        parts.append("OBJECT PHOTO DEMOTED")

    if not parts:
        if can_u:
            return f"CONCEPT {can_u}"
        return "VISUAL SIMILAR"
    # Dedupe preserve order
    seen: set[str] = set()
    out_parts: list[str] = []
    for p in parts:
        key = p.strip().upper()
        if key and key not in seen:
            seen.add(key)
            out_parts.append(p.strip())
    return " | ".join(out_parts[:5])


def attach_result_reasons(
    results: list[Any],
    analysis: dict[str, Any],
) -> list[Any]:
    """Write debug.search_reason for every row (no score change)."""
    for rec in results:
        dbg = dict(getattr(rec, "debug", None) or {})
        dbg["search_reason"] = build_result_reason(rec, analysis)
        dbg["query_meaning"] = {
            "concept_core": analysis.get("concept_core"),
            "colors": list(analysis.get("colors") or []),
            "attributes": analysis.get("attributes") or {},
            "intent_type": analysis.get("intent_type"),
            "context": analysis.get("context"),
            "visual_type": analysis.get("visual_type"),
            "normalized": analysis.get("normalized"),
            "search_text": analysis.get("search_text"),
            "customer": analysis.get("customer") or "",
            "customer_high_confidence": bool(
                analysis.get("customer_high_confidence")
            ),
            "scopes": list(analysis.get("scopes") or []),
            "confidence": analysis.get("confidence"),
            "nl_parse": analysis.get("nl_parse") or {},
        }
        rec.debug = dbg
    return results


def apply_search_intelligence_chain(
    results: list[Any],
    query_text: str,
    *,
    index_db: str = "",
    analysis: dict[str, Any] | None = None,
    customer_registry: Any = None,
    customer: str = "",
    skip_attributes: bool = False,
    skip_object_gate: bool = False,
    skip_discriminative: bool = False,
    skip_variants: bool = False,
    skip_evidence: bool = False,
    skip_color: bool = False,
) -> list[Any]:
    """One orchestration pass — existing modules only, no double primary scoring.

    Order (soft deltas only; concept identity already fixed upstream):
      1) query meaning (analyze)
      2) attributes (+ color via DNA/evidence inside attribute layer)
      3) concept evidence
      4) discriminative rivals
      5) object ≠ pattern
      6) visual variant annotate (no sibling score rewrite)
      7) color evidence only if attribute color channel absent
      8) result reasons
    """
    if not results or not (query_text or "").strip():
        return results

    analysis = analysis or analyze_query_intelligence(
        query_text, customer_registry=customer_registry
    )
    meaning_text = str(analysis.get("search_text") or query_text).strip() or query_text
    attrs = extract_query_attributes(meaning_text)
    layers_run: list[str] = ["query_meaning"]
    cust = " ".join(str(customer or analysis.get("customer") or "").strip().split())

    if not skip_attributes:
        try:
            from core.query_attribute_intel import apply_query_attribute_intel

            results = apply_query_attribute_intel(results, meaning_text, attrs=attrs)
            layers_run.append("query_attribute_intel")
        except Exception:
            pass

    if not skip_evidence and index_db:
        try:
            from core.concept_evidence import apply_concept_evidence_scoring

            results = apply_concept_evidence_scoring(
                results, meaning_text, index_db=index_db, customer_key=cust
            )
            layers_run.append("concept_evidence")
        except Exception:
            pass

    if not skip_discriminative:
        try:
            from core.discriminative_concept_intel import (
                apply_discriminative_concept_intel,
            )

            results = apply_discriminative_concept_intel(results, meaning_text)
            layers_run.append("discriminative_concept_intel")
        except Exception:
            pass

    if not skip_object_gate:
        try:
            from core.object_pattern_gate import apply_object_pattern_gate

            results = apply_object_pattern_gate(results, meaning_text)
            layers_run.append("object_pattern_gate")
        except Exception:
            pass

    if not skip_variants:
        try:
            from core.visual_variant_intel import annotate_visual_variants

            results = annotate_visual_variants(results, meaning_text, attrs=attrs)
            layers_run.append("visual_variant_intel")
        except Exception:
            pass

    # Color soft-score only when attribute layer did not already apply color.
    if not skip_color:
        try:
            from core.color_evidence import apply_color_evidence_scoring

            results = apply_color_evidence_scoring(
                results, meaning_text, customer_key=cust
            )
            layers_run.append("color_evidence")
        except Exception:
            pass

    results = attach_result_reasons(results, analysis)
    layers_run.append("result_reasons")

    if results:
        dbg0 = dict(getattr(results[0], "debug", None) or {})
        dbg0["search_intelligence_chain"] = {
            "analysis": analysis,
            "layers": layers_run,
            "index_db": bool(index_db),
            "unified": True,
            "double_score_guard": True,
            "customer": cust,
        }
        results[0].debug = dbg0
    return results
