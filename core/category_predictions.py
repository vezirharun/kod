"""Human-readable top category predictions from indexed visual metadata."""

from __future__ import annotations

from typing import Any


def category_predictions_from_result(
    result: Any, limit: int = 5
) -> list[dict[str, Any]]:
    debug = dict(getattr(result, "debug", {}) or {})
    texture = dict(debug.get("texture_map") or {})
    merged = {**texture, **debug}
    family = str(
        getattr(result, "pattern_family", "") or merged.get("result_family") or ""
    )
    subtype = str(
        getattr(result, "animal_print_type", "")
        or merged.get("animal_print_type")
        or ""
    )
    # User teach > Pattern DNA / semantic > Global Object AI.
    meta_predictions = category_predictions_from_metadata(
        family, subtype, merged, limit=limit
    )
    if meta_predictions:
        return meta_predictions
    object_meta = merged.get("global_object_intelligence") or {}
    return _object_predictions(object_meta, limit=limit)


def category_predictions_from_metadata(
    family: str,
    subtype: str,
    metadata: dict[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    # Marka kanıtı Monogram/Logo ailesinden bağımsız bir eksen olarak gösterilir.
    brand_name = str(metadata.get("brand_name") or "").strip()
    sem0 = metadata.get("semantic_tags") or {}
    brand_refs = sem0.get("brand_references", []) if isinstance(sem0, dict) else []
    if not brand_name and brand_refs:
        from core.brand_aliases import resolve_brand_alias
        for ref in brand_refs:
            resolved = resolve_brand_alias(str(ref))
            if resolved:
                brand_name = resolved
                break
    brand_prediction = None
    if brand_name:
        brand_label = brand_name.title()
        brand_prediction = {
            "category_path": f"Marka/{brand_label}",
            "family": "Marka",
            "subtype": brand_label,
            "confidence": max(float(metadata.get("category_confidence") or 0.0), 0.90),
            "source": "brand_evidence",
        }

    dna = metadata.get("pattern_dna") or {}
    semantic = metadata.get("semantic_tags") or {}
    manual_path = str(metadata.get("manual_category_path") or "").strip()
    learned = bool(
        metadata.get("learned_concept_exact") or metadata.get("learned_concept")
    )
    learned_name = str(
        metadata.get("learned_canonical") or metadata.get("learned_concept_label") or ""
    ).strip()
    if not learned_name and learned:
        learned_name = str(manual_path or metadata.get("category_path") or "").split("/")[-1].strip()
    if learned_name:
        predictions = [
            {
                "category_path": learned_name,
                "label": learned_name,
                "confidence": 0.99,
                "source": "learned_concept",
            }
        ]
        if brand_prediction and brand_prediction.get("category_path") != learned_name:
            predictions.append(brand_prediction)
        return predictions[: max(1, int(limit))]
    trusted_manual = bool(
        metadata.get("user_labeled")
        or manual_path
        or learned
        or str(metadata.get("category_source") or "")
        in ("manual_user", "gold_dataset", "user_feedback")
    )
    source = ""
    if trusted_manual:
        path = manual_path or str(metadata.get("category_path") or "")
        pred_family = str(metadata.get("pattern_family") or family or "")
        pred_subtype = str(metadata.get("animal_print_type") or subtype or "")
        confidence = max(float(metadata.get("classification_confidence") or 0), 0.95)
        source = "admin_label"
    elif isinstance(dna, dict) and dna.get("family"):
        pred_family = str(dna.get("family") or "")
        pred_subtype = str(dna.get("subfamily") or "").lower().replace(" ", "_")
        confidence = float(dna.get("confidence") or 0)
        path = _family_path(pred_family, pred_subtype)
        source = "pattern_dna"
    elif isinstance(semantic, dict) and semantic.get("family"):
        pred_family = str(semantic.get("family") or "")
        pred_subtype = str(semantic.get("subtype") or "")
        confidence = float(semantic.get("confidence") or 0)
        path = _family_path(pred_family, pred_subtype)
        source = "semantic_tags"
    else:
        return []

    if not path and pred_family and pred_family != "unknown":
        path = _family_path(pred_family, pred_subtype)
    if trusted_manual:
        if not path:
            return [brand_prediction] if brand_prediction else []
    elif not path or not pred_family or pred_family == "unknown" or confidence < 0.38:
        return [brand_prediction] if brand_prediction else []
    if pred_family == "animal_print" and pred_subtype == "leopard":
        evidence = " ".join(
            str(v).lower()
            for v in [
                *(semantic.get("motifs") or []),
                semantic.get("motif", ""),
                dna.get("motif", "") if isinstance(dna, dict) else "",
                dna.get("subfamily", "") if isinstance(dna, dict) else "",
            ]
            if v
        )
        if not any(tag in evidence for tag in ("leopard", "animal print", "animal spot", "organic spot", "rosette")):
            return [brand_prediction] if brand_prediction else []
        semantic_source = str(semantic.get("source") or "")
        dna_source = str(dna.get("source") or "") if isinstance(dna, dict) else ""
        if (
            not trusted_manual
            and semantic_source in ("", "indexed_visual_metadata")
            and dna_source in ("", "heuristic", "semantic_tags")
        ):
            return [brand_prediction] if brand_prediction else []
    predictions = [{
        "category_path": path,
        "label": path.split("/")[-1],
        "confidence": round(confidence, 4),
        "source": source,
    }]
    if brand_prediction and brand_prediction["category_path"] != path:
        predictions.insert(0, brand_prediction)
    return predictions[: max(1, int(limit))]



def _object_predictions(object_meta: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    """Convert Global Object AI detections into UI category cards.

    Each detected instance remains separate; this does not infer identity or
    sensitive attributes such as sex/gender.
    """
    if not isinstance(object_meta, dict):
        return []
    objects = object_meta.get("objects") or []
    if not isinstance(objects, list):
        return []
    labels: dict[str, tuple[str, float, int]] = {}
    display = {
        "person": "İnsan", "car": "Araba", "bicycle": "Bisiklet",
        "motorcycle": "Motosiklet", "bird": "Kuş", "cat": "Kedi",
        "dog": "Köpek", "animal": "Hayvan", "chair": "Sandalye",
        "dining table": "Masa", "table": "Masa", "cell phone": "Telefon",
        "laptop": "Dizüstü Bilgisayar",
    }
    for item in objects:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("label") or "").strip().lower()
        label_tr = str(item.get("label_tr") or display.get(raw) or raw).strip()
        if not label_tr:
            continue
        conf = float(item.get("confidence") or 0.0)
        iid = str(item.get("instance_id") or "")
        count = labels.get(raw, (label_tr, 0.0, 0))[2] + 1
        labels[raw] = (label_tr, max(conf, labels.get(raw, (label_tr, 0.0, 0))[1]), count)
    out = []
    for raw, (label, conf, count) in sorted(labels.items(), key=lambda kv: (-kv[1][1], kv[0])):
        out.append({
            "category_path": f"Global Object/{label}",
            "label": label,
            "confidence": round(conf, 4),
            "count": count,
            "source": "global_object_ai",
        })
    return out[: max(1, int(limit))]


def prediction_label(predictions: list[dict[str, Any]]) -> str:
    if not predictions:
        return "Kararsız"
    top_conf = float(predictions[0].get("confidence", 0))
    if top_conf < 0.38:
        return "Kararsız"
    return str(predictions[0].get("label") or "Kararsız")


def _family_path(family: str, subtype: str = "") -> str:
    labels = {
        "animal_print": "Animal Print",
        "floral": "Floral",
        "geometric": "Geometric",
        "stripe": "Stripe",
        "plaid_check": "Plaid Check",
        "paisley_scarf": "Paisley Scarf",
        "marble_abstract": "Marble Abstract",
        "baroque_ornament": "Baroque Ornament",
        "typography_text": "Typography Text",
        "monogram_logo": "Monogram Logo",
    }
    parent = labels.get(family, family.replace("_", " ").title())
    if family == "animal_print" and subtype:
        child = {
            "leopard": "Leopard",
            "zebra": "Zebra",
            "snake": "Snake Skin",
            "tiger": "Tiger",
            "crocodile": "Crocodile",
        }.get(subtype, subtype.replace("_", " ").title())
        return f"{parent}/{child}"
    return parent
