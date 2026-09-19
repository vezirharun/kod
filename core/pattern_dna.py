"""Pattern DNA — her desen için yapılandırılmış tekstil zekâ profili."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PATTERN_DNA_VERSION = 3

_UNKNOWN = "unknown"


@dataclass
class PatternDNA:
    main_family: str = _UNKNOWN
    family: str = _UNKNOWN
    subfamily: str = ""
    collection: str = ""
    series: str = ""
    motif: str = ""
    pattern_type: str = ""
    repeat: str = ""
    repeat_type: str = ""
    repeat_class: str = ""
    layout: str = ""
    density: str = ""
    scale: str = ""
    orientation: str = ""
    geometry: str = ""
    organic: str = ""
    complexity: str = ""
    text: str = ""
    logo: str = ""
    text_present: bool = False
    logo_present: bool = False
    text_detected: bool = False
    logo_detected: bool = False
    brand_style: str = ""
    designer_style: str = ""
    material_hint: str = ""
    texture: str = ""
    texture_class: str = ""
    motif_class: str = ""
    fabric_type: str = ""
    embroidery_type: str = ""
    needle_type: str = ""
    color_family: str = ""
    dominant_colors: list[str] = field(default_factory=list)
    style: str = ""
    visual_signature: str = ""
    semantic_tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    pattern_dna_confidence: float = 0.0
    source: str = "heuristic"
    version: int = PATTERN_DNA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["version"] = PATTERN_DNA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PatternDNA:
        if not isinstance(data, dict):
            return cls()
        dom = data.get("dominant_colors")
        return cls(
            main_family=str(data.get("main_family") or data.get("family") or _UNKNOWN),
            family=str(data.get("family") or _UNKNOWN),
            subfamily=str(data.get("subfamily") or ""),
            collection=str(data.get("collection") or ""),
            series=str(data.get("series") or ""),
            motif=str(data.get("motif") or ""),
            pattern_type=str(data.get("pattern_type") or ""),
            repeat=str(data.get("repeat") or ""),
            repeat_type=str(data.get("repeat_type") or data.get("repeat") or ""),
            repeat_class=str(data.get("repeat_class") or data.get("repeat_type") or data.get("repeat") or ""),
            layout=str(data.get("layout") or ""),
            density=str(data.get("density") or ""),
            scale=str(data.get("scale") or ""),
            orientation=str(data.get("orientation") or ""),
            geometry=str(data.get("geometry") or ""),
            organic=str(data.get("organic") or ""),
            complexity=str(data.get("complexity") or ""),
            text=str(data.get("text") or ""),
            logo=str(data.get("logo") or ""),
            text_present=bool(data.get("text_present") or data.get("text") == "Yes"),
            logo_present=bool(data.get("logo_present") or data.get("logo") == "Yes"),
            text_detected=bool(
                data.get("text_detected")
                or data.get("text_present")
                or data.get("text") == "Yes"
            ),
            logo_detected=bool(
                data.get("logo_detected")
                or data.get("logo_present")
                or data.get("logo") == "Yes"
            ),
            brand_style=str(data.get("brand_style") or ""),
            designer_style=str(data.get("designer_style") or data.get("style") or data.get("brand_style") or ""),
            material_hint=str(data.get("material_hint") or data.get("texture") or ""),
            texture=str(data.get("texture") or ""),
            texture_class=str(data.get("texture_class") or data.get("texture") or ""),
            motif_class=str(data.get("motif_class") or data.get("motif") or ""),
            fabric_type=str(data.get("fabric_type") or ""),
            embroidery_type=str(data.get("embroidery_type") or ""),
            needle_type=str(data.get("needle_type") or ""),
            color_family=str(data.get("color_family") or ""),
            dominant_colors=list(dom) if isinstance(dom, list) else [],
            style=str(data.get("style") or ""),
            visual_signature=str(data.get("visual_signature") or ""),
            semantic_tags=list(data.get("semantic_tags") or []),
            confidence=float(data.get("confidence") or 0),
            pattern_dna_confidence=float(
                data.get("pattern_dna_confidence") or data.get("confidence") or 0
            ),
            source=str(data.get("source") or "heuristic"),
        )


def _band(value: float, *, low: float, high: float) -> str:
    if value >= high:
        return "High"
    if value >= low:
        return "Medium"
    if value > 0:
        return "Low"
    return ""


def _repeat_type(density: float, stripe: float, scale: float) -> str:
    if stripe >= 0.42:
        return "Stripe"
    if density >= 0.55:
        return "All Over"
    if density >= 0.15:
        return "Half Drop"
    if scale >= 0.35:
        return "Medallion"
    return "Random"


def _motif_label(family: str, subtype: str, semantic: dict[str, Any]) -> str:
    motifs = semantic.get("motifs") or []
    if motifs:
        return str(motifs[0]).title()
    if subtype:
        return subtype.replace("_", " ").title()
    mapping = {
        "animal_print": "Animal Spots",
        "floral": "Botanical",
        "geometric": "Geometric",
        "stripe": "Linear",
        "baroque": "Ornament",
        "marble_abstract": "Fluid",
        "monogram_logo": "Monogram",
        "typography_text": "Typography",
    }
    return mapping.get(family, family.replace("_", " ").title() if family != _UNKNOWN else "")


def _titleish(value: str) -> str:
    return str(value or "").replace("_", " ").strip().title()


def _complexity_label(semantic_terms: list[str], repeat_density: float, contrast: float) -> str:
    score = len([t for t in semantic_terms if t]) + (1 if repeat_density >= 0.45 else 0) + (1 if contrast >= 0.6 else 0)
    if score >= 6:
        return "High"
    if score >= 3:
        return "Medium"
    return "Low"


def build_pattern_dna(
    texture_map: dict[str, Any] | None,
    *,
    semantic_tags: dict[str, Any] | None = None,
    category_path: str = "",
    source: str = "heuristic",
) -> dict[str, Any]:
    """texture_map + semantic_tags'ten Pattern DNA üret."""
    tm = dict(texture_map or {})
    semantic = dict(semantic_tags or tm.get("semantic_tags") or {})
    family = str(semantic.get("family") or tm.get("pattern_family") or _UNKNOWN)
    subtype = str(
        semantic.get("subtype")
        or semantic.get("subfamily")
        or tm.get("animal_print_type")
        or tm.get("pattern_subtype")
        or ""
    )
    if (
        family == "animal_print"
        and subtype == "leopard"
        and semantic
        and not str(semantic.get("subtype") or semantic.get("subfamily") or "")
        and source not in ("user_feedback", "gold_dataset", "manual")
    ):
        subtype = ""
    confidence = max(
        float(tm.get("classification_confidence") or 0),
        float(semantic.get("confidence") or 0),
    )
    repeat_density = float(tm.get("repeat_density", 0) or 0)
    stripe_score = float(tm.get("stripe_score", 0) or 0)
    organic_score = float(tm.get("organic_blob_score", 0) or 0)
    scale_score = float(tm.get("scale_pattern_score", 0) or 0)
    contrast = float(tm.get("contrast_score", 0) or 0) if "contrast_score" in tm else 0.5

    styles = list(semantic.get("styles") or [])
    brand_refs = list(semantic.get("brand_references") or [])
    textures = list(semantic.get("textures") or [])
    flat_semantic_terms: list[str] = []
    for key in (
        "family", "subfamily", "subtype", "motif", "motifs", "repeat_type",
        "color_family", "style", "styles", "themes", "brand_references",
        "objects", "textures",
    ):
        value = semantic.get(key)
        if isinstance(value, list):
            flat_semantic_terms.extend(str(v) for v in value if v)
        elif value:
            flat_semantic_terms.append(str(value))

    repeat_type = str(semantic.get("repeat_type") or "") or _repeat_type(
        repeat_density, stripe_score, scale_score
    )
    text_present = bool(semantic.get("text_present"))
    logo_present = bool(semantic.get("logo_present"))
    pattern_type = str(
        semantic.get("pattern_type")
        or ("allover repeat" if repeat_type.lower() in ("repeat", "all over", "allover") else repeat_type)
        or ""
    )
    material_hint = str(semantic.get("material_hint") or "")
    if not material_hint and any(str(t).lower() in ("jacquard", "jakar") for t in textures):
        material_hint = "jacquard"
    motif_label = _motif_label(family, subtype, semantic)
    family_label = _titleish(subtype or family if family else "")
    style_label = str(styles[0]).title() if styles else ""
    designer_style = str(brand_refs[0]).title() if brand_refs else style_label
    series_bits = [
        brand_refs[0] if brand_refs else "",
        styles[0] if styles else "",
        _band(scale_score, low=0.2, high=0.45),
        "Organic" if organic_score >= 0.3 else "",
    ]
    series_label = " ".join(_titleish(v) for v in series_bits if v).strip()
    visual_signature = "|".join(
        p.lower().replace(" ", "_")
        for p in (
            family or "",
            subtype or "",
            motif_label,
            repeat_type,
            style_label,
            designer_style,
            str(tm.get("color_family") or ""),
        )
        if p
    )
    dna = PatternDNA(
        main_family=family if family != _UNKNOWN else "",
        family=family if family != _UNKNOWN else "",
        subfamily=subtype.replace("_", " ").title() if subtype else "",
        collection=f"{family_label} Collection" if family_label else "",
        series=series_label,
        motif=motif_label,
        pattern_type=pattern_type,
        repeat=repeat_type,
        repeat_type=repeat_type,
        repeat_class=repeat_type,
        layout="Structured" if stripe_score >= 0.42 or scale_score >= 0.35 else "Organic",
        density=_band(repeat_density, low=0.15, high=0.45),
        scale=_band(scale_score, low=0.2, high=0.45),
        orientation="Directional" if stripe_score >= 0.42 else "Random",
        geometry="Geometric" if scale_score >= 0.35 else ("Organic" if organic_score >= 0.3 else ""),
        organic=_band(organic_score, low=0.2, high=0.45),
        complexity=_complexity_label(flat_semantic_terms, repeat_density, contrast),
        text="Yes" if text_present or any(s in ("typography", "text", "letter") for s in styles) else "",
        logo="Yes" if logo_present or family in ("monogram_logo",) or "logo" in " ".join(styles).lower() else "",
        text_present=text_present or any(s in ("typography", "text", "letter") for s in styles),
        logo_present=logo_present or family in ("monogram_logo",) or "logo" in " ".join(styles).lower(),
        text_detected=text_present or any(s in ("typography", "text", "letter") for s in styles),
        logo_detected=logo_present or family in ("monogram_logo",) or "logo" in " ".join(styles).lower(),
        brand_style=str(brand_refs[0]).title() if brand_refs else "",
        designer_style=designer_style,
        material_hint=material_hint,
        texture=str(textures[0]).title() if textures else (
            "Organic" if organic_score >= 0.3 else ""
        ),
        texture_class=str(textures[0]).title() if textures else "",
        motif_class=motif_label,
        fabric_type=str(tm.get("format_family") or ""),
        color_family=str(tm.get("color_family") or ""),
        dominant_colors=[
            str(c) for c in (tm.get("dominant_colors") or [])[:4] if c
        ],
        style=str(styles[0]).title() if styles else "",
        visual_signature=visual_signature,
        semantic_tags=list(dict.fromkeys(v.lower() for v in flat_semantic_terms if v)),
        confidence=round(confidence, 4),
        pattern_dna_confidence=round(confidence, 4),
        source=source,
    )
    if category_path:
        parts = [p.strip() for p in category_path.split("/") if p.strip()]
        if parts and not dna.family:
            dna.family = parts[0]
        if len(parts) > 1 and not dna.subfamily:
            dna.subfamily = parts[-1]
    return dna.to_dict()


def flatten_pattern_dna(dna: dict[str, Any] | None) -> list[str]:
    """FTS blob için DNA terimlerini düzleştir."""
    if not isinstance(dna, dict):
        return []
    values: list[str] = []
    for key, value in dna.items():
        if key in ("version", "confidence", "source", "dominant_colors"):
            continue
        if isinstance(value, list):
            values.extend(str(v) for v in value if v and str(v) != _UNKNOWN)
        elif value and str(value) not in (_UNKNOWN, ""):
            values.append(str(value))
    dom = dna.get("dominant_colors")
    if isinstance(dom, list):
        values.extend(str(c) for c in dom if c)
    fam = str(dna.get("family") or "")
    sub = str(dna.get("subfamily") or "")
    if fam and fam != _UNKNOWN:
        values.append(fam.replace("_", " "))
    if sub:
        values.append(sub.replace("_", " "))
    return list(dict.fromkeys(v.lower() for v in values if v))


def apply_dna_to_texture_map(texture_map: dict[str, Any], dna: dict[str, Any]) -> dict[str, Any]:
    merged = dict(texture_map)
    merged["pattern_dna"] = dna
    if dna.get("family") and not merged.get("user_labeled"):
        merged["pattern_family"] = str(dna["family"])
    if dna.get("subfamily"):
        sub_key = str(dna["subfamily"]).lower().replace(" ", "_")
        if str(merged.get("pattern_family") or "") == "animal_print":
            merged["animal_print_type"] = sub_key
        else:
            merged["pattern_subtype"] = sub_key
    if dna.get("confidence"):
        merged["classification_confidence"] = float(dna["confidence"])
    return merged
