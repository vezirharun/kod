"""Query family'ye göre dinamik sonuç grupları."""

from __future__ import annotations

from dataclasses import dataclass

from core.textile_terms import query_family_hints
from core.texture_profile import TextureProfile

# Genel küme anahtarları
G_EXACT = "exact_same"
G_FORMAT = "format_variant"
G_RESOLUTION = "resolution_variant"
G_CROP = "crop_variant"
G_COLOR = "color_variant"
G_SAME_CLOSE = "same_family_close"
G_SAME_STYLE = "same_family_style"
G_RELATED = "related_family"
G_FAR = "far_texture"
G_UNRELATED = "unrelated"

# Varsayılan açık / kapalı sonuç grupları (spec §6)
DEFAULT_VISIBLE_GROUPS = frozenset(
    {
        G_EXACT,
        G_FORMAT,
        G_RESOLUTION,
        G_CROP,
        G_COLOR,
        G_SAME_CLOSE,
        G_SAME_STYLE,
    }
)
DEFAULT_HIDDEN_GROUPS = frozenset(
    {
        G_RELATED,
        G_FAR,
        G_UNRELATED,
    }
)


LEGACY_CLUSTER_MAP: dict[str, str] = {
    "format_resolution": G_FORMAT,
    "pattern_variant": G_CROP,
    "leopard_brown_similar": G_SAME_CLOSE,
    "leopard_other_color": G_COLOR,
    "related_animal_print": G_RELATED,
    "distant_texture": G_FAR,
    "exact": G_EXACT,
    "format_variant": G_FORMAT,
    "animal_print": G_SAME_CLOSE,
    "textile_texture": G_SAME_STYLE,
    "weak": G_FAR,
}

CLUSTER_RANK: dict[str, int] = {
    G_EXACT: 0,
    G_FORMAT: 1,
    G_RESOLUTION: 2,
    G_CROP: 3,
    G_COLOR: 4,
    G_SAME_CLOSE: 5,
    G_SAME_STYLE: 6,
    G_RELATED: 7,
    G_FAR: 8,
    G_UNRELATED: 9,
}

MARBLE_FAMILIES = frozenset({"marble", "abstract", "marble_abstract", "texture_ground"})


@dataclass(frozen=True)
class QueryContext:
    pattern_family: str = "unknown"
    pattern_subtype: str = ""
    animal_print_type: str = ""
    text_query: str = ""
    classification_confidence: float = 0.0

    @property
    def effective_family(self) -> str:
        from core.pattern_classifier import effective_query_family
        from core.texture_profile import TextureProfile

        prof = TextureProfile(
            pattern_family=self.pattern_family,
            animal_print_type=self.animal_print_type,
            pattern_subtype=self.pattern_subtype,
            classification_confidence=self.classification_confidence,
        )
        hints = query_family_hints(self.text_query) if self.text_query else {}
        fam = effective_query_family(prof)
        if fam in ("unknown", "texture_ground") and hints.get("pattern_family"):
            if self.classification_confidence < 0.55:
                return hints["pattern_family"]
        return fam


def normalize_cluster_key(key: str) -> str:
    if not key:
        return G_FAR
    return LEGACY_CLUSTER_MAP.get(key, key)


def group_checked_by_default(key: str) -> bool:
    norm = normalize_cluster_key(key)
    if norm in DEFAULT_HIDDEN_GROUPS:
        return False
    return True


def resolve_query_context_from_meta(meta: dict) -> QueryContext:
    """Arama meta alanlarından QueryContext üret."""
    from core.textile_terms import query_family_hints

    text = (meta.get("text") or meta.get("text_query") or "").strip()
    hints = meta.get("query_family_hints") or query_family_hints(text)
    prof = TextureProfile(
        pattern_family=meta.get("query_pattern_family")
        or hints.get("pattern_family", "unknown"),
        animal_print_type=meta.get("query_animal_print_type")
        or hints.get("animal_print_type", ""),
        pattern_subtype=meta.get("query_pattern_subtype")
        or hints.get("pattern_type", ""),
        color_family=meta.get("query_color_family", ""),
        classification_confidence=float(meta.get("query_confidence", 0) or 0),
    )
    ts = meta.get("query_texture_summary") or {}
    if ts:
        prof.organic_blob_score = float(ts.get("blob", 0))
        prof.stripe_score = float(ts.get("stripe", 0))
        prof.scale_pattern_score = float(ts.get("scale", 0))
    return resolve_query_context(prof, text)


def resolve_query_context(
    profile: TextureProfile | None = None,
    text_query: str = "",
) -> QueryContext:
    prof = profile or TextureProfile()
    subtype = getattr(prof, "pattern_subtype", "") or prof.animal_print_type or ""
    if prof.pattern_family == "floral" and not subtype:
        if prof.scale_pattern_score > 0.4 and prof.organic_blob_score < 0.35:
            subtype = "small_floral"
    return QueryContext(
        pattern_family=prof.pattern_family or "unknown",
        pattern_subtype=subtype,
        animal_print_type=prof.animal_print_type or "",
        text_query=(text_query or "").strip(),
        classification_confidence=float(
            getattr(prof, "classification_confidence", 0) or 0
        ),
    )


def _animal_label(animal: str) -> str:
    labels = {
        "leopard": "Leopard",
        "zebra": "Zebra",
        "snake": "Yılan Derisi",
        "tiger": "Kaplan",
        "crocodile": "Timsah",
        "cow": "İnek Deseni",
        "cheetah": "Çita",
        "jaguar": "Jaguar",
    }
    return labels.get(animal, "Animal Print")


def group_definitions(ctx: QueryContext) -> list[tuple[str, str]]:
    """Query family için görünür grup sırası ve etiketleri."""
    base_head = [
        (G_EXACT, "Aynı Görsel"),
        (G_FORMAT, "Aynı Desen / Format Varyantı"),
        (G_RESOLUTION, "Aynı Desen / Çözünürlük Varyantı"),
        (G_CROP, "Aynı Desen / Crop Varyantı"),
    ]
    fam = ctx.effective_family
    animal = ctx.animal_print_type or query_family_hints(ctx.text_query).get(
        "animal_print_type", ""
    )

    if fam == "animal_print" and ctx.classification_confidence >= 0.70:
        label = _animal_label(animal or "leopard")
        family_groups = [
            (G_SAME_CLOSE, f"Benzer {label}"),
            (G_COLOR, f"Farklı Renk {label}"),
            (G_RELATED, "Yakın Animal Print"),
        ]
    elif fam == "animal_print" and animal and ctx.text_query:
        label = _animal_label(animal)
        family_groups = [
            (G_SAME_CLOSE, f"Benzer {label}"),
            (G_COLOR, f"Farklı Renk {label}"),
            (G_RELATED, "Yakın Animal Print"),
        ]
    elif fam == "floral":
        family_groups = [
            (G_SAME_CLOSE, "Benzer Çiçek / Floral"),
            (G_COLOR, "Farklı Renk Floral"),
            (G_SAME_STYLE, "Küçük Çiçek / Çıtır Çiçek"),
            (G_RELATED, "Büyük Çiçek / Gül / Papatya"),
        ]
    elif fam in MARBLE_FAMILIES or fam == "marble_abstract":
        family_groups = [
            (G_SAME_CLOSE, "Benzer Mermer / Soyut Doku"),
            (G_SAME_STYLE, "Benzer Renk Akışı"),
        ]
    elif fam == "stripe":
        family_groups = [(G_SAME_CLOSE, "Benzer Çizgi / Stripe")]
    elif fam == "plaid_check":
        family_groups = [(G_SAME_CLOSE, "Benzer Ekose / Kare")]
    elif fam == "paisley":
        family_groups = [(G_SAME_CLOSE, "Benzer Şal / Paisley")]
    elif fam == "lace":
        family_groups = [(G_SAME_CLOSE, "Benzer Dantel / Lace")]
    elif fam == "geometric":
        family_groups = [(G_SAME_CLOSE, "Benzer Geometrik")]
    elif fam == "texture_ground":
        subtype = ctx.pattern_subtype or ""
        if subtype == "military_camo":
            family_groups = [(G_SAME_CLOSE, "Benzer Askeri / Orman Camo")]
        elif subtype == "camouflage":
            family_groups = [(G_SAME_CLOSE, "Benzer Kamuflaj")]
        elif subtype == "ground":
            family_groups = [(G_SAME_STYLE, "Benzer Zemin Doku")]
        else:
            family_groups = [
                (G_SAME_CLOSE, "Benzer Kamuflaj"),
                (G_SAME_STYLE, "Benzer Zemin Doku"),
            ]
    elif fam == "baroque":
        family_groups = [(G_SAME_CLOSE, "Benzer Barok")]
    elif fam == "chain":
        family_groups = [(G_SAME_CLOSE, "Benzer Zincir")]
    elif fam == "monogram_logo":
        family_groups = [(G_SAME_CLOSE, "Benzer Monogram / Logo")]
    else:
        family_groups = [(G_SAME_CLOSE, "Benzer Desen Ailesi")]

    tail = [(G_FAR, "Uzak Doku"), (G_UNRELATED, "Alakasız")]
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    use_generic_color = fam not in ("animal_print", "floral")
    ordered = base_head + family_groups
    if use_generic_color:
        ordered = base_head + [(G_COLOR, "Renk Varyantı")] + family_groups
    for key, label in ordered + tail:
        if key in seen:
            continue
        seen.add(key)
        out.append((key, label))
    return out


def group_label_for_key(key: str, ctx: QueryContext) -> str:
    norm = normalize_cluster_key(key)
    for k, label in group_definitions(ctx):
        if k == norm:
            return label
    return norm.replace("_", " ").title()


def groups_relevant_for_query(key: str, ctx: QueryContext) -> bool:
    """Query family ile alakasız sabit leopard/floral gruplarını gizle."""
    norm = normalize_cluster_key(key)
    fam = ctx.effective_family
    animal = ctx.animal_print_type or query_family_hints(ctx.text_query).get(
        "animal_print_type", ""
    )

    leopard_only_keys = {
        "leopard_brown_similar",
        "leopard_other_color",
        "related_animal_print",
    }
    floral_only_norms = {G_SAME_STYLE}  # küçük çiçek — sadece floral query'de anlamlı

    if fam in MARBLE_FAMILIES or fam == "marble_abstract":
        if key in leopard_only_keys or animal in ("leopard", "zebra", "snake", "tiger"):
            return False
        if norm == G_RELATED and not ctx.text_query:
            return False
    if fam == "floral":
        if key in leopard_only_keys:
            return False
    if fam == "animal_print" or animal:
        if norm in floral_only_norms and fam != "floral":
            return False
    if fam not in ("floral",) and norm in floral_only_norms:
        return False
    return True


def hierarchy_sort_key(
    cluster_group: str, hierarchy_score: float, score: float
) -> tuple[int, float, float]:
    norm = normalize_cluster_key(cluster_group)
    return (CLUSTER_RANK.get(norm, 99), -hierarchy_score, -score)
