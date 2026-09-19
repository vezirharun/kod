"""Kullanıcı benzerlik sınıfları — arama sonuç grupları ve öğrenme."""

from __future__ import annotations

from core.dynamic_groups import (
    G_CROP,
    G_EXACT,
    G_FAR,
    G_FORMAT,
    G_SAME_CLOSE,
    G_SAME_STYLE,
)
from core.textile_terms import normalize_turkish

# (görünen ad, tier_id, cluster_group, pattern_family, pattern_subtype)
SIMILARITY_TIER_CHOICES: list[tuple[str, str, str, str, str]] = [
    ("Aynı Görsel", "exact_same", G_EXACT, "", ""),
    ("Aynı Desen / Format", "format_variant", G_FORMAT, "", ""),
    ("Aynı Desen / Crop", "crop_variant", G_CROP, "", ""),
    ("Benzer Kamuflaj", "camouflage", G_SAME_CLOSE, "texture_ground", "camouflage"),
    (
        "Benzer Askeri / Orman Camo",
        "military_camo",
        G_SAME_CLOSE,
        "texture_ground",
        "military_camo",
    ),
    ("Benzer Zemin Doku", "ground_texture", G_SAME_STYLE, "texture_ground", "ground"),
    ("Uzak Doku", "far_texture", G_FAR, "", ""),
]

_TIER_BY_ID: dict[str, dict[str, str]] = {}
_TIER_ALIASES: dict[str, str] = {}

for label, tier_id, cluster, family, subtype in SIMILARITY_TIER_CHOICES:
    _TIER_BY_ID[tier_id] = {
        "tier_id": tier_id,
        "label": label,
        "cluster_group": cluster,
        "pattern_family": family,
        "pattern_subtype": subtype,
    }
    _TIER_ALIASES[normalize_turkish(label)] = tier_id
    _TIER_ALIASES[normalize_turkish(tier_id)] = tier_id

_EXTRA_ALIASES: dict[str, str] = {
    "ayni gorsel": "exact_same",
    "aynı görsel": "exact_same",
    "ayni desen format": "format_variant",
    "aynı desen / format": "format_variant",
    "ayni desen crop": "crop_variant",
    "aynı desen / crop": "crop_variant",
    "benzer kamuflaj": "camouflage",
    "kamuflaj": "camouflage",
    "camo": "camouflage",
    "camouflage": "camouflage",
    "benzer askeri orman camo": "military_camo",
    "askeri camo": "military_camo",
    "orman camo": "military_camo",
    "military camo": "military_camo",
    "benzer zemin doku": "ground_texture",
    "zemin doku": "ground_texture",
    "ground texture": "ground_texture",
    "uzak doku": "far_texture",
    "distant texture": "far_texture",
}
for alias, tier_id in _EXTRA_ALIASES.items():
    _TIER_ALIASES[normalize_turkish(alias)] = tier_id


def resolve_similarity_tier(text: str) -> dict[str, str]:
    """Kullanıcı metnini benzerlik sınıfına çevir."""
    raw = (text or "").strip()
    if not raw:
        return {}
    norm = normalize_turkish(raw)
    tier_id = _TIER_ALIASES.get(norm, "")
    if not tier_id:
        for alias, tid in _TIER_ALIASES.items():
            if alias in norm or norm in alias:
                tier_id = tid
                break
    if tier_id and tier_id in _TIER_BY_ID:
        out = dict(_TIER_BY_ID[tier_id])
        out["tag"] = raw
        return out
    return {}


def tier_info(tier_id: str) -> dict[str, str]:
    return dict(_TIER_BY_ID.get(tier_id, {}))


def tier_label(tier_id: str) -> str:
    info = _TIER_BY_ID.get(tier_id, {})
    return info.get("label", tier_id.replace("_", " ").title())


def is_builtin_tier_label(text: str) -> bool:
    return bool(resolve_similarity_tier(text).get("tier_id"))
