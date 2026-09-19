"""User-facing search presets and threshold guidance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchPreset:
    key: str
    label: str
    mode: str
    color_mode: str
    threshold: float
    show_near_below: bool
    result_limit: int = 0


SEARCH_PRESETS = (
    SearchPreset("exact", "Birebir Dosya Ara", "exact", "normal", 0.97, False),
    SearchPreset(
        "same_pattern", "Aynı Desen / Varyant Ara", "similar", "normal", 0.80, True
    ),
    SearchPreset("texture", "Benzer Doku Ara", "style", "ignore", 0.60, True),
    SearchPreset(
        "comprehensive", "Daha Kapsamlı Bul", "comprehensive", "ignore", 0.55, True
    ),
    SearchPreset(
        "customer_alternatives",
        "Müşteri İçin Alternatif Bul",
        "comprehensive",
        "normal",
        0.55,
        True,
        0,
    ),
)

PRESET_BY_KEY = {preset.key: preset for preset in SEARCH_PRESETS}


def high_threshold_warning(mode: str, threshold: float) -> str:
    if mode in {"style", "comprehensive"} and threshold >= 0.90:
        return (
            "Bu eşik yalnızca birebir/çok yakın dosyaları gösterir. "
            "Benzer desenler için %55-70 önerilir."
        )
    return ""
