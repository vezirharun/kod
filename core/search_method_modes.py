"""Gelişmiş arama yöntemleri — ayar bayraklarına eşlenir."""

from __future__ import annotations

from typing import Any

from core.settings import AppSettings

SEARCH_METHOD_MODES: dict[str, dict[str, Any]] = {
    "classic": {
        "label": "Klasik Arama",
        "search_mode": "similar",
        "fast_hash_only": False,
        "search_progressive": True,
    },
    "fast": {
        "label": "Hızlı Arama",
        "fast_hash_only": True,
        "search_progressive": True,
    },
    "texture": {
        "label": "Doku Arama",
        "search_mode": "style",
        "color_weight_mode": "ignore",
        "search_texture": True,
    },
    "semantic": {
        "label": "Anlamsal Arama",
        "semantic_text_search_enabled": True,
        "search_text_filename": True,
        "search_text_ocr": True,
    },
    "pattern_dna": {
        "label": "Desen DNA Arama",
        "semantic_text_search_enabled": True,
        "search_texture": True,
    },
    "ocr": {
        "label": "Yazı/OCR Arama",
        "search_text_ocr": True,
        "search_text_filename": False,
        "semantic_text_search_enabled": True,
    },
    "brand": {
        "label": "Marka/Logo Arama",
        "semantic_text_search_enabled": True,
        "search_mode": "comprehensive",
    },
    "logo": {
        "label": "Logo Arama",
        "search_mode": "similar",
        "semantic_text_search_enabled": True,
    },
    "embroidery": {
        "label": "Nakış Arama",
        "search_mode": "style",
        "semantic_text_search_enabled": True,
    },
    "hybrid": {
        "label": "Hibrit Arama",
        "search_progressive": True,
        "semantic_text_search_enabled": True,
        "search_text_visual": True,
    },
    "project": {
        "label": "Proje Arama",
        "search_mode": "comprehensive",
    },
    "deep": {
        "label": "Derin Arama",
        "search_mode": "comprehensive",
        "search_progressive": False,
        "fine_detail_enabled": True,
        "fast_hash_only": False,
    },
    "exact": {
        "label": "Birebir Arama",
        "search_mode": "exact",
        "similarity_threshold": 0.97,
    },
    "near_variant": {
        "label": "Yakın Varyant",
        "search_mode": "similar",
        "similarity_threshold": 0.80,
    },
    "family": {
        "label": "Aile Araması",
        "search_mode": "comprehensive",
        "similarity_threshold": 0.55,
        "show_unrelated_results": False,
    },
    "ai_auto": {
        "label": "AI Otomatik",
        "search_progressive": True,
        "semantic_text_search_enabled": True,
        "ai_embedding_enabled": True,
        "fast_hash_only": False,
    },
}

DEFAULT_SEARCH_METHOD = "hybrid"

PRIMARY_SEARCH_METHOD_KEYS = (
    "classic",
    "fast",
    "texture",
    "semantic",
    "pattern_dna",
    "ocr",
    "brand",
    "logo",
    "embroidery",
    "hybrid",
)


def search_method_labels() -> list[tuple[str, str]]:
    return [(key, str(meta["label"])) for key, meta in SEARCH_METHOD_MODES.items()]


def primary_search_method_labels() -> list[tuple[str, str]]:
    return [
        (key, str(SEARCH_METHOD_MODES[key]["label"]))
        for key in PRIMARY_SEARCH_METHOD_KEYS
        if key in SEARCH_METHOD_MODES
    ]


def apply_search_method_mode(settings: AppSettings, mode_key: str) -> None:
    """Seçilen arama yöntemini settings'e uygula (kalıcı alanlar)."""
    meta = SEARCH_METHOD_MODES.get(mode_key) or SEARCH_METHOD_MODES[DEFAULT_SEARCH_METHOD]
    for key, value in meta.items():
        if key == "label":
            continue
        if hasattr(settings, key):
            setattr(settings, key, value)
    settings.search_method_mode = mode_key
