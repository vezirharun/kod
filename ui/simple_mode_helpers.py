"""Basit Mod — kullanıcı dostu metinler."""

from __future__ import annotations


def simple_status_line(
    *,
    searchable: int = 0,
    index_percent: int = 0,
    indexing: bool = False,
    searching: bool = False,
    search_count: int = 0,
    refining: bool = False,
    search_phase: str = "",
    search_kind: str = "",
    empty_result: bool = False,
) -> str:
    """Alt durum çubuğu — teknik terim yok."""
    if searching:
        if search_phase == "starting":
            if search_kind == "text":
                return "Arama başladı — metin taranıyor…"
            if search_kind == "visual":
                return "Arama başladı — görsel analiz ediliyor…"
            return "Arama başladı…"
        if search_phase == "scanning":
            return "Adaylar taranıyor…"
        if search_phase == "preparing":
            return "Sonuçlar hazırlanıyor…"
        if empty_result:
            return (
                "Sonuç bulunamadı — semantik aramayı veya kapsamı genişletmeyi deneyin"
            )
        if search_count > 0:
            if refining:
                return (
                    f"Aranıyor… {search_count:,} sonuç — "
                    f"benzerlik sıralaması güncelleniyor"
                )
            return f"{search_count:,} sonuç bulundu"
        if search_kind == "text":
            return "Metin aranıyor…"
        if search_kind == "visual":
            return "Görsel aranıyor…"
        return "Aranıyor… benzer desenler taranıyor"
    if empty_result:
        return (
            "Sonuç bulunamadı — semantik aramayı veya kapsamı genişletmeyi deneyin"
        )
    if search_count > 0 and not searching:
        return f"{search_count:,} sonuç bulundu · {searchable:,} desen aranabilir"
    base = f"Arama hazır · {searchable:,} desen aranabilir"
    if indexing:
        if index_percent > 0:
            return f"{base} · Index arka planda %{index_percent}"
        return f"{base} · Index arka planda devam ediyor"
    return base


def simple_header_line(
    searchable: int = 0,
    index_percent: int = 0,
    indexing: bool = False,
    *,
    searching: bool = False,
    search_phase: str = "",
    search_count: int = 0,
) -> str:
    if searching:
        if search_phase == "starting":
            return "Arama başladı…"
        if search_phase == "scanning":
            return "Adaylar taranıyor…"
        if search_phase == "preparing":
            return "Sonuçlar hazırlanıyor…"
        if search_count > 0:
            return f"{search_count:,} sonuç bulundu"
        return "Arama devam ediyor…"
    if indexing and index_percent > 0:
        return (
            f"Index arka planda devam ediyor. "
            f"Şu anda {searchable:,} desen aranabilir (%{index_percent})."
        )
    if indexing:
        return (
            f"Index arka planda devam ediyor. Şu anda {searchable:,} desen aranabilir."
        )
    return f"{searchable:,} desen aranabilir"
