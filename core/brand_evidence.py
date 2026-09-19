"""Ortak marka kanıtı çıkarımı.

Amaç: belirli marka araması ile genel `marka` aramasının aynı kanıt kümesini
kullanmasını sağlamak. Böylece `marka` sonucu, tek tek marka sonuçlarının
alt kümesini kaybetmez.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [raw]
    return [str(value)]


def _flatten_strings(value: Any, *, max_depth: int = 5) -> list[str]:
    """Nested metadata içindeki tüm metinleri güvenli biçimde çıkar."""
    out: list[str] = []
    if max_depth < 0 or value is None:
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            # Anahtar isimleri de kanıt olabilir: brand_name, brand_references vb.
            if key in {
                "brand", "brand_name", "brand_style", "brand_references",
                "category_path", "manual_category_path", "category_aliases",
                "label", "path", "name", "title", "text", "ocr",
            }:
                out.extend(_flatten_strings(item, max_depth=max_depth - 1))
            else:
                out.extend(_flatten_strings(item, max_depth=max_depth - 1))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(_flatten_strings(item, max_depth=max_depth - 1))
    elif isinstance(value, str):
        if value.strip():
            out.append(value)
    else:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _known_brand_aliases(db_path: str | None = None) -> tuple[tuple[str, str], ...]:
    """Marka ağacındaki öğrenilmiş markaları da alias sözlüğüne dahil et."""
    from core.category_tree import CATEGORY_TREE
    from core.brand_aliases import BRAND_ALIASES, normalize_brand_key, _dynamic_brand_aliases

    pairs: list[tuple[str, str]] = [
        (normalize_brand_key(a), normalize_brand_key(c))
        for a, c in BRAND_ALIASES.items()
    ]
    for brand, aliases in (CATEGORY_TREE.get("Marka", {}) or {}).items():
        canonical = normalize_brand_key(brand)
        if canonical:
            pairs.append((canonical, canonical))
        for alias in aliases or ():
            a = normalize_brand_key(str(alias))
            if a:
                pairs.append((a, canonical))
    if db_path:
        for alias, canonical in _dynamic_brand_aliases(db_path).items():
            a = normalize_brand_key(alias)
            c = normalize_brand_key(canonical)
            if a and c:
                pairs.append((a, c))
    # Uzun aliaslar önce: "saint laurent" kısa "saint"ten önce bulunmalı.
    return tuple(sorted(set(pairs), key=lambda x: (-len(x[0]), x[0], x[1])))


def extract_brand_evidence(rec: dict[str, Any], db_path: str | None = None) -> set[str]:
    """Kayıttaki bütün marka kanıtlarını tek, kanonik kümeye indirger.

    Önemli: `Marka/<isim>` yolu, manuel öğrenme, OCR, semantic/DNA, kategori
    alias'ları, arama blob'u ve dosya adı aynı kanıt havuzuna girer. Böylece
    `amiri`, `dior` ve genel `marka` sorguları farklı veri kümeleri üretmez.
    Dosya adı tek başına yalnızca bilinen/ayırt edici marka adıyla eşleştiğinde
    zayıf kanıt olarak kullanılır.
    """
    from core.brand_aliases import normalize_brand_key, resolve_brand_alias, tokenize_brand_text

    if not isinstance(rec, dict):
        return set()

    found: set[str] = set()
    try:
        fid = int(rec.get("id") or rec.get("file_id") or 0)
    except (TypeError, ValueError):
        fid = 0
    if fid > 0 and db_path:
        try:
            from core.search_memory import taught_brand_keys_for_file

            found.update(taught_brand_keys_for_file(db_path, fid))
        except Exception:
            pass
    explicit: list[str] = []
    free_text: list[str] = []

    # Bütün kayıt metadata'sını tara. Legacy indekslerde marka bilgisi farklı
    # katmanlarda kalabildiği için yalnızca birkaç sabit alanı okumak yetersiz.
    for key, value in rec.items():
        values = _flatten_strings(value)
        if key in {
            "brand", "brand_name", "brand_style", "brand_references",
            "category_path", "manual_category_path", "category_aliases",
        }:
            explicit.extend(values)
        else:
            free_text.extend(values)

    # Texture map / semantic / DNA zaten rec içinde olabilir; yukarıdaki
    # recursive tarama bunları kapsar. Explicit yolları önce değerlendir.
    for value in explicit:
        n = normalize_brand_key(value)
        if not n:
            continue
        if n.startswith("marka/"):
            child = n.split("/", 1)[1].strip()
            canonical = resolve_brand_alias(child, db_path) or child
            if canonical:
                found.add(normalize_brand_key(canonical))
            continue
        canonical = resolve_brand_alias(n, db_path)
        if canonical:
            found.add(normalize_brand_key(canonical))

    # Bilinen marka adlarını serbest metinde de yakala. Kısa aliaslar (lv/gg/dg)
    # serbest dosya isminde tek başına marka kanıtı sayılmaz.
    known = _known_brand_aliases(db_path)
    for value in free_text:
        n = normalize_brand_key(value)
        if not n:
            continue
        padded = f" {n} "
        # "D&G.jpg" → compact "dg" so kısa alias kanıtı dosya adında tutulur.
        compact = normalize_brand_key(re.sub(r"[^a-z0-9]+", "", n))
        spaced = normalize_brand_key(re.sub(r"[&+._/-]+", " ", n))
        padded_spaced = f" {spaced} "
        amp_tokens = {
            normalize_brand_key(t)
            for t in tokenize_brand_text(str(value).replace("&", "").replace("+", ""))
        }
        for alias, canonical in known:
            if not alias or not canonical:
                continue
            if len(alias) < 4:
                if alias == compact or f" {alias} " in padded_spaced or alias in amp_tokens:
                    found.add(canonical)
                continue
            if alias == n or f" {alias} " in padded or alias in n:
                found.add(canonical)

    return found
