"""Renk indeksi — dominant/ortalama/palet/HSV + moda renk adı araması."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Moda renk adı → RGB prototip + aile
FASHION_COLOR_PROTOTYPES: dict[str, tuple[tuple[int, int, int], str]] = {
    "siyah": ((20, 20, 20), "black_white"),
    "black": ((20, 20, 20), "black_white"),
    "beyaz": ((245, 245, 245), "black_white"),
    "white": ((245, 245, 245), "black_white"),
    "krem": ((245, 235, 210), "cream"),
    "cream": ((245, 235, 210), "cream"),
    "ivory": ((255, 255, 240), "cream"),
    "bej": ((210, 180, 140), "beige"),
    "beige": ((210, 180, 140), "beige"),
    "camel": ((193, 154, 107), "camel"),
    "deve": ((193, 154, 107), "camel"),
    "kahverengi": ((139, 90, 43), "brown_tan"),
    "brown": ((139, 90, 43), "brown_tan"),
    "bordo": ((128, 0, 32), "burgundy"),
    "burgundy": ((128, 0, 32), "burgundy"),
    "wine": ((114, 47, 55), "burgundy"),
    "sarap": ((114, 47, 55), "burgundy"),
    "haki": ((128, 128, 80), "khaki"),
    "khaki": ((128, 128, 80), "khaki"),
    "olive": ((107, 142, 35), "khaki"),
    "zeytin": ((107, 142, 35), "khaki"),
    "altin": ((212, 175, 55), "gold"),
    "altın": ((212, 175, 55), "gold"),
    "gold": ((212, 175, 55), "gold"),
    "golden": ((212, 175, 55), "gold"),
    "mavi": ((30, 90, 180), "blue"),
    "blue": ((30, 90, 180), "blue"),
    "kirmizi": ((180, 30, 40), "red_pink"),
    "kırmızı": ((180, 30, 40), "red_pink"),
    "red": ((180, 30, 40), "red_pink"),
    "pembe": ((220, 120, 150), "red_pink"),
    "pink": ((220, 120, 150), "red_pink"),
    "yesil": ((40, 140, 70), "green"),
    "yeşil": ((40, 140, 70), "green"),
    "green": ((40, 140, 70), "green"),
    "sari": ((230, 200, 50), "yellow"),
    "sarı": ((230, 200, 50), "yellow"),
    "yellow": ((230, 200, 50), "yellow"),
    "turuncu": ((230, 120, 40), "orange"),
    "orange": ((230, 120, 40), "orange"),
    "gri": ((140, 140, 140), "grayscale"),
    "gray": ((140, 140, 140), "grayscale"),
    "grey": ((140, 140, 140), "grayscale"),
    "lacivert": ((20, 40, 100), "blue"),
    "navy": ((20, 40, 100), "blue"),
    "mor": ((120, 50, 160), "purple"),
    "purple": ((120, 50, 160), "purple"),
    "turkuaz": ((40, 180, 180), "turquoise"),
    "turquoise": ((40, 180, 180), "turquoise"),
    "dark_brown": ((70, 40, 20), "brown_tan"),
    "dark_red": ((110, 20, 25), "red_pink"),
    "dark_green": ((20, 80, 40), "green"),
    "dark_blue": ((15, 30, 90), "blue"),
    "light_gray": ((190, 190, 190), "grayscale"),
}

# Aile → arama etiketleri (sorgu genişletme)
FAMILY_SEARCH_LABELS: dict[str, tuple[str, ...]] = {
    "cream": ("krem", "cream", "ivory", "bej", "beige"),
    "beige": ("bej", "beige", "krem", "cream", "camel"),
    "camel": ("camel", "deve", "bej", "beige", "kahverengi"),
    "brown_tan": ("kahverengi", "brown", "bej", "beige", "camel", "tan"),
    "burgundy": ("bordo", "burgundy", "wine", "sarap", "kirmizi"),
    "khaki": ("haki", "khaki", "olive", "zeytin", "yesil"),
    "gold": ("altin", "altın", "gold", "golden", "sari"),
    "black_white": ("siyah", "black", "beyaz", "white"),
    "blue": ("mavi", "blue", "lacivert", "navy", "dark_blue"),
    "red_pink": ("kirmizi", "red", "pembe", "pink", "bordo", "dark_red"),
    "green": ("yesil", "green", "haki", "olive", "dark_green"),
    "yellow": ("sari", "yellow", "altin", "gold"),
    "orange": ("turuncu", "orange"),
    "grayscale": ("gri", "gray", "grey", "light_gray"),
    "purple": ("mor", "purple"),
    "turquoise": ("turkuaz", "turquoise"),
}

# Eski coarse aile ↔ moda aile yakınlığı
FAMILY_COMPAT: dict[str, tuple[str, ...]] = {
    "cream": ("cream", "beige", "brown_tan", "yellow", "black_white"),
    "beige": ("beige", "cream", "camel", "brown_tan"),
    "camel": ("camel", "beige", "brown_tan", "orange"),
    "burgundy": ("burgundy", "red_pink", "brown_tan"),
    "khaki": ("khaki", "green", "brown_tan"),
    "gold": ("gold", "yellow", "orange", "brown_tan"),
    "brown_tan": ("brown_tan", "beige", "camel", "cream", "khaki", "gold"),
    "black_white": ("black_white", "grayscale"),
}


def _rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(rn, gn, bn), min(rn, gn, bn)
    diff = mx - mn
    v = mx
    s = 0.0 if mx < 1e-8 else diff / mx
    if diff < 1e-8:
        h = 0.0
    elif mx == rn:
        h = (60 * ((gn - bn) / diff) + 360) % 360
    elif mx == gn:
        h = (60 * ((bn - rn) / diff) + 120) % 360
    else:
        h = (60 * ((rn - gn) / diff) + 240) % 360
    return h, s * 100.0, v * 100.0


def _dist(a: tuple[float, float, float], b: tuple[int, int, int]) -> float:
    return float(
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
    )


def classify_rgb(r: float, g: float, b: float, *, max_dist: float = 95.0) -> list[str]:
    """RGB → moda renk adları (yakınlık sırasıyla)."""
    scored: list[tuple[float, str, str]] = []
    for name, (proto, family) in FASHION_COLOR_PROTOTYPES.items():
        d = _dist((r, g, b), proto)
        if d <= max_dist:
            scored.append((d, name, family))
    scored.sort(key=lambda x: x[0])
    # Aynı aileden tek etiket + en iyi isimler
    out: list[str] = []
    seen_family: set[str] = set()
    for _d, name, family in scored:
        if name not in out:
            out.append(name)
        if family not in seen_family:
            seen_family.add(family)
        if len(out) >= 4:
            break
    return out


def classify_family(r: float, g: float, b: float) -> str:
    labels = classify_rgb(r, g, b)
    if not labels:
        return "unknown"
    name = labels[0]
    return FASHION_COLOR_PROTOTYPES.get(name, ((0, 0, 0), "unknown"))[1]


def compute_hsv_histogram(image: np.ndarray) -> list[float]:
    """Normalize HSV histogram (H32+S16+V16 = 64 float)."""
    try:
        if HAS_CV2:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
            v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
            for arr in (h, s, v):
                total = float(arr.sum()) + 1e-8
                arr /= total
            return [round(float(x), 5) for x in np.concatenate([h, s, v])]
        small = image[::8, ::8].reshape(-1, 3).astype(np.float32)
        hs = [_rgb_to_hsv(float(p[0]), float(p[1]), float(p[2])) for p in small]
        h_vals = [x[0] for x in hs]
        s_vals = [x[1] for x in hs]
        v_vals = [x[2] for x in hs]
        h_hist, _ = np.histogram(h_vals, bins=32, range=(0, 360), density=True)
        s_hist, _ = np.histogram(s_vals, bins=16, range=(0, 100), density=True)
        v_hist, _ = np.histogram(v_vals, bins=16, range=(0, 100), density=True)
        return [round(float(x), 5) for x in np.concatenate([h_hist, s_hist, v_hist])]
    except Exception:
        return []


def build_color_index(
    image: np.ndarray | None = None,
    *,
    dominant_colors: list[list[int]] | None = None,
) -> dict[str, Any]:
    """
    İndeks kaydı için renk özeti:
    dominant_colors, average_rgb, palette, hsv_hist, search_labels, color_family
    """
    dom = list(dominant_colors or [])
    avg = [0, 0, 0]
    hsv_hist: list[float] = []

    if image is not None and getattr(image, "size", 0) > 0:
        sample = image[::8, ::8].reshape(-1, 3).astype(np.float32)
        if len(sample):
            mean = sample.mean(axis=0)
            avg = [int(round(float(mean[0]))), int(round(float(mean[1]))), int(round(float(mean[2])))]
        if not dom:
            # basit örnekleme
            step = max(1, len(sample) // 5)
            dom = [[int(p[0]), int(p[1]), int(p[2])] for p in sample[::step][:5]]
        hsv_hist = compute_hsv_histogram(image)
    elif dom:
        arr = np.array(dom, dtype=np.float32)
        mean = arr.mean(axis=0)
        avg = [int(round(float(mean[0]))), int(round(float(mean[1]))), int(round(float(mean[2])))]

    palette_labels: list[str] = []
    for c in dom[:5]:
        if not c or len(c) < 3:
            continue
        palette_labels.extend(classify_rgb(float(c[0]), float(c[1]), float(c[2])))
    if avg != [0, 0, 0]:
        palette_labels.extend(classify_rgb(float(avg[0]), float(avg[1]), float(avg[2])))

    # Sıralı tekilleştir
    palette = list(dict.fromkeys(palette_labels))
    primary_family = classify_family(float(avg[0]), float(avg[1]), float(avg[2])) if avg else "unknown"
    if palette:
        first = palette[0]
        primary_family = FASHION_COLOR_PROTOTYPES.get(first, ((0, 0, 0), primary_family))[1]

    search_labels: list[str] = list(palette)
    for fam in {primary_family, *(
        FASHION_COLOR_PROTOTYPES.get(p, ((0, 0, 0), ""))[1] for p in palette
    )}:
        if not fam:
            continue
        search_labels.extend(FAMILY_SEARCH_LABELS.get(fam, ()))
        search_labels.append(fam)

    return {
        "version": 1,
        "dominant_colors": [[int(x) for x in c[:3]] for c in dom[:5]],
        "average_rgb": avg,
        "palette": palette[:8],
        "color_family": primary_family,
        "hsv_hist": hsv_hist,
        "search_labels": list(dict.fromkeys(search_labels))[:24],
    }


def flatten_color_index(color_index: dict[str, Any] | None) -> list[str]:
    if not isinstance(color_index, dict):
        return []
    out: list[str] = []
    out.extend(str(x) for x in (color_index.get("palette") or []) if x)
    out.extend(str(x) for x in (color_index.get("search_labels") or []) if x)
    fam = color_index.get("color_family")
    if fam:
        out.append(str(fam))
    return list(dict.fromkeys(out))


def resolve_color_query(token: str) -> str:
    """Sorgu token → moda renk ailesi."""
    key = (token or "").strip().lower().replace("ı", "i").replace("ş", "s").replace("ğ", "g")
    key = key.replace("ü", "u").replace("ö", "o").replace("ç", "c")
    # doğrudan
    for name, (_proto, family) in FASHION_COLOR_PROTOTYPES.items():
        n = name.replace("ı", "i").replace("ş", "s").replace("ğ", "g")
        n = n.replace("ü", "u").replace("ö", "o").replace("ç", "c")
        if n == key or name == token.strip().lower():
            return family
    # etiket sözlüğünden
    for family, labels in FAMILY_SEARCH_LABELS.items():
        if key in labels or token.strip().lower() in labels:
            return family
    return ""


def color_index_match_score(query_color: str, color_index: dict[str, Any] | None) -> float:
    """Sorgu rengi ile kayıt renk indeksi eşleşmesi."""
    if not query_color or not isinstance(color_index, dict):
        return 0.0
    q = query_color.strip().lower()
    q_family = resolve_color_query(q) or q

    labels = {str(x).lower() for x in flatten_color_index(color_index)}
    cand_family = str(color_index.get("color_family") or "").lower()

    if q in labels or q_family in labels:
        return 0.92
    if cand_family and (cand_family == q_family or cand_family == q):
        return 0.90
    compat = FAMILY_COMPAT.get(q_family, ())
    if cand_family in compat:
        return 0.72
    # average_rgb üzerinden anlık sınıflama
    avg = color_index.get("average_rgb") or []
    if len(avg) >= 3:
        avg_fam = classify_family(float(avg[0]), float(avg[1]), float(avg[2]))
        if avg_fam == q_family:
            return 0.86
        if avg_fam in compat:
            return 0.68
    return 0.0


_LOCKED_COLOR_SOURCES = frozenset({"teach_me", "manual_user", "user"})

# Eşdeğer aile adları (UI / kaba analiz / moda indeks)
COLOR_FAMILY_ALIASES: dict[str, str] = {
    "navy": "navy_blue",
    "blue": "navy_blue",
    "neon_multicolor": "multicolor",
    "grey": "grayscale",
    "gray": "grayscale",
}


def canonicalize_color_family(value: Any) -> str:
    key = str(value or "").strip().lower()
    if not key or key in ("unknown", "none", "null", "belirsiz"):
        return ""
    return COLOR_FAMILY_ALIASES.get(key, key)


def color_is_user_locked(texture_map: dict[str, Any] | None) -> bool:
    tm = texture_map if isinstance(texture_map, dict) else {}
    src = str(tm.get("color_source") or "").strip().lower()
    return src in _LOCKED_COLOR_SOURCES


def resolved_color_family(
    texture_map: dict[str, Any] | None,
    *,
    file_color: str = "",
) -> str:
    """Öğretilmiş renk kilitliyse onu, değilse renk indeksini kullan."""
    tm = texture_map if isinstance(texture_map, dict) else {}
    locked = canonicalize_color_family(tm.get("color_family"))
    if color_is_user_locked(tm) and locked:
        return locked
    ci = tm.get("color_index") if isinstance(tm.get("color_index"), dict) else {}
    for cand in (
        ci.get("color_family") if isinstance(ci, dict) else "",
        tm.get("color_family"),
        file_color,
    ):
        fam = canonicalize_color_family(cand)
        if fam:
            return fam
    return ""


def apply_auto_color(texture_map: dict[str, Any] | None) -> dict[str, Any]:
    """Kullanıcı kilidi yoksa color_family'yi renk indeksinden düzelt."""
    tm = dict(texture_map or {})
    if color_is_user_locked(tm):
        fam = canonicalize_color_family(tm.get("color_family"))
        if fam:
            tm["color_family"] = fam
        return tm
    fam = resolved_color_family(tm)
    if fam:
        tm["color_family"] = fam
        if str(tm.get("color_source") or "") not in _LOCKED_COLOR_SOURCES:
            tm["color_source"] = "auto_index"
    return tm
