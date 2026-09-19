"""Sıkı tekstil desen sınıflandırması — animal / floral / marble ayrımı."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.taxonomy import infer_subtype, normalize_family

ANIMAL_PRINT_MIN_CONFIDENCE = 0.70
LEOPARD_MIN_CONFIDENCE = 0.75
SUBTYPE_MIN_CONFIDENCE = 0.70

ANIMAL_PRINT_TYPES = (
    "leopard",
    "jaguar",
    "cheetah",
    "tiger",
    "zebra",
    "snake",
    "crocodile",
    "cow",
    "unknown_animal",
)

BROWN_FAMILIES = {"brown_tan", "orange", "yellow"}
LEOPARD_FAMILY = {"leopard", "jaguar", "cheetah"}


@dataclass
class ClassificationResult:
    pattern_family: str = "unknown"
    animal_print_type: str = ""
    pattern_subtype: str = ""
    confidence: float = 0.0
    spot_stripe_structure: str = "unknown"
    animal_score: float = 0.0
    floral_score: float = 0.0
    marble_score: float = 0.0
    product_score: float = 0.0
    why_animal_rejected: str = ""
    why_family_selected: str = ""


def _token_in_name(name: str, token: str) -> bool:
    import re

    name = name.lower()
    token = token.lower()
    if not token:
        return False
    pattern = r"(^|[^a-z0-9])" + re.escape(token) + r"([^a-z0-9]|$)"
    return bool(re.search(pattern, name))


def filename_animal_hint(filename: str, path: str) -> str:
    from core.texture_profile import FILENAME_TOKENS

    name = Path(filename).name.lower() if filename else Path(path).name.lower()
    for animal, tokens in FILENAME_TOKENS.items():
        if animal in ("floral", "document", "icon_logo_non_textile", "marble"):
            continue
        if any(_token_in_name(name, t) for t in tokens):
            return animal
    return ""


def filename_pattern_hint(filename: str, path: str) -> str:
    from core.texture_profile import FILENAME_TOKENS

    name = Path(filename).name.lower() if filename else Path(path).name.lower()
    for family, tokens in FILENAME_TOKENS.items():
        if family in ANIMAL_PRINT_TYPES or family in (
            "document",
            "icon_logo_non_textile",
        ):
            continue
        if any(_token_in_name(name, t) for t in tokens):
            return family
    return ""


def product_photo_score(
    rgb: np.ndarray, gray: np.ndarray, bg_fg: float, edge: float, contrast: float
) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    score = 0.0
    h, w = gray.shape
    if bg_fg > 0.62:
        score += 0.30
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(np.mean(border)) > 195:
        score += 0.28
    if edge < 0.055 and contrast < 0.14:
        score += 0.12
    if rgb is not None and rgb.size:
        center = rgb[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        if center.size:
            c_std = float(np.std(center.astype(np.float32)))
            full_std = float(np.std(rgb.astype(np.float32)))
            if full_std > 0 and c_std < full_std * 0.72:
                score += 0.18
    return min(1.0, score)


def marble_abstract_score(
    rgb: np.ndarray | None,
    gray: np.ndarray | None,
    blob: float,
    stripe: float,
    scale: float,
    edge: float,
    contrast: float,
    color_family: str,
    repeat: float,
) -> float:
    score = 0.0
    if color_family in ("neon_multicolor", "blue", "red_pink", "orange"):
        score += 0.32
    if (
        color_family in ("blue", "red_pink", "neon_multicolor")
        and stripe < 0.28
        and edge < 0.14
    ):
        score += 0.22
    if contrast > 0.12 and stripe < 0.36 and 0.10 < blob < 0.58 and scale < 0.38:
        score += 0.32
    if edge < 0.09 and contrast > 0.10 and repeat < 0.28:
        score += 0.22
    if blob < 0.50 and stripe < 0.32 and repeat < 0.22:
        score += 0.12
    if rgb is not None and rgb.size:
        px = rgb.reshape(-1, 3).astype(np.float32)
        spread = float(np.std(px, axis=0).mean())
        if spread > 28 and stripe < 0.35:
            score += 0.15
    return min(1.0, score)


def _hue_diversity(rgb: np.ndarray | None) -> float:
    """0–1: kromatik piksellerde hue yayılımı. Leopar düşük, çiçek yüksek."""
    if rgb is None or rgb.size == 0 or rgb.ndim != 3 or rgb.shape[2] < 3:
        return 0.0
    arr = rgb.reshape(-1, 3).astype(np.float32)
    if arr.shape[0] > 8000:
        arr = arr[:: max(1, arr.shape[0] // 8000)]
    mx = arr.max(axis=1)
    chroma = mx - arr.min(axis=1)
    mask = chroma > 18.0
    if float(np.mean(mask)) < 0.08:
        return 0.0
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    hue = np.zeros(arr.shape[0], dtype=np.float32)
    rc = (g - b) / np.maximum(chroma, 1e-3)
    gc = 2.0 + (b - r) / np.maximum(chroma, 1e-3)
    bc = 4.0 + (r - g) / np.maximum(chroma, 1e-3)
    hue = np.where((mx == r) & mask, rc, hue)
    hue = np.where((mx == g) & mask, gc, hue)
    hue = np.where((mx == b) & mask, bc, hue)
    hue = ((hue / 6.0) % 1.0)[mask]
    hist, _ = np.histogram(hue, bins=12, range=(0.0, 1.0))
    occupied = int(np.count_nonzero(hist > max(3, hue.size * 0.03)))
    return min(1.0, occupied / 8.0)


def floral_score(
    rgb: np.ndarray | None,
    gray: np.ndarray | None,
    blob: float,
    stripe: float,
    scale: float,
    edge: float,
    repeat: float,
    color_family: str,
) -> float:
    score = 0.0
    if color_family in ("green", "red_pink", "neon_multicolor", "yellow"):
        score += 0.26
    hue_div = _hue_diversity(rgb)
    if hue_div >= 0.42:
        # Leopar 2-3 sıcak tonda kalır; çiçekte hue çeşitliliği yüksektir.
        score += 0.22
    if 0.12 < blob < 0.62 and stripe < 0.32 and scale < 0.50:
        score += 0.30
    if repeat >= 0.14 and edge >= 0.04:
        score += 0.12
    if rgb is not None and rgb.size and gray is not None and gray.shape[0] >= 32:
        green_px = np.count_nonzero(
            (rgb[:, :, 1] > rgb[:, :, 0] + 15) & (rgb[:, :, 1] > rgb[:, :, 2] + 10)
        )
        if green_px / max(rgb.shape[0] * rgb.shape[1], 1) > 0.22:
            score += 0.16
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        ch, cw = max(1, h // 4), max(1, w // 4)
        center = rgb[cy - ch : cy + ch, cx - cw : cx + cw]
        if center.size:
            c_std = float(np.std(center.astype(np.float32)))
            corners = np.concatenate(
                [
                    rgb[: h // 4, : w // 4].reshape(-1, 3),
                    rgb[-h // 4 :, -w // 4 :].reshape(-1, 3),
                ]
            )
            if (
                corners.size
                and c_std > float(np.std(corners.astype(np.float32))) * 1.04
            ):
                score += 0.14
    return min(1.0, score)


def _leopard_evidence(
    blob, stripe, scale, edge, contrast, color_family, repeat
) -> float:
    if blob < 0.18 or stripe > 0.42:
        return 0.0
    # Yüksek blob + düşük stripe → rosette/benek (scale yüksek olsa bile)
    if blob > 0.65 and stripe < 0.20:
        score = 0.72 + min(0.12, (blob - 0.65) * 0.3)
        if color_family in BROWN_FAMILIES or color_family == "black_white":
            score += 0.06
        return min(1.0, score)
    if scale > 0.72 and blob < 0.40:
        return 0.0
    score = min(1.0, blob) * 0.30
    if 0.20 <= blob <= 0.70:
        score += 0.18
    if stripe < 0.25:
        score += 0.14
    if color_family in BROWN_FAMILIES or color_family == "black_white":
        score += 0.14
    if repeat >= 0.10:
        score += 0.08
    if 0.04 <= edge <= 0.22:
        score += 0.08
    return min(1.0, score)


def _zebra_evidence(blob, stripe, scale, color_family, repeat) -> float:
    if stripe < 0.40 or blob > 0.40:
        return 0.0
    score = stripe * 0.45
    if color_family == "black_white":
        score += 0.28
    if repeat >= 0.12:
        score += 0.10
    if scale < 0.30:
        score += 0.08
    return min(1.0, score)


def _snake_evidence(blob, stripe, scale, edge, repeat) -> float:
    # Diamond/scale tessellation: scale+repeat high. Spots (leopard) are high blob, low scale.
    if stripe > 0.38:
        return 0.0
    if scale < 0.34:
        return 0.0
    if blob > 0.68:
        return 0.0
    score = scale * 0.40 + edge * 0.16
    if 0.06 < blob < 0.55:
        score += 0.10
    if repeat >= 0.16:
        score += 0.14
    if scale >= 0.42 and stripe < 0.28:
        score += 0.08
    return min(1.0, score)


def _tiger_evidence(blob, stripe, scale, color_family) -> float:
    if stripe < 0.30 or color_family not in ("orange", "brown_tan", "yellow"):
        return 0.0
    if color_family == "black_white":
        return 0.0
    score = stripe * 0.38 + blob * 0.22
    if scale < 0.35:
        score += 0.10
    return min(1.0, score)


def _crocodile_evidence(blob, stripe, scale, repeat) -> float:
    if scale < 0.45 or blob < 0.10 or blob > 0.55:
        return 0.0
    score = scale * 0.40 + blob * 0.18
    if repeat >= 0.15 and stripe < 0.28:
        score += 0.12
    return min(1.0, score)


def animal_subtype_scores(
    blob: float,
    stripe: float,
    scale: float,
    edge: float,
    contrast: float,
    color_family: str,
    repeat: float,
) -> dict[str, float]:
    return {
        "leopard": _leopard_evidence(
            blob, stripe, scale, edge, contrast, color_family, repeat
        ),
        "jaguar": _leopard_evidence(
            blob, stripe, scale, edge, contrast, color_family, repeat
        )
        * 0.95,
        "cheetah": _leopard_evidence(
            blob, stripe, scale, edge, contrast, color_family, repeat
        )
        * 0.92,
        "zebra": _zebra_evidence(blob, stripe, scale, color_family, repeat),
        "snake": _snake_evidence(blob, stripe, scale, edge, repeat),
        "tiger": _tiger_evidence(blob, stripe, scale, color_family),
        "crocodile": _crocodile_evidence(blob, stripe, scale, repeat),
        "cow": min(
            1.0, blob * 0.35 + (0.25 if color_family == "black_white" else 0.05)
        ),
    }


def classify_texture(
    *,
    rgb: np.ndarray | None,
    gray: np.ndarray | None,
    blob: float,
    stripe: float,
    scale: float,
    edge: float,
    contrast: float,
    color_family: str,
    repeat: float,
    bg_fg: float,
    filename: str = "",
    path: str = "",
) -> ClassificationResult:
    result = ClassificationResult()
    fname_hint = filename_animal_hint(filename, path)
    pattern_hint = filename_pattern_hint(filename, path)

    if edge < 0.02 and contrast < 0.08:
        result.pattern_family = "plain"
        result.spot_stripe_structure = "flat"
        result.confidence = 0.65
        result.why_family_selected = "düz düşük kontrast"
        return result

    result.product_score = product_photo_score(rgb, gray, bg_fg, edge, contrast)
    if result.product_score >= 0.58:
        result.pattern_family = "garment_photo"
        result.confidence = result.product_score
        result.why_family_selected = "ürün/mockup fotoğrafı"
        result.why_animal_rejected = "ürün fotoğrafı — desen yüzeyi net değil"
        return result

    result.floral_score = floral_score(
        rgb, gray, blob, stripe, scale, edge, repeat, color_family
    )
    result.marble_score = marble_abstract_score(
        rgb,
        gray,
        blob,
        stripe,
        scale,
        edge,
        contrast,
        color_family,
        repeat,
    )
    floral_name = pattern_hint == "floral" or any(
        t in pattern_hint for t in ("flower", "cicek", "çiçek", "fiori", "fiore")
    )
    if floral_name:
        result.floral_score = max(result.floral_score, 0.62)
    if pattern_hint == "marble":
        result.marble_score = max(result.marble_score, 0.50)

    animals = animal_subtype_scores(
        blob, stripe, scale, edge, contrast, color_family, repeat
    )
    open_floral_motif = (
        result.floral_score >= 0.35 and bg_fg < 0.55 and scale < 0.82
    ) or floral_name
    if open_floral_motif:
        for animal in LEOPARD_FAMILY:
            animals[animal] = min(animals.get(animal, 0), 0.50)
    # Yüksek benek + düşük çizgi → leopard ailesi baskın (scale yanıltmasın)
    # Çiçek motifleri de yüksek blob üretir; floral kanıt varken zorlama.
    # Pul/tessellation (yüksek scale+repeat) yılan olabilir; leopar zorlama.
    tessellated_skin = scale >= 0.40 and repeat >= 0.16 and stripe < 0.30
    if blob > 0.60 and stripe < 0.22 and not open_floral_motif and not tessellated_skin:
        animals["leopard"] = max(animals.get("leopard", 0), 0.76)
        animals["jaguar"] = max(animals.get("jaguar", 0), 0.74)
        animals["cheetah"] = max(animals.get("cheetah", 0), 0.72)
        animals["snake"] = min(animals.get("snake", 0), 0.35)
        animals["crocodile"] = min(animals.get("crocodile", 0), 0.40)
    best_animal = max(animals, key=animals.get)
    result.animal_score = animals[best_animal]

    leopard_spotted = (
        blob > 0.55
        and stripe < 0.22
        and best_animal in LEOPARD_FAMILY
        and not open_floral_motif
    )

    # Dosya adı tek başına animal yapmaz — sadece görsel kanıt varsa küçük boost
    if fname_hint in animals and animals.get(fname_hint, 0) >= 0.40:
        animals[fname_hint] = min(1.0, animals[fname_hint] + 0.08)
        best_animal = max(animals, key=animals.get)
        result.animal_score = animals[best_animal]
        leopard_spotted = leopard_spotted or (
            blob > 0.55 and stripe < 0.22 and best_animal in LEOPARD_FAMILY
        )

    # Çok renkli soyut palet → marble önceliği (leopard paleti değil)
    if color_family in ("neon_multicolor", "blue", "red_pink") and stripe < 0.28:
        result.marble_score = max(result.marble_score, 0.55 + contrast * 0.15)
        if edge < 0.13:
            result.marble_score = max(result.marble_score, 0.62)

    reject_reasons: list[str] = []
    if (
        result.marble_score >= 0.48
        and result.marble_score >= result.animal_score - 0.05
    ):
        reject_reasons.append("mermer/soyut akış baskın")
    if (
        result.floral_score >= 0.50
        and result.floral_score >= result.animal_score - 0.06
    ):
        reject_reasons.append("floral motif baskın")
    if stripe > 0.45 and best_animal not in ("zebra", "tiger"):
        reject_reasons.append("yüksek stripe — animal print değil")
    if (
        scale > 0.45
        and best_animal not in ("snake", "crocodile")
        and not leopard_spotted
    ):
        reject_reasons.append("yüksek scale — pul/grid değil")
    if blob < 0.15 and best_animal in LEOPARD_FAMILY:
        reject_reasons.append("yetersiz benek/blob")
    if (
        color_family in ("neon_multicolor", "blue", "red_pink")
        and best_animal in LEOPARD_FAMILY
        and not leopard_spotted
    ):
        reject_reasons.append("neon/çok renkli palet — leopard değil")

    result.why_animal_rejected = "; ".join(reject_reasons) if reject_reasons else ""

    # Güçlü leopard benek kanıtı — reject cezalarını uygulama
    if leopard_spotted and result.animal_score >= LEOPARD_MIN_CONFIDENCE - 0.02:
        reject_reasons = []
        result.why_animal_rejected = ""

    # Marble / floral öncelik — animal'dan önce
    if result.marble_score >= 0.50 and (
        result.marble_score >= result.animal_score
        or result.marble_score >= 0.58
        or (
            color_family in ("neon_multicolor", "blue", "red_pink")
            and result.marble_score >= 0.52
        )
    ):
        result.pattern_family = "marble_abstract"
        result.confidence = result.marble_score
        result.why_family_selected = "akışkan boya / mermer damar heuristiği"
        sub, conf = infer_subtype(
            "marble_abstract",
            organic_blob=blob,
            stripe=stripe,
            scale=scale,
            contrast=contrast,
            filename=filename,
        )
        result.pattern_subtype = sub
        result.confidence = max(result.confidence, conf)
        result.spot_stripe_structure = "fluid"
        return result

    if (
        result.floral_score >= 0.52
        and result.floral_score >= result.animal_score - 0.04
    ):
        result.pattern_family = "floral"
        result.confidence = result.floral_score
        result.why_family_selected = "çiçek/floral motif heuristiği"
        sub, conf = infer_subtype(
            "floral",
            organic_blob=blob,
            stripe=stripe,
            scale=scale,
            filename=filename,
        )
        result.pattern_subtype = sub
        result.confidence = max(result.confidence, conf)
        result.spot_stripe_structure = "floral"
        return result

    if stripe > 0.48 and stripe > blob and stripe > scale:
        result.pattern_family = "stripe"
        result.confidence = 0.68
        result.why_family_selected = "baskın stripe yapısı"
        result.spot_stripe_structure = "striped"
        return result

    if scale > 0.42 and repeat > 0.30 and blob < 0.20:
        result.pattern_family = "plaid_check"
        result.confidence = 0.62
        result.why_family_selected = "kare/ekose tekrar"
        return result

    # Animal print — sıkı eşik
    min_conf = (
        LEOPARD_MIN_CONFIDENCE
        if best_animal in LEOPARD_FAMILY
        else SUBTYPE_MIN_CONFIDENCE
    )
    if reject_reasons and not leopard_spotted:
        result.animal_score = min(result.animal_score, min_conf - 0.15)

    if result.animal_score < min_conf:
        if result.marble_score >= 0.38:
            result.pattern_family = "marble_abstract"
            result.confidence = result.marble_score
            result.why_family_selected = "düşük animal kanıt — mermer/soyut"
        elif result.floral_score >= 0.40:
            result.pattern_family = "floral"
            result.confidence = result.floral_score
            result.why_family_selected = "düşük animal kanıt — floral"
        elif blob > 0.25:
            result.pattern_family = "texture_ground"
            result.confidence = 0.45
            result.why_family_selected = "belirsiz organik doku"
        else:
            result.pattern_family = "unknown"
            result.confidence = 0.40
            result.why_family_selected = "yetersiz kanıt"
        if not result.why_animal_rejected:
            result.why_animal_rejected = (
                f"animal skor {result.animal_score:.2f} < {min_conf:.2f}"
            )
        return result

    if result.animal_score < ANIMAL_PRINT_MIN_CONFIDENCE:
        result.pattern_family = "texture_ground"
        result.confidence = result.animal_score
        result.why_family_selected = "animal kanıtı eşik altı"
        result.why_animal_rejected = (
            f"confidence {result.animal_score:.2f} < {ANIMAL_PRINT_MIN_CONFIDENCE}"
        )
        return result

    result.pattern_family = "animal_print"
    result.animal_print_type = best_animal
    result.confidence = result.animal_score
    result.why_family_selected = f"güçlü {best_animal} görsel kanıtı"
    result.spot_stripe_structure = (
        "spotted" if blob > stripe else "striped" if stripe > scale else "scaled"
    )
    sub, conf = infer_subtype(
        "animal_print",
        best_animal,
        organic_blob=blob,
        stripe=stripe,
        scale=scale,
        contrast=contrast,
        filename=filename,
    )
    result.pattern_subtype = sub
    result.confidence = max(result.confidence, conf)
    return result


def is_confident_animal_print(profile) -> bool:
    from core.taxonomy import normalize_family

    fam = normalize_family(getattr(profile, "pattern_family", "") or "")
    conf = float(getattr(profile, "classification_confidence", 0) or 0)
    return fam == "animal_print" and conf >= ANIMAL_PRINT_MIN_CONFIDENCE


def effective_query_family(profile) -> str:
    from core.taxonomy import normalize_family

    fam = normalize_family(getattr(profile, "pattern_family", "") or "unknown")
    conf = float(getattr(profile, "classification_confidence", 0) or 0)
    if fam == "animal_print" and conf < ANIMAL_PRINT_MIN_CONFIDENCE:
        return "unknown"
    if fam in ("marble", "abstract"):
        return "marble_abstract"
    return fam or "unknown"
