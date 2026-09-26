"""Tekstil doku profili — sınıflandırma, renk ailesi, küme atama."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from core.cv2_runtime import harden_cv2_runtime

    # Harden sets OPENCV_OPENCL_DEVICE before cv2 first-import.
    if not harden_cv2_runtime():
        raise ImportError('cv2 unavailable')
    import cv2  # noqa: F401

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Sürüm — indexer ile senkron
TEXTURE_MAP_VERSION = 3
FEATURE_VERSION = 3
THUMBNAIL_VERSION = 1

# Küme grupları — query family'ye göre etiketler dynamic_groups'ta
CLUSTER_EXACT = "exact_same"
CLUSTER_FORMAT = "format_variant"
CLUSTER_RESOLUTION = "resolution_variant"
CLUSTER_VARIANT = "crop_variant"
CLUSTER_COLOR = "color_variant"
CLUSTER_SAME_CLOSE = "same_family_close"
CLUSTER_SAME_STYLE = "same_family_style"
CLUSTER_RELATED = "related_family"
CLUSTER_DISTANT = "far_texture"
CLUSTER_UNRELATED = "unrelated"

# Geriye uyumluluk
CLUSTER_FORMAT_RESOLUTION = CLUSTER_FORMAT
CLUSTER_LEOPARD_SAME_COLOR = CLUSTER_SAME_CLOSE
CLUSTER_LEOPARD_OTHER_COLOR = CLUSTER_COLOR
CLUSTER_RELATED_ANIMAL = CLUSTER_RELATED

CLUSTER_RANK: dict[str, int] = {
    CLUSTER_EXACT: 0,
    CLUSTER_FORMAT: 1,
    CLUSTER_RESOLUTION: 2,
    CLUSTER_VARIANT: 3,
    CLUSTER_COLOR: 4,
    CLUSTER_SAME_CLOSE: 5,
    CLUSTER_SAME_STYLE: 6,
    CLUSTER_RELATED: 7,
    CLUSTER_DISTANT: 8,
    CLUSTER_UNRELATED: 9,
    # legacy keys
    "format_resolution": 1,
    "pattern_variant": 3,
    "leopard_brown_similar": 5,
    "leopard_other_color": 4,
    "related_animal_print": 7,
    "distant_texture": 8,
}

CLUSTER_LABELS: dict[str, str] = {
    CLUSTER_EXACT: "Aynı Görsel",
    CLUSTER_FORMAT: "Aynı Desen / Format Varyantı",
    CLUSTER_VARIANT: "Aynı Desen / Crop Varyantı",
    CLUSTER_COLOR: "Renk Varyantı",
    CLUSTER_SAME_CLOSE: "Benzer Desen",
    CLUSTER_SAME_STYLE: "Benzer Tarz",
    CLUSTER_RELATED: "Yakın Aile",
    CLUSTER_DISTANT: "Uzak Doku",
    CLUSTER_UNRELATED: "Alakasız",
}

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

PATTERN_FAMILIES = (
    "animal_print",
    "floral",
    "paisley",
    "lace",
    "geometric",
    "plaid_check",
    "stripe",
    "polka_dot",
    "monogram_logo",
    "baroque",
    "chain",
    "scarf_border",
    "marble",
    "marble_abstract",
    "abstract",
    "texture_ground",
    "garment_photo",
    "document",
    "icon_logo_non_textile",
    "plain",
    "unknown",
)

COLOR_FAMILIES = (
    "brown_tan",
    "black_white",
    "red_pink",
    "blue",
    "green",
    "yellow",
    "orange",
    "neon_multicolor",
    "grayscale",
    "unknown",
)

BROWN_FAMILIES = {"brown_tan", "orange", "yellow"}
LEOPARD_FAMILY = {"leopard", "jaguar", "cheetah"}
RELATED_ANIMALS = {
    "leopard",
    "jaguar",
    "cheetah",
    "tiger",
    "zebra",
    "snake",
    "crocodile",
    "cow",
    "unknown_animal",
}

FILENAME_TOKENS: dict[str, tuple[str, ...]] = {
    "leopard": ("leopard", "leo", "leopar"),
    "jaguar": ("jaguar",),
    "cheetah": ("cheetah", "çita", "cita"),
    "tiger": ("tiger", "kaplan", "tigerprint"),
    "zebra": ("zebra",),
    "snake": ("snake", "yilan", "yılan", "python", "serpent"),
    "crocodile": ("crocodile", "croco", "timsah", "alligator"),
    "cow": ("cow", "inek", "holstein"),
    "floral": (
        "floral",
        "flower",
        "cicek",
        "çiçek",
        "rose",
        "bloom",
        "fiori",
        "fiore",
        "fleur",
        "gül",
        "gul",
        "papatya",
    ),
    "paisley": ("paisley", "şal", "sal", "boteh"),
    "lace": ("lace", "dantel", "crochet", "openwork", "guipure"),
    "plaid_check": ("plaid", "check", "ekose", "tartan", "gingham"),
    "stripe": ("stripe", "çizgi", "cizgi", "striped", "pin"),
    "polka_dot": ("polka", "puantiye", "dot"),
    "monogram_logo": ("monogram", "logo", "lv", "louis", "vuitton", "gucci", "chanel"),
    "baroque": ("baroque", "versace", "ornate", "barok"),
    "chain": ("chain", "zincir", "link", "catene", "catena"),
    "scarf_border": ("scarf", "eşarp", "esarp", "border", "bordür", "bordur"),
    "marble": ("marble", "mermer"),
    "garment_photo": (
        "shirt",
        "tshirt",
        "tisort",
        "tişört",
        "dress",
        "elbise",
        "mockup",
        "product",
    ),
    "document": ("document", "doc", "pdf", "scan", "invoice"),
    "icon_logo_non_textile": ("logo", "icon", "brand", "marka"),
}


@dataclass
class TextureProfile:
    """Görsel doku haritası özeti."""

    texture_map: dict[str, Any] = field(default_factory=dict)
    pattern_family: str = "unknown"
    animal_print_type: str = ""
    color_family: str = "unknown"
    dominant_palette: list[str] = field(default_factory=list)
    organic_blob_score: float = 0.0
    stripe_score: float = 0.0
    scale_pattern_score: float = 0.0
    edge_density: float = 0.0
    contrast_score: float = 0.0
    repeat_density: float = 0.0
    background_foreground_ratio: float = 0.0
    patch_cluster_signature: str = ""
    spot_stripe_structure: str = "unknown"
    texture_family: str = "unknown"
    pattern_subtype: str = ""
    classification_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": TEXTURE_MAP_VERSION,
            **self.texture_map,
            "pattern_family": self.pattern_family,
            "pattern_subtype": self.pattern_subtype,
            "classification_confidence": round(self.classification_confidence, 4),
            "animal_print_type": self.animal_print_type,
            "color_family": self.color_family,
            "dominant_palette": self.dominant_palette,
            "organic_blob_score": round(self.organic_blob_score, 4),
            "stripe_score": round(self.stripe_score, 4),
            "scale_pattern_score": round(self.scale_pattern_score, 4),
            "edge_density": round(self.edge_density, 4),
            "contrast_score": round(self.contrast_score, 4),
            "repeat_density": round(self.repeat_density, 4),
            "background_foreground_ratio": round(self.background_foreground_ratio, 4),
            "patch_cluster_signature": self.patch_cluster_signature,
            "spot_stripe_structure": self.spot_stripe_structure,
            "texture_family": self.texture_family,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TextureProfile:
        if not data:
            return cls()
        from core.color_index import resolved_color_family

        color = resolved_color_family(data) or str(data.get("color_family") or "unknown")
        return cls(
            texture_map=dict(data),
            pattern_family=data.get("pattern_family", "unknown"),
            animal_print_type=data.get("animal_print_type", ""),
            color_family=color,
            dominant_palette=list(data.get("dominant_palette", [])),
            organic_blob_score=float(data.get("organic_blob_score", 0)),
            stripe_score=float(data.get("stripe_score", 0)),
            scale_pattern_score=float(data.get("scale_pattern_score", 0)),
            edge_density=float(data.get("edge_density", 0)),
            contrast_score=float(data.get("contrast_score", 0)),
            repeat_density=float(data.get("repeat_density", 0)),
            background_foreground_ratio=float(
                data.get("background_foreground_ratio", 0)
            ),
            patch_cluster_signature=str(data.get("patch_cluster_signature", "")),
            spot_stripe_structure=data.get("spot_stripe_structure", "unknown"),
            texture_family=data.get("texture_family", "unknown"),
            pattern_subtype=data.get("pattern_subtype", ""),
            classification_confidence=float(data.get("classification_confidence", 0)),
        )


class TextureAnalyzer:
    """Klasik heuristic doku / animal print analizi."""

    @staticmethod
    def analyze(
        image: np.ndarray,
        filename: str = "",
        path: str = "",
        patch_metas: list[dict[str, Any]] | None = None,
        dominant_colors: list[list[int]] | None = None,
    ) -> TextureProfile:
        if image is None or image.size == 0:
            return TextureProfile()

        rgb = image[:, :, :3] if image.ndim == 3 else np.stack([image] * 3, axis=-1)
        gray = TextureAnalyzer._to_gray(rgb)
        h, w = gray.shape

        edge_density = TextureAnalyzer._edge_density(gray)
        contrast = float(np.std(gray)) / 128.0
        blob_score = TextureAnalyzer._blob_score(gray)
        stripe_score = TextureAnalyzer._stripe_score(gray)
        scale_score = TextureAnalyzer._scale_score(gray)
        repeat_density = TextureAnalyzer._repeat_density(gray)
        bg_fg = TextureAnalyzer._bg_fg_ratio(gray)

        color_family, palette = TextureAnalyzer._color_family(
            rgb, dominant_colors or []
        )

        from core.pattern_classifier import classify_texture

        cls = classify_texture(
            rgb=rgb,
            gray=gray,
            blob=blob_score,
            stripe=stripe_score,
            scale=scale_score,
            edge=edge_density,
            contrast=min(1.0, contrast),
            color_family=color_family,
            repeat=repeat_density,
            bg_fg=bg_fg,
            filename=filename,
            path=path,
        )
        animal_type = cls.animal_print_type
        pattern_family = cls.pattern_family
        spot_stripe = cls.spot_stripe_structure

        patch_sig = TextureAnalyzer._patch_signature(patch_metas or [])
        texture_family = pattern_family
        if animal_type and pattern_family == "animal_print":
            texture_family = f"animal_{animal_type}"

        profile = TextureProfile(
            pattern_family=pattern_family,
            animal_print_type=animal_type,
            color_family=color_family,
            dominant_palette=palette,
            organic_blob_score=blob_score,
            stripe_score=stripe_score,
            scale_pattern_score=scale_score,
            edge_density=edge_density,
            contrast_score=min(1.0, contrast),
            repeat_density=repeat_density,
            background_foreground_ratio=bg_fg,
            patch_cluster_signature=patch_sig,
            spot_stripe_structure=spot_stripe,
            texture_family=texture_family,
            pattern_subtype=cls.pattern_subtype,
            classification_confidence=cls.confidence,
        )
        if pattern_family in ("marble", "abstract", "marble_abstract"):
            profile.pattern_family = "marble_abstract"
        profile.texture_map = {
            **profile.to_dict(),
            "animal_score": round(cls.animal_score, 4),
            "floral_score": round(cls.floral_score, 4),
            "marble_score": round(cls.marble_score, 4),
            "product_score": round(cls.product_score, 4),
            "why_animal_rejected": cls.why_animal_rejected,
            "why_family_selected": cls.why_family_selected,
        }
        return profile

    @staticmethod
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        if HAS_CV2:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    @staticmethod
    def _edge_density(gray: np.ndarray) -> float:
        if HAS_CV2:
            edges = cv2.Canny(gray, 50, 150)
            return float(np.count_nonzero(edges)) / max(edges.size, 1)
        return float(np.std(gray)) / 128.0 * 0.5

    @staticmethod
    def _blob_score(gray: np.ndarray) -> float:
        if not HAS_CV2 or gray.shape[0] < 32:
            return 0.0
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        areas = [cv2.contourArea(c) for c in contours]
        total = gray.shape[0] * gray.shape[1]
        small_blobs = sum(1 for a in areas if 0.001 * total < a < 0.05 * total)
        return min(1.0, small_blobs / 25.0)

    @staticmethod
    def _stripe_score(gray: np.ndarray) -> float:
        if gray.shape[0] < 16:
            return 0.0
        # Yatay gradyan enerjisi vs dikey
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1)).mean()
        gy = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean()
        if gx + gy < 1e-6:
            return 0.0
        stripe = abs(gx - gy) / (gx + gy)
        # FFT yüksek frekans şeritleri
        if HAS_CV2:
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            mag = np.log1p(np.abs(fshift))
            cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
            band = mag[cy - 5 : cy + 5, :].mean() / (mag.mean() + 1e-6)
            stripe = min(1.0, stripe * 0.6 + min(1.0, band / 10.0) * 0.4)
        return float(min(1.0, stripe))

    @staticmethod
    def _scale_score(gray: np.ndarray) -> float:
        if not HAS_CV2 or gray.shape[0] < 32:
            return 0.0
        small = cv2.resize(gray, (64, 64))
        lap = cv2.Laplacian(small, cv2.CV_64F)
        local_var = cv2.blur(lap**2, (8, 8))
        cells = (local_var > local_var.mean()).astype(np.uint8)
        # Hücresel tekrar yoğunluğu
        return min(1.0, float(cells.mean()) * 1.8)

    @staticmethod
    def _repeat_density(gray: np.ndarray) -> float:
        if gray.shape[0] < 32:
            return 0.0
        block = 16
        h, w = gray.shape
        gh = (h - block) // block
        gw = (w - block) // block
        if gh < 1 or gw < 1:
            return 0.0
        cropped = gray[: gh * block, : gw * block]
        tiles = cropped.reshape(gh, block, gw, block).astype(np.float32)
        vars_ = tiles.std(axis=(1, 3))
        return min(1.0, float(vars_.std()) / 40.0)

    @staticmethod
    def _bg_fg_ratio(gray: np.ndarray) -> float:
        thresh = np.median(gray)
        fg = np.count_nonzero(gray < thresh - 15) + np.count_nonzero(gray > thresh + 15)
        return min(1.0, fg / max(gray.size, 1))

    @staticmethod
    def _color_family(
        rgb: np.ndarray,
        dominant: list[list[int]],
    ) -> tuple[str, list[str]]:
        if dominant:
            centers = np.array(dominant[:5], dtype=np.float32)
        else:
            pixels = rgb.reshape(-1, 3).astype(np.float32)
            step = max(1, len(pixels) // 2000)
            sample = pixels[::step]
            if len(sample) < 10:
                return "unknown", []
            if HAS_CV2:
                _, _, centers = cv2.kmeans(
                    sample,
                    min(5, len(sample)),
                    None,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                    3,
                    cv2.KMEANS_PP_CENTERS,
                )
            else:
                centers = sample[:5]

        palette: list[str] = []
        families: list[str] = []
        for c in centers:
            r, g, b = float(c[0]), float(c[1]), float(c[2])
            label = TextureAnalyzer._rgb_to_family(r, g, b)
            palette.append(label)
            families.append(label)

        if not families:
            return "unknown", []
        # En sık renk ailesi
        from collections import Counter

        top = Counter(families).most_common(1)[0][0]
        return top, palette[:5]

    @staticmethod
    def _rgb_to_family(r: float, g: float, b: float) -> str:
        mx, mn = max(r, g, b), min(r, g, b)
        if mx - mn < 25:
            if mx < 60:
                return "black_white"
            if mx > 200:
                return "black_white"
            return "grayscale"
        if r > 180 and g < 100 and b < 100:
            return "red_pink"
        if b > r + 30 and b > g:
            return "blue"
        if g > r and g > b:
            return "green"
        if r > 150 and g > 100 and b < 100:
            return "orange" if r > g + 20 else "brown_tan"
        if r > 100 and g > 80 and b < 80:
            return "brown_tan"
        if r > 200 and g > 200 and b < 120:
            return "yellow"
        if mx > 200 and (r > 200 or g > 200) and (b > 180):
            return "neon_multicolor"
        return "brown_tan"

    @staticmethod
    def _token_in_name(name: str, token: str) -> bool:
        """Dosya adında token — 'leo' floral içinde yanlış eşleşmesin."""
        name = name.lower()
        token = token.lower()
        if not token:
            return False
        pattern = r"(^|[^a-z0-9])" + re.escape(token) + r"([^a-z0-9]|$)"
        return bool(re.search(pattern, name))

    @staticmethod
    def _filename_animal_hint(filename: str, path: str) -> str:
        name = Path(filename).name.lower() if filename else Path(path).name.lower()
        for animal, tokens in FILENAME_TOKENS.items():
            if animal in ("floral", "document", "icon_logo_non_textile"):
                continue
            if any(TextureAnalyzer._token_in_name(name, t) for t in tokens):
                return animal
        return ""

    @staticmethod
    def _filename_pattern_hint(filename: str, path: str) -> str:
        name = Path(filename).name.lower() if filename else Path(path).name.lower()
        for family, tokens in FILENAME_TOKENS.items():
            if family in ANIMAL_PRINT_TYPES or family in (
                "document",
                "icon_logo_non_textile",
            ):
                continue
            if any(TextureAnalyzer._token_in_name(name, t) for t in tokens):
                return family
        return ""

    @staticmethod
    def _filename_token_boost(filename: str, path: str, pattern_family: str) -> float:
        """Token tek başına sınıf değiştirmez — görsel ile desteklenirse boost."""
        name = Path(filename).name.lower() if filename else Path(path).name.lower()
        tokens = FILENAME_TOKENS.get(pattern_family, ())
        if any(TextureAnalyzer._token_in_name(name, t) for t in tokens):
            return 0.15
        return 0.0

    @staticmethod
    def _floral_score(
        rgb: np.ndarray,
        gray: np.ndarray,
        blob: float,
        stripe: float,
        scale: float,
        edge: float,
        repeat: float,
        color_family: str,
    ) -> float:
        """Çiçek/floral desen heuristiği — animal print'ten ayrılır."""
        score = 0.0
        if color_family in ("green", "red_pink", "neon_multicolor", "yellow"):
            score += 0.28
        # Floral: orta benek, düşük şerit/scale, yeşil zemin üzerinde renkli motif
        if 0.12 < blob < 0.62 and stripe < 0.32 and scale < 0.50:
            score += 0.32
        if repeat >= 0.15 and edge >= 0.04:
            score += 0.12
        if rgb is not None and rgb.size > 0 and gray.shape[0] >= 32:
            h, w = gray.shape
            ch, cw = max(1, h // 4), max(1, w // 4)
            cy, cx = h // 2, w // 2
            center = rgb[cy - ch : cy + ch, cx - cw : cx + cw]
            corners = np.concatenate(
                [
                    rgb[: h // 4, : w // 4].reshape(-1, 3),
                    rgb[: h // 4, -w // 4 :].reshape(-1, 3),
                    rgb[-h // 4 :, : w // 4].reshape(-1, 3),
                    rgb[-h // 4 :, -w // 4 :].reshape(-1, 3),
                ]
            )
            if center.size and corners.size:
                c_std = float(np.std(center.astype(np.float32)))
                corner_std = float(np.std(corners.astype(np.float32)))
                if c_std > corner_std * 1.05:
                    score += 0.18
            # Yeşil zemin oranı — çiçek desenlerinde yaygın
            green_px = np.count_nonzero(
                (rgb[:, :, 1] > rgb[:, :, 0] + 15) & (rgb[:, :, 1] > rgb[:, :, 2] + 10)
            )
            if green_px / max(rgb.shape[0] * rgb.shape[1], 1) > 0.25:
                score += 0.15
        return min(1.0, score)

    @staticmethod
    def _classify_pattern(
        blob: float,
        stripe: float,
        scale: float,
        edge: float,
        contrast: float,
        color_family: str,
        fname_hint: str,
        pattern_hint: str = "",
        rgb: np.ndarray | None = None,
        gray: np.ndarray | None = None,
        repeat: float = 0.0,
    ) -> tuple[str, str, str]:
        """animal_type, pattern_family, spot_stripe_structure"""
        # Belge / logo / düz
        if edge < 0.02 and contrast < 0.08:
            return "", "plain", "flat"
        if edge < 0.04 and blob < 0.05 and stripe < 0.15:
            if color_family in ("black_white", "grayscale"):
                return "", "document", "flat"

        floral_score = 0.0
        if rgb is not None and gray is not None:
            floral_score = TextureAnalyzer._floral_score(
                rgb,
                gray,
                blob,
                stripe,
                scale,
                edge,
                repeat,
                color_family,
            )
        if (
            pattern_hint == "floral"
            or "floral" in pattern_hint
            or "flower" in pattern_hint
        ):
            floral_score = max(floral_score, 0.65)
        if "cicek" in pattern_hint or "çiçek" in pattern_hint:
            floral_score = max(floral_score, 0.70)

        scores: dict[str, float] = {
            "leopard": blob * 0.55 + (1 - stripe) * 0.25 + edge * 0.2,
            "zebra": stripe * 0.65
            + (1 if color_family == "black_white" else 0.3) * 0.25,
            "snake": scale * 0.55 + edge * 0.25 + blob * 0.1,
            "tiger": stripe * 0.45
            + (1 if color_family in ("orange", "brown_tan") else 0.2) * 0.35,
            "crocodile": scale * 0.5 + blob * 0.2 + (1 - stripe) * 0.2,
            "cow": blob * 0.35 + (1 if color_family == "black_white" else 0.1) * 0.4,
            "cheetah": blob * 0.5 + edge * 0.3,
            "jaguar": blob * 0.52 + edge * 0.28,
        }

        if fname_hint and fname_hint in scores:
            scores[fname_hint] += 0.25

        best_animal = max(scores, key=scores.get)
        best_score = scores[best_animal]

        if scale > 0.45 and blob < 0.15 and stripe < 0.20 and repeat >= 0.20:
            floral_score = max(floral_score, 0.48)

        # Floral, animal print'ten önce — çiçek benekleri leopard ile karışmasın
        if floral_score >= 0.50 and floral_score >= best_score - 0.08:
            return "", "floral", "floral"
        if floral_score >= 0.62:
            return "", "floral", "floral"

        if best_score < 0.25:
            if floral_score >= 0.38:
                return "", "floral", "floral"
            if stripe > 0.42:
                return "", "stripe", "striped"
            if scale > 0.40 and repeat > 0.25:
                return "", "plaid_check", "check"
            if blob > 0.3:
                return "", "texture_ground", "organic"
            if pattern_hint in PATTERN_FAMILIES:
                return "", pattern_hint, "flat"
            return "", "abstract", "mixed"

        # Non-animal pattern families from heuristics
        if stripe > 0.48 and stripe > blob and stripe > scale:
            if fname_hint:
                return "", "stripe", "striped"
            return "", "stripe", "striped"
        if scale > 0.42 and repeat > 0.30 and blob < 0.20:
            return "", "plaid_check", "check"
        if blob > 0.35 and edge > 0.06 and stripe < 0.25 and scale < 0.35:
            if floral_score < 0.45:
                pass  # continue to animal
        if pattern_hint == "paisley" and blob > 0.20:
            return "", "paisley", "organic"
        if pattern_hint == "lace" or (
            edge > 0.11 and repeat > 0.30 and blob < 0.30 and stripe < 0.24
        ):
            return "", "lace", "openwork"
        if pattern_hint in (
            "monogram_logo",
            "baroque",
            "chain",
            "scarf_border",
            "marble",
        ):
            return "", pattern_hint, "structured"

        # Yeşil zemin + düşük leopard skoru → floral
        if color_family == "green" and best_score < 0.45 and floral_score >= 0.35:
            return "", "floral", "floral"

        # Düşük benek — araba/foto gibi görseller animal print sayılmasın
        if blob < 0.12:
            if fname_hint in scores and fname_hint in ANIMAL_PRINT_TYPES:
                return fname_hint, "animal_print", "spotted"
            if floral_score >= 0.30:
                return "", "floral", "floral"
            if stripe > 0.35:
                return "", "stripe", "striped"
            return "", "abstract", "mixed"

        spot_stripe = "spotted" if blob > stripe else "striped"
        if scale > max(blob, stripe):
            spot_stripe = "scaled"

        return best_animal, "animal_print", spot_stripe

    @staticmethod
    def _patch_signature(patch_metas: list[dict[str, Any]]) -> str:
        if not patch_metas:
            return ""
        parts = []
        for p in patch_metas[:9]:
            ph = (p.get("phash") or "")[:4]
            parts.append(ph)
        return "-".join(parts)


def palette_similarity(fam_a: str, fam_b: str) -> float:
    if not fam_a or not fam_b or fam_a == "unknown" or fam_b == "unknown":
        return 0.5
    if fam_a == fam_b:
        return 1.0
    if fam_a in BROWN_FAMILIES and fam_b in BROWN_FAMILIES:
        return 0.85
    if fam_a == "black_white" and fam_b == "grayscale":
        return 0.75
    return 0.2


def animal_family_similarity(type_a: str, type_b: str) -> float:
    if not type_a or not type_b:
        return 0.3
    if type_a == type_b:
        return 1.0
    if type_a in LEOPARD_FAMILY and type_b in LEOPARD_FAMILY:
        return 0.9
    if type_a in LEOPARD_FAMILY and type_b in ("tiger", "cheetah", "jaguar"):
        return 0.75
    if type_a in LEOPARD_FAMILY and type_b in ("zebra", "snake", "crocodile", "cow"):
        return 0.55
    if type_a in RELATED_ANIMALS and type_b in RELATED_ANIMALS:
        return 0.5
    return 0.15


def texture_family_score(prof_a: TextureProfile, prof_b: TextureProfile) -> float:
    if (
        prof_a.pattern_family == prof_b.pattern_family
        and prof_a.pattern_family != "unknown"
    ):
        base = 0.7
    else:
        base = 0.25
    blob = 1.0 - abs(prof_a.organic_blob_score - prof_b.organic_blob_score)
    stripe = 1.0 - abs(prof_a.stripe_score - prof_b.stripe_score)
    scale = 1.0 - abs(prof_a.scale_pattern_score - prof_b.scale_pattern_score)
    struct = blob * 0.4 + stripe * 0.35 + scale * 0.25
    animal = animal_family_similarity(
        prof_a.animal_print_type, prof_b.animal_print_type
    )
    return min(1.0, base * 0.4 + struct * 0.35 + animal * 0.25)


def assign_cluster_group(
    query: TextureProfile,
    candidate: TextureProfile,
    *,
    is_self: bool = False,
    score: float = 0.0,
    phash_sim: float = 0.0,
    dhash_sim: float = 0.0,
    patch_sim: float = 0.0,
    different_extension: bool = False,
    is_color_variant: bool = False,
    filename: str = "",
) -> tuple[str, str]:
    """Küme grubu ve kısa açıklama döndür."""
    if is_self or score >= 0.99:
        return CLUSTER_EXACT, "Aynı dosya"

    from core.taxonomy import normalize_family

    qf = query.pattern_family or "unknown"
    cf = candidate.pattern_family or "unknown"
    qf_n = normalize_family(qf)
    cf_n = normalize_family(cf)
    # Bilinmeyen aileler birbirine "uyumlu" kabul edilmez.
    # Aksi halde genel fotoğraflar aynı desen ailesine sızabiliyor.
    _non_family = {"", "unknown", "texture_ground", "plain", "document", "garment_photo", "icon_logo_non_textile"}
    family_compatible = (
        qf_n == cf_n
        and qf_n not in _non_family
    )

    if phash_sim >= 0.98:
        if different_extension:
            return CLUSTER_FORMAT, "Aynı görsel — farklı format/çözünürlük"
        return CLUSTER_EXACT, "Birebir / neredeyse aynı desen"
    if dhash_sim >= 0.98 and patch_sim >= 0.85 and family_compatible:
        if different_extension:
            return CLUSTER_FORMAT, "Aynı görsel — farklı format/çözünürlük"
        return CLUSTER_EXACT, "Birebir / neredeyse aynı desen"

    if family_compatible and (
        phash_sim >= 0.92 or (patch_sim >= 0.80 and dhash_sim >= 0.88)
    ):
        if different_extension:
            return CLUSTER_FORMAT, "Aynı desen — format veya çözünürlük farkı"
        return CLUSTER_VARIANT, "Aynı desen varyantı (crop/scale/renk)"
    if phash_sim >= 0.92 and dhash_sim >= 0.90:
        if different_extension:
            return CLUSTER_FORMAT, "Aynı desen — format veya çözünürlük farkı"
        return CLUSTER_VARIANT, "Aynı görsel — family etiketi farklı olabilir"

    strong_color_variant = is_color_variant and patch_sim >= 0.85 and dhash_sim >= 0.75
    strong_structural_variant = patch_sim >= 0.90 and dhash_sim >= 0.78
    if (family_compatible or strong_color_variant or strong_structural_variant) and (
        phash_sim >= 0.78 or patch_sim >= 0.65
    ):
        if is_color_variant:
            return CLUSTER_VARIANT, "Aynı motif — renk/kontrast varyantı"
        if patch_sim >= 0.60 and dhash_sim >= 0.70:
            return CLUSTER_VARIANT, "Aynı repeat/motif — kesit farklı"

    q_animal = query.animal_print_type
    c_animal = candidate.animal_print_type
    pal = palette_similarity(query.color_family, candidate.color_family)

    # Marble query — animal print adaylarını aşağı indir
    if qf_n == "marble_abstract":
        if cf_n == "animal_print" or c_animal:
            if phash_sim >= 0.92 or patch_sim >= 0.82:
                return CLUSTER_VARIANT, "Aynı görsel — family etiketi farklı"
            return CLUSTER_UNRELATED, "Alakasız — animal print (mermer sorgusu)"
        if cf_n == "marble_abstract" or cf in ("marble", "abstract", "texture_ground"):
            if patch_sim >= 0.45 or texture_family_score(query, candidate) >= 0.5:
                return CLUSTER_SAME_CLOSE, "Benzer mermer / soyut doku"
            return CLUSTER_SAME_STYLE, "Benzer renk akışı"
        if cf_n == "floral":
            return CLUSTER_UNRELATED, "Alakasız — floral desen"

    # Floral query — animal bastır
    if qf_n == "floral":
        if cf_n == "animal_print" or c_animal:
            if phash_sim >= 0.92 or patch_sim >= 0.80:
                return CLUSTER_VARIANT, "Aynı görsel — family etiketi farklı"
            return CLUSTER_UNRELATED, "Alakasız — hayvan deseni (floral sorgu)"
        if cf_n == "floral":
            if patch_sim >= 0.45:
                return CLUSTER_SAME_CLOSE, "Benzer çiçek deseni"
            return CLUSTER_SAME_STYLE, "Benzer floral motif"
        return CLUSTER_DISTANT, "Uzak doku"

    if candidate.pattern_family == "floral" or candidate.texture_family == "floral":
        if query.pattern_family == "floral":
            if phash_sim >= 0.92 or patch_sim >= 0.80:
                return CLUSTER_VARIANT, "Aynı floral desen — crop/ölçek"
            if phash_sim >= 0.70 or patch_sim >= 0.55:
                return CLUSTER_SAME_STYLE, "Benzer floral desen"
            return CLUSTER_DISTANT, "Zayıf floral benzerliği"
        if query.pattern_family not in ("floral", "unknown"):
            return CLUSTER_UNRELATED, "Alakasız — çiçek/floral desen"

    # Lace — ince delikli doku
    if cf_n == "lace":
        if qf_n == "lace":
            if patch_sim >= 0.45 or phash_sim >= 0.70:
                return CLUSTER_SAME_CLOSE, "Benzer dantel / lace"
            return CLUSTER_SAME_STYLE, "Benzer dantel dokusu"
        if qf_n not in ("unknown", "texture_ground", "geometric"):
            return CLUSTER_UNRELATED, "Alakasız — dantel desen"

    # Animal print — aile içi renk varyantı generic Renk Varyantı olmasın
    from core.pattern_classifier import is_confident_animal_print

    q_animal_conf = is_confident_animal_print(query)
    if q_animal_conf and (
        qf_n == "animal_print"
        or q_animal in LEOPARD_FAMILY
        or q_animal in RELATED_ANIMALS
    ):
        if cf_n == "animal_print" or c_animal in RELATED_ANIMALS:
            if q_animal and c_animal and q_animal == c_animal:
                if pal >= 0.70 and patch_sim >= 0.45:
                    return CLUSTER_SAME_CLOSE, f"Benzer {c_animal} deseni"
                if patch_sim >= 0.40 or phash_sim >= 0.72:
                    return CLUSTER_SAME_CLOSE, f"Benzer {c_animal} — ton farkı"
                return CLUSTER_COLOR, f"Farklı renk {c_animal}"
            if q_animal in LEOPARD_FAMILY and c_animal in LEOPARD_FAMILY:
                if patch_sim >= 0.40 or phash_sim >= 0.70:
                    return CLUSTER_SAME_CLOSE, "Benzer leopard benek yapısı"
                return CLUSTER_COLOR, "Farklı renk leopard"
            if c_animal or cf_n == "animal_print":
                return CLUSTER_RELATED, f"Yakın animal print — {c_animal or 'benzer'}"
        if cf_n in ("floral", "document", "plain", "monogram_logo", "lace"):
            return CLUSTER_UNRELATED, f"Alakasız — {cf_n}"

    # Aynı desen ailesi
    if qf_n == cf_n and qf_n not in ("unknown", "plain", "document"):
        if is_color_variant or candidate.color_family != query.color_family:
            if pal < 0.55:
                return CLUSTER_COLOR, f"Renk varyantı — {cf_n}"
        if patch_sim >= 0.50 or phash_sim >= 0.72:
            label = c_animal or query.pattern_subtype or cf_n
            return CLUSTER_SAME_CLOSE, f"Benzer {label} doku"
        if patch_sim >= 0.35:
            return CLUSTER_SAME_STYLE, f"Benzer {cf_n} tarzı"

    if score < 0.40:
        return CLUSTER_UNRELATED, "Düşük skor — alakasız"

    if cf_n in ("texture_ground", "abstract", "marble", "marble_abstract"):
        return CLUSTER_DISTANT, "Uzak organik doku"

    return CLUSTER_DISTANT, "Zayıf ilişkili doku"


def hierarchy_sort_key(
    cluster_group: str,
    hierarchy_score: float,
    score: float,
) -> tuple[int, float, float]:
    from core.dynamic_groups import hierarchy_sort_key as _dg_sort

    return _dg_sort(cluster_group, hierarchy_score, score)


def compute_hierarchy_score(
    query: TextureProfile,
    candidate: TextureProfile,
    visual: float,
    patch: float,
    palette_sim: float,
    texture_fam: float,
) -> float:
    return min(
        1.0,
        (
            0.30 * visual
            + 0.25 * patch
            + 0.20 * texture_fam
            + 0.15 * palette_sim
            + 0.10
            * animal_family_similarity(
                query.animal_print_type, candidate.animal_print_type
            )
        ),
    )
