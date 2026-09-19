"""texture_quality test veri seti — leopard senaryoları + ek family klasörleri."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent / "sample_data" / "texture_quality"
LEOPARD = ROOT / "leopard"


def _spots(
    w: int, h: int, seed: int, base: tuple, spot: tuple, n: int = 40
) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), base)
    d = ImageDraw.Draw(img)
    for _ in range(n):
        x, y = rng.randint(15, w - 15), rng.randint(15, h - 15)
        rx, ry = rng.randint(10, 26), rng.randint(8, 20)
        d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=spot)
        d.ellipse([x - rx // 2, y - ry // 2, x + rx // 2, y + ry // 2], fill=base)
    return img


def _zebra(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    for x in range(0, w, 22):
        d.rectangle([x, 0, x + 11, h], fill=(15, 15, 15))
    return img


def _snake(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (55, 85, 45))
    d = ImageDraw.Draw(img)
    for gy in range(0, h, 16):
        for gx in range(0, w, 16):
            c = (140, 130, 90) if (gx // 16 + gy // 16) % 2 == 0 else (90, 80, 55)
            d.rectangle([gx, gy, gx + 15, gy + 15], fill=c)
    return img


def _floral(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (35, 110, 55))
    d = ImageDraw.Draw(img)
    for cx, cy in [(90, 90), (200, 150)]:
        for i in range(8):
            a = math.radians(i * 45)
            dx, dy = int(math.cos(a) * 28), int(math.sin(a) * 28)
            d.ellipse(
                [cx + dx - 10, cy + dy - 10, cx + dx + 10, cy + dy + 10],
                fill=(255, 90, 140),
            )
    return img


def generate_leopard_set() -> None:
    LEOPARD.mkdir(parents=True, exist_ok=True)
    base = _spots(512, 512, 100, (200, 150, 80), (35, 20, 10))
    base.save(LEOPARD / "leopard_original.jpg", quality=92)
    base.resize((256, 256)).save(
        LEOPARD / "leopard_same_different_resolution.jpg", quality=90
    )
    base.save(LEOPARD / "leopard_same_tif.tif")
    base.crop((100, 100, 412, 412)).save(LEOPARD / "leopard_crop.jpg", quality=90)
    base.resize((640, 640)).save(LEOPARD / "leopard_scaled.jpg", quality=90)
    _spots(512, 512, 101, (190, 140, 75), (30, 18, 8)).save(
        LEOPARD / "leopard_color_variant.jpg", quality=90
    )
    _spots(512, 512, 43, (200, 150, 80), (35, 20, 10)).save(
        LEOPARD / "leopard_similar_01.jpg", quality=90
    )
    _spots(512, 512, 44, (195, 145, 78), (38, 22, 12)).save(
        LEOPARD / "leopard_similar_02.jpg", quality=90
    )
    _spots(512, 512, 102, (160, 40, 50), (20, 10, 10)).save(
        LEOPARD / "red_leopard.jpg", quality=90
    )
    _spots(512, 512, 103, (50, 70, 180), (15, 15, 40)).save(
        LEOPARD / "blue_leopard.jpg", quality=90
    )
    _spots(512, 512, 104, (40, 140, 70), (10, 30, 15)).save(
        LEOPARD / "green_leopard.jpg", quality=90
    )
    _zebra(512, 512).save(LEOPARD / "zebra.jpg", quality=90)
    _snake(512, 512).save(LEOPARD / "snake.jpg", quality=90)
    _floral(300, 300).save(LEOPARD / "floral_unrelated.jpg", quality=90)
    Image.new("RGB", (300, 300), (255, 255, 255)).save(LEOPARD / "logo_unrelated.jpg")
    Image.new("RGB", (300, 300), (250, 250, 245)).save(
        LEOPARD / "document_unrelated.jpg"
    )


def generate_extra_families() -> None:
    """Ek tekstil family örnekleri — genişletilmiş taksonomi."""
    for name in (
        "floral",
        "paisley",
        "plaid",
        "monogram",
        "baroque",
        "scarf_border",
        "mixed_noise",
    ):
        d = ROOT / name
        d.mkdir(parents=True, exist_ok=True)
        for variant in (
            "exact",
            "resolution",
            "crop",
            "scale",
            "color_variant",
            "similar",
            "unrelated",
        ):
            img = Image.new("RGB", (256, 256), (80 + hash(name) % 100, 60, 90))
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), f"{name}_{variant}", fill=(255, 255, 255))
            img.save(d / f"{name}_{variant}.png")


def main() -> None:
    generate_leopard_set()
    generate_extra_families()
    print(f"Generated leopard set: {LEOPARD}")
    print(f"Extra families under: {ROOT}")


if __name__ == "__main__":
    main()
