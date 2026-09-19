"""Animal print test veri seti — kahverengi leopard, renk varyantları, yakın aileler."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = (
    Path(__file__).resolve().parent.parent
    / "sample_data"
    / "texture_quality"
    / "animal_print"
)


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


def _tiger(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (210, 120, 40))
    d = ImageDraw.Draw(img)
    for y in range(0, h, 28):
        d.rectangle([0, y, w, y + 10], fill=(20, 20, 20))
    return img


def _snake(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (55, 85, 45))
    d = ImageDraw.Draw(img)
    for gy in range(0, h, 16):
        for gx in range(0, w, 16):
            c = (140, 130, 90) if (gx // 16 + gy // 16) % 2 == 0 else (90, 80, 55)
            d.rectangle([gx, gy, gx + 15, gy + 15], fill=c)
    return img


def _crocodile(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (70, 90, 50))
    d = ImageDraw.Draw(img)
    for gy in range(0, h, 24):
        for gx in range(0, w, 24):
            d.rectangle(
                [gx + 2, gy + 2, gx + 20, gy + 20], outline=(40, 50, 30), width=2
            )
    return img


def _cow(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (240, 240, 240))
    d = ImageDraw.Draw(img)
    for x, y, rx, ry in [(80, 100, 60, 50), (200, 180, 70, 55), (150, 300, 50, 45)]:
        d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(15, 15, 15))
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


def generate() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    base = _spots(512, 512, 100, (200, 150, 80), (35, 20, 10))
    base.save(ROOT / "brown_leopard_exact.jpg", quality=92)
    base.resize((256, 256)).save(ROOT / "brown_leopard_same_resolution.jpg", quality=90)
    base.resize((768, 768)).save(
        ROOT / "brown_leopard_different_resolution.jpg", quality=90
    )
    base.save(ROOT / "brown_leopard_same_tif.tif")
    base.crop((100, 100, 412, 412)).save(ROOT / "brown_leopard_crop.jpg", quality=90)
    base.resize((640, 640)).save(ROOT / "brown_leopard_scale.jpg", quality=90)
    _spots(512, 512, 100, (190, 140, 75), (30, 18, 8)).save(
        ROOT / "brown_leopard_color_adjusted.jpg", quality=90
    )
    _spots(512, 512, 43, (200, 150, 80), (35, 20, 10)).save(
        ROOT / "brown_leopard_similar_01.jpg", quality=90
    )
    _spots(512, 512, 44, (195, 145, 78), (38, 22, 12)).save(
        ROOT / "brown_leopard_similar_02.jpg", quality=90
    )

    # Farklı seed — renk varyantları kahverengi benzerlerinden sonra gelsin
    _spots(512, 512, 101, (160, 40, 50), (20, 10, 10)).save(
        ROOT / "red_leopard.jpg", quality=90
    )
    _spots(512, 512, 102, (50, 70, 180), (15, 15, 40)).save(
        ROOT / "blue_leopard.jpg", quality=90
    )
    _spots(512, 512, 103, (40, 140, 70), (10, 30, 15)).save(
        ROOT / "green_leopard.jpg", quality=90
    )
    _spots(512, 512, 104, (220, 220, 220), (30, 30, 30)).save(
        ROOT / "black_white_leopard.jpg", quality=90
    )

    _zebra(512, 512).save(ROOT / "zebra.jpg", quality=90)
    _tiger(512, 512).save(ROOT / "tiger.jpg", quality=90)
    _snake(512, 512).save(ROOT / "snake.jpg", quality=90)
    _crocodile(512, 512).save(ROOT / "crocodile.jpg", quality=90)
    _cow(512, 512).save(ROOT / "cow.jpg", quality=90)

    _floral(300, 300).save(ROOT / "floral_unrelated.jpg", quality=90)
    Image.new("RGB", (300, 300), (255, 255, 255)).save(ROOT / "logo_unrelated.jpg")
    Image.new("RGB", (300, 300), (250, 250, 245)).save(ROOT / "document_unrelated.jpg")
    Image.new("RGB", (200, 200), (180, 50, 50)).save(ROOT / "plain_color_unrelated.jpg")
    print(f"Animal print samples: {ROOT}")


if __name__ == "__main__":
    generate()
