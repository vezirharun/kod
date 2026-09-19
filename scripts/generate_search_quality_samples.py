"""Arama kalite test veri seti oluşturucu."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent / "sample_data" / "search_quality"


def _flower(
    draw: ImageDraw.ImageDraw, cx: int, cy: int, color: tuple, size: int = 40
) -> None:
    for i in range(8):
        angle = i * 45
        import math

        dx = int(math.cos(math.radians(angle)) * size)
        dy = int(math.sin(math.radians(angle)) * size)
        draw.ellipse(
            [cx + dx - 15, cy + dy - 15, cx + dx + 15, cy + dy + 15], fill=color
        )
    draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 220, 0))


def _leopard_spots(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    draw.rectangle([0, 0, w, h], fill=(200, 150, 80))
    spots = [(60, 60), (180, 90), (120, 180), (220, 200), (80, 220)]
    for x, y in spots:
        draw.ellipse([x - 20, y - 15, x + 20, y + 15], fill=(40, 20, 10))


def _geometric(draw: ImageDraw.ImageDraw, w: int, h: int, color: tuple) -> None:
    draw.rectangle([0, 0, w, h], fill=(240, 240, 240))
    for x in range(0, w, 40):
        for y in range(0, h, 40):
            if (x // 40 + y // 40) % 2 == 0:
                draw.rectangle([x, y, x + 40, y + 40], fill=color)


def generate() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    size = (300, 300)

    # Çiçek seti
    img = Image.new("RGB", size, (30, 120, 60))
    d = ImageDraw.Draw(img)
    _flower(d, 150, 150, (255, 100, 150))
    img.save(ROOT / "flower_original.png")

    img.save(ROOT / "flower_same_copy_different_name.jpg", quality=95)

    img2 = Image.new("RGB", size, (30, 120, 60))
    d2 = ImageDraw.Draw(img2)
    _flower(d2, 150, 150, (100, 80, 200))  # renk varyantı
    img2.save(ROOT / "flower_color_variant.png")

    cropped = img.crop((50, 50, 250, 250)).resize(size)
    cropped.save(ROOT / "flower_cropped.png")

    rotated = img.rotate(8, expand=False, fillcolor=(30, 120, 60))
    rotated.save(ROOT / "flower_rotated_small.png")

    # Leopar
    img3 = Image.new("RGB", size, (200, 150, 80))
    d3 = ImageDraw.Draw(img3)
    _leopard_spots(d3, *size)
    img3.save(ROOT / "leopard_original.png")

    img4 = Image.new("RGB", size, (180, 130, 70))
    d4 = ImageDraw.Draw(img4)
    _leopard_spots(d4, *size)
    img4.save(ROOT / "leopard_color_variant.png")

    # Geometrik
    img5 = Image.new("RGB", size)
    d5 = ImageDraw.Draw(img5)
    _geometric(d5, *size, (50, 100, 200))
    img5.save(ROOT / "geometric_original.png")

    img6 = Image.new("RGB", size)
    d6 = ImageDraw.Draw(img6)
    _geometric(d6, *size, (55, 105, 195))
    img6.save(ROOT / "geometric_similar.png")

    # Alakasız
    car = Image.new("RGB", size, (100, 100, 100))
    dc = ImageDraw.Draw(car)
    dc.rectangle([50, 120, 250, 180], fill=(200, 50, 50))
    dc.ellipse([60, 200, 100, 240], fill=(20, 20, 20))
    dc.ellipse([200, 200, 240, 240], fill=(20, 20, 20))
    car.save(ROOT / "unrelated_car.png")

    plain = Image.new("RGB", size, (180, 180, 180))
    plain.save(ROOT / "unrelated_plain_color.png")

    (ROOT / "broken_file.tif").write_bytes(b"NOT A TIFF")

    expectations = {
        "flower_original.png": {
            "query": "flower_original.png",
            "must_top1": ["flower_original.png"],
            "must_top5": [
                "flower_same_copy_different_name.jpg",
                "flower_color_variant.png",
            ],
            "must_top10": ["flower_cropped.png"],
            "should_not_top10": ["unrelated_car.png", "unrelated_plain_color.png"],
        },
        "leopard_original.png": {
            "query": "leopard_original.png",
            "must_top1": ["leopard_original.png"],
            "must_top5": ["leopard_color_variant.png"],
            "should_not_top10": ["flower_original.png", "unrelated_car.png"],
        },
    }
    with open(ROOT / "expectations.json", "w", encoding="utf-8") as f:
        json.dump(expectations, f, indent=2, ensure_ascii=False)

    print(f"Oluşturuldu: {ROOT}")


if __name__ == "__main__":
    generate()
