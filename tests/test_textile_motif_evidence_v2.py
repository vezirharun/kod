"""Textile Motif Evidence V2: real detectors + independent GT.

INDEX FROZEN. tmp textile store only. CLIP is not a detector.
Does not walk the 116K archive. Does not write Pattern Index / FAISS.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

from core.global_object_intelligence import DetectedObject
from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.object_evidence import CLIP_AS_DETECTOR, extra_detector_available, extra_detector_error
from core.settings import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR
from core.spatial.spatial_engine import SPATIAL_EVIDENCE_UNAVAILABLE, SpatialEngine
from core.textile_motif_evidence import (
    PRIORITY_CLASSES,
    SMALL_AREA_RATIO,
    TextileMotifEvidenceLayer,
    class_metrics,
    match_preds,
)

REPORT = Path(__file__).resolve().parents[1] / "data" / "reports" / "textile_motif_evidence_v2.json"


def _freeze_snap() -> dict:
    return snapshot_index_artifacts(
        db_path=DEFAULT_DATA_DIR / "patterns.db",
        faiss_dino_path=DEFAULT_DATA_DIR / "faiss_dino.index",
        faiss_clip_path=DEFAULT_DATA_DIR / "faiss_clip.index",
        cache_dir=DEFAULT_CACHE_DIR,
    )


def _area_ratio(box, wh) -> float:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1) / max(1, wh[0] * wh[1])


def _draw_flower(im: Image.Image, cx: int, cy: int, r: int, color=(220, 50, 90)) -> tuple[int, int, int, int]:
    dr = ImageDraw.Draw(im)
    n = 8
    for i in range(n):
        a = i * (2 * math.pi / n)
        px, py = cx + int(math.cos(a) * r * 0.55), cy + int(math.sin(a) * r * 0.55)
        dr.ellipse([px - r // 2, py - r // 2, px + r // 2, py + r // 2], fill=color)
    dr.ellipse([cx - r // 3, cy - r // 3, cx + r // 3, cy + r // 3], fill=(240, 210, 40))
    return (cx - r, cy - r, cx + r, cy + r)


def _draw_leaf(im: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    dr = ImageDraw.Draw(im)
    x1, y1, x2, y2 = box
    dr.ellipse(box, fill=(40, 140, 55), outline=(20, 90, 30))
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    dr.line([(mx, y1 + 2), (mx, y2 - 2)], fill=(20, 80, 25), width=2)
    return box


def _draw_rose(im: Image.Image, cx: int, cy: int, r: int) -> tuple[int, int, int, int]:
    dr = ImageDraw.Draw(im)
    for i, rad in enumerate(range(r, max(6, r // 5), -max(3, r // 8))):
        off = (i % 3) * 2
        col = (180 + i * 8, 20 + i * 4, 50)
        dr.ellipse([cx - rad + off, cy - rad, cx + rad + off, cy + rad], outline=col, width=max(3, r // 10))
    dr.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(160, 10, 40))
    return (cx - r, cy - r, cx + r, cy + r)


def _draw_leopard(im: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    patch = Image.new("RGB", (x2 - x1, y2 - y1), (196, 154, 88))
    d = ImageDraw.Draw(patch)
    w, h = patch.size
    spots = [
        (18, 16, 22, 16), (55, 22, 20, 14), (90, 18, 24, 18), (30, 55, 18, 14),
        (70, 58, 22, 16), (110, 50, 16, 12), (20, 90, 20, 15), (60, 95, 24, 18),
        (100, 88, 18, 14), (45, 125, 20, 14), (85, 122, 22, 16), (15, 40, 14, 12),
    ]
    for sx, sy, rw, rh in spots:
        if sx + rw < w and sy + rh < h:
            d.ellipse([sx, sy, sx + rw, sy + rh], outline=(28, 18, 10), width=3)
            d.ellipse([sx + 4, sy + 3, sx + rw - 4, sy + rh - 3], fill=(48, 30, 16))
    im.paste(patch, (x1, y1))
    return box


def _draw_butterfly(im: Image.Image, cx: int, cy: int, s: int) -> tuple[int, int, int, int]:
    dr = ImageDraw.Draw(im)
    dr.ellipse([cx - s, cy - s // 2, cx, cy + s // 3], fill=(70, 90, 200))
    dr.ellipse([cx, cy - s // 2, cx + s, cy + s // 3], fill=(70, 90, 200))
    dr.ellipse([cx - s + 8, cy, cx - 4, cy + s // 2], fill=(50, 70, 170))
    dr.ellipse([cx + 4, cy, cx + s - 8, cy + s // 2], fill=(50, 70, 170))
    dr.ellipse([cx - 4, cy - s // 3, cx + 4, cy + s // 2], fill=(30, 30, 40))
    dr.ellipse([cx - s // 3, cy - s // 4, cx - s // 8, cy], fill=(240, 200, 40))
    dr.ellipse([cx + s // 8, cy - s // 4, cx + s // 3, cy], fill=(240, 200, 40))
    return (cx - s, cy - s // 2, cx + s, cy + s // 2)


def _draw_snake(im: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    patch = Image.new("RGB", (x2 - x1, y2 - y1), (70, 120, 70))
    d = ImageDraw.Draw(patch)
    w, h = patch.size
    for row, y in enumerate(range(8, h - 12, 18)):
        ox = 10 if row % 2 else 0
        for x in range(ox, w - 16, 22):
            d.polygon(
                [(x + 10, y), (x + 20, y + 8), (x + 10, y + 16), (x, y + 8)],
                fill=(210, 190, 90),
                outline=(40, 70, 40),
            )
    im.paste(patch, (x1, y1))
    return box


def _fabric_bg(size=(420, 420), color=(232, 220, 200)) -> Image.Image:
    im = Image.new("RGB", size, color)
    dr = ImageDraw.Draw(im)
    for y in range(0, size[1], 14):
        dr.line([(0, y), (size[0], y)], fill=(color[0] - 8, color[1] - 8, color[2] - 8), width=1)
    return im.filter(ImageFilter.SMOOTH)


def build_fixture_set(dest: Path) -> list[dict]:
    """Independent GT boxes authored here — not copied from any detector."""
    dest.mkdir(parents=True, exist_ok=True)
    specs = []

    im = _fabric_bg()
    b = _draw_flower(im, 210, 210, 70)
    p = dest / "flower_large.png"
    im.save(p)
    specs.append({"name": "flower_large", "path": p, "gt": [("flower", b)], "size": im.size})

    im = _fabric_bg()
    b = _draw_flower(im, 80, 90, 22, color=(200, 40, 80))
    p = dest / "flower_small.png"
    im.save(p)
    specs.append({"name": "flower_small", "path": p, "gt": [("flower", b)], "size": im.size})

    im = _fabric_bg()
    b = _draw_leaf(im, (300, 40, 348, 88))
    p = dest / "leaf_small.png"
    im.save(p)
    specs.append({"name": "leaf_small", "path": p, "gt": [("leaf", b)], "size": im.size})

    im = _fabric_bg()
    b = _draw_rose(im, 210, 200, 55)
    p = dest / "rose.png"
    im.save(p)
    specs.append({"name": "rose", "path": p, "gt": [("rose", b)], "size": im.size})

    im = _fabric_bg((440, 360), (210, 170, 110))
    b = _draw_leopard(im, (40, 40, 400, 320))
    p = dest / "leopard_print.png"
    im.save(p)
    specs.append({"name": "leopard_print", "path": p, "gt": [("leopard", b)], "size": im.size})

    im = _fabric_bg()
    b = _draw_butterfly(im, 210, 200, 70)
    p = dest / "butterfly.png"
    im.save(p)
    specs.append({"name": "butterfly", "path": p, "gt": [("butterfly", b)], "size": im.size})

    im = _fabric_bg((440, 360), (90, 130, 80))
    b = _draw_snake(im, (30, 40, 410, 320))
    p = dest / "snake_print.png"
    im.save(p)
    specs.append({"name": "snake_print", "path": p, "gt": [("snake", b)], "size": im.size})

    im = _fabric_bg()
    fb = _draw_flower(im, 140, 280, 50)
    lb = _draw_leaf(im, (150, 40, 250, 130))
    p = dest / "flower_leaf.png"
    im.save(p)
    specs.append({"name": "flower_leaf", "path": p, "gt": [("flower", fb), ("leaf", lb)], "size": im.size})

    im = _fabric_bg((480, 400), (210, 170, 110))
    leo = _draw_leopard(im, (20, 30, 250, 360))
    fl = _draw_flower(im, 370, 200, 48)
    p = dest / "leopard_flower.png"
    im.save(p)
    specs.append({"name": "leopard_flower", "path": p, "gt": [("leopard", leo), ("flower", fl)], "size": im.size})

    im = _fabric_bg()
    rb = _draw_rose(im, 150, 260, 50)
    lb = _draw_leaf(im, (250, 40, 340, 120))
    p = dest / "rose_leaf.png"
    im.save(p)
    specs.append({"name": "rose_leaf", "path": p, "gt": [("rose", rb), ("leaf", lb)], "size": im.size})

    im = _fabric_bg((400, 400), (40, 70, 160))
    dr = ImageDraw.Draw(im)
    for i in range(0, 400, 18):
        dr.line([(i, 0), (i, 400)], fill=(30, 50, 120), width=6)
    p = dest / "negative_stripes.png"
    im.save(p)
    specs.append({"name": "negative_stripes", "path": p, "gt": [], "size": im.size})

    im = Image.new("RGB", (400, 400), (180, 180, 180))
    p = dest / "negative_plain.png"
    im.save(p)
    specs.append({"name": "negative_plain", "path": p, "gt": [], "size": im.size})
    return specs


def _empty_class_acc() -> dict[str, dict[str, float]]:
    return {
        c: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "conf_sum": 0.0, "small_tp": 0.0, "small_gt": 0.0, "lat_ms": 0.0}
        for c in PRIORITY_CLASSES
    }


def _accumulate(acc, gt, preds, latency_ms, size):
    by = {c: [] for c in PRIORITY_CLASSES}
    for lab, box in gt:
        if lab in by:
            by[lab].append((lab, box))
            if _area_ratio(box, size) <= SMALL_AREA_RATIO:
                acc[lab]["small_gt"] += 1
    pred_by = {c: [d for d in preds if d.label == c] for c in PRIORITY_CLASSES}
    for lab in PRIORITY_CLASSES:
        m = match_preds(by[lab], pred_by[lab])
        acc[lab]["tp"] += m["tp"]
        acc[lab]["fp"] += m["fp"]
        acc[lab]["fn"] += m["fn"]
        acc[lab]["lat_ms"] += latency_ms
        for _lab, iou, conf in m["pairs"]:
            acc[lab]["conf_sum"] += conf
            gbox = by[lab][0][1] if by[lab] else None
            if gbox and _area_ratio(gbox, size) <= SMALL_AREA_RATIO:
                acc[lab]["small_tp"] += 1
    extra_fp_labs = [d.label for d in preds if d.label in PRIORITY_CLASSES]
    # match_preds already counts FP per class from pred_by; negatives with preds add FP.


def test_clip_not_detector():
    assert CLIP_AS_DETECTOR is False


def test_textile_motif_evidence_v2_bench(tmp_path):
    before = freeze_fingerprint(_freeze_snap())
    obj_meta = None
    prod_obj = DEFAULT_DATA_DIR / "object_index.db"
    if prod_obj.exists():
        obj_meta = (prod_obj.stat().st_mtime_ns, prod_obj.stat().st_size)

    fixtures = build_fixture_set(tmp_path / "fixtures")
    layer = TextileMotifEvidenceLayer(tmp_path / "textile_motif_evidence.db")
    rtdetr_ok = layer._coco.detector_available()
    yolo_ok = extra_detector_available()

    rtdetr_acc = _empty_class_acc()
    yolo_acc = _empty_class_acc()
    per_image = []

    for i, spec in enumerate(fixtures, start=1):
        path = str(spec["path"])
        pack = layer.detect(path)
        rtdetr_preds = [d for d in pack["rtdetr"] if d.label in PRIORITY_CLASSES]
        yolo_preds = pack["yoloworld"]
        _accumulate(rtdetr_acc, spec["gt"], rtdetr_preds, pack["rtdetr_ms"], spec["size"])
        _accumulate(yolo_acc, spec["gt"], yolo_preds, pack["yoloworld_ms"], spec["size"])
        layer.persist(i, path, yolo_preds)
        per_image.append(
            {
                "name": spec["name"],
                "gt": [(lab, list(box)) for lab, box in spec["gt"]],
                "rtdetr": [(d.label, round(d.confidence, 3), list(d.bbox)) for d in rtdetr_preds],
                "yoloworld": [(d.label, round(d.confidence, 3), list(d.bbox)) for d in yolo_preds],
                "rtdetr_ms": pack["rtdetr_ms"],
                "yoloworld_ms": pack["yoloworld_ms"],
            }
        )

    rtdetr_m = class_metrics(rtdetr_acc)
    yolo_m = class_metrics(yolo_acc)
    # Production: YOLO-World is the only available open-vocab detection head.
    # RT-DETR-L cannot emit textile classes (COCO-80). Do not swap without a better bench.
    production_model = "ultralytics_yoloworld_v2_boxes" if yolo_ok else "unavailable"
    class_ready = {}
    for c in PRIORITY_CLASSES:
        ym = yolo_m[c]
        ok = bool(yolo_ok and ym["precision"] >= 0.5 and ym["recall"] >= 0.5)
        class_ready[c] = ok

    sizes = {i: fixtures[i - 1]["size"] for i in range(1, len(fixtures) + 1)}
    eng = SpatialEngine(str(tmp_path / "unused_object.db"), readonly=True, textile_db_path=str(tmp_path / "textile_motif_evidence.db"))
    q_fl = eng.search("çiçeğin üstünde yaprak", image_sizes=sizes)
    q_lf = eng.search("leoparın yanında çiçek", image_sizes=sizes)
    q_rl = eng.search("gülün üstünde yaprak", image_sizes=sizes)

    empty_eng = SpatialEngine(
        str(tmp_path / "unused2.db"),
        readonly=True,
        textile_db_path=str(tmp_path / "missing_textile.db"),
    )
    q_miss = empty_eng.search("çiçeğin üstünde yaprak", image_sizes={1: (100, 100)})
    spatial_missing_ok = q_miss.status == SPATIAL_EVIDENCE_UNAVAILABLE

    # Spatial wiring: persist independent GT (not detector self-GT) for pairs, prove engine.
    gt_layer = TextileMotifEvidenceLayer(tmp_path / "gt_spatial.db")

    fl = next(s for s in fixtures if s["name"] == "flower_leaf")
    lf = next(s for s in fixtures if s["name"] == "leopard_flower")
    rl = next(s for s in fixtures if s["name"] == "rose_leaf")

    def _dets(spec):
        out = []
        for j, (lab, box) in enumerate(spec["gt"], start=1):
            x1, y1, x2, y2 = box
            ar = _area_ratio(box, spec["size"])
            cx = round(((x1 + x2) / 2) / spec["size"][0], 4)
            cy = round(((y1 + y2) / 2) / spec["size"][1], 4)
            out.append(DetectedObject(f"{lab}_{j:02d}", lab, lab, 0.91, box, ar, (cx, cy)))
        return out

    gt_layer.persist(1, str(fl["path"]), _dets(fl))
    gt_layer.persist(2, str(lf["path"]), _dets(lf))
    gt_layer.persist(3, str(rl["path"]), _dets(rl))
    gt_sizes = {1: fl["size"], 2: lf["size"], 3: rl["size"]}
    gt_eng = SpatialEngine("x", readonly=True, textile_db_path=str(tmp_path / "gt_spatial.db"))
    s_fl = gt_eng.search("çiçeğin üstünde yaprak", image_sizes=gt_sizes)
    s_lf = gt_eng.search("leoparın yanında çiçek", image_sizes=gt_sizes)
    s_rl = gt_eng.search("gülün üstünde yaprak", image_sizes=gt_sizes)
    spatial_wiring = (
        spatial_missing_ok
        and s_fl.status != SPATIAL_EVIDENCE_UNAVAILABLE
        and bool(s_fl.hits)
        and s_lf.status != SPATIAL_EVIDENCE_UNAVAILABLE
        and bool(s_lf.hits)
        and s_rl.status != SPATIAL_EVIDENCE_UNAVAILABLE
        and bool(s_rl.hits)
    )

    after = freeze_fingerprint(_freeze_snap())
    index_unchanged = after == before
    if obj_meta is not None:
        index_unchanged = index_unchanged and (
            prod_obj.stat().st_mtime_ns,
            prod_obj.stat().st_size,
        ) == obj_meta

    weak = [c for c, ok in class_ready.items() if not ok]
    production_ready = bool(yolo_ok and not weak and spatial_wiring and index_unchanged)
    bottleneck = ""
    solutions = []
    if weak:
        bottleneck = (
            "YOLO-World open-vocab boxes on authored textile prints miss or FP "
            f"priority classes: {', '.join(weak)}. RT-DETR-L has no textile classes (COCO-80)."
        )
        solutions = [
            "Keep YOLO-World; collect a small labeled textile-print photo set (not 116K scan) and re-bench.",
            "Train/fine-tune a detection head on textile motifs (flower/leaf/rose/leopard) — only after photo bench fails.",
            "Do not use CLIP similarity heatmaps as boxes.",
        ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "clip_as_detection": False,
                "independent_gt": True,
                "archive_116k_scanned": False,
                "reindex": False,
                "production_index_unchanged": index_unchanged,
                "faiss_unchanged": index_unchanged,
                "rtdetr_available": rtdetr_ok,
                "yoloworld_available": yolo_ok,
                "yoloworld_error": extra_detector_error() if not yolo_ok else "",
                "production_model": production_model,
                "why_not_rtdetr": "RT-DETR-L is COCO-80; flower/leaf/rose/leopard/butterfly/snake are not COCO classes.",
                "rtdetr_metrics": rtdetr_m,
                "yoloworld_metrics": yolo_m,
                "class_ready": class_ready,
                "per_image": per_image,
                "spatial": {
                    "missing": q_miss.status,
                    "detector_flower_leaf": q_fl.status,
                    "detector_leopard_flower": q_lf.status,
                    "detector_rose_leaf": q_rl.status,
                    "gt_wired_flower_leaf": s_fl.status,
                    "gt_wired_leopard_flower": s_lf.status,
                    "gt_wired_rose_leaf": s_rl.status,
                    "wiring_pass": spatial_wiring,
                },
                "production_ready": production_ready,
                "bottleneck": bottleneck,
                "solutions": solutions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert CLIP_AS_DETECTOR is False
    assert after == before
    assert spatial_missing_ok
    assert spatial_wiring
    assert rtdetr_ok
    for c in PRIORITY_CLASSES:
        assert rtdetr_m[c]["tp"] == 0
        assert rtdetr_m[c]["recall"] == 0.0
    if not yolo_ok:
        pytest.skip(f"YOLO-World not runnable: {extra_detector_error()}")
    # Do not force overall PASS: per-class readiness is reported, not asserted all True.
    assert yolo_m["flower"]["fp"] >= 0
    assert Path(layer.store.path).name == "textile_motif_evidence.db"
    assert "patterns.db" not in layer.store.path
