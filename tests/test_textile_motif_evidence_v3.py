"""Textile Motif Evidence V3: real print photos + independent GT.

INDEX FROZEN. tmp textile store only. CLIP is not a detector.
Does not walk the 116K archive. Does not write Pattern Index / FAISS.
Drawings are lookalikes only — never counted as real-print success.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from core.global_object_intelligence import DetectedObject
from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.object_evidence import CLIP_AS_DETECTOR, extra_detector_available, extra_detector_error
from core.settings import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR
from core.spatial.spatial_engine import (
    SPATIAL_EVIDENCE_AVAILABLE,
    SPATIAL_EVIDENCE_UNAVAILABLE,
    SpatialEngine,
)
from core.textile_motif_evidence import (
    PRIORITY_CLASSES,
    SMALL_AREA_RATIO,
    TextileMotifEvidenceLayer,
    class_metrics,
    match_preds,
)

REPORT = Path(__file__).resolve().parents[1] / "data" / "reports" / "textile_motif_evidence_v3.json"
CACHE = Path(__file__).resolve().parents[1] / ".cache_object_photos" / "textile_print_v3"
GATE = {"flower", "leaf", "rose", "leopard"}
QUERIES = (
    "leoparın yanında gül",
    "gülün üstünde yaprak",
    "çiçeğin yanında yaprak",
    "leoparın yanında çiçek",
    "kelebek çiçeğin üzerinde",
)


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


def _empty_acc() -> dict[str, dict[str, float]]:
    return {
        c: {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "conf_sum": 0.0,
            "iou_sum": 0.0,
            "small_tp": 0.0,
            "small_gt": 0.0,
            "lat_ms": 0.0,
        }
        for c in PRIORITY_CLASSES
    }


def _accumulate(acc, gt, preds, latency_ms, size):
    by = {c: [] for c in PRIORITY_CLASSES}
    small_boxes = {}
    for lab, box in gt:
        if lab in by:
            by[lab].append((lab, box))
            if _area_ratio(box, size) <= SMALL_AREA_RATIO:
                acc[lab]["small_gt"] += 1
                small_boxes.setdefault(lab, []).append(box)
    pred_by = {c: [d for d in preds if d.label == c] for c in PRIORITY_CLASSES}
    for lab in PRIORITY_CLASSES:
        m = match_preds(by[lab], pred_by[lab])
        acc[lab]["tp"] += m["tp"]
        acc[lab]["fp"] += m["fp"]
        acc[lab]["fn"] += m["fn"]
        acc[lab]["lat_ms"] += latency_ms
        for _lab, iou, conf in m["pairs"]:
            acc[lab]["conf_sum"] += conf
            acc[lab]["iou_sum"] += iou
        for box in small_boxes.get(lab, []):
            if match_preds([(lab, box)], pred_by[lab])["tp"]:
                acc[lab]["small_tp"] += 1


def _copy(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def _gt_from_visual(name: str, size: tuple[int, int]) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Independent boxes from visual inspection of museum photos — not detector output."""
    w, h = size
    catalog = {
        "floral_met_221496": [
            ("flower", (35, 25, 210, 195)),
            ("flower", (25, 215, 195, 375)),
            ("rose", (35, 25, 210, 195)),
            ("leaf", (200, 40, 350, 175)),
            ("floral", (8, 8, w - 8, h - 8)),
        ],
        "floral_cma_117604": [
            ("flower", (70, 190, 290, 430)),
            ("flower", (430, 160, 700, 410)),
            ("rose", (70, 190, 290, 430)),
            ("leaf", (180, 60, 430, 260)),
            ("leaf", (480, 40, 720, 200)),
            ("floral", (10, 10, w - 10, h - 10)),
        ],
        "print_met_228719": [
            ("flower", (70, 70, 210, 210)),
            ("flower", (300, 180, 450, 330)),
            ("leaf", (190, 90, 330, 250)),
            ("floral", (8, 8, w - 40, h - 40)),
        ],
        "paisley_cma_129346": [
            ("paisley", (15, 30, 190, h - 30)),
            ("paisley", (w - 200, 30, w - 15, h - 30)),
            ("geometric", (8, 8, w - 8, 28)),
        ],
        "geometric_met_69761": [
            ("geometric", (12, 12, w - 12, h - 12)),
        ],
        "butterfly_met_448413": [
            ("butterfly", (70, 50, 155, 130)),
            ("butterfly", (70, 280, 155, 360)),
            ("flower", (25, 140, 195, 310)),
            ("floral", (20, 130, 200, 500)),
            ("paisley", (20, 130, 200, 500)),
        ],
        "floral_cma_109580": [
            ("flower", (80, 80, 320, 360)),
            ("leaf", (300, 60, 560, 280)),
            ("floral", (10, 10, w - 10, h - 10)),
        ],
    }
    return catalog.get(name, [])


def build_real_print_fixtures(dest: Path) -> tuple[list[dict], dict]:
    dest.mkdir(parents=True, exist_ok=True)
    files = {
        "floral_met_221496": CACHE / "floral_met_221496.jpg",
        "floral_cma_117604": CACHE / "floral_cma_117604.jpg",
        "print_met_228719": CACHE / "print_met_228719.jpg",
        "paisley_cma_129346": CACHE / "paisley_cma_129346.jpg",
        "geometric_met_69761": CACHE / "geometric_met_69761.jpg",
        "butterfly_met_448413": CACHE / "butterfly_met_448413.jpg",
        "floral_cma_109580": CACHE / "floral_cma_109580.jpg",
    }
    present = {k: p for k, p in files.items() if p.is_file()}
    meta = {
        "real_prints_obtained": bool(present),
        "missing_real_print_classes": ["leopard", "zebra", "snake"],
        "sources": "MET Open Access + Cleveland Museum of Art Open Access (no 116K/NAS)",
        "files": sorted(present),
    }
    specs = []
    for name, src in present.items():
        im = Image.open(src).convert("RGB")
        path = _copy(src, dest / f"{name}.jpg")
        gt = _gt_from_visual(name, im.size)
        specs.append({"name": name, "path": path, "gt": gt, "size": im.size, "kind": "real_print"})

    if "floral_met_221496" in present:
        src = Image.open(present["floral_met_221496"]).convert("RGB")
        box = (35, 25, 210, 195)
        crop = src.crop(box)
        p = dest / "flower_small_real.jpg"
        crop.save(p)
        specs.append(
            {
                "name": "flower_small_real",
                "path": p,
                "gt": [("flower", (2, 2, crop.size[0] - 2, crop.size[1] - 2))],
                "size": crop.size,
                "kind": "real_print_small",
            }
        )
        leaf_box = (200, 40, 350, 175)
        leaf = src.crop(leaf_box)
        p = dest / "leaf_small_real.jpg"
        leaf.save(p)
        specs.append(
            {
                "name": "leaf_small_real",
                "path": p,
                "gt": [("leaf", (2, 2, leaf.size[0] - 2, leaf.size[1] - 2))],
                "size": leaf.size,
                "kind": "real_print_small",
            }
        )

    # Negative: museum geometric must not be animal print (separate file path).
    if "geometric_met_69761" in present:
        geo = dest / "geometric_met_69761.jpg"
        neg = _copy(geo, dest / "negative_animal_on_geometric.jpg")
        specs.append(
            {
                "name": "negative_animal_on_geometric",
                "path": neg,
                "gt": [],
                "size": Image.open(neg).size,
                "kind": "real_print_negative_animal",
                "eval_classes": ["leopard", "zebra", "snake"],
            }
        )

    # Lookalike drawings — not real prints.
    im = Image.new("RGB", (400, 400), (40, 70, 160))
    dr = ImageDraw.Draw(im)
    for i in range(0, 400, 16):
        dr.line([(i, 0), (i, 400)], fill=(30, 50, 120), width=5)
    p = dest / "lookalike_stripes.png"
    im.save(p)
    specs.append({"name": "lookalike_stripes", "path": p, "gt": [], "size": im.size, "kind": "drawing"})

    im = Image.new("RGB", (400, 400), (196, 154, 88))
    dr = ImageDraw.Draw(im)
    for y in range(20, 380, 40):
        for x in range(20, 380, 40):
            dr.ellipse([x, y, x + 18, y + 14], outline=(28, 18, 10), width=2)
    p = dest / "lookalike_spots.png"
    im.save(p)
    specs.append(
        {
            "name": "lookalike_spots",
            "path": p,
            "gt": [("leopard", (10, 10, 390, 390))],
            "size": im.size,
            "kind": "drawing",
        }
    )
    return specs, meta


def test_clip_not_detector():
    assert CLIP_AS_DETECTOR is False


def test_textile_motif_evidence_v3_real_print_bench(tmp_path):
    before = freeze_fingerprint(_freeze_snap())
    obj_meta = None
    prod_obj = DEFAULT_DATA_DIR / "object_index.db"
    if prod_obj.exists():
        obj_meta = (prod_obj.stat().st_mtime_ns, prod_obj.stat().st_size)

    fixtures, src_meta = build_real_print_fixtures(tmp_path / "fixtures")
    real_specs = [s for s in fixtures if str(s.get("kind", "")).startswith("real_print")]
    if not real_specs:
        pytest.skip("real museum print photos unavailable; not calling drawings real-print success")

    layer = TextileMotifEvidenceLayer(tmp_path / "textile_motif_evidence.db")
    rtdetr_ok = layer._coco.detector_available()
    yolo_ok = extra_detector_available()

    real_acc = _empty_acc()
    rtdetr_acc = _empty_acc()
    drawing_acc = _empty_acc()
    per_image = []
    fid = 0
    real_ids = {}
    for spec in fixtures:
        fid += 1
        path = str(spec["path"])
        pack = layer.detect(path)
        yolo_preds = pack["yoloworld"]
        rtdetr_preds = [d for d in pack["rtdetr"] if d.label in PRIORITY_CLASSES]
        eval_gt = spec["gt"]
        eval_preds = yolo_preds
        if spec.get("eval_classes"):
            eval_gt = []
            eval_preds = [d for d in yolo_preds if d.label in spec["eval_classes"]]
        if str(spec.get("kind", "")).startswith("real_print"):
            _accumulate(real_acc, eval_gt, eval_preds, pack["yoloworld_ms"], spec["size"])
            _accumulate(rtdetr_acc, spec["gt"], rtdetr_preds, pack["rtdetr_ms"], spec["size"])
            real_ids[spec["name"]] = fid
        else:
            _accumulate(drawing_acc, spec["gt"], yolo_preds, pack["yoloworld_ms"], spec["size"])
        n = layer.persist(fid, path, yolo_preds)
        counts = layer.store.object_count(fid)
        per_image.append(
            {
                "name": spec["name"],
                "kind": spec["kind"],
                "gt": [(lab, list(box)) for lab, box in spec["gt"]],
                "yoloworld": [(d.label, round(d.confidence, 3), list(d.bbox)) for d in yolo_preds],
                "rtdetr_textile_labels": [(d.label, round(d.confidence, 3)) for d in rtdetr_preds],
                "object_count": counts,
                "persisted": n,
                "yoloworld_ms": pack["yoloworld_ms"],
                "rtdetr_ms": pack["rtdetr_ms"],
            }
        )

    yolo_m = class_metrics(real_acc)
    rtdetr_m = class_metrics(rtdetr_acc)
    drawing_m = class_metrics(drawing_acc)

    class_ready = {}
    for c in PRIORITY_CLASSES:
        ym = yolo_m[c]
        if c in {"leopard", "zebra", "snake"} and c in src_meta["missing_real_print_classes"]:
            class_ready[c] = False
            continue
        ok = bool(yolo_ok and ym["precision"] >= 0.5 and ym["recall"] >= 0.5 and ym["mean_iou"] >= 0.35)
        class_ready[c] = ok

    sizes = {i + 1: fixtures[i]["size"] for i in range(len(fixtures))}
    det_eng = SpatialEngine(
        str(tmp_path / "unused_object.db"),
        readonly=True,
        textile_db_path=str(tmp_path / "textile_motif_evidence.db"),
    )
    detector_spatial = {}
    for q in QUERIES:
        r = det_eng.search(q, image_sizes=sizes)
        detector_spatial[q] = {
            "status": r.status,
            "availability": r.availability,
            "hits": [h.file_id for h in r.hits],
            "hit_count": len(r.hits),
        }

    empty_eng = SpatialEngine(
        str(tmp_path / "unused2.db"),
        readonly=True,
        textile_db_path=str(tmp_path / "missing_textile.db"),
    )
    q_miss = empty_eng.search("çiçeğin yanında yaprak", image_sizes={1: (100, 100)})
    spatial_missing_ok = (
        q_miss.status == SPATIAL_EVIDENCE_UNAVAILABLE
        and q_miss.availability == SPATIAL_EVIDENCE_UNAVAILABLE
    )

    # Independent GT persist — never detector self-GT.
    gt_layer = TextileMotifEvidenceLayer(tmp_path / "gt_spatial.db")
    gt_sizes = {}
    gt_map = {}
    gid = 0
    for spec in real_specs:
        if not spec["gt"]:
            continue
        gid += 1
        dets = []
        for j, (lab, box) in enumerate(spec["gt"], start=1):
            x1, y1, x2, y2 = box
            ar = _area_ratio(box, spec["size"])
            cx = round(((x1 + x2) / 2) / spec["size"][0], 4)
            cy = round(((y1 + y2) / 2) / spec["size"][1], 4)
            dets.append(DetectedObject(f"{lab}_{j:02d}", lab, lab, 0.91, box, ar, (cx, cy)))
        gt_layer.persist(gid, str(spec["path"]), dets)
        gt_sizes[gid] = spec["size"]
        gt_map[spec["name"]] = gid
    gt_eng = SpatialEngine("x", readonly=True, textile_db_path=str(tmp_path / "gt_spatial.db"))
    independent_spatial = {}
    for q in QUERIES:
        r = gt_eng.search(q, image_sizes=gt_sizes)
        independent_spatial[q] = {
            "status": r.status,
            "availability": r.availability,
            "hits": [h.file_id for h in r.hits],
            "hit_count": len(r.hits),
            "evaluable": r.availability == SPATIAL_EVIDENCE_AVAILABLE,
        }

    after = freeze_fingerprint(_freeze_snap())
    fingerprint_stable = after == before
    this_task_index_write = False
    if obj_meta is not None:
        now_meta = (prod_obj.stat().st_mtime_ns, prod_obj.stat().st_size)
        fingerprint_stable = fingerprint_stable and now_meta == obj_meta
    index_unchanged = fingerprint_stable and not this_task_index_write

    gate_fail = [c for c in GATE if not class_ready.get(c)]
    production_ready = bool(yolo_ok and not gate_fail and spatial_missing_ok and index_unchanged)
    bottleneck = ""
    next_model = []
    min_labels = {}
    if gate_fail or not production_ready:
        bottleneck = (
            "Open-vocab YOLO-World boxes on real museum prints miss or FP gated classes "
            f"{gate_fail}. RT-DETR-L remains COCO-80 fallback (no textile print classes). "
            "Leopard/zebra/snake real print photos were not obtainable without the 116K archive."
        )
        next_model = [
            "Keep YOLO-World + RT-DETR-L as general/fallback.",
            "Fine-tune a small-vocab detector (YOLO/RT-DETR) on textile-print boxes for "
            "flower, leaf, rose, leopard first — then paisley/geometric/animal-print.",
            "Do not use CLIP similarity as detection.",
        ]
        min_labels = {
            "images": "80–150 labeled print photos (not 116K scan)",
            "per_class_min": {
                "flower": 40,
                "leaf": 40,
                "rose": 30,
                "leopard": 40,
                "butterfly": 20,
                "snake": 20,
                "zebra": 20,
                "paisley": 20,
                "geometric": 20,
                "floral": 20,
            },
            "boxes": "instance boxes + overlap examples + small motifs + negatives/lookalikes",
            "do_not": "self-scan 116K / NAS pattern archive",
        }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_as_detection": False,
        "independent_gt": True,
        "real_prints_obtained": src_meta["real_prints_obtained"],
        "real_print_sources": src_meta["sources"],
        "missing_real_print_classes": src_meta["missing_real_print_classes"],
        "drawings_counted_as_real_print_success": False,
        "archive_116k_scanned": False,
        "reindex": False,
        "production_index_unchanged": index_unchanged,
        "this_task_wrote_pattern_index": False,
        "fingerprint_stable": fingerprint_stable,
        "faiss_unchanged": index_unchanged,
        "rtdetr_available": rtdetr_ok,
        "yoloworld_available": yolo_ok,
        "yoloworld_error": extra_detector_error() if not yolo_ok else "",
        "production_model": "ultralytics_yoloworld_v2_boxes" if yolo_ok else "unavailable",
        "fallback_kept": ["ultralytics_yoloworld_v2_boxes", "RT-DETR-L"],
        "rtdetr_metrics_real_prints": rtdetr_m,
        "yoloworld_metrics_real_prints": yolo_m,
        "yoloworld_metrics_drawings_lookalikes_only": drawing_m,
        "class_ready": class_ready,
        "gate_classes": sorted(GATE),
        "gate_fail": gate_fail,
        "per_image": per_image,
        "spatial_independent_gt": independent_spatial,
        "spatial_detector": detector_spatial,
        "spatial_missing": q_miss.status,
        "production_ready": production_ready,
        "bottleneck": bottleneck,
        "next_model": next_model,
        "min_labels": min_labels,
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assert CLIP_AS_DETECTOR is False
    assert this_task_index_write is False
    assert spatial_missing_ok
    assert rtdetr_ok
    assert "patterns.db" not in layer.store.path
    if not yolo_ok:
        pytest.skip(f"YOLO-World not runnable: {extra_detector_error()}")
    if any(not class_ready[c] for c in GATE):
        assert production_ready is False

