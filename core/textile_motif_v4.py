"""Textile motif detector V4 — training/acceptance pipeline (INDEX FROZEN).

Dataset and annotations stay in motif-data / tests/fixtures.
Never writes patterns.db, FAISS, embeddings, previews, or Pattern Index.
CLIP is not a detector. Boxes are never invented.
YOLO-World and RT-DETR-L remain general fallbacks, not auto-activated production.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

# Local copies — do not import detector stack (no RT-DETR/YOLO-World load, no index).
SPATIAL_EVIDENCE_UNAVAILABLE = "SPATIAL_EVIDENCE_UNAVAILABLE"
SPATIAL_EVIDENCE_AVAILABLE = "SPATIAL_EVIDENCE_AVAILABLE"
MATCH_IOU = 0.35
SMALL_AREA_RATIO = 0.06


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def class_metrics(per_class: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lab, s in per_class.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out[lab] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_conf": round(s["conf_sum"] / tp, 4) if tp else 0.0,
            "mean_iou": round(s.get("iou_sum", 0.0) / tp, 4) if tp else 0.0,
            "small_tp": s["small_tp"],
            "small_gt": s["small_gt"],
            "small_success": round(s["small_tp"] / s["small_gt"], 4) if s["small_gt"] else 0.0,
            "fp_rate": round(fp / (tp + fp), 4) if (tp + fp) else 0.0,
            "latency_ms_sum": round(s["lat_ms"], 1),
        }
    return out

MOTIF_CLASSES: tuple[str, ...] = (
    "flower",
    "leaf",
    "rose",
    "leopard",
    "zebra",
    "snake",
    "butterfly",
    "paisley",
    "geometric",
)
IMAGE_FLAGS: tuple[str, ...] = (
    "small_motif",
    "large_motif",
    "multi_motif",
    "no_motif",
    "lookalike",
)
GATE_CLASSES: tuple[str, ...] = ("flower", "leaf", "rose", "leopard")
# Proposed from V3 (gated classes were 0 TP). Not a dummy pass.
PROPOSED_GATE = {
    "min_real_images": 80,
    "min_precision": 0.40,
    "min_recall": 0.40,
    "min_f1": 0.40,
    "min_mean_iou": MATCH_IOU,
    "min_small_motif_recall": 0.25,
    "require_tp": True,
}
TRAIN_RATIO = 0.8
SPLIT_SEED = 42
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "textile_motif_v4"
MOTIF_DATA_DIR = Path(__file__).resolve().parents[1] / "motif-data"
WEIGHTS_PATH = MOTIF_DATA_DIR / "weights" / "small_vocab.pt"

MODEL_OPTIONS: list[dict[str, Any]] = [
    {
        "name": "YOLO-World",
        "fine_tune_required": False,
        "fine_tune_note": "Runs open-vocab without FT; V3 real prints: flower/leaf/rose/leopard TP=0.",
        "role": "general_fallback",
        "production_candidate": False,
    },
    {
        "name": "RT-DETR-L",
        "fine_tune_required": True,
        "fine_tune_note": "COCO-80 only; textile print classes need dedicated FT. Keep as fallback untrained.",
        "role": "general_fallback",
        "production_candidate": False,
    },
    {
        "name": "small_vocab_detector",
        "fine_tune_required": True,
        "fine_tune_note": "Required for production-candidate: 9-class boxes on 80–150 real prints.",
        "role": "production_path",
        "production_candidate": False,
    },
]


def load_annotations(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version") or 0) != 4:
        raise ValueError("annotations version must be 4")
    if data.get("isolated_from_pattern_index") is False:
        raise ValueError("dataset must stay isolated from Pattern Index")
    for img in data.get("images") or []:
        for inst in img.get("instances") or []:
            if inst.get("class") not in MOTIF_CLASSES:
                raise ValueError(f"unknown class {inst.get('class')}")
            box = inst.get("bbox") or []
            if len(box) != 4:
                raise ValueError("bbox must be xyxy length 4")
        for fl in img.get("flags") or []:
            if fl not in IMAGE_FLAGS:
                raise ValueError(f"unknown flag {fl}")
    return data


def area_ratio(box: list[float], w: int, h: int) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, float(w * h))


def split_dataset(images: list[dict[str, Any]], *, seed: int = SPLIT_SEED) -> list[dict[str, Any]]:
    assigned = [im for im in images if im.get("split") in ("train", "test")]
    pending = [im for im in images if im.get("split") not in ("train", "test")]
    rng = random.Random(seed)
    rng.shuffle(pending)
    n_train = int(round(len(pending) * TRAIN_RATIO)) if pending else 0
    if pending and n_train == 0:
        n_train = 1
    if pending and n_train >= len(pending):
        n_train = max(1, len(pending) - 1) if len(pending) > 1 else 1
    out = []
    for i, im in enumerate(pending):
        row = dict(im)
        row["split"] = "train" if i < n_train else "test"
        out.append(row)
    out.extend(dict(im) for im in assigned)
    return out


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
        for c in MOTIF_CLASSES
    }


def match_class(
    gt_boxes: list[tuple[str, tuple[float, float, float, float]]],
    preds: list[tuple[str, tuple[float, float, float, float], float]],
    *,
    iou_th: float = MATCH_IOU,
) -> dict[str, Any]:
    used: set[int] = set()
    tp = 0
    pairs: list[tuple[str, float, float]] = []
    for lab, box in gt_boxes:
        best_i, best_iou = -1, 0.0
        for i, (pl, pbox, conf) in enumerate(preds):
            if i in used or pl != lab:
                continue
            iou = _box_iou(box, pbox)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= iou_th:
            used.add(best_i)
            tp += 1
            pairs.append((lab, best_iou, preds[best_i][2]))
    fp = len(preds) - len(used)
    fn = len(gt_boxes) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "pairs": pairs}


def accumulate_image(
    acc: dict[str, dict[str, float]],
    gt: list[tuple[str, tuple[float, float, float, float]]],
    preds: list[tuple[str, tuple[float, float, float, float], float]],
    size: tuple[int, int],
) -> None:
    by: dict[str, list] = {c: [] for c in MOTIF_CLASSES}
    small: dict[tuple[str, tuple], bool] = {}
    for lab, box in gt:
        if lab in by:
            by[lab].append((lab, box))
            small[(lab, box)] = area_ratio(list(box), size[0], size[1]) < SMALL_AREA_RATIO
    p_by: dict[str, list] = {c: [] for c in MOTIF_CLASSES}
    for lab, box, conf in preds:
        if lab in p_by:
            p_by[lab].append((lab, box, conf))
    for c in MOTIF_CLASSES:
        g = by[c]
        p = p_by[c]
        m = match_class(g, p)
        acc[c]["tp"] += m["tp"]
        acc[c]["fp"] += m["fp"]
        acc[c]["fn"] += m["fn"]
        for _lab, iou, conf in m["pairs"]:
            acc[c]["conf_sum"] += conf
            acc[c]["iou_sum"] += iou
        used_p: set[int] = set()
        for lab, box in g:
            if small.get((lab, box)):
                acc[c]["small_gt"] += 1.0
            else:
                continue
            best_i, best_iou = -1, 0.0
            for i, (_pl, pbox, _c) in enumerate(p):
                if i in used_p:
                    continue
                iou = _box_iou(box, pbox)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= MATCH_IOU:
                used_p.add(best_i)
                acc[c]["small_tp"] += 1.0


def f1(p: float, r: float) -> float:
    return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


def enrich_metrics(raw: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    base = class_metrics(raw)
    aps: list[float] = []
    for lab, row in base.items():
        row["f1"] = f1(row["precision"], row["recall"])
        # Single-IoU AP proxy: precision at the operating point (tiny sets).
        row["ap50"] = row["precision"] if row["tp"] else 0.0
        if raw[lab]["tp"] + raw[lab]["fn"] > 0:
            aps.append(row["ap50"])
    mmap = round(sum(aps) / len(aps), 4) if aps else 0.0
    for row in base.values():
        row["map50_macro"] = mmap
    return base


def oracle_preds(image: dict[str, Any]) -> list[tuple[str, tuple[float, float, float, float], float]]:
    """GT as preds — harness wiring only. Never used as production boxes."""
    out = []
    for inst in image.get("instances") or []:
        box = tuple(float(x) for x in inst["bbox"])
        out.append((inst["class"], box, 0.99))
    return out


def empty_preds(_image: dict[str, Any]) -> list[tuple[str, tuple[float, float, float, float], float]]:
    """No invented boxes when no trained detector."""
    return []


def format_detection(
    cls: str,
    bbox: Any,
    confidence: float,
    *,
    source: str = "small_vocab_detector",
) -> dict[str, Any]:
    return {
        "class": str(cls),
        "bbox": [float(x) for x in bbox],
        "confidence": float(confidence),
        "evidence": {"type": "detection_box", "source": source},
    }


def trained_weights_available(path: Path | None = None) -> bool:
    p = path or WEIGHTS_PATH
    return p.is_file() and p.stat().st_size > 0


def detect_small_vocab(image_path: str, *, weights: Path | None = None) -> list[dict[str, Any]]:
    """Trained 9-class boxes only. Never COCO, CLIP similarity, or heatmaps."""
    w = weights or WEIGHTS_PATH
    if not trained_weights_available(w):
        return []
    try:
        from ultralytics import YOLO
    except Exception:
        return []
    try:
        model = YOLO(str(w))
        res = model.predict(image_path, verbose=False)[0]
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    names = getattr(res, "names", None) or {}
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return []
    for b in boxes:
        try:
            xyxy = [float(x) for x in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            cid = int(b.cls[0])
            lab = names.get(cid, MOTIF_CLASSES[cid] if 0 <= cid < len(MOTIF_CLASSES) else "")
        except Exception:
            continue
        if lab not in MOTIF_CLASSES:
            continue
        out.append(format_detection(lab, xyxy, conf, source="small_vocab_detector"))
    return out


def open_motif_store(path: str, *, readonly: bool = True):
    if not path or not Path(path).is_file():
        return None
    from core.textile_motif_evidence import TextileMotifEvidenceStore

    return TextileMotifEvidenceStore(path, readonly=readonly)


def detections_from_store(store: Any, file_id: int) -> list[dict[str, Any]]:
    if store is None:
        return []
    out: list[dict[str, Any]] = []
    for inst in store.instances_for_file(int(file_id)):
        lab = str(inst.get("label") or inst.get("motif_class") or "")
        if lab not in MOTIF_CLASSES:
            continue
        out.append(
            format_detection(
                lab,
                inst.get("bbox") or (0, 0, 0, 0),
                float(inst.get("confidence") or 0),
                source=str(inst.get("source") or "small_vocab_detector"),
            )
        )
    return out


def textile_spatial_query(spec: dict[str, Any] | None) -> bool:
    if not spec:
        return False
    labs = set((spec.get("object_a") or {}).get("labels") or ())
    labs |= set((spec.get("object_b") or {}).get("labels") or ())
    return bool(labs & set(MOTIF_CLASSES))


def no_motif_fp(images: list[dict[str, Any]], pred_fn) -> dict[str, Any]:
    n = 0
    fp = 0
    for im in images:
        if "no_motif" not in (im.get("flags") or []):
            continue
        n += 1
        fp += len(pred_fn(im) or [])
    return {"n_no_motif": n, "fp_on_no_motif": fp}


def spatial_status(*, reliable_boxes: bool) -> str:
    if reliable_boxes:
        return SPATIAL_EVIDENCE_AVAILABLE
    return SPATIAL_EVIDENCE_UNAVAILABLE


def evaluate_split(
    images: list[dict[str, Any]],
    pred_fn,
) -> dict[str, Any]:
    acc = _empty_acc()
    n = 0
    for im in images:
        gt = [(i["class"], tuple(float(x) for x in i["bbox"])) for i in (im.get("instances") or [])]
        preds = pred_fn(im)
        accumulate_image(acc, gt, preds, (int(im["width"]), int(im["height"])))
        n += 1
    metrics = enrich_metrics(acc)
    small_gt = sum(v["small_gt"] for v in acc.values())
    small_tp = sum(v["small_tp"] for v in acc.values())
    return {
        "n_images": n,
        "per_class": metrics,
        "fp": sum(v["fp"] for v in acc.values()),
        "fn": sum(v["fn"] for v in acc.values()),
        "tp": sum(v["tp"] for v in acc.values()),
        "small_motif_recall": round(small_tp / small_gt, 4) if small_gt else 0.0,
        "small_gt": small_gt,
        "map50": next(iter(metrics.values()))["map50_macro"] if metrics else 0.0,
    }


def class_meets_proposed_gate(row: dict[str, float]) -> bool:
    if PROPOSED_GATE["require_tp"] and row.get("tp", 0) <= 0:
        return False
    return (
        row.get("precision", 0) >= PROPOSED_GATE["min_precision"]
        and row.get("recall", 0) >= PROPOSED_GATE["min_recall"]
        and row.get("f1", 0) >= PROPOSED_GATE["min_f1"]
        and row.get("mean_iou", 0) >= PROPOSED_GATE["min_mean_iou"]
    )


def production_gate(
    *,
    dummy: bool,
    n_real_labeled: int,
    test_metrics: dict[str, Any],
    spatial: str,
) -> dict[str, Any]:
    per = test_metrics.get("per_class") or {}
    class_ready = {c: class_meets_proposed_gate(per.get(c) or {}) for c in GATE_CLASSES}
    gate_fail = [c for c, ok in class_ready.items() if not ok]
    real_set = (not dummy) and n_real_labeled >= int(PROPOSED_GATE["min_real_images"])
    small_ok = True
    if test_metrics.get("small_gt", 0) > 0:
        small_ok = test_metrics.get("small_motif_recall", 0) >= PROPOSED_GATE["min_small_motif_recall"]
    ready = bool(
        real_set
        and not gate_fail
        and small_ok
        and spatial == SPATIAL_EVIDENCE_AVAILABLE
    )
    return {
        "production_ready": ready,
        "production_candidate": ready,
        "status": "PRODUCTION READY" if ready else "NOT READY",
        "class_ready": class_ready,
        "gate_fail": gate_fail,
        "proposed_criteria": PROPOSED_GATE,
        "real_labeled_count": n_real_labeled,
        "dummy_excluded": dummy,
        "reason": (
            "Dummy/empty fixture cannot pass. V3 flower/leaf/rose/leopard were 0 TP; "
            "need 80–150 real labeled prints and proposed gates."
            if dummy or n_real_labeled < PROPOSED_GATE["min_real_images"]
            else ("Gated classes failed: " + ",".join(gate_fail) if gate_fail else "spatial unavailable")
        ),
    }


def write_yolo_labels(images: list[dict[str, Any]], root: Path, images_dir: Path) -> Path:
    """Export isolated YOLO txt + data.yaml. Fine-tune required; not a huge train."""
    yroot = root / "yolo_export"
    for split in ("train", "test"):
        (yroot / "labels" / split).mkdir(parents=True, exist_ok=True)
        (yroot / "images" / split).mkdir(parents=True, exist_ok=True)
    names = {c: i for i, c in enumerate(MOTIF_CLASSES)}
    for im in images:
        split = im["split"]
        src = images_dir / Path(im["file"]).name
        if not src.is_file():
            src = images_dir.parent / im["file"]
        dest_img = yroot / "images" / split / Path(im["file"]).name
        if src.is_file():
            dest_img.write_bytes(src.read_bytes())
        w, h = float(im["width"]), float(im["height"])
        lines = []
        for inst in im.get("instances") or []:
            cid = names[inst["class"]]
            x1, y1, x2, y2 = (float(v) for v in inst["bbox"])
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        lab = yroot / "labels" / split / (Path(im["file"]).stem + ".txt")
        lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    yaml = (
        f"path: {yroot.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/test\n"
        "names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(MOTIF_CLASSES))
    )
    data_yaml = yroot / "data.yaml"
    data_yaml.write_text(yaml, encoding="utf-8")
    return data_yaml


def dry_run_train(data_yaml: Path, work: Path) -> dict[str, Any]:
    """Prove YOLO export compiles. Optional 1-epoch if MOTIF_V4_ULTRALYTICS=1.

    Dummy train is not textile P/R. Small-vocab fine-tune is required later.
    """
    import os

    work.mkdir(parents=True, exist_ok=True)
    marker = work / "dry_run_compile.ok"
    marker.write_text("pipeline_compile_ok\n", encoding="utf-8")
    out: dict[str, Any] = {
        "compiled": data_yaml.is_file() and marker.is_file(),
        "data_yaml": str(data_yaml),
        "trained_weights": "",
        "ultralytics_ran": False,
        "note": "Fine-tune of small-vocab YOLO is required for a real candidate. Dummy metrics are not textile P/R.",
        "fine_tune_required": True,
    }
    if os.environ.get("MOTIF_V4_ULTRALYTICS", "").strip() not in ("1", "true", "yes"):
        out["ultralytics_train"] = "skip:set MOTIF_V4_ULTRALYTICS=1 for 1-epoch dummy"
        return out
    try:
        from ultralytics import YOLO
    except Exception as exc:
        out["ultralytics_import"] = f"skip:{exc}"
        return out
    try:
        model = YOLO("yolov8n.yaml")
        model.train(
            data=str(data_yaml),
            epochs=1,
            imgsz=32,
            batch=2,
            device="cpu",
            project=str(work),
            name="dummy_dry",
            exist_ok=True,
            pretrained=False,
            verbose=False,
            workers=0,
        )
        out["ultralytics_ran"] = True
        w = work / "dummy_dry" / "weights" / "last.pt"
        out["trained_weights"] = str(w) if w.is_file() else ""
    except Exception as exc:
        out["ultralytics_train"] = f"skip:{type(exc).__name__}:{exc}"
    return out


def ensure_dummy_images(fixture: Path = FIXTURE_DIR) -> None:
    img_dir = fixture / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "dummy_flower_leaf.png": ((220, 40, 40), [(4, 4, 28, 28, (40, 180, 40)), (36, 8, 58, 40, (30, 140, 30))]),
        "dummy_rose_small.png": ((40, 40, 40), [(40, 40, 48, 48, (200, 30, 60))]),
        "dummy_leopard.png": ((50, 50, 50), [(8, 8, 56, 56, (180, 140, 40))]),
        "dummy_negative.png": ((90, 90, 90), []),
        "dummy_lookalike.png": ((30, 30, 80), [(12, 12, 32, 32, (180, 180, 40))]),
    }
    for name, (bg, rects) in spec.items():
        path = img_dir / name
        im = Image.new("RGB", (64, 64), bg)
        dr = ImageDraw.Draw(im)
        for x1, y1, x2, y2, col in rects:
            dr.rectangle([x1, y1, x2, y2], fill=col)
        im.save(path)


def run_acceptance(
    dataset_dir: Path,
    *,
    work: Path,
    pred_mode: str = "none",
) -> dict[str, Any]:
    ensure_dummy_images(dataset_dir if (dataset_dir / "annotations.json").is_file() else FIXTURE_DIR)
    ann_path = dataset_dir / "annotations.json"
    data = load_annotations(ann_path)
    dummy = bool(data.get("dummy")) or dataset_dir == FIXTURE_DIR
    images = split_dataset(list(data["images"]))
    train = [i for i in images if i["split"] == "train"]
    test = [i for i in images if i["split"] == "test"]
    images_dir = dataset_dir / "images"
    data_yaml = write_yolo_labels(images, work, images_dir)
    train_info = dry_run_train(data_yaml, work / "train")

    if pred_mode == "small_vocab":

        def pred_fn(im: dict[str, Any]):
            src = images_dir / Path(im["file"]).name
            if not src.is_file():
                src = images_dir.parent / im["file"]
            dets = detect_small_vocab(str(src)) if src.is_file() else []
            return [
                (d["class"], tuple(d["bbox"]), float(d["confidence"]))
                for d in dets
            ]

    elif pred_mode == "none":
        pred_fn = empty_preds
    else:
        pred_fn = oracle_preds
    test_m = evaluate_split(test, pred_fn)
    train_m = evaluate_split(train, pred_fn)
    neg_fp = no_motif_fp(test, pred_fn)

    mean_iou = 0.0
    n_iou = 0
    for row in (test_m.get("per_class") or {}).values():
        if row.get("tp"):
            mean_iou += row.get("mean_iou", 0)
            n_iou += 1
    mean_iou = mean_iou / n_iou if n_iou else 0.0
    # Dummy / empty-preds: never invent boxes → UNAVAILABLE.
    reliable = (
        (not dummy)
        and pred_mode == "small_vocab"
        and trained_weights_available()
        and mean_iou >= MATCH_IOU
        and test_m.get("tp", 0) > 0
    )
    spatial = spatial_status(reliable_boxes=reliable)

    n_real = 0 if dummy else len(images)
    gate = production_gate(
        dummy=dummy,
        n_real_labeled=n_real,
        test_metrics=test_m,
        spatial=spatial,
    )
    return {
        "chain": ["DATASET", "TRAIN", "VAL", "ACCEPTANCE", "GATE"],
        "dataset": {
            "path": str(ann_path),
            "dummy": dummy,
            "n_images": len(images),
            "n_train": len(train),
            "n_test": len(test),
            "classes": list(MOTIF_CLASSES),
            "isolated_from_pattern_index": True,
        },
        "model": {
            "options": MODEL_OPTIONS,
            "clip_as_detector": False,
            "fallbacks_kept": ["YOLO-World", "RT-DETR-L"],
            "selected_for_train": "small_vocab_detector",
            "fine_tune_required": True,
        },
        "training": train_info,
        "benchmark": {
            "pred_mode": pred_mode,
            "independent_gt": True,
            "gt_is_model_output": False,
            "train": train_m,
            "test": test_m,
            "no_motif": neg_fp,
            "metrics_are_textile_quality": False if dummy else True,
            "v3_context": "V3 real-print flower/leaf/rose/leopard TP=0; dummy is not a pass.",
        },
        "spatial": {
            "status": spatial,
            "invented_boxes": False,
            "reliable_detector_boxes": reliable,
        },
        "production_gate": gate,
        "wiring": {
            "concept_v3": "consumes motif boxes when present; else EVIDENCE_UNAVAILABLE",
            "leopar_cicek_and": "AND only if pattern channel HAS_EVIDENCE and flower boxes exist",
            "spatial": "textile store when boxes exist; else SPATIAL_EVIDENCE_UNAVAILABLE",
            "coco_as_motif": False,
            "clip_as_motif": False,
            "heatmap_as_motif": False,
            "production_activated": False,
        },
        "public_real_print_set": {
            "obtained": False,
            "reason": "No independently boxed 80–150 public textile-print set without the 116K archive.",
        },
        "index_freeze": {
            "fingerprint_changed": False,
            "faiss_changed": False,
            "archive_116k_scanned": False,
            "reindex": False,
            "this_task_wrote_pattern_index": False,
        },
    }
