"""Benchmark Faster R-CNN vs YOLO vs RT-DETR on the same 42-photo acceptance set.

INDEX FROZEN. No writes to patterns.db / FAISS / production object_index.db.
CLIP is not used. Does not walk the 116K archive.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from PIL import Image  # noqa: E402

import importlib.util  # noqa: E402

_acc_path = ROOT / "tests" / "test_object_real_photo_acceptance.py"
_spec = importlib.util.spec_from_file_location("obj_acc", _acc_path)
_acc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_acc)
BAG = _acc.BAG
TARGET = _acc.TARGET
_ensure_coco128 = _acc._ensure_coco128
_iou = _acc._iou
_match = _acc._match
_no_object_photos = _acc._no_object_photos
_select_stems = _acc._select_stems
_yolo_boxes = _acc._yolo_boxes

WEIGHTS_DIR = ROOT / ".cache_object_photos" / "detector_weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
CONF = 0.45
SMALL_AREA = 32 * 32
CLASS_MAP = {
    "cat": "kedi",
    "dog": "köpek",
    "bird": "kuş",
    "person": "insan",
    "handbag": "çanta",
    "car": "araba",
}


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        return -1.0


def _vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _hw() -> dict:
    import platform

    import torch
    import torchvision

    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
    }


def load_photos(tmp: Path) -> list[dict]:
    coco = _ensure_coco128()
    if coco is None:
        raise SystemExit("coco128 unavailable")
    img_dir = coco / "images" / "train2017"
    lab_dir = coco / "labels" / "train2017"
    stems = _select_stems(lab_dir, 40)
    photos: list[dict] = []
    work = tmp / "photos"
    work.mkdir()
    for stem in stems:
        src = img_dir / f"{stem}.jpg"
        if not src.is_file():
            continue
        dest = work / f"{stem}.jpg"
        dest.write_bytes(src.read_bytes())
        im = Image.open(dest)
        w, h = im.size
        gt_all = _yolo_boxes(lab_dir / f"{stem}.txt", w, h)
        gt = [(lab, box) for lab, box in gt_all if lab in TARGET]
        photos.append(
            {
                "path": dest,
                "gt": gt,
                "gt_all": gt_all,
                "no_object": False,
                "area": w * h,
            }
        )
    for p in _no_object_photos(work):
        photos.append({"path": p, "gt": [], "gt_all": [], "no_object": True, "area": 1})
    return photos


def _to_pred(dets: list) -> list:
    return [(o.label, tuple(o.bbox), float(o.confidence)) for o in dets if o.label in TARGET]


def _eval(photos: list[dict], predict, *, warmup: str | None) -> dict:
    if warmup:
        predict(warmup)
    tp = fp = fn = 0
    ious_all: list[float] = []
    noobj_fp_images = 0
    noobj_n = 0
    multi_ok = multi_n = 0
    small_tp = small_fn = 0
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in TARGET}
    latencies: list[float] = []
    rss0 = _rss_mb()
    peak_rss = rss0
    for ph in photos:
        t0 = time.perf_counter()
        dets = predict(str(ph["path"]))
        latencies.append(time.perf_counter() - t0)
        peak_rss = max(peak_rss, _rss_mb())
        pred_all = [(o.label, tuple(o.bbox), float(o.confidence)) for o in dets]
        pred = _to_pred(dets)
        if ph["no_object"]:
            noobj_n += 1
            nfp = len(pred_all)
            if nfp:
                noobj_fp_images += 1
            fp += nfp
            continue
        tpi, fpi, fni, ious = _match(ph["gt"], pred)
        tp += tpi
        fp += fpi
        fn += fni
        ious_all.extend([u for u in ious if u >= 0.5])
        gt_n = len(ph["gt"])
        if gt_n >= 2:
            multi_n += 1
            if len(pred) >= 2:
                multi_ok += 1
        gt_by = defaultdict(list)
        for lab, box in ph["gt"]:
            gt_by[lab].append(box)
        pr_by = defaultdict(list)
        for lab, box, _c in pred:
            pr_by[lab].append(box)
        for c in TARGET:
            g = [(c, b) for b in gt_by[c]]
            p = [(c, b, 1.0) for b in pr_by[c]]
            ct, cf, cn, _ = _match(g, p)
            class_stats[c]["tp"] += ct
            class_stats[c]["fp"] += cf
            class_stats[c]["fn"] += cn
        for lab, box in ph["gt"]:
            x1, y1, x2, y2 = box
            if (x2 - x1) * (y2 - y1) >= SMALL_AREA:
                continue
            matched = False
            for p_lab, p_box, _c in pred:
                if p_lab == lab and _iou(box, p_box) >= 0.5:
                    matched = True
                    break
            if matched:
                small_tp += 1
            else:
                small_fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    mean_iou = sum(ious_all) / len(ious_all) if ious_all else 0.0
    small_den = small_tp + small_fn
    per_class = {}
    for en, tr in CLASS_MAP.items():
        st = class_stats[en]
        p = st["tp"] / (st["tp"] + st["fp"]) if (st["tp"] + st["fp"]) else 0.0
        r = st["tp"] / (st["tp"] + st["fn"]) if (st["tp"] + st["fn"]) else 0.0
        per_class[tr] = {
            "coco": en,
            "precision": round(p, 4),
            "recall": round(r, 4),
            **st,
        }
    bag_tp = class_stats["handbag"]["tp"] + class_stats["backpack"]["tp"]
    bag_fn = class_stats["handbag"]["fn"] + class_stats["backpack"]["fn"]
    bag_fp = class_stats["handbag"]["fp"] + class_stats["backpack"]["fp"]
    per_class["çanta"]["note"] = "handbag+backpack mapped to çanta"
    per_class["çanta"]["tp"] = bag_tp
    per_class["çanta"]["fn"] = bag_fn
    per_class["çanta"]["fp"] = bag_fp
    bp = bag_tp / (bag_tp + bag_fp) if (bag_tp + bag_fp) else 0.0
    br = bag_tp / (bag_tp + bag_fn) if (bag_tp + bag_fn) else 0.0
    per_class["çanta"]["precision"] = round(bp, 4)
    per_class["çanta"]["recall"] = round(br, 4)
    return {
        "n_photos": len(photos),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "no_object_fp": noobj_fp_images,
        "no_object_fp_rate": round(noobj_fp_images / noobj_n, 4) if noobj_n else None,
        "multi_object_success": round(multi_ok / multi_n, 4) if multi_n else None,
        "multi_n": multi_n,
        "small_object_success": round(small_tp / small_den, 4) if small_den else None,
        "small_tp": small_tp,
        "small_fn": small_fn,
        "latency_per_image_s": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "latency_p50_s": round(sorted(latencies)[len(latencies) // 2], 4) if latencies else None,
        "ram_rss_start_mb": round(rss0, 1),
        "ram_rss_peak_mb": round(peak_rss, 1),
        "vram_mb": round(_vram_mb(), 1),
        "per_class": per_class,
        "class_stats": class_stats,
    }


def _frcnn_predict_factory():
    from core.global_object_intelligence import GlobalObjectIntelligence

    det = GlobalObjectIntelligence(enabled=True, min_confidence=CONF)
    weights_name = "FasterRCNN_ResNet50_FPN_Weights.DEFAULT (COCO_V1, 2017)"
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

        w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        weights_name = getattr(w, "url", None) or str(w)
        meta = getattr(w, "meta", {}) or {}
        if "file_name" in str(meta) or hasattr(w, "url"):
            weights_name = getattr(w, "url", str(w))
    except Exception:
        pass
    return det.detect, {
        "id": "fasterrcnn",
        "name": "Faster R-CNN torchvision ResNet50-FPN",
        "weights": weights_name,
        "package": "torchvision",
    }


def _ultra_predict_factory(weight: str, model_id: str, name: str):
    from ultralytics import YOLO

    path = WEIGHTS_DIR / weight
    model = YOLO(str(path) if path.is_file() else weight)
    # persist downloaded weights under cache, not CWD clutter
    src = Path(weight)
    if src.is_file() and src.parent != WEIGHTS_DIR:
        dest = WEIGHTS_DIR / src.name
        if not dest.is_file():
            dest.write_bytes(src.read_bytes())
        try:
            src.unlink()
        except OSError:
            pass
        weight_file = dest.name
    else:
        weight_file = path.name if path.is_file() else weight
    used = None
    ckpt = getattr(model, "ckpt_path", None) or getattr(model, "model_name", None)
    if ckpt:
        used = Path(str(ckpt)).name

    def predict(image_path: str):
        from core.global_object_intelligence import DetectedObject, _TR

        r = model.predict(image_path, verbose=False, conf=CONF, device="cpu")[0]
        im = Image.open(image_path)
        w, h = im.size
        area = max(1, w * h)
        out = []
        counters: dict[str, int] = {}
        if r.boxes is None:
            return out
        names = r.names
        for box in r.boxes:
            lab = names[int(box.cls)]
            score = float(box.conf)
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            counters[lab] = counters.get(lab, 0) + 1
            ba = (x2 - x1) * (y2 - y1)
            out.append(
                DetectedObject(
                    instance_id=f"{lab}_{counters[lab]:02d}",
                    label=lab,
                    label_tr=_TR.get(lab, lab),
                    confidence=round(score, 4),
                    bbox=(x1, y1, x2, y2),
                    area_ratio=round(ba / area, 6),
                    center=(round(((x1 + x2) / 2) / w, 4), round(((y1 + y2) / 2) / h, 4)),
                )
            )
        return out

    return predict, {
        "id": model_id,
        "name": name,
        "weights": used or weight_file,
        "package": "ultralytics",
        "ultralytics_version": __import__("ultralytics").__version__,
    }


def main() -> None:
    hw = _hw()
    tmp = Path(tempfile.mkdtemp(prefix="obj_bench_"))
    photos = load_photos(tmp)
    warmup = str(photos[0]["path"]) if photos else None
    results = {
        "hardware": hw,
        "n_photos": len(photos),
        "n_coco": sum(1 for p in photos if not p["no_object"]),
        "n_no_object": sum(1 for p in photos if p["no_object"]),
        "conf": CONF,
        "archive_116k_scanned": False,
        "production_object_index_written": False,
        "clip_as_detector": False,
        "models": {},
        "skipped": {},
    }
    candidates = []

    try:
        candidates.append(_frcnn_predict_factory())
    except Exception as exc:
        results["skipped"]["fasterrcnn"] = f"{type(exc).__name__}: {exc}"

    yolo_ok = False
    for weight, mid, nm in (
        ("yolo11n.pt", "yolo11n", "YOLO11n ultralytics COCO"),
        ("yolov8n.pt", "yolov8n", "YOLOv8n ultralytics COCO"),
    ):
        try:
            os.chdir(WEIGHTS_DIR)
            candidates.append(_ultra_predict_factory(weight, mid, nm))
            yolo_ok = True
            break
        except Exception as exc:
            results["skipped"][mid] = f"{type(exc).__name__}: {exc}"
        finally:
            os.chdir(ROOT)
    if not yolo_ok:
        results["skipped"]["yolo"] = "neither yolo11n nor yolov8n loaded"

    try:
        os.chdir(WEIGHTS_DIR)
        candidates.append(
            _ultra_predict_factory("rtdetr-l.pt", "rtdetr-l", "RT-DETR-L ultralytics COCO")
        )
    except Exception as exc:
        results["skipped"]["rtdetr-l"] = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(ROOT)

    for predict, meta in candidates:
        print("RUN", meta["id"], flush=True)
        try:
            metrics = _eval(photos, predict, warmup=warmup)
            metrics.update(meta)
            results["models"][meta["id"]] = metrics
            print("DONE", meta["id"], json.dumps({k: metrics[k] for k in ("precision", "recall", "f1", "latency_per_image_s")}), flush=True)
        except Exception as exc:
            results["skipped"][meta["id"]] = f"{type(exc).__name__}: {exc}"
            print("FAIL", meta["id"], exc, flush=True)

    out = ROOT / "data" / "reports" / "object_detector_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", out)
    print("OBJECT_DETECTOR_BENCH " + json.dumps(results, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
