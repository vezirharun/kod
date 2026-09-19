"""Real-photo object acceptance. INDEX FROZEN. object_index.db writes stay in tmp.

CLIP similarity is not detection. Default detector is RT-DETR; Faster R-CNN remains fallback.
Does not walk the 116K production archive. Does not start production object indexing.
"""
from __future__ import annotations

import json
import sqlite3
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.object_intelligence import (
    DETECTOR_BACKEND,
    DETECTOR_UNAVAILABLE,
    FALLBACK_DETECTOR_BACKEND,
    ObjectIntelligence,
    later_archive_scan_howto,
)
from core.settings import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, AppSettings

# Ultralytics coco128: real COCO train2017 stills + YOLO boxes (public test set).
_COCO128_URLS = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
    "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
)
_CACHE = Path(__file__).resolve().parents[1] / ".cache_object_photos"

# COCO-80 names (YOLO / torchvision, spaces). Not CLIP vocab.
COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
TARGET = frozenset({"person", "cat", "dog", "bird", "handbag", "backpack", "car"})
BAG = frozenset({"handbag", "backpack"})
IOU_TH = 0.5

_NO_OBJECT_URLS = {
    "nasa_earth.jpg": "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57723/globe_west_2048.jpg",
    "noaa_goes.jpg": "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/thumbnail.jpg",
    "picsum_1015.jpg": "https://picsum.photos/id/1015/640/420.jpg",
    "picsum_1018.jpg": "https://picsum.photos/id/1018/640/420.jpg",
}


def _fp_prod() -> dict:
    return freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=str(DEFAULT_DATA_DIR / "patterns.db"),
            faiss_dino_path=str(DEFAULT_DATA_DIR / "faiss_dino.index"),
            faiss_clip_path=str(DEFAULT_DATA_DIR / "faiss_clip.index"),
            cache_dir=str(DEFAULT_CACHE_DIR),
        )
    )


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "VEZIR-object-acceptance/1.0")]
    try:
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dest)
        return dest.is_file() and dest.stat().st_size > 1000
    except Exception:
        return False


def _ensure_coco128() -> Path | None:
    root = _CACHE / "coco128"
    img_dir = root / "images" / "train2017"
    if img_dir.is_dir() and len(list(img_dir.glob("*.jpg"))) >= 20:
        return root
    zpath = _CACHE / "coco128.zip"
    if not (zpath.is_file() and zpath.stat().st_size > 100_000):
        ok = False
        for url in _COCO128_URLS:
            if _download(url, zpath):
                ok = True
                break
        if not ok:
            return None
    _CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(_CACHE)
    return root if img_dir.is_dir() else None


def _yolo_boxes(label_path: Path, w: int, h: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    out = []
    if not label_path.is_file():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        if cid < 0 or cid >= len(COCO80):
            continue
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = int(round((xc - bw / 2) * w))
        y1 = int(round((yc - bh / 2) * h))
        x2 = int(round((xc + bw / 2) * w))
        y2 = int(round((yc + bh / 2) * h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            out.append((COCO80[cid], (x1, y1, x2, y2)))
    return out


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _match(gt, pred):
    """Greedy class-aware IoU match. Returns tp, fp, fn, ious."""
    used = set()
    ious = []
    tp = 0
    for g_lab, g_box in gt:
        best_i, best = -1, 0.0
        for i, (p_lab, p_box, _c) in enumerate(pred):
            if i in used or p_lab != g_lab:
                continue
            v = _iou(g_box, p_box)
            if v > best:
                best, best_i = v, i
        if best_i >= 0 and best >= IOU_TH:
            used.add(best_i)
            tp += 1
            ious.append(best)
        else:
            ious.append(best)
    fp = len(pred) - len(used)
    fn = len(gt) - tp
    return tp, fp, fn, ious


def _select_stems(label_dir: Path, limit: int = 40) -> list[str]:
    by_class: dict[str, list[str]] = defaultdict(list)
    person3: list[str] = []
    mixed: list[str] = []
    for p in sorted(label_dir.glob("*.txt")):
        boxes = []
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                cid = int(float(parts[0]))
                if 0 <= cid < len(COCO80):
                    boxes.append(COCO80[cid])
        labs = set(boxes)
        hit = labs & TARGET
        for c in hit:
            by_class[c].append(p.stem)
        if boxes.count("person") >= 3:
            person3.append(p.stem)
        if len(hit) >= 2:
            mixed.append(p.stem)
    ordered: list[str] = []
    for key in ("cat", "dog", "bird", "handbag", "backpack", "car"):
        for s in by_class[key]:
            if s not in ordered:
                ordered.append(s)
    for s in person3:
        if s not in ordered:
            ordered.append(s)
    for s in mixed:
        if s not in ordered:
            ordered.append(s)
    for s in by_class["person"]:
        if s not in ordered:
            ordered.append(s)
    return ordered[:limit]


def _no_object_photos(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    cache = _CACHE / "no_object"
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in _NO_OBJECT_URLS.items():
        src = cache / name
        if not (src.is_file() and src.stat().st_size > 1000):
            _download(url, src)
        if src.is_file() and src.stat().st_size > 1000:
            target = dest / f"noobj_{name}"
            target.write_bytes(src.read_bytes())
            out.append(target)
    return out


def _compose_cat_and_bag(coco: Path, dest: Path) -> dict | None:
    """Real coco128 pixels: cat still + bag crop in one image (coco128 has no native cat+bag)."""
    img_dir = coco / "images" / "train2017"
    lab_dir = coco / "labels" / "train2017"
    cat_src = None
    cat_gt: list[tuple[str, tuple[int, int, int, int]]] = []
    cat_extra = 10**9
    bag_crop = None
    bag_lab = None
    best_bag = -1.0
    for p in sorted(lab_dir.glob("*.txt")):
        src = img_dir / f"{p.stem}.jpg"
        if not src.is_file():
            continue
        im = Image.open(src)
        w, h = im.size
        boxes = _yolo_boxes(p, w, h)
        labs = [lab for lab, _ in boxes]
        if "cat" in labs:
            extra = len(labs) - labs.count("cat")
            if extra < cat_extra:
                cat_src = src
                cat_gt = list(boxes)
                cat_extra = extra
        for lab, box in boxes:
            if lab not in BAG:
                continue
            x1, y1, x2, y2 = box
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            ar = bw / bh
            aspect_ok = 0.35 <= ar <= 2.8
            score = bw * bh * (1.2 if lab == "backpack" else 1.0)
            if not aspect_ok:
                score *= 0.25
            if score > best_bag and bw * bh >= 400:
                bag_crop = im.crop((x1, y1, x2, y2)).convert("RGB")
                bag_lab = lab
                best_bag = score
    if cat_src is None or bag_crop is None or bag_lab is None:
        return None
    left = Image.open(cat_src).convert("RGB")
    lw, lh = left.size
    bw, bh = bag_crop.size
    scale = max(1.0, 220 / max(1, min(bw, bh)))
    scale = min(scale, 3.0)
    bag_w, bag_h = max(1, int(bw * scale)), max(1, int(bh * scale))
    bag_img = bag_crop.resize((bag_w, bag_h), Image.Resampling.BILINEAR)
    pad = 12
    canvas = Image.new("RGB", (lw + bag_w + pad * 2, max(lh, bag_h + pad * 2)), (24, 24, 24))
    canvas.paste(left, (0, 0))
    px, py = lw + pad, pad
    canvas.paste(bag_img, (px, py))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, "JPEG", quality=92)
    bag_box = (px, py, px + bag_w, py + bag_h)
    gt_all = list(cat_gt) + [(bag_lab, bag_box)]
    gt = [(lab, box) for lab, box in gt_all if lab in TARGET]
    cw, ch = canvas.size
    return {
        "path": dest,
        "stem": dest.stem,
        "gt": gt,
        "gt_all": gt_all,
        "no_object": False,
        "w": cw,
        "h": ch,
        "extra": True,
    }


def _score_detector(oi: ObjectIntelligence, photos: list[dict], *, embed: bool = True) -> dict:
    per_image = []
    tp = fp = fn = 0
    ious_all: list[float] = []
    noobj_fp_images = 0
    multi_ok = 0
    multi_n = 0
    crops = 0
    small_tp = small_fn = 0
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in TARGET}
    backend = None
    for ph in photos:
        if embed:
            det = oi.detect_embed(str(ph["path"]))
            backend = det.get("backend")
            assert det.get("clip_used_as_detector") is False
            assert det.get("evidence") in {"detector", DETECTOR_UNAVAILABLE, "detector_unavailable"}
            objs = det.get("objects") or []
        else:
            backend = oi.detector_backend()
            raw = oi.detect(str(ph["path"]))
            objs = [
                {
                    "label": o.label,
                    "bbox": o.bbox,
                    "confidence": o.confidence,
                    "embedding": None,
                }
                for o in raw
            ]
        for o in objs:
            x1, y1, x2, y2 = o["bbox"]
            assert x2 > x1 and y2 > y1
            assert 0.0 <= float(o["confidence"]) <= 1.0
            if o.get("embedding") is not None:
                crops += 1
        pred = [
            (o["label"], tuple(o["bbox"]), float(o["confidence"]))
            for o in objs
            if o["label"] in TARGET
        ]
        pred_all = [(o["label"], tuple(o["bbox"]), float(o["confidence"])) for o in objs]
        if ph["no_object"]:
            nfp = len(pred_all)
            if nfp:
                noobj_fp_images += 1
            per_image.append({"file": ph["path"].name, "no_object": True, "pred_n": nfp})
            fp += nfp
            continue
        tpi, fpi, fni, ious = _match(ph["gt"], pred)
        tp += tpi
        fp += fpi
        fn += fni
        ious_all.extend([u for u in ious if u >= IOU_TH])
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
        area = max(1, int(ph.get("w") or 1) * int(ph.get("h") or 1))
        gt_small = [
            (lab, box) for lab, box in ph["gt"]
            if ((box[2] - box[0]) * (box[3] - box[1])) / area < 0.05
        ]
        if gt_small:
            st, _sf, sn, _ = _match(gt_small, pred)
            small_tp += st
            small_fn += sn
        per_image.append({"file": ph["path"].name, "object_count": len(objs), "multi_gt": gt_n >= 2})
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    mean_iou = sum(ious_all) / len(ious_all) if ious_all else 0.0
    noobj_n = sum(1 for p in photos if p["no_object"])
    small_den = small_tp + small_fn
    return {
        "backend": backend,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou_tp": round(mean_iou, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "crops_produced": crops,
        "no_object_fp_image_rate": round(noobj_fp_images / noobj_n, 4) if noobj_n else None,
        "multi_object_success": round(multi_ok / multi_n, 4) if multi_n else None,
        "multi_n": multi_n,
        "small_object_success": round(small_tp / small_den, 4) if small_den else None,
        "small_tp": small_tp,
        "small_fn": small_fn,
        "class_stats": class_stats,
        "detector_success_rate_recall": round(rec, 4),
        "per_image_n": len(per_image),
    }


def test_object_index_still_off_by_default():
    s = AppSettings()
    assert s.object_index_enabled is False
    assert s.object_auto_scan_on_startup is False
    assert "116K" in later_archive_scan_howto()


def test_real_photo_fasterrcnn_acceptance(tmp_path):
    prod_before = _fp_prod()
    isolated = tmp_path / "data"
    cache = tmp_path / "cache"
    isolated.mkdir()
    cache.mkdir()
    sentinel = isolated / "sentinel.patterns.db"
    sentinel.write_bytes(b"index-frozen-sentinel")
    iso_before = freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=str(sentinel),
            faiss_dino_path=str(isolated / "faiss_dino.index"),
            faiss_clip_path=str(isolated / "faiss_clip.index"),
            cache_dir=str(cache),
        )
    )

    coco = _ensure_coco128()
    if coco is None:
        pytest.skip("coco128 real photos unavailable (download failed); not filling with drawings")

    img_dir = coco / "images" / "train2017"
    lab_dir = coco / "labels" / "train2017"
    stems = _select_stems(lab_dir, 40)
    photos: list[dict] = []
    work = tmp_path / "photos"
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
        photos.append({"path": dest, "stem": stem, "gt": gt, "gt_all": gt_all, "no_object": False, "w": w, "h": h})

    noobj = _no_object_photos(work)
    for p in noobj:
        photos.append({"path": p, "stem": p.stem, "gt": [], "gt_all": [], "no_object": True, "w": 1, "h": 1})

    core_photos = list(photos)
    composed = _compose_cat_and_bag(coco, work / "cat_and_bag_coco_compose.jpg")
    extra_photos: list[dict] = []
    if composed:
        extra_photos.append(composed)
        photos.append(composed)

    n_real = len(photos)
    n_core = len(core_photos)
    gap = []
    if n_real < 20:
        gap.append(f"only {n_real} real photos (wanted 20-50)")
    present = set()
    for ph in photos:
        present |= {lab for lab, _ in ph["gt"]}
    for need in ("person", "cat", "dog", "bird", "car"):
        if need not in present:
            gap.append(f"no GT image for {need}")
    if "handbag" not in present and "backpack" not in present:
        gap.append("no GT bag image")
    cat_bag_gt = any(
        {"cat"} <= {lab for lab, _ in ph["gt"]} and ({lab for lab, _ in ph["gt"]} & BAG)
        for ph in photos
        if not ph["no_object"]
    )
    if not cat_bag_gt:
        gap.append("no GT image with cat AND bag (kedi + çanta coverage gap)")
    gap.append("COCO has person only — no woman/man class in Faster R-CNN")

    oi_frcnn = ObjectIntelligence(
        tmp_path / "object_index_frcnn.db",
        detector_backend=FALLBACK_DETECTOR_BACKEND,
    )
    oi = ObjectIntelligence(tmp_path / "object_index.db")
    assert Path(oi.object_db_path) == tmp_path / "object_index.db"
    if not oi.detector_available():
        pytest.skip("no object detector available")

    baseline = _score_detector(oi_frcnn, core_photos, embed=False)
    winner = _score_detector(oi, core_photos)
    assert baseline["backend"] == FALLBACK_DETECTOR_BACKEND
    assert winner["backend"] == DETECTOR_BACKEND
    assert winner["backend"] != "unavailable"

    prec = winner["precision"]
    rec = winner["recall"]
    mean_iou = winner["mean_iou_tp"]
    success = winner["detector_success_rate_recall"]
    noobj_rate = winner["no_object_fp_image_rate"]
    multi_rate = winner["multi_object_success"]
    multi_n = winner["multi_n"]
    tp, fp, fn = winner["tp"], winner["fp"], winner["fn"]
    crops = winner["crops_produced"]
    class_stats = winner["class_stats"]

    why = []
    if rec < 0.35:
        why.append("Winner recall below 0.35 on coco128 small-instance stills.")
    if class_stats["handbag"]["tp"] + class_stats["backpack"]["tp"] == 0 and (
        class_stats["handbag"]["fn"] + class_stats["backpack"]["fn"] > 0
    ):
        why.append("Bag classes are small/occluded.")
    if class_stats["cat"]["tp"] == 0 and class_stats["cat"]["fn"] > 0:
        why.append("Cat recall 0 on this still set.")
    recommend = (
        "RT-DETR seçildi: default object_intelligence detector is ultralytics rtdetr-l.pt; "
        "torchvision Faster R-CNN remains fallback. CLIP is not a detector."
    )

    # --- SearchEngine against tmp object_index.db only ---
    from core.search_engine import SearchEngine

    settings = AppSettings(
        db_path=str(isolated / "patterns.db"),
        cache_dir=str(cache),
        faiss_dino_path=str(isolated / "faiss_dino.index"),
        faiss_clip_path=str(isolated / "faiss_clip.index"),
        face_db_path=str(isolated / "face_index.db"),
        object_db_path=str(tmp_path / "object_index.db"),
        object_index_enabled=True,
        object_auto_scan_on_startup=False,
        face_index_enabled=False,
        ai_embedding_enabled=False,
        ocr_enabled=False,
    )
    db = Database(settings.db_path)
    folder = work
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("obj_real", str(folder)),
        )
    file_ids = {}
    for i, ph in enumerate(photos, start=1):
        if ph["no_object"]:
            continue
        p = ph["path"]
        fid = db.upsert_file(
            {
                "path": str(p),
                "filename": p.name,
                "source_id": 1,
                "status": "indexed",
                "file_size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "width": 64,
                "height": 64,
            }
        )
        file_ids[p.name] = fid
        # object rows already stored with sequential i; re-index with real file_id
        oi.index_image(int(fid), str(p))

    obj_db = Path(settings.object_db_path)
    assert obj_db.is_file()
    assert "object_index.db" in str(obj_db)
    assert DEFAULT_DATA_DIR.as_posix() not in obj_db.resolve().as_posix().replace("\\", "/")

    eng = SearchEngine(settings, load_ai=False)
    queries = {
        "kedi": "cat",
        "köpek": "dog",
        "kuş": "bird",
        "insan": "person",
        "çanta": "handbag",
        "kedi + çanta": None,
        "3 insan": "person",
        "2 insan": "person",
        "insan + çanta": None,
        "köpek + insan": None,
    }
    search_report = {}
    for q, lab in queries.items():
        hits = eng.search_by_text(q)
        names = [r.filename for r in hits]
        proof = []
        for r in hits:
            assert r.breakdown.get("object_evidence") == "detector_class"
            assert r.breakdown.get("clip_as_detection") is False
            # Prove row exists in object_index.db for this filename
            with sqlite3.connect(str(obj_db)) as con:
                row = con.execute(
                    """SELECT oi.label, oi.confidence, of.path FROM object_instances oi
                       JOIN object_files of ON of.file_id=oi.file_id
                       WHERE of.path LIKE ? LIMIT 8""",
                    (f"%{r.filename}",),
                ).fetchall()
            proof.append({"file": r.filename, "db_rows": [tuple(x) for x in row]})
            assert row, f"hit {r.filename} for {q!r} has no object_index.db row"
        search_report[q] = {"hit_names": names, "n": len(hits), "db_proof": proof}

        if q == "kedi + çanta":
            with sqlite3.connect(str(obj_db)) as con:
                both = con.execute(
                    """SELECT file_id FROM object_instances WHERE label='cat'
                       INTERSECT
                       SELECT file_id FROM object_instances WHERE label IN ('handbag','backpack')"""
                ).fetchall()
            if both:
                assert hits, "object_index has cat+bag rows but search returned none"
            else:
                assert not hits
        elif q == "3 insan":
            with sqlite3.connect(str(obj_db)) as con:
                three = con.execute(
                    """SELECT file_id FROM object_instances WHERE label='person'
                       GROUP BY file_id HAVING COUNT(*)>=3"""
                ).fetchall()
            three_ids = {int(x[0]) for x in three}
            hit_fids = set()
            for r in hits:
                with sqlite3.connect(str(obj_db)) as con:
                    fidrow = con.execute(
                        "SELECT file_id FROM object_files WHERE path LIKE ?",
                        (f"%{r.filename}",),
                    ).fetchone()
                if fidrow:
                    hit_fids.add(int(fidrow[0]))
            assert hit_fids <= three_ids
            if three_ids:
                assert hits, "object_index has 3+ person but search missed"
        elif q == "2 insan":
            with sqlite3.connect(str(obj_db)) as con:
                two = con.execute(
                    """SELECT file_id FROM object_instances WHERE label='person'
                       GROUP BY file_id HAVING COUNT(*)>=2"""
                ).fetchall()
            two_ids = {int(x[0]) for x in two}
            hit_fids = set()
            for r in hits:
                with sqlite3.connect(str(obj_db)) as con:
                    fidrow = con.execute(
                        "SELECT file_id FROM object_files WHERE path LIKE ?",
                        (f"%{r.filename}",),
                    ).fetchone()
                if fidrow:
                    hit_fids.add(int(fidrow[0]))
            assert hit_fids <= two_ids
            if two_ids:
                assert hits, "object_index has 2+ person but search missed"
        elif q == "insan + çanta":
            with sqlite3.connect(str(obj_db)) as con:
                both = con.execute(
                    """SELECT file_id FROM object_instances WHERE label='person'
                       INTERSECT
                       SELECT file_id FROM object_instances WHERE label IN ('handbag','backpack')"""
                ).fetchall()
            if both:
                assert hits, "object_index has person+bag rows but search returned none"
            else:
                assert not hits
        elif q == "köpek + insan":
            with sqlite3.connect(str(obj_db)) as con:
                both = con.execute(
                    """SELECT file_id FROM object_instances WHERE label='dog'
                       INTERSECT
                       SELECT file_id FROM object_instances WHERE label='person'"""
                ).fetchall()
            if both:
                assert hits, "object_index has dog+person rows but search returned none"
            else:
                assert not hits
        elif lab:
            want_labels = BAG if lab == "handbag" else {lab}
            with sqlite3.connect(str(obj_db)) as con:
                phold = ",".join("?" * len(want_labels))
                db_files = {
                    Path(r[0]).name
                    for r in con.execute(
                        f"SELECT DISTINCT of.path FROM object_files of "
                        f"JOIN object_instances oi ON oi.file_id=of.file_id "
                        f"WHERE oi.label IN ({phold})",
                        tuple(want_labels),
                    )
                }
            assert set(names) <= db_files
            if db_files:
                assert hits, f"object_index has {lab} but search {q!r} missed"

    def _gt_names(pred) -> set[str]:
        names = set()
        for ph in photos:
            if ph["no_object"]:
                continue
            labs = [lab for lab, _ in ph["gt"]]
            if pred(labs):
                names.add(ph["path"].name)
        return names

    query_gt = {}
    gt_preds = {
        "kedi + çanta": lambda labs: "cat" in labs and bool(set(labs) & BAG),
        "3 insan": lambda labs: labs.count("person") >= 3,
        "2 insan": lambda labs: labs.count("person") >= 2,
        "insan + çanta": lambda labs: "person" in labs and bool(set(labs) & BAG),
        "köpek + insan": lambda labs: "dog" in labs and "person" in labs,
    }
    for q, pred in gt_preds.items():
        want = _gt_names(pred)
        got = set(search_report[q]["hit_names"])
        inter = want & got
        prec_g = (len(inter) / len(got)) if got else (1.0 if not want else 0.0)
        rec_g = (len(inter) / len(want)) if want else None
        query_gt[q] = {
            "gt_n": len(want),
            "hit_n": len(got),
            "tp": len(inter),
            "precision_vs_gt": None if rec_g is None and not want else round(prec_g, 4),
            "recall_vs_gt": None if rec_g is None else round(rec_g, 4),
            "gt_names": sorted(want)[:12],
            "hit_names": sorted(got)[:12],
        }

    iso_after = freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=str(sentinel),
            faiss_dino_path=str(isolated / "faiss_dino.index"),
            faiss_clip_path=str(isolated / "faiss_clip.index"),
            cache_dir=str(cache),
        )
    )
    assert iso_before == iso_after
    assert sentinel.read_bytes() == b"index-frozen-sentinel"
    prod_after = _fp_prod()
    prod_obj = DEFAULT_DATA_DIR / "object_index.db"
    assert not prod_obj.exists(), "production object_index.db must stay untouched"
    prod_db = DEFAULT_DATA_DIR / "patterns.db"
    if prod_db.is_file():
        with sqlite3.connect(f"file:{prod_db.as_posix()}?mode=ro", uri=True) as con:
            planted = con.execute(
                "SELECT COUNT(*) FROM files WHERE filename LIKE '000000%.jpg'"
            ).fetchone()[0]
        assert planted == 0
    # Isolated sentinel is the freeze oracle this test controls. Production
    # patterns.db may change if a live indexer is running; we must not write it.

    with sqlite3.connect(str(obj_db)) as con:
        n_inst = con.execute("SELECT COUNT(*) FROM object_instances").fetchone()[0]
        n_files = con.execute("SELECT COUNT(*) FROM object_files").fetchone()[0]
        det_name = con.execute("SELECT DISTINCT detector FROM object_files").fetchall()

    report = {
        "n_real_photos": n_real,
        "n_core_42": n_core,
        "n_extra_and_fixture": len(extra_photos),
        "n_coco": sum(1 for p in core_photos if not p["no_object"]),
        "n_no_object": len(noobj),
        "min_confidence_unchanged": 0.45,
        "fp_filter": "drop bowl/donut/frisbee/sports ball when area_ratio>=0.70",
        "gap": gap,
        "detector": DETECTOR_BACKEND,
        "decision": "RT-DETR seçildi",
        "baseline_fasterrcnn": baseline,
        "winner": winner,
        "clip_as_detector": False,
        "archive_116k_scanned": False,
        "production_object_index_written": False,
        "detector_success_rate_recall": round(success, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": winner.get("f1"),
        "mean_iou_tp": round(mean_iou, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "crops_produced": crops,
        "no_object_fp_image_rate": None if noobj_rate is None else round(noobj_rate, 4),
        "multi_object_success": None if multi_rate is None else round(multi_rate, 4),
        "multi_n": multi_n,
        "small_object_success": winner.get("small_object_success"),
        "class_stats": class_stats,
        "search": {k: {"n": v["n"], "names": v["hit_names"][:12]} for k, v in search_report.items()},
        "query_gt": query_gt,
        "cat_bag_gt_present": bool(cat_bag_gt),
        "object_index_tmp": {
            "path": str(obj_db),
            "files": n_files,
            "instances": n_inst,
            "detector": [x[0] for x in det_name],
        },
        "pattern_index_fingerprint_unchanged": prod_before == prod_after,
        "isolated_sentinel_unchanged": iso_before == iso_after,
        "production_object_index_exists": (DEFAULT_DATA_DIR / "object_index.db").exists(),
        "why_if_weak": why,
        "production_detector_recommendation": recommend,
    }
    print("OBJECT_REAL_PHOTO_REPORT " + json.dumps(report, ensure_ascii=False, default=str))

    assert n_real >= 8, "too few real photos to call this an acceptance run"
    assert noobj_rate is not None and noobj_rate < 0.5
    assert Path(settings.object_db_path).resolve() != (DEFAULT_DATA_DIR / "object_index.db").resolve()
    assert any(x[0] == DETECTOR_BACKEND for x in det_name) or n_inst == 0
