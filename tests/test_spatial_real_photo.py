"""Spatial V1 real-photo acceptance. coco128 GT boxes → tmp object DB only.

Does not write production object_index.db / patterns.db / FAISS.
Does not walk the 116K archive. CLIP is not used as a relation detector.
Flower/leaf/rose/leopard are not COCO classes — reported honestly if unevaluable.
"""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pytest
from PIL import Image

from core.object_index import ObjectIndexStore
from core.settings import DEFAULT_DATA_DIR
from core.spatial.spatial_config import SpatialConfig
from core.spatial.spatial_engine import SpatialEngine
from core.spatial.spatial_geometry import normalize_bbox
from core.spatial.spatial_relation import Relation, evaluate_relation

_ACC = Path(__file__).resolve().parent / "test_object_real_photo_acceptance.py"
_spec = importlib.util.spec_from_file_location("_object_real_photo_acc", _ACC)
_acc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_acc)
_ensure_coco128 = _acc._ensure_coco128
_select_stems = _acc._select_stems
_yolo_boxes = _acc._yolo_boxes

BAG = {"handbag", "backpack"}
PERSON = {"person"}
REPORT = Path(__file__).resolve().parents[1] / "data" / "reports" / "spatial_real_photo_v1.json"


def _gt_relation(person_box, bag_box, rel: Relation, cfg: SpatialConfig, w: int, h: int) -> bool:
    pb = normalize_bbox(person_box, w, h)
    bb = normalize_bbox(bag_box, w, h)
    if pb is None or bb is None:
        return False
    return evaluate_relation(bb, pb, rel, cfg).ok


def test_spatial_real_photo_coco128(tmp_path):
    prod_obj = DEFAULT_DATA_DIR / "object_index.db"
    obj_meta = (prod_obj.exists(), prod_obj.stat().st_mtime_ns if prod_obj.exists() else 0)
    coco = _ensure_coco128()
    if coco is None:
        pytest.skip("coco128 real photos unavailable (download failed); not filling with drawings")
    img_dir = coco / "images" / "train2017"
    lab_dir = coco / "labels" / "train2017"
    stems = _select_stems(lab_dir, limit=42)
    pair_stems = []
    for p in sorted(lab_dir.glob("*.txt")):
        labs = []
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                cid = int(float(parts[0]))
                if 0 <= cid < 80:
                    labs.append(_acc.COCO80[cid])
        if "person" in labs and (set(labs) & BAG):
            pair_stems.append(p.stem)
    ordered = list(dict.fromkeys(pair_stems + stems))[:42]
    stems = ordered
    assert len(stems) >= 20, f"wanted ~42-set, got {len(stems)}"

    store = ObjectIndexStore(tmp_path / "object_index.db")
    cfg = SpatialConfig()
    file_ids = []
    sizes = {}
    gt_near = set()
    gt_left = set()
    n_pair = 0
    for i, stem in enumerate(stems, start=1):
        img_p = img_dir / f"{stem}.jpg"
        if not img_p.is_file():
            continue
        im = Image.open(img_p)
        w, h = im.size
        boxes = _yolo_boxes(lab_dir / f"{stem}.txt", w, h)
        objs = []
        counts: dict[str, int] = defaultdict(int)
        people = []
        bags = []
        for lab, box in boxes:
            counts[lab] += 1
            inst = f"{lab}_{counts[lab]:02d}"
            x1, y1, x2, y2 = box
            area = max(0, x2 - x1) * max(0, y2 - y1) / max(1, w * h)
            objs.append(
                {
                    "label": lab,
                    "label_tr": lab,
                    "confidence": 0.99,
                    "bbox": box,
                    "area_ratio": area,
                    "instance_id": inst,
                }
            )
            if lab in PERSON:
                people.append(box)
            if lab in BAG:
                bags.append(box)
        if not objs:
            continue
        store.replace_file_objects(i, str(img_p), objs, detector="coco128_gt_readonly")
        file_ids.append(i)
        sizes[i] = (w, h)
        if people and bags:
            n_pair += 1
            if any(_gt_relation(p, b, Relation.NEAR, cfg, w, h) for p in people for b in bags):
                gt_near.add(i)
            if any(_gt_relation(p, b, Relation.LEFT_OF, cfg, w, h) for p in people for b in bags):
                gt_left.add(i)

    eng = SpatialEngine(str(tmp_path / "object_index.db"), readonly=True)
    near = eng.search("insanın yanında çanta", image_sizes=sizes)
    left = eng.search("insanın solunda çanta", image_sizes=sizes)
    pred_near = {h.file_id for h in near.hits}
    pred_left = {h.file_id for h in left.hits}

    def _prf(gt: set[int], pred: set[int]):
        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return tp, fp, fn, p, r

    n_tp, n_fp, n_fn, n_p, n_r = _prf(gt_near, pred_near)
    l_tp, l_fp, l_fn, l_p, l_r = _prf(gt_left, pred_left)
    rel_acc_num = n_tp + l_tp
    rel_acc_den = (n_tp + n_fp + n_fn) + (l_tp + l_fp + l_fn)
    rel_acc = rel_acc_num / rel_acc_den if rel_acc_den else 0.0

    flower = eng.search("çiçeğin üstünde yaprak", image_sizes=sizes)
    gul = eng.search("gülün altında yaprak", image_sizes=sizes)
    leo = eng.search("leoparın yanında gül", image_sizes=sizes)
    flower_unevaluable = flower.status == "SPATIAL_EVIDENCE_UNAVAILABLE"
    gul_unevaluable = gul.status == "SPATIAL_EVIDENCE_UNAVAILABLE"
    leo_unevaluable = leo.status == "SPATIAL_EVIDENCE_UNAVAILABLE"

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "n_images": len(file_ids),
                "n_person_bag_pairs": n_pair,
                "near": {"tp": n_tp, "fp": n_fp, "fn": n_fn, "precision": n_p, "recall": n_r},
                "left_of": {"tp": l_tp, "fp": l_fp, "fn": l_fn, "precision": l_p, "recall": l_r},
                "relation_accuracy": rel_acc,
                "flower_leaf_unevaluable": flower_unevaluable,
                "rose_leaf_unevaluable": gul_unevaluable,
                "leopard_rose_unevaluable": leo_unevaluable,
                "clip_as_relation": False,
                "production_object_index_exists": (DEFAULT_DATA_DIR / "object_index.db").exists(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert Path(store.path).resolve() != prod_obj.resolve()
    now_meta = (prod_obj.exists(), prod_obj.stat().st_mtime_ns if prod_obj.exists() else 0)
    assert now_meta == obj_meta

    gaps = []
    if n_pair < 3:
        gaps.append(f"too few person+bag GT pairs ({n_pair}) for spatial real-photo")
    if n_pair >= 3 and n_p < 0.85:
        gaps.append(f"NEAR precision {n_p:.2f} < 0.85")
    if n_pair >= 3 and n_r < 0.85:
        gaps.append(f"NEAR recall {n_r:.2f} < 0.85")
    if n_pair >= 3 and rel_acc < 0.80:
        gaps.append(f"relation accuracy {rel_acc:.2f} < 0.80")
    if gaps:
        pytest.fail("; ".join(gaps) + f" report={REPORT}")
    assert n_pair >= 3
    assert n_p >= 0.85 and n_r >= 0.85
    assert flower_unevaluable and gul_unevaluable and leo_unevaluable
