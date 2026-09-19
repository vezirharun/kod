"""Spatial V2 object evidence layer (not Pattern Index, not segmentation).

RT-DETR-L supplies COCO-80 boxes. Optional YOLO-World supplies open-vocab boxes
from a detection head (CLIP is only the class-name encoder — never CLIP-similarity
heatmaps as detections). Callers pass the object DB path; tests use tmp.
Does not walk the 116K archive. Does not write patterns.db or FAISS.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from core.global_object_intelligence import DetectedObject, _TR
from core.object_index import ObjectIndexStore
from core.object_intelligence import (
    DETECTOR_BACKEND,
    ObjectIntelligence,
    _nms_mixed,
)

logger = logging.getLogger(__name__)

EXTRA_DETECTOR_BACKEND = "ultralytics_yoloworld_v2_boxes"
WORLD_WEIGHTS = "yolov8s-worldv2.pt"
CLIP_AS_DETECTOR = False

# Open-vocab textile + spatial classes. Not COCO-80 (except zebra/person/handbag overlap).
OPEN_VOCAB_CLASSES: tuple[str, ...] = (
    "flower",
    "leaf",
    "rose",
    "leopard",
    "butterfly",
    "snake",
    "zebra",
    "paisley",
    "geometric pattern",
    "floral",
    "person",
    "handbag",
)
TEXTILE_OPEN: frozenset[str] = frozenset(
    {
        "flower",
        "leaf",
        "rose",
        "leopard",
        "butterfly",
        "snake",
        "zebra",
        "paisley",
        "geometric",
        "floral",
    }
)
_TR_EXTRA = {
    "flower": "çiçek",
    "leaf": "yaprak",
    "rose": "gül",
    "leopard": "leopar",
    "butterfly": "kelebek",
    "snake": "yılan",
    "zebra": "zebra",
    "paisley": "paisley",
    "geometric": "geometrik",
    "geometric pattern": "geometrik",
    "floral": "çiçekli",
}
_LABEL_CANON = {"geometric pattern": "geometric"}

_world_lock = threading.Lock()
_world_model = None
_world_err = ""
_world_loaded = ""


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _world_weight_path() -> Path:
    cached = _root() / ".cache_object_photos" / "detector_weights" / WORLD_WEIGHTS
    if cached.is_file():
        return cached
    cwd = _root() / WORLD_WEIGHTS
    if cwd.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(cwd), str(cached))
            return cached
        except OSError:
            return cwd
    return cached


def extra_detector_available() -> bool:
    return _load_world()


def extra_detector_backend() -> str:
    return EXTRA_DETECTOR_BACKEND if _load_world() else "unavailable"


def extra_detector_error() -> str:
    return _world_err


def _load_world() -> bool:
    global _world_model, _world_err, _world_loaded
    if _world_model is not None:
        return True
    if _world_err:
        return False
    with _world_lock:
        if _world_model is not None:
            return True
        try:
            from ultralytics import YOLOWorld

            wpath = _world_weight_path()
            model = YOLOWorld(str(wpath) if wpath.is_file() else WORLD_WEIGHTS)
            model.set_classes(list(OPEN_VOCAB_CLASSES))
            _world_model = model
            _world_loaded = str(wpath if wpath.is_file() else WORLD_WEIGHTS)
            return True
        except Exception as exc:
            _world_err = f"{type(exc).__name__}: {exc}"
            logger.warning("YOLO-World extra detector unavailable: %s", _world_err)
            _world_model = None
            return False


def _to_dets(image_path: str, result, *, min_confidence: float, max_detections: int) -> list[DetectedObject]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_area = max(1, width * height)
    detections: list[DetectedObject] = []
    counters: dict[str, int] = {}
    if result.boxes is None:
        return detections
    names = result.names
    for box in result.boxes:
        lab = _LABEL_CANON.get(str(names[int(box.cls)]), str(names[int(box.cls)]))
        score = float(box.conf)
        if score < min_confidence:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        counters[lab] = counters.get(lab, 0) + 1
        area = (x2 - x1) * (y2 - y1)
        detections.append(
            DetectedObject(
                instance_id=f"{lab}_{counters[lab]:02d}",
                label=lab,
                label_tr=_TR_EXTRA.get(lab, _TR.get(lab, lab)),
                confidence=round(score, 4),
                bbox=(x1, y1, x2, y2),
                area_ratio=round(area / image_area, 6),
                center=(round(((x1 + x2) / 2) / width, 4), round(((y1 + y2) / 2) / height, 4)),
            )
        )
        if len(detections) >= max_detections:
            break
    return detections


def detect_open_vocab(
    image_path: str,
    *,
    min_confidence: float = 0.25,
    max_detections: int = 50,
) -> list[DetectedObject]:
    """YOLO-World boxes. Empty if model missing — never CLIP-similarity boxes."""
    if CLIP_AS_DETECTOR:
        return []
    if not image_path or not os.path.isfile(image_path):
        return []
    if not _load_world():
        return []
    _world_model.set_classes(list(OPEN_VOCAB_CLASSES))
    result = _world_model.predict(
        image_path,
        verbose=False,
        conf=min_confidence,
        iou=0.5,
        max_det=max_detections,
        imgsz=640,
        device="cpu",
    )[0]
    return _to_dets(image_path, result, min_confidence=min_confidence, max_detections=max_detections)


def _nms_same_label(dets: list[DetectedObject], iou_th: float = 0.65) -> list[DetectedObject]:
    by: dict[str, list[DetectedObject]] = {}
    for d in dets:
        by.setdefault(d.label, []).append(d)
    out: list[DetectedObject] = []
    for group in by.values():
        out.extend(_nms_mixed(group, iou_th))
    return out


def merge_evidence(
    coco: list[DetectedObject], extra: list[DetectedObject]
) -> list[DetectedObject]:
    """Keep textile extra labels even when they overlap a COCO potted-plant box."""
    return _nms_same_label(list(extra) + list(coco))


class ObjectEvidenceLayer:
    """Detect + persist object evidence into an isolated object_index (tmp in tests)."""

    def __init__(
        self,
        object_db_path: str | Path,
        *,
        coco_min_confidence: float = 0.45,
        extra_min_confidence: float = 0.25,
        max_detections: int = 50,
    ):
        self.object_db_path = str(object_db_path)
        self.store = ObjectIndexStore(self.object_db_path)
        self.coco_min_confidence = float(coco_min_confidence)
        self.extra_min_confidence = float(extra_min_confidence)
        self.max_detections = max(1, min(200, int(max_detections)))
        self._coco = ObjectIntelligence(
            self.object_db_path,
            min_confidence=self.coco_min_confidence,
            max_detections=self.max_detections,
        )

    def detect_coco(self, image_path: str) -> list[DetectedObject]:
        return self._coco.detect(image_path)

    def detect_extra(self, image_path: str) -> list[DetectedObject]:
        return detect_open_vocab(
            image_path,
            min_confidence=self.extra_min_confidence,
            max_detections=self.max_detections,
        )

    def detect(self, image_path: str) -> dict[str, Any]:
        coco = self.detect_coco(image_path)
        extra = self.detect_extra(image_path) if extra_detector_available() else []
        merged = merge_evidence(coco, extra)
        backends = [self._coco.detector_backend()]
        if extra:
            backends.append(EXTRA_DETECTOR_BACKEND)
        elif extra_detector_available():
            backends.append(EXTRA_DETECTOR_BACKEND + ":empty")
        return {
            "coco": coco,
            "extra": extra,
            "merged": merged,
            "backend": "+".join(backends),
            "clip_as_detection": False,
            "coco_backend": self._coco.detector_backend() or DETECTOR_BACKEND,
            "extra_backend": extra_detector_backend(),
        }

    def persist(self, file_id: int, image_path: str, objects: list[DetectedObject] | None = None) -> int:
        pack = None
        dets = objects
        if dets is None:
            pack = self.detect(image_path)
            dets = pack["merged"]
        rows = []
        for d in dets:
            rows.append(
                {
                    "label": d.label,
                    "label_tr": d.label_tr,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "area_ratio": d.area_ratio,
                    "instance_id": d.instance_id,
                }
            )
        detector = (pack or {}).get("backend") or (
            EXTRA_DETECTOR_BACKEND if extra_detector_available() else DETECTOR_BACKEND
        )
        return self.store.replace_file_objects(
            int(file_id),
            str(image_path),
            rows,
            detector=str(detector),
        )
