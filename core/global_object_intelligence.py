"""Global Object Intelligence (optional, fail-safe).

Provides broad object detection using torchvision's COCO detector when available.
The detector is lazy-loaded and never required for the existing pattern-search
pipeline. Human instances are separated by detection boxes; this module does
NOT perform face recognition, identity matching, or biometric identification.

Fine-grained species/brand/model remains the existing zero-shot CLIP layer.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any

from core.face_detection import FaceDetector

logger = logging.getLogger(__name__)

try:  # optional runtime dependency already present in the supplied environment
    import torch
    from PIL import Image
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    _TORCHVISION_OK = True
except Exception:  # pragma: no cover - fail-safe import
    torch = None
    Image = None
    FasterRCNN_ResNet50_FPN_Weights = None
    fasterrcnn_resnet50_fpn = None
    _TORCHVISION_OK = False


COCO_LABELS = (
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

_TR = {
    "person": "insan", "bicycle": "bisiklet", "car": "araba", "motorcycle": "motosiklet", "airplane": "uçak",
    "bus": "otobüs", "train": "tren", "truck": "kamyon", "boat": "tekne", "bird": "kuş", "cat": "kedi",
    "dog": "köpek", "horse": "at", "sheep": "koyun", "cow": "inek", "elephant": "fil", "bear": "ayı",
    "zebra": "zebra", "giraffe": "zürafa", "backpack": "sırt çantası", "bench": "bank", "handbag": "çanta",
    "suitcase": "valiz", "skateboard": "kaykay", "surfboard": "sörf tahtası", "bottle": "şişe", "cup": "kupa",
    "fork": "çatal", "knife": "bıçak", "spoon": "kaşık", "bowl": "kase", "banana": "muz", "apple": "elma",
    "orange": "portakal", "pizza": "pizza", "cake": "pasta", "chair": "sandalye", "couch": "koltuk",
    "potted plant": "saksı bitkisi", "bed": "yatak", "dining table": "yemek masası", "tv": "televizyon",
    "laptop": "dizüstü bilgisayar", "mouse": "bilgisayar faresi", "keyboard": "klavye", "cell phone": "telefon",
    "book": "kitap", "clock": "saat", "scissors": "makas", "teddy bear": "oyuncak ayı",
}


@dataclass(frozen=True)
class DetectedObject:
    instance_id: str
    label: str
    label_tr: str
    confidence: float
    bbox: tuple[int, int, int, int]
    area_ratio: float
    center: tuple[float, float]

    @property
    def is_person(self) -> bool:
        return self.label == "person"


class GlobalObjectIntelligence:
    """Lazy, optional COCO detector + safe instance separation."""

    _model: Any = None
    _transform: Any = None
    _lock = threading.Lock()
    _load_error: str = ""

    def __init__(self, *, enabled: bool = True, min_confidence: float = 0.45, max_detections: int = 50):
        self.enabled = bool(enabled)
        self.min_confidence = max(0.05, min(0.99, float(min_confidence)))
        self.max_detections = max(1, min(200, int(max_detections)))

    @classmethod
    def capability(cls) -> dict[str, Any]:
        return {
            "available": bool(_TORCHVISION_OK),
            "backend": "torchvision_fasterrcnn_coco" if _TORCHVISION_OK else "unavailable",
            "global_common_object_detection": "supported" if _TORCHVISION_OK else "optional_dependency",
            "multi_object_detection": "supported" if _TORCHVISION_OK else "unsupported",
            "object_counting": "supported" if _TORCHVISION_OK else "unsupported",
            "human_instance_separation": "supported" if _TORCHVISION_OK else "unsupported",
            "human_instance_counting": "supported" if _TORCHVISION_OK else "unsupported",
            "person_identity_matching": "supported_by_face_index",
            "gender_classification_from_image": "unsupported",
            "face_recognition": "optional_dependency",
            "face_detection": "supported" if _TORCHVISION_OK else "optional_dependency",
            "biometric_identification": "unsupported",
            "gender_classification_from_face": "supported_by_face_index",
            "fine_grained_species_brand_model": "experimental_zero_shot",
        }

    @classmethod
    def _load(cls) -> bool:
        if cls._model is not None:
            return True
        if not _TORCHVISION_OK:
            cls._load_error = "torchvision_not_available"
            return False
        with cls._lock:
            if cls._model is not None:
                return True
            try:
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                cls._model = fasterrcnn_resnet50_fpn(weights=weights)
                cls._model.eval()
                cls._transform = weights.transforms()
                return True
            except Exception as exc:  # no hard failure in search
                cls._load_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Global Object AI yüklenemedi: %s", cls._load_error)
                cls._model = None
                cls._transform = None
                return False

    def detect(self, image_path: str) -> list[DetectedObject]:
        if not self.enabled or not image_path or not os.path.isfile(image_path):
            return []
        if not self._load():
            return []
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            tensor = self._transform(image)
            with torch.inference_mode():
                out = self._model([tensor])[0]
            labels = out["labels"].detach().cpu().tolist()
            scores = out["scores"].detach().cpu().tolist()
            boxes = out["boxes"].detach().cpu().tolist()
        except Exception as exc:
            logger.warning("Global Object AI görüntü analizi atlandı: %s", exc)
            return []

        detections: list[DetectedObject] = []
        counters: dict[str, int] = {}
        image_area = max(1, width * height)
        for label_id, score, box in zip(labels, scores, boxes):
            score = float(score)
            if score < self.min_confidence:
                continue
            if not 0 < int(label_id) < len(COCO_LABELS):
                continue
            label = COCO_LABELS[int(label_id)]
            if label in ("N/A", "background"):
                continue
            counters[label] = counters.get(label, 0) + 1
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            detections.append(
                DetectedObject(
                    instance_id=f"{label}_{counters[label]:02d}",
                    label=label,
                    label_tr=_TR.get(label, label),
                    confidence=round(score, 4),
                    bbox=(x1, y1, x2, y2),
                    area_ratio=round(area / image_area, 6),
                    center=(round(((x1 + x2) / 2) / width, 4), round(((y1 + y2) / 2) / height, 4)),
                )
            )
            if len(detections) >= self.max_detections:
                break
        return detections

    def analyze(self, image_path: str) -> dict[str, Any]:
        detections = self.detect(image_path)
        by_label: dict[str, int] = {}
        for obj in detections:
            by_label[obj.label_tr] = by_label.get(obj.label_tr, 0) + 1
        people = [o for o in detections if o.is_person]
        face_meta = FaceDetector(enabled=True).analyze(image_path)
        return {
            "enabled": self.enabled,
            "capability": self.capability(),
            "objects": [asdict(o) for o in detections],
            "counts": by_label,
            "person_count": len(people),
            "person_instances": [asdict(o) for o in people],
            "faces": face_meta.get("faces", []),
            "face_count": int(face_meta.get("face_count", 0)),
            "face_capability": face_meta.get("capability", {}),
            "face_identity_matching": "optional_dependency",
            "gender_classification": "unsupported",
            "note": "İnsanlar ayrı örnekler olarak ayrılır; yüz tespiti vardır. Kimlik eşleştirme yalnızca opsiyonel FaceIdentityEngine ile yapılır.",
        }
