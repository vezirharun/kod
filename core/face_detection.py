"""Fail-safe face detection for General Vision.

Detection only: no identity, gender, or other biometric inference is performed here.
Uses OpenCV's bundled Haar cascade when available; the existing pattern-search
pipeline is unaffected if OpenCV/cascade is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import os

try:
    import cv2
    _CV2_OK = True
except Exception:
    cv2 = None
    _CV2_OK = False


@dataclass(frozen=True)
class FaceDetection:
    face_id: str
    bbox: tuple[int, int, int, int]
    confidence: float
    area_ratio: float
    center: tuple[float, float]

    @classmethod
    def capability(cls) -> dict[str, str]:
        return {
            "face_detection": "supported" if _CV2_OK else "optional_dependency",
            "face_counting": "supported" if _CV2_OK else "unsupported",
            "face_identity_matching": "unsupported",
            "gender_classification_from_face": "unsupported",
        }


class FaceDetector:
    _cascade = None

    def __init__(self, *, enabled: bool = True, scale_factor: float = 1.1,
                 min_neighbors: int = 5, min_size: tuple[int, int] = (40, 40),
                 max_faces: int = 50):
        self.enabled = bool(enabled)
        self.scale_factor = max(1.01, float(scale_factor))
        self.min_neighbors = max(2, int(min_neighbors))
        self.min_size = (max(20, int(min_size[0])), max(20, int(min_size[1])))
        self.max_faces = max(1, min(200, int(max_faces)))

    @classmethod
    def _load(cls) -> bool:
        if cls._cascade is not None:
            return True
        if not _CV2_OK:
            return False
        try:
            path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(str(path))
            if cascade.empty():
                return False
            cls._cascade = cascade
            return True
        except Exception:
            return False

    def detect(self, image_path: str) -> list[FaceDetection]:
        if not self.enabled or not image_path or not os.path.isfile(image_path) or not self._load():
            return []
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            boxes = self._cascade.detectMultiScale(
                gray, scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors, minSize=self.min_size,
            )
        except Exception:
            return []
        image_area = max(1, w * h)
        out: list[FaceDetection] = []
        for i, (x, y, bw, bh) in enumerate(boxes[: self.max_faces], start=1):
            area = max(1, int(bw) * int(bh))
            out.append(FaceDetection(
                face_id=f"face_{i:03d}",
                bbox=(int(x), int(y), int(x + bw), int(y + bh)),
                confidence=0.0,  # Haar does not expose calibrated confidence.
                area_ratio=round(area / image_area, 6),
                center=(round((x + bw / 2) / w, 4), round((y + bh / 2) / h, 4)),
            ))
        return out

    def analyze(self, image_path: str) -> dict[str, Any]:
        faces = self.detect(image_path)
        return {
            "enabled": self.enabled,
            "capability": self.capability(),
            "face_count": len(faces),
            "faces": [asdict(face) for face in faces],
            "identity_matching": "unsupported",
            "gender_classification": "unsupported",
        }
