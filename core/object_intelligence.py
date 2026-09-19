"""Object Intelligence: RT-DETR detection (Faster R-CNN fallback) + crop embeddings + isolated DB.

CLIP is never used as a detector. Crop embeddings are ResNet18 (ImageNet) when
available, otherwise a labeled color-grid fallback — not zero-shot classification.
Auto-scan of the production archive is not performed by this module.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from core.global_object_intelligence import DetectedObject, GlobalObjectIntelligence, _TORCHVISION_OK, _TR
from core.object_index import ObjectIndexStore
from core.object_search import ObjectSearch, parse_object_query

logger = logging.getLogger(__name__)

FALLBACK_DETECTOR_BACKEND = "torchvision_fasterrcnn_coco"
DETECTOR_BACKEND = "ultralytics_rtdetr_l_coco"
RTDETR_WEIGHTS = "rtdetr-l.pt"
RTDETR_WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt"
)
DETECTOR_UNAVAILABLE = "DETECTOR_UNAVAILABLE"
DETECTOR_FAILED = "DETECTOR_FAILED"
EMBED_RESNET = "resnet18_imagenet_crop"
EMBED_COLORGRID = "colorgrid_crop_fallback"
_NMS_IOU = 0.65
# Globe/satellite disks are often labeled bowl/donut at 0.9+; raising conf does not help.
# Drop only near-full-frame compact disk classes (close-up real bowls in-frame stay).
_DISK_FP_LABELS = frozenset({"bowl", "donut", "frisbee", "sports ball"})
_DISK_FP_AREA = 0.70

_ultra_lock = threading.Lock()
_ultra_model = None
_ultra_err = ""
_ultra_loaded_path = ""

_embed_lock = threading.Lock()
_resnet = None
_resnet_tf = None
_resnet_err = ""


def _rtdetr_weight_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    cached = root / ".cache_object_photos" / "detector_weights" / RTDETR_WEIGHTS
    if cached.is_file() and cached.stat().st_size > 1_000_000:
        return cached
    alt = root / "data" / "detector_weights" / RTDETR_WEIGHTS
    if alt.is_file() and alt.stat().st_size > 1_000_000:
        return alt
    return cached


def ensure_rtdetr_weights() -> Path:
    dest = _rtdetr_weight_path()
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pt.part")
    logger.info("RT-DETR-L weights downloading: %s -> %s", RTDETR_WEIGHTS_URL, dest)
    import urllib.request

    urllib.request.urlretrieve(RTDETR_WEIGHTS_URL, str(tmp))
    tmp.replace(dest)
    logger.info("RT-DETR-L weights ready: %s (%s bytes)", dest, dest.stat().st_size)
    return dest


def reset_rtdetr_cache() -> None:
    global _ultra_model, _ultra_err, _ultra_loaded_path
    with _ultra_lock:
        _ultra_model = None
        _ultra_err = ""
        _ultra_loaded_path = ""


def rtdetr_status() -> dict[str, str]:
    return {
        "backend": DETECTOR_BACKEND if _ultra_model is not None else "",
        "weights": _ultra_loaded_path,
        "error": _ultra_err,
    }


def _load_rtdetr() -> bool:
    global _ultra_model, _ultra_err, _ultra_loaded_path
    if _ultra_model is not None:
        return True
    with _ultra_lock:
        if _ultra_model is not None:
            return True
        try:
            from ultralytics import YOLO

            wpath = ensure_rtdetr_weights()
            if not wpath.is_file():
                raise FileNotFoundError(str(wpath))
            _ultra_model = YOLO(str(wpath))
            _ultra_loaded_path = str(wpath)
            _ultra_err = ""
            logger.info(
                "Object detector backend selected: %s weights=%s",
                DETECTOR_BACKEND,
                _ultra_loaded_path,
            )
            return True
        except Exception as exc:
            _ultra_err = f"{type(exc).__name__}: {exc}"
            _ultra_model = None
            logger.error(
                "RT-DETR-L failed to load (%s). Faster R-CNN fallback is NOT used "
                "unless explicitly allowed. Fix ultralytics / rtdetr-l.pt.",
                _ultra_err,
            )
            return False


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(by2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_mixed(dets: list[DetectedObject], iou_th: float = _NMS_IOU) -> list[DetectedObject]:
    """Keep one label per region so different classes are not mixed on the same box."""
    kept: list[DetectedObject] = []
    for d in sorted(dets, key=lambda x: x.confidence, reverse=True):
        if any(_box_iou(d.bbox, k.bbox) >= iou_th for k in kept):
            continue
        kept.append(d)
    return kept


def _is_fullframe_disk_fp(d: DetectedObject) -> bool:
    """True for disk-class boxes that cover most of the frame (Earth/GOES false positives)."""
    return d.label in _DISK_FP_LABELS and float(d.area_ratio) >= _DISK_FP_AREA


def _detect_rtdetr(image_path: str, *, min_confidence: float, max_detections: int) -> list[DetectedObject]:
    if not image_path or not os.path.isfile(image_path):
        return []
    if not _load_rtdetr():
        raise RuntimeError(_ultra_err or DETECTOR_UNAVAILABLE)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    r = _ultra_model.predict(
        image_path,
        verbose=False,
        conf=min_confidence,
        iou=0.5,
        max_det=max_detections,
        imgsz=640,
        device="cpu",
    )[0]
    detections: list[DetectedObject] = []
    counters: dict[str, int] = {}
    image_area = max(1, width * height)
    if r.boxes is None:
        return detections
    names = r.names
    for box in r.boxes:
        lab = str(names[int(box.cls)])
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
                label_tr=_TR.get(lab, lab),
                confidence=round(score, 4),
                bbox=(x1, y1, x2, y2),
                area_ratio=round(area / image_area, 6),
                center=(round(((x1 + x2) / 2) / width, 4), round(((y1 + y2) / 2) / height, 4)),
            )
        )
        if len(detections) >= max_detections:
            break
    return _nms_mixed(detections)


def later_archive_scan_howto() -> str:
    return (
        "Object auto-scan is off. After review, index an explicit path list only; "
        "do not walk the 116K production archive from app start."
    )


def _l2(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return v
    return v / n


def _colorgrid(crop: Image.Image, cells: int = 8) -> np.ndarray:
    small = crop.convert("RGB").resize((cells, cells), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    return _l2(arr.reshape(-1))


def _load_resnet() -> bool:
    global _resnet, _resnet_tf, _resnet_err
    if _resnet is not None:
        return True
    if _resnet_err:
        return False
    with _embed_lock:
        if _resnet is not None:
            return True
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            _resnet = model
            _resnet_tf = weights.transforms()
            return True
        except Exception as exc:
            _resnet_err = f"{type(exc).__name__}: {exc}"
            logger.warning("Object crop ResNet18 unavailable: %s", _resnet_err)
            return False


def encode_crop(crop: Image.Image) -> tuple[np.ndarray, str]:
    """Dedicated crop encoder. Not CLIP. Not used as an object detector."""
    if _load_resnet():
        try:
            import torch

            tensor = _resnet_tf(crop.convert("RGB"))
            with torch.inference_mode():
                vec = _resnet(tensor.unsqueeze(0))[0].detach().cpu().numpy()
            return _l2(vec), EMBED_RESNET
        except Exception as exc:
            logger.warning("ResNet crop encode failed, using colorgrid: %s", exc)
    return _colorgrid(crop), EMBED_COLORGRID


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size == 0:
        return 0.0
    return float(np.dot(_l2(a), _l2(b)))


@dataclass(frozen=True)
class ObjectRecord:
    instance_id: str
    label: str
    label_tr: str
    confidence: float
    bbox: tuple[int, int, int, int]
    area_ratio: float
    embedding_dim: int
    embedding_backend: str
    evidence: str = "detector"


class ObjectIntelligence:
    """Detect (RT-DETR default, Faster R-CNN fallback) → crop-embed → persist/search."""

    def __init__(
        self,
        object_db_path: str | Path,
        *,
        min_confidence: float = 0.45,
        max_detections: int = 50,
        detector: Callable[[str], list[DetectedObject]] | None = None,
        detector_backend: str | None = None,
    ):
        self.object_db_path = str(object_db_path)
        self.store = ObjectIndexStore(self.object_db_path)
        self.min_confidence = float(min_confidence)
        self.max_detections = max(1, min(200, int(max_detections)))
        self._detector_fn = detector
        self._requested_backend = detector_backend
        self._last_error = ""
        self._coco = GlobalObjectIntelligence(
            enabled=True,
            min_confidence=min_confidence,
            max_detections=max_detections,
        )
        self.search = ObjectSearch(self.object_db_path, readonly=True)

    def _finalize(self, dets: list[DetectedObject]) -> list[DetectedObject]:
        out = [
            d
            for d in dets
            if float(d.confidence) >= self.min_confidence and not _is_fullframe_disk_fp(d)
        ]
        return _nms_mixed(out)[: self.max_detections]

    def detector_available(self) -> bool:
        if self._detector_fn is not None:
            return True
        if self._requested_backend == "unavailable":
            return False
        if self._requested_backend == FALLBACK_DETECTOR_BACKEND:
            return bool(_TORCHVISION_OK)
        if self._requested_backend == DETECTOR_BACKEND:
            return bool(_load_rtdetr())
        return bool(_load_rtdetr())

    def detector_backend(self) -> str:
        if self._detector_fn is not None:
            return "injected_detector"
        if self._requested_backend == "unavailable":
            return "unavailable"
        if self._requested_backend == FALLBACK_DETECTOR_BACKEND:
            chosen = FALLBACK_DETECTOR_BACKEND if _TORCHVISION_OK else "unavailable"
            logger.info("Object detector backend selected: %s (explicit fallback)", chosen)
            return chosen
        if _load_rtdetr():
            return DETECTOR_BACKEND
        if self._requested_backend == DETECTOR_BACKEND:
            logger.error("RT-DETR-L required but unavailable: %s", _ultra_err)
            return "unavailable"
        logger.error(
            "RT-DETR-L unavailable (%s); not silently switching to Faster R-CNN",
            _ultra_err or DETECTOR_UNAVAILABLE,
        )
        return "unavailable"

    def detect(self, image_path: str) -> list[DetectedObject]:
        self._last_error = ""
        if self._detector_fn is not None:
            return self._finalize(list(self._detector_fn(image_path) or []))
        backend = self.detector_backend()
        if backend == "unavailable":
            self._last_error = DETECTOR_UNAVAILABLE
            return []
        if backend == DETECTOR_BACKEND:
            try:
                return self._finalize(
                    _detect_rtdetr(
                        image_path,
                        min_confidence=self.min_confidence,
                        max_detections=self.max_detections,
                    )
                )
            except Exception as exc:
                self._last_error = DETECTOR_FAILED
                logger.error(
                    "RT-DETR-L inference failed (%s); Faster R-CNN fallback is NOT used",
                    exc,
                )
                return []
        if not _TORCHVISION_OK:
            self._last_error = DETECTOR_UNAVAILABLE
            return []
        return self._finalize(self._coco.detect(image_path))

    def detect_embed(self, image_path: str) -> dict[str, Any]:
        backend = self.detector_backend()
        if backend == "unavailable" or not self.detector_available():
            self._last_error = DETECTOR_UNAVAILABLE
            return {
                "ok": False,
                "backend": "unavailable",
                "error": DETECTOR_UNAVAILABLE,
                "embedding_kind": "none",
                "evidence": DETECTOR_UNAVAILABLE,
                "clip_used_as_detector": False,
                "objects": [],
                "object_count": 0,
            }
        dets = self.detect(image_path)
        if self._last_error in {DETECTOR_UNAVAILABLE, DETECTOR_FAILED}:
            return {
                "ok": False,
                "backend": backend,
                "error": self._last_error,
                "embedding_kind": "none",
                "evidence": self._last_error,
                "clip_used_as_detector": False,
                "objects": [],
                "object_count": 0,
            }
        records: list[dict[str, Any]] = []
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return {
                "ok": False,
                "backend": backend,
                "error": f"{type(exc).__name__}: {exc}",
                "clip_used_as_detector": False,
                "objects": [],
                "object_count": 0,
            }
        w, h = image.size
        embed_backend = ""
        for obj in dets:
            x1, y1, x2, y2 = obj.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            vec, embed_backend = encode_crop(crop)
            rec = ObjectRecord(
                instance_id=obj.instance_id,
                label=obj.label,
                label_tr=obj.label_tr,
                confidence=float(obj.confidence),
                bbox=obj.bbox,
                area_ratio=float(obj.area_ratio),
                embedding_dim=int(vec.size),
                embedding_backend=embed_backend,
                evidence="detector",
            )
            payload = asdict(rec)
            payload["embedding"] = vec
            records.append(payload)
        return {
            "ok": True,
            "backend": backend,
            "embedding_kind": embed_backend or EMBED_COLORGRID,
            "clip_used_as_detector": False,
            "evidence": "detector",
            "objects": records,
            "object_count": len(records),
        }

    def index_image(self, file_id: int, image_path: str) -> dict[str, Any]:
        det = self.detect_embed(image_path)
        if not det.get("ok"):
            return det
        st = os.stat(image_path) if os.path.isfile(image_path) else None
        n = self.store.replace_file_objects(
            int(file_id),
            str(image_path),
            det["objects"],
            mtime=float(st.st_mtime if st else 0),
            file_size=int(st.st_size if st else 0),
            detector=str(det.get("backend") or ""),
        )
        det["indexed"] = n
        det["file_id"] = int(file_id)
        return det

    def index_paths(self, items: list[tuple[int, str]]) -> dict[str, Any]:
        """Index an explicit list only. Does not walk an archive root."""
        ok = 0
        fail = 0
        for file_id, path in items:
            r = self.index_image(int(file_id), str(path))
            if r.get("ok"):
                ok += 1
            else:
                fail += 1
        return {"ok": fail == 0, "indexed_files": ok, "failed": fail, "auto_scan": False}

    def query_images(self, text: str, limit: int = 400) -> list[int]:
        return self.search.file_ids(text, limit=limit)

    def object_count(self, file_id: int, label: str | None = None) -> int:
        counts = self.store.counts_for_file(int(file_id))
        if label:
            return int(counts.get(label, 0))
        return int(sum(counts.values()))

    def search_similar_image(self, image_path: str, *, threshold: float = 0.55, limit: int = 50) -> list[dict[str, Any]]:
        det = self.detect_embed(image_path)
        if not det.get("ok") or not det["objects"]:
            return []
        gallery = self.store.all_embeddings()
        hits: dict[int, dict[str, Any]] = {}
        for q in det["objects"]:
            qv = q.get("embedding")
            if qv is None:
                continue
            for file_id, inst_i, label, gv in gallery:
                score = _cosine(qv, gv)
                if score < threshold:
                    continue
                prev = hits.get(file_id)
                if prev is None or score > float(prev["score"]):
                    hits[file_id] = {
                        "file_id": file_id,
                        "score": round(float(score), 4),
                        "query_label": q["label"],
                        "match_label": label,
                        "instance_index": inst_i,
                        "evidence": "crop_embedding",
                        "embedding_backend": q.get("embedding_backend"),
                    }
        ranked = sorted(hits.values(), key=lambda x: x["score"], reverse=True)
        return ranked[: int(limit)]
