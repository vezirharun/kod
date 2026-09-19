"""Index-time object evidence from the existing Feature Preview only.

RT-DETR (or explicit Faster R-CNN fallback) writes boxes into object_index.db.
Does not create, rewrite, or re-encode previews. Does not enqueue Hızlı Index jobs.
Does not read NAS/TIFF originals. Missing preview: skip (preview bekleniyor).
CLIP is never used as a detector.
Does not rebuild FAISS or wipe patterns.db.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from core.visual_concept import english_lemma

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_engines: dict[str, Any] = {}


def _engine(object_db_path: str, min_confidence: float):
    key = f"{object_db_path}|{min_confidence:.3f}"
    with _lock:
        eng = _engines.get(key)
        if eng is not None:
            return eng
        from core.object_intelligence import ObjectIntelligence

        eng = ObjectIntelligence(object_db_path, min_confidence=min_confidence)
        _engines[key] = eng
        return eng


def attach_preview_object_evidence(
    *,
    file_id: int,
    preview_path: str,
    object_db_path: str,
    min_confidence: float = 0.45,
    detector=None,
    concepts: list[dict[str, Any]] | None = None,
    force: bool = False,
    require_rtdetr: bool = False,
) -> dict[str, Any] | None:
    """Detect on a local preview; persist object_index + GOI + Kavram DNA. None = skipped."""
    from core.index_freeze import in_search_session, process_search_active
    from core.object_index import ObjectIndexStore

    if process_search_active() or in_search_session():
        return None
    if int(file_id) <= 0 or not preview_path or not os.path.isfile(preview_path):
        return None
    if not object_db_path:
        return None
    if not force:
        probe = ObjectIndexStore(object_db_path, readonly=True)
        if probe.has_scan(int(file_id)):
            return None
    try:
        if detector is not None:
            from core.object_intelligence import ObjectIntelligence

            oi = ObjectIntelligence(object_db_path, min_confidence=min_confidence, detector=detector)
        else:
            oi = _engine(object_db_path, min_confidence)
            if require_rtdetr:
                from core.object_intelligence import DETECTOR_BACKEND, _load_rtdetr, _ultra_err

                if not _load_rtdetr():
                    raise RuntimeError(
                        f"RT-DETR-L required: {_ultra_err or 'ultralytics/rtdetr-l.pt missing'}"
                    )
                if oi.detector_backend() != DETECTOR_BACKEND:
                    raise RuntimeError(
                        f"RT-DETR-L required, got backend={oi.detector_backend()}"
                    )
            if not oi.detector_available() and not concepts:
                from core.visual_concept_dna import build_visual_concept_dna

                dna = build_visual_concept_dna(objects=[], concepts=[], backend="unavailable")
                try:
                    st = os.stat(preview_path)
                    oi.store.replace_file_objects(
                        int(file_id),
                        preview_path,
                        [],
                        mtime=float(st.st_mtime),
                        file_size=int(st.st_size),
                        detector="unavailable",
                    )
                    oi.store.replace_file_concepts(int(file_id), [])
                except Exception:
                    logger.warning("object_index empty scan skipped", exc_info=True)
                return {
                    "enabled": False,
                    "object_detection_available": False,
                    "clip_used_as_detector": False,
                    "clip_as_detector": False,
                    "objects": [],
                    "concepts": [],
                    "object_count": 0,
                    "backend": "unavailable",
                    "visual_concept_dna": dna,
                }
        dets = []
        if detector is not None or (oi.detector_available() if detector is None else False):
            try:
                dets = oi.detect(preview_path)
            except Exception:
                dets = []
        objects: list[dict[str, Any]] = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            objects.append(
                {
                    "instance_id": d.instance_id,
                    "label": d.label,
                    "label_tr": d.label_tr,
                    "canonical_name": english_lemma(d.label),
                    "confidence": float(d.confidence),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "area_ratio": float(d.area_ratio),
                    "evidence": "object_detector",
                    "evidence_source": "object_detector",
                }
            )
        backend = str(oi.detector_backend() if detector is None else "injected_detector")
        from core.visual_concept_dna import build_visual_concept_dna

        dna = build_visual_concept_dna(
            objects=objects,
            concepts=list(concepts or []),
            backend=backend,
        )
        try:
            st = os.stat(preview_path)
            oi.store.replace_file_objects(
                int(file_id),
                preview_path,
                objects,
                mtime=float(st.st_mtime),
                file_size=int(st.st_size),
                detector=backend,
            )
            oi.store.replace_file_concepts(
                int(file_id),
                list(dna.get("objects") or []) + list(dna.get("concepts") or []),
            )
        except Exception:
            logger.warning("object_index write skipped", exc_info=True)
        return {
            "enabled": True,
            "object_detection_available": bool(objects),
            "clip_used_as_detector": False,
            "clip_as_detector": False,
            "backend": backend,
            "objects": objects,
            "concepts": list(dna.get("concepts") or []),
            "object_count": len(objects),
            "visual_concept_dna": dna,
        }
    except Exception:
        logger.warning("preview object evidence skipped; CLIP job continues", exc_info=True)
        return None
