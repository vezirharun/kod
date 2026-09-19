"""Face Intelligence façade: InsightFace/ArcFace only, isolated face DB.

Never writes patterns.db, FAISS, preview/thumbnail cache, or Pattern DNA.
Do not point this at the 116K production archive until a reviewed batch scan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.face_identity import (
    FaceIdentityEngine,
    FaceObservation,
    _INSIGHTFACE_OK,
    normalize_gender,
)
from core.face_index import FaceIndexStore
from core.face_search import FaceSearch, parse_face_query

# ArcFace cosine thresholds (not DINO/CLIP). Stricter than settings 0.56 to
# cut false person merges; margin rejects two close gallery candidates.
MATCH_THRESHOLD = 0.62
MIN_MARGIN = 0.05
MIN_DET_SCORE = 0.50
GENDER_CONF_MIN = 0.60
BACKEND = "insightface_arcface_buffalo_l"

_GENDER_TR = {"FEMALE": "kadın", "MALE": "erkek"}


def kisi_label(person_id: str) -> str:
    """person_0017 -> 'Kişi 017'."""
    digits = "".join(ch for ch in str(person_id) if ch.isdigit())
    n = int(digits) if digits else 0
    return f"Kişi {n:03d}"


def gender_tr(value: Any, confidence: float = 1.0) -> str:
    g = normalize_gender(value)
    if g in {"FEMALE", "MALE"} and float(confidence) >= GENDER_CONF_MIN:
        return _GENDER_TR[g]
    return "bilinmiyor"


def insightface_import_ok() -> bool:
    return bool(_INSIGHTFACE_OK)


@dataclass(frozen=True)
class FaceRecord:
    face_id: str
    bbox: tuple[int, int, int, int]
    embedding_dim: int
    confidence: float
    gender: str
    gender_tr: str
    person_id: str
    kisi_label: str


class PersonEmbeddingCache:
    """In-memory view of persisted person centroids (face DB only)."""

    def __init__(self) -> None:
        self._by_person: dict[str, np.ndarray] = {}

    def reload(self, store: FaceIndexStore) -> None:
        with store._connect() as con:
            rows = con.execute(
                "SELECT person_id,centroid FROM persons WHERE centroid IS NOT NULL AND sample_count>0"
            ).fetchall()
        self._by_person = {str(r["person_id"]): np.frombuffer(r["centroid"], dtype=np.float32).copy() for r in rows}

    def get(self, person_id: str) -> np.ndarray | None:
        return self._by_person.get(str(person_id))

    def persons(self) -> dict[str, np.ndarray]:
        return dict(self._by_person)

    def __len__(self) -> int:
        return len(self._by_person)


class FaceIntelligence:
    """Detect → embed (InsightFace) → person match → query. Face DB only."""

    def __init__(
        self,
        face_db_path: str | Path,
        *,
        threshold: float = MATCH_THRESHOLD,
        min_margin: float = MIN_MARGIN,
        min_det_score: float = MIN_DET_SCORE,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: int = 640,
    ):
        self.face_db_path = str(face_db_path)
        self.min_det_score = float(min_det_score)
        self.engine = FaceIdentityEngine(
            threshold=threshold,
            min_margin=min_margin,
            model_name=model_name,
            providers=providers or ["CPUExecutionProvider"],
            det_size=det_size,
            allow_vision_fallback=False,
        )
        self.store = FaceIndexStore(
            self.face_db_path,
            threshold=self.engine.effective_threshold,
            min_margin=min_margin,
            identity_engine=BACKEND,
        )
        self.search = FaceSearch.__new__(FaceSearch)
        self.search.engine = self.engine
        self.search.store = self.store
        self.cache = PersonEmbeddingCache()
        self._unavailable_reason = ""

    def ready(self) -> bool:
        """True only if InsightFace buffalo actually prepared. Never a stub."""
        if not insightface_import_ok():
            self._unavailable_reason = "insightface_not_installed"
            return False
        ok = bool(self.engine.available() and self.engine._app is not None)
        if not ok:
            self._unavailable_reason = "buffalo_weights_or_onnxruntime_failed"
            return False
        if self.engine.backend != "insightface_arcface_v11":
            self._unavailable_reason = f"backend_not_insightface:{self.engine.backend}"
            return False
        self._unavailable_reason = ""
        return True

    def unavailable_reason(self) -> str:
        if self.ready():
            return ""
        return self._unavailable_reason or "insightface_unavailable"

    def _read_bgr(self, image_path: str) -> np.ndarray | None:
        try:
            import cv2
        except Exception:
            return None
        if not image_path or not Path(image_path).is_file():
            return None
        image = cv2.imread(image_path)
        return image if image is not None else None

    def _filter_obs(self, observations: list[FaceObservation]) -> list[FaceObservation]:
        out: list[FaceObservation] = []
        for obs in observations:
            if obs.embedding is None:
                continue
            if float(obs.det_score) < self.min_det_score:
                continue
            g = obs.gender if float(obs.gender_confidence) >= GENDER_CONF_MIN else "UNKNOWN"
            out.append(
                FaceObservation(
                    face_id=obs.face_id,
                    bbox=obs.bbox,
                    embedding=obs.embedding,
                    gender=g,
                    gender_confidence=obs.gender_confidence if g != "UNKNOWN" else 0.0,
                    gender_confidence_band=obs.gender_confidence_band if g != "UNKNOWN" else "UNKNOWN",
                    det_score=obs.det_score,
                )
            )
        return out

    def detect_embed(self, image_path: str) -> dict[str, Any]:
        """Per-image faces: embedding, confidence, bbox, count, gender (TR)."""
        if not self.ready():
            return {
                "ok": False,
                "reason": self.unavailable_reason(),
                "backend": self.engine.backend,
                "face_count": 0,
                "faces": [],
            }
        image = self._read_bgr(image_path)
        if image is None:
            return {"ok": False, "reason": "image_unreadable", "face_count": 0, "faces": []}
        raw = self.engine.analyze_image(image)
        faces = self._filter_obs(raw)
        payload = []
        for obs in faces:
            payload.append({
                "face_id": obs.face_id,
                "bbox": list(obs.bbox),
                "confidence": float(obs.det_score),
                "embedding_dim": int(np.asarray(obs.embedding).size),
                "gender": obs.gender,
                "gender_tr": gender_tr(obs.gender, obs.gender_confidence),
                "has_embedding": True,
            })
        return {
            "ok": True,
            "backend": BACKEND,
            "face_count": len(payload),
            "faces": payload,
        }

    def index_image(self, file_id: int, image_path: str, *, source_path: str = "") -> dict[str, Any]:
        """Index one file into the face DB. Caller supplies the path; no archive walk."""
        if not self.ready():
            return {"ok": False, "reason": self.unavailable_reason(), "person_ids": []}
        image = self._read_bgr(image_path)
        if image is None:
            return {"ok": False, "reason": "image_unreadable", "person_ids": []}
        path = source_path or image_path
        p = Path(image_path)
        mtime = p.stat().st_mtime if p.is_file() else 0.0
        size = p.stat().st_size if p.is_file() else 0
        faces = self._filter_obs(self.engine.analyze_image(image))
        assigned = self.store.replace_file_faces(int(file_id), path, mtime, size, faces)
        self.cache.reload(self.store)
        records = []
        for obs, pid in zip(faces, assigned):
            records.append(asdict(FaceRecord(
                face_id=obs.face_id,
                bbox=obs.bbox,
                embedding_dim=int(np.asarray(obs.embedding).size),
                confidence=float(obs.det_score),
                gender=obs.gender,
                gender_tr=gender_tr(obs.gender, obs.gender_confidence),
                person_id=pid,
                kisi_label=kisi_label(pid),
            )))
        return {
            "ok": True,
            "face_count": len(assigned),
            "person_ids": assigned,
            "kisi_labels": [kisi_label(p) for p in assigned],
            "faces": records,
        }

    def name_person(self, person_id: str, display_name: str) -> None:
        """Kişi 017 = Ahmet — stored in face DB only."""
        self.store.label_person(person_id, display_name)

    def match_new_image(self, image_path: str) -> dict[str, Any]:
        """Compare a new photo to existing persons (cache + exemplars)."""
        if not self.ready():
            return {"ok": False, "reason": self.unavailable_reason(), "matches": []}
        det = self.detect_embed(image_path)
        if not det.get("ok"):
            return {"ok": False, "reason": det.get("reason"), "matches": []}
        self.cache.reload(self.store)
        image = self._read_bgr(image_path)
        faces = self._filter_obs(self.engine.analyze_image(image))
        matches = []
        gallery = self.cache.persons()
        th = self.engine.effective_threshold
        for obs in faces:
            hit = FaceIdentityEngine.best_match(
                obs.embedding, gallery, threshold=th, min_margin=self.engine.min_margin,
            )
            if hit:
                matches.append({
                    "face_id": obs.face_id,
                    "person_id": hit.person_id,
                    "kisi_label": kisi_label(hit.person_id),
                    "similarity": hit.similarity,
                    "confidence": hit.confidence,
                    "gender_tr": gender_tr(obs.gender, obs.gender_confidence),
                })
            else:
                matches.append({
                    "face_id": obs.face_id,
                    "person_id": None,
                    "kisi_label": None,
                    "similarity": 0.0,
                    "confidence": "UNKNOWN",
                    "gender_tr": gender_tr(obs.gender, obs.gender_confidence),
                })
        file_hits = self.search.image_matches(image_path, threshold=th)
        return {"ok": True, "matches": matches, "file_ids": file_hits}

    def query_images(self, text: str, *, limit: int = 400) -> list[int]:
        """'Kişi 017' / kadın / erkek / bilinmiyor / display name → file_ids (face DB)."""
        return self.search.file_ids(text, limit=limit)

    def images_for_kisi(self, n: int, *, limit: int = 400) -> list[int]:
        return self.store.search(person_id=f"person_{int(n):04d}", limit=limit)

    def filter_gender(self, gender_tr_value: str, *, limit: int = 400) -> list[int]:
        spec = parse_face_query(gender_tr_value)
        if not spec or spec.get("kind") != "gender":
            return []
        return self.store.search(gender=spec["value"], limit=limit)

    def list_persons(self, limit: int = 10000) -> list[dict[str, Any]]:
        rows = self.store.list_persons(limit=limit)
        out = []
        for row in rows:
            pid = str(row["person_id"])
            name = str(row.get("display_name") or "").strip()
            out.append({
                **row,
                "kisi_label": kisi_label(pid),
                "display_or_kisi": name or kisi_label(pid),
                "gender_tr": gender_tr(row.get("gender"), float(row.get("gender_confidence") or 0)),
            })
        return out


def later_archive_scan_howto() -> str:
    """High-level only. Do not run on production until review."""
    return (
        "After review: enable FaceBackgroundIndexer / FaceIndexScanner.scan_incremental "
        "against preview/thumbnail paths already in the pattern index (read-only files table). "
        "Writes go only to face_index.db. Batch small (e.g. 25), never reindex FAISS/patterns.db, "
        "never open the 116K source archive if previews exist."
    )
