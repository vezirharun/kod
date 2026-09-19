"""Production face identity backend with ArcFace and local vision fallback.

ArcFace/InsightFace is preferred. When it is not installed, the engine falls
back to the same local DINO/CLIP vision stack already used by Pattern Search,
using a detected face crop as the identity embedding. This is an appearance
identity fallback, not a biometric claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_OK = True
except Exception:
    FaceAnalysis = None
    _INSIGHTFACE_OK = False

try:
    import cv2
    _CV2_OK = True
except Exception:
    cv2 = None
    _CV2_OK = False


@dataclass(frozen=True)
class FaceObservation:
    face_id: str
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray | None
    gender: str = "UNKNOWN"
    gender_confidence: float = 0.0
    gender_confidence_band: str = "UNKNOWN"
    det_score: float = 0.0


@dataclass(frozen=True)
class FaceMatch:
    person_id: str
    similarity: float
    confidence: str
    margin: float = 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8 or a.size != b.size:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def confidence_band(similarity: float) -> str:
    if similarity >= 0.72:
        return "VERY_HIGH"
    if similarity >= 0.62:
        return "HIGH"
    if similarity >= 0.52:
        return "MEDIUM"
    if similarity >= 0.42:
        return "LOW"
    return "UNKNOWN"


def gender_band(confidence: float) -> str:
    if confidence >= 0.90:
        return "VERY_HIGH"
    if confidence >= 0.75:
        return "HIGH"
    if confidence >= 0.60:
        return "MEDIUM"
    if confidence >= 0.50:
        return "LOW"
    return "UNKNOWN"


def normalize_gender(value: Any, confidence: float = 0.0) -> str:
    if isinstance(value, str):
        v = value.strip().upper()
        if v in {"FEMALE", "WOMAN", "KADIN"}: return "FEMALE"
        if v in {"MALE", "MAN", "ERKEK"}: return "MALE"
        return "UNKNOWN"
    try:
        v = int(value)
    except Exception:
        return "UNKNOWN"
    return "FEMALE" if v == 0 else "MALE" if v == 1 else "UNKNOWN"


class FaceIdentityEngine:
    """ArcFace first; DINO/CLIP face-crop fallback when ArcFace is unavailable."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        threshold: float = 0.62,
        min_margin: float = 0.05,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: int = 640,
        allow_vision_fallback: bool = True,
    ):
        self.enabled = bool(enabled)
        self.requested_threshold = float(threshold)
        self.min_margin = float(min_margin)
        self.model_name = str(model_name or "buffalo_l")
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.det_size = max(160, int(det_size or 640))
        self.allow_vision_fallback = bool(allow_vision_fallback)
        self._app: Any = None
        self._fallback_extractor: Any = None
        self._cascade: Any = None
        if _INSIGHTFACE_OK:
            self.backend = "insightface_arcface_v11"
        elif self.allow_vision_fallback:
            self.backend = "vision_face_crop_v11"
        else:
            self.backend = "unavailable"

    @classmethod
    def capability(cls) -> dict[str, str]:
        if _INSIGHTFACE_OK:
            return {
                "face_recognition": "supported",
                "face_embedding": "supported",
                "person_identity_matching": "supported",
                "gender_classification_from_face": "supported",
                "persistent_person_gallery": "supported",
            }
        # The fallback is available if cv2 + the project's DINO/CLIP stack can be loaded.
        return {
            "face_recognition": "optional_dependency",
            "face_embedding": "optional_dependency",
            "person_identity_matching": "optional_dependency",
            "gender_classification_from_face": "optional_dependency",
            "persistent_person_gallery": "supported",
        }

    def available(self) -> bool:
        """Return whether a real face embedding backend is currently usable."""
        return bool(self._load())

    @property
    def effective_threshold(self) -> float:
        if _INSIGHTFACE_OK:
            return self.requested_threshold
        # DINO/CLIP crop similarities are not on ArcFace's scale.
        return max(0.66, self.requested_threshold)

    def _load(self) -> bool:
        if not self.enabled:
            return False
        if _INSIGHTFACE_OK:
            if self._app is not None: return True
            try:
                self._app = FaceAnalysis(name=self.model_name, providers=self.providers)
                self._app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
                self.backend = "insightface_arcface_v11"
                return True
            except Exception:
                self._app = None
        if self.allow_vision_fallback:
            return self._load_fallback()
        self.backend = "unavailable"
        return False

    def _load_fallback(self) -> bool:
        if not _CV2_OK:
            return False
        if self._fallback_extractor is not None and self._cascade is not None:
            return True
        try:
            from core.feature_extractor import FeatureExtractor
            self._fallback_extractor = FeatureExtractor(use_ai=True, use_gpu=False, fast_hash_only=False)
            if not self._fallback_extractor.ai_available:
                self._fallback_extractor = None
                return False
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                self._cascade = None
                return False
            self.backend = "vision_face_crop_v11"
            return True
        except Exception:
            self._fallback_extractor = None
            self._cascade = None
            return False

    def _fallback_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """Build a stable appearance embedding from several face views.

        OpenCV images are BGR, while FeatureExtractor expects RGB.  V10 fed
        the BGR crop directly into DINO/CLIP, which silently degraded the
        fallback similarity.  V11 fixes the color order and uses tight,
        contextual and mirrored views so hairstyle/lighting/background changes
        have less influence on the stored appearance signature.
        """
        try:
            if crop is None or crop.size == 0:
                return None
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if h < 32 or w < 32:
                return None

            views: list[np.ndarray] = []
            # Tight face view.
            views.append(rgb)
            # Slightly expanded context keeps hair/jaw/neck cues without
            # allowing the original background to dominate.
            pad_x = max(1, int(w * 0.10))
            pad_y = max(1, int(h * 0.10))
            x0, y0 = pad_x, pad_y
            x1, y1 = max(x0 + 1, w - pad_x), max(y0 + 1, h - pad_y)
            views.append(rgb[y0:y1, x0:x1])
            # Horizontal flip makes the appearance representation less
            # sensitive to small pose asymmetries.
            views.append(cv2.flip(rgb, 1))

            parts: list[np.ndarray] = []
            for view in views:
                feats = self._fallback_extractor.extract_from_array(
                    view, include_patches=False, deep_analysis=False,
                    compute={"dino", "clip"},
                )
                subparts: list[np.ndarray] = []
                if feats.dino_embedding:
                    subparts.append(_norm(np.frombuffer(feats.dino_embedding, dtype=np.float32)))
                if feats.clip_embedding:
                    subparts.append(_norm(np.frombuffer(feats.clip_embedding, dtype=np.float32)))
                if subparts:
                    parts.append(_norm(np.concatenate(subparts).astype(np.float32)))

            if not parts:
                return None

            # Keep all views as evidence instead of averaging them away.
            return _norm(np.concatenate(parts).astype(np.float32))
        except Exception:
            return None

    def _detect_fallback_boxes(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Multi-cascade face detection for the local fallback.

        Haar frontal-only detection misses side faces, older photos and
        low-contrast studio portraits. We combine frontal/profile cascades,
        CLAHE and a modest upscale, then deduplicate overlapping boxes.
        """
        if self._cascade is None or image_bgr is None or image_bgr.size == 0:
            return []
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        cascades = [self._cascade]
        try:
            for name in (
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_frontalface_alt.xml",
                "haarcascade_profileface.xml",
            ):
                c = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
                if not c.empty():
                    cascades.append(c)
        except Exception:
            pass

        boxes: list[tuple[int, int, int, int]] = []
        for source in (gray, clahe):
            for cascade in cascades:
                try:
                    found = cascade.detectMultiScale(
                        source, scaleFactor=1.06, minNeighbors=4,
                        minSize=(36, 36), maxSize=(max(40, source.shape[1] // 2), max(40, source.shape[0] // 2)),
                    )
                    boxes.extend((int(x), int(y), int(x+w), int(y+h)) for x, y, w, h in found)
                except Exception:
                    continue

        # Upscale small/medium faces once; map boxes back.
        if min(gray.shape[:2]) < 1200:
            scale = 1.5
            up = cv2.resize(clahe, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for cascade in cascades[:3]:
                try:
                    found = cascade.detectMultiScale(up, scaleFactor=1.06, minNeighbors=4, minSize=(45,45))
                    boxes.extend(
                        (int(x/scale), int(y/scale), int((x+w)/scale), int((y+h)/scale))
                        for x, y, w, h in found
                    )
                except Exception:
                    continue

        if not boxes:
            return []

        def iou(a, b):
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ix0, iy0 = max(ax0,bx0), max(ay0,by0)
            ix1, iy1 = min(ax1,bx1), min(ay1,by1)
            inter = max(0,ix1-ix0)*max(0,iy1-iy0)
            aa = max(1,ax1-ax0)*max(1,ay1-ay0)
            ab = max(1,bx1-bx0)*max(1,by1-by0)
            return inter / max(1, aa+ab-inter)

        # Larger boxes first; suppress near-duplicates.
        kept: list[tuple[int,int,int,int]] = []
        for box in sorted(boxes, key=lambda b: -((b[2]-b[0])*(b[3]-b[1]))):
            if any(iou(box, k) >= 0.45 for k in kept):
                continue
            kept.append(box)
            if len(kept) >= 50:
                break
        return kept

    def _analyze_fallback(self, image_bgr: np.ndarray) -> list[FaceObservation]:
        if not self._load_fallback():
            return []
        boxes = self._detect_fallback_boxes(image_bgr)
        out: list[FaceObservation] = []
        h, w = image_bgr.shape[:2]
        for i, (x, y, x1, y1) in enumerate(boxes[:50], start=1):
            bw, bh = max(1, x1-x), max(1, y1-y)
            pad_x = int(bw * 0.45)
            pad_y = int(bh * 0.65)
            x0 = max(0, int(x-pad_x)); y0 = max(0, int(y-pad_y))
            xa = min(w, int(x1+pad_x)); ya = min(h, int(y1+pad_y))
            crop = image_bgr[y0:ya, x0:xa]
            emb = self._fallback_embedding(crop)
            if emb is None:
                continue
            out.append(FaceObservation(
                face_id=f"face_{i:03d}",
                bbox=(int(x), int(y), int(x1), int(y1)),
                embedding=emb,
                gender="UNKNOWN",
                gender_confidence=0.0,
                gender_confidence_band="UNKNOWN",
            ))
        return out

    def analyze_image(self, image_bgr: np.ndarray) -> list[FaceObservation]:
        if image_bgr is None or not self.enabled: return []
        if _INSIGHTFACE_OK and self._load() and self._app is not None:
            try:
                faces=self._app.get(image_bgr)
                out=[]
                for i,face in enumerate(faces,start=1):
                    bbox_raw=getattr(face,"bbox",None)
                    if bbox_raw is None or len(bbox_raw)<4: continue
                    bbox=tuple(int(round(float(x))) for x in bbox_raw[:4])
                    emb=getattr(face,"embedding",None)
                    emb_arr=np.asarray(emb,dtype=np.float32) if emb is not None else None
                    raw_gender=getattr(face,"gender",None)
                    g=normalize_gender(raw_gender)
                    gconf=float(getattr(face,"gender_score",0.0) or 0.0)
                    if g!="UNKNOWN" and gconf<=0.0: gconf=0.50
                    det=float(getattr(face,"det_score",0.0) or 0.0)
                    out.append(FaceObservation(
                        f"face_{i:03d}", bbox, emb_arr, g,
                        round(min(1.0, max(0.0, gconf)), 4), gender_band(gconf),
                        det_score=round(min(1.0, max(0.0, det)), 4),
                    ))
                return out
            except Exception:
                pass
        if self.allow_vision_fallback:
            return self._analyze_fallback(image_bgr)
        return []

    def embeddings(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        return [o.embedding for o in self.analyze_image(image_bgr) if o.embedding is not None]

    @staticmethod
    def best_match(query: np.ndarray, gallery: dict[str,np.ndarray], threshold: float=0.62, min_margin: float=0.05) -> FaceMatch | None:
        scores=sorted(((pid,_cosine(query,vec)) for pid,vec in gallery.items()), key=lambda x:x[1], reverse=True)
        if not scores or scores[0][1]<threshold: return None
        margin=scores[0][1]-(scores[1][1] if len(scores)>1 else 0.0)
        if len(scores)>1 and margin<min_margin: return None
        pid,score=scores[0]
        return FaceMatch(pid,round(score,5),confidence_band(score),round(margin,5))

    @staticmethod
    def filter_observations(observations:list[FaceObservation], gender:str)->list[FaceObservation]:
        g=str(gender or "ALL").strip().upper()
        if g in {"ALL",""}: return list(observations)
        return [o for o in observations if o.gender==g]

    @staticmethod
    def cluster(embeddings:dict[str,np.ndarray], threshold:float=0.62)->dict[str,str]:
        labels={}; centroids=[]; next_id=1
        for item_id,vector in embeddings.items():
            best=None; best_score=-1.0
            for idx,(cid,centroid,count) in enumerate(centroids):
                score=_cosine(vector,centroid)
                if score>best_score: best_score,best=score,idx
            if best is not None and best_score>=threshold:
                cid,centroid,count=centroids[best]
                new=(centroid*count+np.asarray(vector,dtype=np.float32))/(count+1)
                centroids[best]=(cid,new,count+1); labels[item_id]=cid
            else:
                cid=f"person_{next_id:04d}"; next_id+=1
                centroids.append((cid,np.asarray(vector,dtype=np.float32).copy(),1)); labels[item_id]=cid
        return labels
