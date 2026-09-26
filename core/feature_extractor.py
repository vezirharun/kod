"""Görsel özellik çıkarma — hash, renk, doku, AI embedding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from core.logger import setup_logger
from core.thumbnailer import Thumbnailer

logger = setup_logger(__name__)

try:
    import imagehash

    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

# Intentionally NO cv2 on this path — avoids 0xC0000374 with torch/DINO/CLIP.
# Color/texture use core.safe_image_ops (numpy/sklearn).
HAS_CV2 = False

# AI modelleri — opsiyonel
HAS_TORCH = False
HAS_DINO = False
HAS_CLIP = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore

try:
    import open_clip

    HAS_CLIP = True
except ImportError:
    open_clip = None  # type: ignore


@dataclass
class ExtractedFeatures:
    phash: str = ""
    dhash: str = ""
    whash: str = ""
    color_hist: bytes = b""
    dominant_colors: list[list[int]] = field(default_factory=list)
    texture_features: list[float] = field(default_factory=list)
    dino_embedding: bytes = b""
    clip_embedding: bytes = b""
    patch_embeddings: list[bytes] = field(default_factory=list)
    patch_embeddings_meta: list[dict[str, Any]] = field(default_factory=list)
    texture_map: dict[str, Any] = field(default_factory=dict)


class FeatureExtractor:
    DINO_DIM = 384
    CLIP_DIM = 512

    def __init__(
        self,
        use_ai: bool = True,
        use_gpu: bool = False,
        fast_hash_only: bool = False,
    ):
        self.use_ai = use_ai
        self.use_gpu = use_gpu and HAS_TORCH
        self.fast_hash_only = fast_hash_only
        self.device = "cuda" if self.use_gpu else "cpu"

        self._dino_model = None
        self._dino_transform = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

        if use_ai and not fast_hash_only:
            self._init_ai_models()

    def _init_ai_models(self) -> None:
        if HAS_TORCH:
            try:
                self._dino_model = torch.hub.load(
                    "facebookresearch/dinov2",
                    "dinov2_vits14",
                    trust_repo=True,
                )
                self._dino_model.eval()
                self._dino_model.to(self.device)
                from torchvision import transforms

                self._dino_transform = transforms.Compose(
                    [
                        transforms.Resize(
                            256, interpolation=transforms.InterpolationMode.BICUBIC
                        ),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                        ),
                    ]
                )
                HAS_DINO_LOCAL = True
                logger.info("DINOv2 yüklendi (%s)", self.device)
            except Exception as exc:
                HAS_DINO_LOCAL = False
                logger.warning("DINOv2 yüklenemedi: %s", exc)
                self._dino_model = None

        if HAS_CLIP and HAS_TORCH:
            try:
                self._clip_model, _, self._clip_preprocess = (
                    open_clip.create_model_and_transforms(
                        "ViT-B-32",
                        pretrained="openai",
                    )
                )
                self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self._clip_model.eval()
                self._clip_model.to(self.device)
                logger.info("OpenCLIP yüklendi (%s)", self.device)
            except Exception as exc:
                logger.warning("OpenCLIP yüklenemedi: %s", exc)
                self._clip_model = None

    @property
    def ai_available(self) -> bool:
        return self._dino_model is not None or self._clip_model is not None

    def extract_from_path(
        self,
        image_path: str,
        include_patches: bool = True,
        source_path: str = "",
        *,
        deep_analysis: bool = True,
        compute: set[str] | None = None,
    ) -> ExtractedFeatures:
        image = Thumbnailer.load_image(image_path)
        if image is None:
            return ExtractedFeatures()
        from pathlib import Path as _Path

        meta_path = source_path or image_path
        return self.extract_from_array(
            image,
            include_patches=include_patches,
            filename=_Path(meta_path).name,
            path=meta_path,
            deep_analysis=deep_analysis,
            compute=compute,
        )

    def extract_from_array(
        self,
        image: np.ndarray,
        include_patches: bool = True,
        filename: str = "",
        path: str = "",
        *,
        deep_analysis: bool = True,
        compute: set[str] | None = None,
        seed_dominant: list[list[int]] | None = None,
        seed_color_hist: bytes | None = None,
    ) -> ExtractedFeatures:
        """compute=None → tüm vision; aksi halde yalnız istenen modüller.

        RC2 sparse: örn. compute={'patch'} yalnız patch; dino/clip/texture atlanır.
        HASH already stored color/hashes can be seeded so TEXTURE skips kmeans.
        """
        features = ExtractedFeatures()
        if self.fast_hash_only:
            features.phash, features.dhash, features.whash = self._compute_hashes(image)
            return features

        # None = legacy full; boş set = hiç vision (yalnız semantic/dna yolu)
        do_all = compute is None
        want = compute if compute is not None else set()

        def _need(name: str) -> bool:
            return do_all or name in want

        if _need("hash") or do_all:
            features.phash, features.dhash, features.whash = self._compute_hashes(image)
        if _need("color") or do_all:
            features.color_hist, features.dominant_colors = self._compute_color(image)
        elif seed_dominant:
            features.dominant_colors = list(seed_dominant)
            features.color_hist = seed_color_hist or b""
        if _need("texture") or do_all:
            features.texture_features = self._compute_texture(image)
        if include_patches and deep_analysis and (_need("patch") or do_all):
            features.patch_embeddings_meta = self._compute_multiscale_patch_features(
                image
            )

        color_index: dict[str, Any] = {}
        if deep_analysis and (_need("texture") or _need("color") or do_all):
            from core.color_index import build_color_index

            color_index = build_color_index(
                image, dominant_colors=features.dominant_colors
            )

        if deep_analysis and (_need("texture") or do_all):
            from core.texture_profile import TextureAnalyzer

            profile = TextureAnalyzer.analyze(
                image,
                filename=filename,
                path=path,
                patch_metas=features.patch_embeddings_meta,
                dominant_colors=features.dominant_colors,
            )
            features.texture_map = profile.to_dict()
            features.texture_map["color_index"] = color_index
            from core.color_index import apply_auto_color

            features.texture_map = apply_auto_color(features.texture_map)
            try:
                from core.color_evidence import (
                    attach_color_evidence,
                    extract_palette_clusters,
                )

                _cl, weights = extract_palette_clusters(image, k=5)
                features.texture_map = attach_color_evidence(
                    features.texture_map,
                    features.dominant_colors or _cl,
                    overwrite_ai=True,
                    cluster_weights=weights or None,
                )
            except Exception:
                pass
            if color_index.get("palette"):
                features.texture_map["dominant_palette"] = list(
                    dict.fromkeys(
                        list(features.texture_map.get("dominant_palette") or [])
                        + list(color_index["palette"])
                    )
                )[:8]
        else:
            features.texture_map = {"color_index": color_index}

        if self.use_ai and not self.fast_hash_only and (
            _need("dino") or _need("clip") or do_all
        ):
            pil = Image.fromarray(image)
            if self._dino_model is not None and (_need("dino") or do_all):
                emb = self._embed_dino(pil)
                if emb is not None:
                    features.dino_embedding = emb.astype(np.float32).tobytes()
            if self._clip_model is not None and (_need("clip") or do_all):
                emb = self._embed_clip_image(pil)
                if emb is not None:
                    features.clip_embedding = emb.astype(np.float32).tobytes()
            # 3x3 DINO-on-patch bytes are never upserted (DB column is JSON
            # patch_embeddings_meta). Search ranking uses meta phash/texture.
            # Skipping them does not change AI Final or Pattern Search scores.

        return features

    def _patch_meta_from_array(
        self, patch: np.ndarray, tag: str, index: int
    ) -> dict[str, Any]:
        ph, dh, wh = self._compute_hashes(patch)
        texture = self._compute_texture(patch)
        gray_sig = self._grayscale_texture_signature(patch, texture=texture)
        mean_rgb = np.mean(patch.reshape(-1, 3), axis=0).astype(float).tolist()
        return {
            "index": index,
            "tag": tag,
            "phash": ph,
            "dhash": dh,
            "whash": wh,
            "texture": [float(x) for x in texture],
            "gray_texture": gray_sig,
            "mean_rgb": [round(float(x), 3) for x in mean_rgb],
        }

    def _grayscale_texture_signature(
        self, image: np.ndarray, texture: list[float] | None = None
    ) -> list[float]:
        """Renk bağımsız doku imzası — LBP/edge ağırlıklı."""
        tex = texture if texture is not None else self._compute_texture(image)
        if image.ndim == 3:
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = image.astype(np.float32)
        edge = tex[2] if len(tex) > 2 else 0.0
        lbp = tex[4] if len(tex) > 4 else 0.0
        return [float(edge), float(lbp), float(np.std(gray) / 128.0)]

    def _compute_multiscale_patch_features(
        self, image: np.ndarray
    ) -> list[dict[str, Any]]:
        from core.multiscale_patch import extract_multiscale_patches

        def factory(patch, tag, idx):
            return self._patch_meta_from_array(patch, tag, idx)

        return extract_multiscale_patches(image, factory)

    def embed_text(self, text: str) -> np.ndarray | None:
        if self._clip_model is None or self._clip_tokenizer is None or not HAS_TORCH:
            return None
        try:
            with torch.no_grad():
                tokens = self._clip_tokenizer([text]).to(self.device)
                emb = self._clip_model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb.cpu().numpy().flatten()
        except Exception as exc:
            logger.warning("CLIP metin embedding hatası: %s", exc)
            return None

    def _compute_hashes(self, image: np.ndarray) -> tuple[str, str, str]:
        if not HAS_IMAGEHASH:
            return "", "", ""
        try:
            pil = Image.fromarray(image)
            return (
                str(imagehash.phash(pil)),
                str(imagehash.dhash(pil)),
                str(imagehash.whash(pil)),
            )
        except Exception as exc:
            logger.warning("Hash hesaplama hatası: %s", exc)
            return "", "", ""

    def _compute_color(self, image: np.ndarray) -> tuple[bytes, list[list[int]]]:
        try:
            from core.safe_image_ops import hsv_hist_rgb

            combined = hsv_hist_rgb(image, 32, 32, 32)

            dominant = self._dominant_colors(image)
            return combined.astype(np.float32).tobytes(), dominant
        except Exception as exc:
            logger.warning("Renk özelliği hatası: %s", exc)
            return b"", []

    @staticmethod
    def _dominant_colors(image: np.ndarray, k: int = 5) -> list[list[int]]:
        try:
            from core.color_evidence import extract_palette_clusters

            clusters, _weights = extract_palette_clusters(image, k=k)
            if clusters:
                return clusters
        except Exception:
            pass
        small = image[::8, ::8].reshape(-1, 3).astype(np.float32)
        if len(small) < k:
            return small.astype(int).tolist()
        from core.safe_image_ops import kmeans_centers

        centers, labels = kmeans_centers(small, k)
        if len(centers) == 0:
            return small[:k].astype(int).tolist()
        counts = np.bincount(labels.flatten(), minlength=len(centers))
        order = np.argsort(-counts)
        return [centers[i].astype(int).tolist() for i in order]

    def _texture_core(self, image: np.ndarray) -> tuple[list[float], np.ndarray]:
        """Contrast/brightness/edge/laplacian — same numbers as _compute_texture."""
        from core.safe_image_ops import edge_density as _edge_density
        from core.safe_image_ops import laplacian_var, rgb_to_gray

        gray = rgb_to_gray(image)
        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))
        edge_density = _edge_density(gray)
        lap_var = laplacian_var(gray)
        return [contrast, brightness, edge_density, lap_var], gray

    def _compute_texture(self, image: np.ndarray) -> list[float]:
        try:
            core, gray = self._texture_core(image)
            lbp = self._simple_lbp(gray)
            return core + [lbp]
        except Exception as exc:
            logger.warning("Doku özelliği hatası: %s", exc)
            return [0.0] * 5

    @staticmethod
    def _simple_lbp(gray: np.ndarray) -> float:
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.0
        center = gray[1:-1, 1:-1]
        code = np.zeros_like(center, dtype=np.uint8)
        code |= (gray[:-2, :-2] >= center).astype(np.uint8) << 7
        code |= (gray[:-2, 1:-1] >= center).astype(np.uint8) << 6
        code |= (gray[:-2, 2:] >= center).astype(np.uint8) << 5
        code |= (gray[1:-1, 2:] >= center).astype(np.uint8) << 4
        code |= (gray[2:, 2:] >= center).astype(np.uint8) << 3
        code |= (gray[2:, 1:-1] >= center).astype(np.uint8) << 2
        code |= (gray[2:, :-2] >= center).astype(np.uint8) << 1
        code |= (gray[1:-1, :-2] >= center).astype(np.uint8)
        hist, _ = np.histogram(code, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-8
        return float(-np.sum(hist * np.log2(hist + 1e-8)))

    def _embed_dino(self, pil: Image.Image) -> np.ndarray | None:
        if self._dino_model is None or self._dino_transform is None:
            return None
        try:
            tensor = self._dino_transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self._dino_model(tensor)
                if hasattr(out, "norm"):
                    out = out / out.norm(dim=-1, keepdim=True)
                return out.cpu().numpy().flatten()
        except Exception as exc:
            logger.warning("DINO embedding hatası: %s", exc)
            return None

    def _embed_clip_image(self, pil: Image.Image) -> np.ndarray | None:
        if self._clip_model is None or self._clip_preprocess is None:
            return None
        try:
            tensor = self._clip_preprocess(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self._clip_model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb.cpu().numpy().flatten()
        except Exception as exc:
            logger.warning("CLIP görsel embedding hatası: %s", exc)
            return None

    @staticmethod
    def color_similarity(hist_a: bytes, hist_b: bytes) -> float:
        if not hist_a or not hist_b:
            return 0.0
        a = np.frombuffer(hist_a, dtype=np.float32)
        b = np.frombuffer(hist_b, dtype=np.float32)
        if len(a) != len(b):
            return 0.0
        # Histogram korelasyonu
        a = a - a.mean()
        b = b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-8:
            return 0.0
        return float(max(0.0, np.dot(a, b) / denom))

    @staticmethod
    def texture_similarity(tex_a: list[float], tex_b: list[float]) -> float:
        if not tex_a or not tex_b or len(tex_a) != len(tex_b):
            return 0.0
        a = np.array(tex_a, dtype=np.float32)
        b = np.array(tex_b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-8:
            return 0.0
        return float(max(0.0, np.dot(a, b) / denom))

    @staticmethod
    def embedding_similarity(emb_a: bytes, emb_b: bytes) -> float:
        if not emb_a or not emb_b:
            return 0.0
        # Legacy V3 test records may contain arbitrary/non-float32 BLOBs.
        # Ignore those records instead of aborting the whole search.
        try:
            if len(emb_a) % 4 != 0 or len(emb_b) % 4 != 0:
                return 0.0
            a = np.frombuffer(emb_a, dtype=np.float32)
            b = np.frombuffer(emb_b, dtype=np.float32)
            if len(a) == 0 or len(a) != len(b):
                return 0.0
        except (TypeError, ValueError):
            return 0.0
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-8:
            return 0.0
        return float(max(0.0, np.dot(a, b) / denom))

    @staticmethod
    def combined_phash_similarity(
        phash: str, dhash: str, whash: str, q_phash: str, q_dhash: str, q_whash: str
    ) -> float:
        from core.utils import phash_similarity

        scores = []
        for a, b in [(phash, q_phash), (dhash, q_dhash), (whash, q_whash)]:
            if a and b:
                scores.append(phash_similarity(a, b))
        return sum(scores) / len(scores) if scores else 0.0
