"""Multi-scale patch extraction — crop/scale dayanıklı desen eşleşmesi."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from core.thumbnailer import Thumbnailer


def _patch_texture_weight(texture: list[float]) -> float:
    """Düz zemin patchlerini ele, yoğun doku patchlerini öne al."""
    if not texture or len(texture) < 4:
        return 0.0
    edge = float(texture[2]) if len(texture) > 2 else 0.0
    lap = float(texture[3]) if len(texture) > 3 else 0.0
    contrast = float(texture[0])
    return min(1.0, edge * 2.5 + lap / 500.0 + contrast / 80.0)


def extract_multiscale_patches(
    image: np.ndarray,
    compute_meta: Callable[[np.ndarray, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    3x3, 4x4, center, köşeler, sliding window patch meta listesi.
    compute_meta(patch_array, tag, index) -> dict with phash, dhash, whash, texture, mean_rgb
    """
    if image is None or image.size == 0:
        return []

    metas: list[dict[str, Any]] = []
    idx = 0

    for grid in (3, 4):
        patches = Thumbnailer.extract_patches(image, grid=grid)
        for i, patch in enumerate(patches):
            meta = compute_meta(patch, f"g{grid}_{i}", idx)
            tw = _patch_texture_weight(meta.get("texture", []))
            if tw < 0.08:
                continue
            meta["scale"] = f"grid_{grid}"
            meta["texture_weight"] = round(tw, 4)
            meta["region"] = f"g{grid}_{i}"
            metas.append(meta)
            idx += 1

    h, w = image.shape[:2]
    ch, cw = int(h * 0.6), int(w * 0.6)
    cy, cx = (h - ch) // 2, (w - cw) // 2
    center = image[cy : cy + ch, cx : cx + cw]
    if center.size > 0:
        meta = compute_meta(center, "center_crop", idx)
        meta["scale"] = "center"
        meta["texture_weight"] = round(
            _patch_texture_weight(meta.get("texture", [])), 4
        )
        meta["region"] = "center_crop"
        metas.append(meta)
        idx += 1

    size = min(h, w) // 3
    if size >= 16:
        regions = [
            (0, 0, "tl"),
            (w - size, 0, "tr"),
            (0, h - size, "bl"),
            (w - size, h - size, "br"),
            ((w - size) // 2, (h - size) // 2, "center"),
        ]
        for x, y, tag in regions:
            x, y = max(0, x), max(0, y)
            patch = image[y : y + size, x : x + size]
            if patch.size == 0:
                continue
            meta = compute_meta(patch, tag, idx)
            tw = _patch_texture_weight(meta.get("texture", []))
            if tw < 0.06 and tag != "center":
                continue
            meta["scale"] = "corner"
            meta["texture_weight"] = round(tw, 4)
            meta["region"] = tag
            metas.append(meta)
            idx += 1

    win = max(32, min(h, w) // 5)
    step = max(16, win // 2)
    for y in range(0, max(1, h - win), step):
        for x in range(0, max(1, w - win), step):
            patch = image[y : y + win, x : x + win]
            if patch.size == 0:
                continue
            meta = compute_meta(patch, f"sw_{y}_{x}", idx)
            tw = _patch_texture_weight(meta.get("texture", []))
            if tw < 0.12:
                continue
            meta["scale"] = "sliding"
            meta["texture_weight"] = round(tw, 4)
            meta["region"] = f"sw_{y}_{x}"
            metas.append(meta)
            idx += 1
            if idx > 80:
                break
        if idx > 80:
            break

    return metas


def multiscale_patch_similarity(
    query_patches: list[dict[str, Any]],
    candidate_patches: list[dict[str, Any]],
    hash_sim_fn,
    texture_sim_fn,
    color_sim_fn,
) -> tuple[float, float]:
    """
    patch_score (en iyi eşleşmeler) ve multi_scale_patch_score (scale çeşitliliği).
    """
    if not query_patches or not candidate_patches:
        return 0.0, 0.0

    best_scores: list[float] = []
    scale_hits: dict[str, float] = {}

    for qp in query_patches:
        q_weight = float(qp.get("texture_weight", 0.5) or 0.5)
        patch_scores: list[float] = []
        for cp in candidate_patches:
            hash_scores = []
            for key in ("phash", "dhash", "whash"):
                if qp.get(key) and cp.get(key):
                    hash_scores.append(hash_sim_fn(qp.get(key, ""), cp.get(key, "")))
            hash_score = sum(hash_scores) / len(hash_scores) if hash_scores else 0.0
            texture_score = texture_sim_fn(
                qp.get("texture") or [],
                cp.get("texture") or [],
            )
            color_score = color_sim_fn(
                qp.get("mean_rgb") or [],
                cp.get("mean_rgb") or [],
            )
            # Renk bağımsız modda color düşük ağırlık
            combined = 0.48 * hash_score + 0.40 * texture_score + 0.12 * color_score
            c_weight = float(cp.get("texture_weight", 0.5) or 0.5)
            patch_scores.append(combined * (0.5 + 0.5 * min(q_weight, c_weight)))

        if patch_scores:
            best = max(patch_scores)
            best_scores.append(best)
            scale = str(qp.get("scale", "grid"))
            scale_hits[scale] = max(scale_hits.get(scale, 0.0), best)

    best_scores.sort(reverse=True)
    top_n = best_scores[: max(1, min(6, len(best_scores)))]
    patch_score = float(sum(top_n) / len(top_n)) if top_n else 0.0

    if scale_hits:
        multi_scale = sum(scale_hits.values()) / len(scale_hits)
        # Farklı scale'lerde tutarlı eşleşme bonusu
        if len(scale_hits) >= 3:
            multi_scale = min(1.0, multi_scale * 1.05)
    else:
        multi_scale = patch_score

    return patch_score, multi_scale
