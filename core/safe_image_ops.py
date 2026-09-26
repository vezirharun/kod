"""Numpy/PIL image ops used instead of OpenCV on V3 feature paths.

Avoids loading cv2.pyd into the main process during DINO/CLIP heavy extract
(Windows heap corruption 0xC0000374). API is intentionally small and local.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans

    _HAS_SK = True
except Exception:  # pragma: no cover
    MiniBatchKMeans = None  # type: ignore
    _HAS_SK = False


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.uint8, copy=False)
    return np.dot(rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)


def resize_gray(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize to (width, height) — OpenCV-compatible order."""
    w, h = int(size[0]), int(size[1])
    if gray.shape[1] == w and gray.shape[0] == h:
        return gray
    ys = (np.linspace(0, gray.shape[0] - 1, h)).astype(np.int32)
    xs = (np.linspace(0, gray.shape[1] - 1, w)).astype(np.int32)
    return gray[ys][:, xs]


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy)


def edge_density(gray: np.ndarray, low: float = 50.0, high: float = 150.0) -> float:
    mag = sobel_mag(gray)
    # Hysteresis-lite: strong edges + weak connected-ish via threshold band.
    strong = mag >= high
    weak = (mag >= low) & (mag < high)
    edges = strong | (weak & (mag >= (low + high) * 0.5))
    return float(np.count_nonzero(edges)) / max(edges.size, 1)


def laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    # 4-neighbour Laplacian kernel
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = (
        4.0 * g[1:-1, 1:-1]
        - g[:-2, 1:-1]
        - g[2:, 1:-1]
        - g[1:-1, :-2]
        - g[1:-1, 2:]
    )
    return float(lap.var())


def box_blur(img: np.ndarray, k: int = 5) -> np.ndarray:
    k = max(1, int(k) | 1)  # odd
    pad = k // 2
    x = img.astype(np.float32)
    # separable cumulative sum blur
    c = np.pad(x, ((pad, pad), (pad, pad)), mode="edge")
    cs = np.cumsum(np.cumsum(c, axis=0), axis=1)
    # integral image trick with zero pad row/col
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    h, w = x.shape
    out = (
        cs[k : k + h, k : k + w]
        - cs[0:h, k : k + w]
        - cs[k : k + h, 0:w]
        + cs[0:h, 0:w]
    ) / float(k * k)
    return out


def blob_score(gray: np.ndarray) -> float:
    if gray.shape[0] < 32:
        return 0.0
    blur = box_blur(gray, 5)
    # Otsu-ish via median threshold (stable, no cv2)
    thresh = float(np.median(blur))
    binary = (blur < thresh).astype(np.uint8)
    # connected components via simple run labeling on downscaled grid
    small = binary[::2, ::2]
    h, w = small.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_id = 0
    areas: list[int] = []
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for y in range(h):
        for x in range(w):
            if small[y, x] == 0:
                continue
            left = labels[y, x - 1] if x > 0 else 0
            up = labels[y - 1, x] if y > 0 else 0
            if left == 0 and up == 0:
                next_id += 1
                labels[y, x] = next_id
                parent[next_id] = next_id
            elif left and not up:
                labels[y, x] = left
            elif up and not left:
                labels[y, x] = up
            else:
                labels[y, x] = left
                union(left, up)

    counts: dict[int, int] = {}
    for y in range(h):
        for x in range(w):
            lab = labels[y, x]
            if lab == 0:
                continue
            root = find(lab)
            counts[root] = counts.get(root, 0) + 1
    areas = list(counts.values())
    if not areas:
        return 0.0
    total = h * w
    small_blobs = sum(1 for a in areas if 0.001 * total < a < 0.05 * total)
    return min(1.0, small_blobs / 25.0)


def scale_score(gray: np.ndarray) -> float:
    if gray.shape[0] < 32:
        return 0.0
    small = resize_gray(gray, (64, 64)).astype(np.float32)
    lap = np.zeros_like(small)
    lap[1:-1, 1:-1] = (
        4.0 * small[1:-1, 1:-1]
        - small[:-2, 1:-1]
        - small[2:, 1:-1]
        - small[1:-1, :-2]
        - small[1:-1, 2:]
    )
    local_var = box_blur(lap ** 2, 8)
    cells = (local_var > local_var.mean()).astype(np.uint8)
    return min(1.0, float(cells.mean()) * 1.8)


def kmeans_centers(
    samples: np.ndarray,
    k: int,
    *,
    max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (centers[k,d], labels[n]) with sklearn MiniBatchKMeans or quantize fallback."""
    pts = np.asarray(samples, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, pts.shape[-1] if pts.ndim == 2 else 0), dtype=np.float32), np.zeros(0, dtype=np.int32)
    kk = max(1, min(int(k), len(pts)))
    if _HAS_SK and kk >= 2 and len(pts) >= kk:
        try:
            model = MiniBatchKMeans(
                n_clusters=kk,
                max_iter=max_iter,
                n_init=3,
                random_state=0,
                batch_size=min(1024, len(pts)),
            )
            labels = model.fit_predict(pts).astype(np.int32)
            return model.cluster_centers_.astype(np.float32), labels
        except Exception:
            pass
    # Frequency-quantize fallback (8-step RGB buckets when d==3)
    if pts.shape[1] == 3:
        rounded = np.round(pts / 8.0).astype(np.int32) * 8
        keys = (
            (rounded[:, 0] << 16) | (rounded[:, 1] << 8) | rounded[:, 2]
        ).astype(np.int32)
        uniq, counts = np.unique(keys, return_counts=True)
        order = np.argsort(-counts)[:kk]
        centers = []
        for i in order:
            key = int(uniq[i])
            centers.append([(key >> 16) & 255, (key >> 8) & 255, key & 255])
        centers_arr = np.asarray(centers, dtype=np.float32)
        # assign labels by nearest center
        d = ((pts[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1).astype(np.int32)
        return centers_arr, labels
    step = max(1, len(pts) // kk)
    centers_arr = pts[::step][:kk]
    d = ((pts[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
    labels = d.argmin(axis=1).astype(np.int32)
    return centers_arr.astype(np.float32), labels


def hsv_hist_rgb(image: np.ndarray, h_bins: int = 32, s_bins: int = 32, v_bins: int = 32) -> np.ndarray:
    """HSV histogram from RGB uint8 image (OpenCV H range mapped 0..180)."""
    rgb = image[..., :3].astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-8
    # hue in degrees 0..360 then map to OpenCV 0..180
    rc = ((mx - r) / (diff + 1e-8))
    gc = ((mx - g) / (diff + 1e-8))
    bc = ((mx - b) / (diff + 1e-8))
    h = np.where((mx == r) & mask, (60.0 * (gc - bc)) % 360.0, h)
    h = np.where((mx == g) & mask, 60.0 * (bc - rc) + 120.0, h)
    h = np.where((mx == b) & mask, 60.0 * (rc - gc) + 240.0, h)
    h = (h / 2.0)  # OpenCV scale
    s = np.where(mx > 1e-8, diff / (mx + 1e-8), 0.0) * 255.0
    v = mx * 255.0
    hh, _ = np.histogram(h, bins=h_bins, range=(0, 180))
    ss, _ = np.histogram(s, bins=s_bins, range=(0, 256))
    vv, _ = np.histogram(v, bins=v_bins, range=(0, 256))
    out = np.concatenate([hh, ss, vv]).astype(np.float32)
    total = float(out.sum()) + 1e-8
    out /= total
    return out
