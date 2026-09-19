"""Profile Patch vs DINO sub-steps on 18 local previews. No DB writes."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    import numpy as np
    from PIL import Image

    from core.db import Database
    from core.feature_extractor import FeatureExtractor
    from core.settings import AppSettings
    from core.thumbnailer import Thumbnailer

    s = AppSettings.load()
    db = Database(s.db_path)
    paths: list[str] = []
    with db.connect() as conn:
        for rec in conn.execute(
            """
            SELECT feature_preview_path FROM files
            WHERE source_id IN (8,9)
              AND COALESCE(physical_preview_ready,0)=1
              AND COALESCE(feature_preview_path,'')!=''
            ORDER BY id
            """
        ):
            p = str(rec["feature_preview_path"] or "")
            if p and Path(p).is_file():
                paths.append(p)
            if len(paths) >= 18:
                break
    print(f"files {len(paths)}", flush=True)
    ext = FeatureExtractor(use_ai=True, use_gpu=bool(s.use_gpu), fast_hash_only=False)

    buckets: dict[str, list[float]] = defaultdict(list)

    def timed(name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        buckets[name].append(time.perf_counter() - t0)
        return out

    # warmup
    ext.extract_from_path(paths[0], include_patches=False, compute={"dino"})

    work = paths[:18]
    for src in work:
        img = timed("decode", lambda: Thumbnailer.load_image(src))
        if img is None:
            continue
        timed("dino_full", lambda i=img: ext._embed_dino(Image.fromarray(i)))
        timed(
            "multiscale_meta",
            lambda i=img: ext._compute_multiscale_patch_features(i),
        )
        patches = timed(
            "extract_grid3", lambda i=img: Thumbnailer.extract_patches(i, grid=3)
        )
        n_dino = 0
        t_d = time.perf_counter()
        for patch in patches:
            ext._embed_dino(Image.fromarray(patch))
            n_dino += 1
        buckets["dino_x_grid3"].append(time.perf_counter() - t_d)
        buckets["dino_grid_count"].append(float(n_dino))
        t_np = time.perf_counter()
        Image.fromarray(img)
        buckets["numpy_to_pil"].append(time.perf_counter() - t_np)
        if ext.device != "cpu":
            t_x = time.perf_counter()
            ten = ext._dino_transform(Image.fromarray(img)).unsqueeze(0).to(ext.device)
            _ = ten.cpu()
            buckets["cpu_gpu_roundtrip"].append(time.perf_counter() - t_x)

    def agg(name: str) -> dict:
        xs = buckets.get(name) or []
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "sum_sec": round(sum(xs), 3),
            "avg_ms": round(1000 * sum(xs) / len(xs), 2),
            "max_ms": round(1000 * max(xs), 2),
        }

    t_full_patch = 0.0
    t_dino_job = 0.0
    for src in work:
        t0 = time.perf_counter()
        ext.extract_from_path(src, include_patches=False, compute={"dino"})
        t_dino_job += time.perf_counter() - t0
        t0 = time.perf_counter()
        ext.extract_from_path(src, include_patches=True, compute={"patch"})
        t_full_patch += time.perf_counter() - t0

    out = {
        "files": len(work),
        "device": ext.device,
        "substeps": {k: agg(k) for k in (
            "decode", "dino_full", "multiscale_meta", "extract_grid3",
            "dino_x_grid3", "numpy_to_pil", "cpu_gpu_roundtrip",
        )},
        "job_dino_sum_sec": round(t_dino_job, 3),
        "job_patch_sum_sec": round(t_full_patch, 3),
        "patch_per_file_avg_sec": round(t_full_patch / len(work), 3),
        "dino_per_file_avg_sec": round(t_dino_job / len(work), 3),
        "findings": {
            "patch_dino_bytes_persisted": False,
            "search_uses": "patch_embeddings_meta hashes/texture, not patch DINO vectors",
            "has_dino_flag_read_elsewhere": False,
        },
    }
    path = ROOT / "data" / "reports" / "patch_profile_18.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
