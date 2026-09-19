"""18-file Patch optimize after-bench. No production DB writes."""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    import psutil

    from core.db import Database
    from core.feature_extractor import FeatureExtractor
    from core.settings import AppSettings

    s = AppSettings.load()
    db = Database(s.db_path)
    rows = []
    with db.connect() as conn:
        for rec in conn.execute(
            """
            SELECT id, feature_preview_path FROM files
            WHERE source_id IN (8,9)
              AND COALESCE(physical_preview_ready,0)=1
              AND COALESCE(feature_preview_path,'')!=''
            ORDER BY id
            """
        ):
            p = str(rec["feature_preview_path"] or "")
            if p and Path(p).is_file():
                rows.append((int(rec["id"]), p))
            if len(rows) >= 18:
                break
    ext = FeatureExtractor(use_ai=True, use_gpu=bool(s.use_gpu), fast_hash_only=False)
    ext.extract_from_path(rows[0][1], include_patches=False, compute={"dino"})
    proc = psutil.Process()
    proc.cpu_percent(None)
    cpu: list[float] = []
    dino_t = 0.0
    patch_t = 0.0
    patch_lens = []
    t_all = time.perf_counter()
    for fid, src in rows:
        t0 = time.perf_counter()
        d = ext.extract_from_path(src, include_patches=False, compute={"dino"})
        dino_t += time.perf_counter() - t0
        t0 = time.perf_counter()
        p = ext.extract_from_path(src, include_patches=True, compute={"patch"})
        patch_t += time.perf_counter() - t0
        patch_lens.append(len(p.patch_embeddings_meta or []))
        cpu.append(float(proc.cpu_percent(None)))
        assert d.dino_embedding, fid
        assert p.patch_embeddings_meta, fid
        assert not p.patch_embeddings, "discarded DINO patch bytes must stay empty"
    wall = time.perf_counter() - t_all
    n = len(rows)
    out = {
        "files": n,
        "before": {
            "job_dino_sum_sec": 8.538,
            "job_patch_sum_sec": 88.434,
            "patch_avg_sec": 4.913,
            "dino_avg_sec": 0.474,
        },
        "after": {
            "wall_sec": round(wall, 3),
            "files_per_min": round(n / max(wall, 1e-6) * 60.0, 2),
            "dino_sum_sec": round(dino_t, 3),
            "patch_sum_sec": round(patch_t, 3),
            "dino_avg_sec": round(dino_t / n, 3),
            "patch_avg_sec": round(patch_t / n, 3),
            "cpu_avg": round(sum(cpu) / len(cpu), 2) if cpu else None,
            "patch_meta_avg_len": round(sum(patch_lens) / n, 1),
        },
        "speedup_patch_pct": round((88.434 / patch_t - 1) * 100.0, 1) if patch_t else None,
        "db_faiss_overhead": "not in this extract-only bench (no upsert/FAISS)",
    }
    path = ROOT / "data" / "reports" / "patch_optimize_18.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
