"""CLIP map recovery probe: DINO-prefix vs DB vector match. Read-only."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.faiss_store import load_faiss_id_map, resolve_faiss_id_map_path
from core.settings import AppSettings
import faiss


def norm_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-8)


def main() -> None:
    s = AppSettings.load()
    clip_index = faiss.read_index(s.faiss_clip_path)
    dino_ids = load_faiss_id_map(resolve_faiss_id_map_path(s.faiss_dino_path))
    n = int(clip_index.ntotal)
    d = int(clip_index.d)
    print("clip", n, d, "dino_ids", len(dino_ids))

    prefix = dino_ids[:n]
    # reconstruct all clip vectors
    xb = np.zeros((n, d), dtype=np.float32)
    clip_index.reconstruct_n(0, n, xb)
    xb = norm_rows(xb)

    con = sqlite3.connect(f"file:{Path(s.db_path).as_posix()}?mode=ro", uri=True)
    # prefix alignment
    ok = 0
    missing = 0
    bad = []
    step = max(1, n // 40)
    for i in list(range(0, n, step)) + [n - 1]:
        fid = int(prefix[i])
        row = con.execute("SELECT clip_embedding FROM features WHERE file_id=?", (fid,)).fetchone()
        if not row or not row[0]:
            missing += 1
            continue
        dbv = np.frombuffer(row[0], dtype=np.float32)
        if dbv.size != d:
            bad.append((i, fid, "dim"))
            continue
        dbv = dbv / max(np.linalg.norm(dbv), 1e-8)
        sim = float(np.dot(xb[i], dbv))
        if sim >= 0.999:
            ok += 1
        else:
            bad.append((i, fid, round(sim, 4)))
    print("prefix_sample ok", ok, "bad", len(bad), "missing", missing)
    print("bad sample", bad[:12])

    # Full DB matrix match for a 200-vector sample of CLIP positions
    rows = con.execute(
        "SELECT file_id, clip_embedding FROM features "
        "WHERE clip_embedding IS NOT NULL AND length(clip_embedding)=?"
        " ORDER BY file_id",
        (d * 4,),
    ).fetchall()
    fids = np.array([int(r[0]) for r in rows], dtype=np.int64)
    dbm = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    dbm = norm_rows(dbm)
    print("db clip matrix", dbm.shape)

    sample = np.linspace(0, n - 1, num=min(200, n), dtype=int)
    q = xb[sample]
    sims = q @ dbm.T
    best = sims.argmax(axis=1)
    best_s = sims.max(axis=1)
    recovered = fids[best]
    print("sample200 mean_best", float(best_s.mean()), "min", float(best_s.min()), "frac>=0.999", float((best_s >= 0.999).mean()))
    print("sample recovered unique", len(set(recovered.tolist())))
    # prefix vs recovered agreement
    agree = sum(int(recovered[k]) == int(prefix[int(sample[k])]) for k in range(len(sample)))
    print("sample agree with dino-prefix ids", agree, "/", len(sample))
    con.close()


if __name__ == "__main__":
    main()
