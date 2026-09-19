"""Full CLIP slot → DB embedding unique-match recovery probe. Read-only."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.settings import AppSettings


def norm_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-8)


def main() -> None:
    s = AppSettings.load()
    index = faiss.read_index(s.faiss_clip_path)
    n, d = int(index.ntotal), int(index.d)
    xb = np.zeros((n, d), dtype=np.float32)
    index.reconstruct_n(0, n, xb)
    norms = np.linalg.norm(xb, axis=1)
    print("clip slots", n, "zero_or_tiny", int((norms < 1e-6).sum()), "norm_mean", float(norms.mean()))

    con = sqlite3.connect(f"file:{Path(s.db_path).as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT file_id, clip_embedding FROM features "
        "WHERE clip_embedding IS NOT NULL AND length(clip_embedding)=?",
        (d * 4,),
    ).fetchall()
    fids = np.array([int(r[0]) for r in rows], dtype=np.int64)
    dbm = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    dbm = norm_rows(dbm)
    xb_n = norm_rows(xb)
    live = norms >= 1e-6
    print("live slots", int(live.sum()), "db", dbm.shape)

    # Match live slots in chunks to limit RAM
    recovered = np.full(n, -1, dtype=np.int64)
    scores = np.zeros(n, dtype=np.float32)
    margins = np.zeros(n, dtype=np.float32)
    idx_live = np.where(live)[0]
    chunk = 500
    unique_ok = 0
    for start in range(0, len(idx_live), chunk):
        sl = idx_live[start : start + chunk]
        sims = xb_n[sl] @ dbm.T
        part = np.argpartition(sims, -2, axis=1)[:, -2:]
        # order the two
        for row, pos in enumerate(sl):
            a, b = part[row]
            sa, sb = float(sims[row, a]), float(sims[row, b])
            if sa >= sb:
                best_i, best_s, second_s = a, sa, sb
            else:
                best_i, best_s, second_s = b, sb, sa
            scores[pos] = best_s
            margins[pos] = best_s - second_s
            if best_s >= 0.999 and (best_s - second_s) >= 0.001:
                recovered[pos] = int(fids[best_i])
                unique_ok += 1
        print("chunk", start, "unique_ok", unique_ok)
    con.close()

    mapped = recovered[recovered >= 0]
    print("unique_ok", unique_ok, "unique_ids", len(set(mapped.tolist())))
    print("score live p50", float(np.median(scores[live])), "p10", float(np.percentile(scores[live], 10)))
    dup = len(mapped) - len(set(mapped.tolist()))
    print("duplicate_assigned_ids", dup)
    out = {
        "clip_ntotal": n,
        "zero_slots": int((~live).sum()),
        "unique_ok": unique_ok,
        "unique_ids": len(set(mapped.tolist())),
        "duplicate_assigned": dup,
        "recoverable_frac_of_index": round(unique_ok / max(n, 1), 4),
        "recoverable_frac_of_live": round(unique_ok / max(int(live.sum()), 1), 4),
    }
    dest = ROOT / "data" / "reports" / "clip_map_recovery_probe.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
