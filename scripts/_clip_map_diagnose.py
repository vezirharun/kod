"""Read-only CLIP map recovery diagnosis. Does not write production files."""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.faiss_store import (
    HAS_FAISS,
    inspect_faiss_id_map,
    load_faiss_id_map,
    resolve_faiss_id_map_path,
)
from core.settings import AppSettings

try:
    import faiss
except ImportError:
    faiss = None


def main() -> None:
    s = AppSettings.load()
    clip_idx = Path(s.faiss_clip_path)
    dino_idx = Path(s.faiss_dino_path)
    print("clip_index", clip_idx, "exists", clip_idx.exists(), "size", clip_idx.stat().st_size if clip_idx.exists() else 0)
    print("dino_index", dino_idx, "exists", dino_idx.exists(), "size", dino_idx.stat().st_size if dino_idx.exists() else 0)
    clip_map = resolve_faiss_id_map_path(clip_idx)
    dino_map = resolve_faiss_id_map_path(dino_idx)
    print("clip_map", inspect_faiss_id_map(clip_map))
    print("dino_map", inspect_faiss_id_map(dino_map))
    legacy_clip = clip_idx.with_name("faiss_clip_map.npy")
    print("legacy_clip_map exists", legacy_clip.exists(), inspect_faiss_id_map(legacy_clip) if legacy_clip.exists() else None)

    con = sqlite3.connect(f"file:{Path(s.db_path).as_posix()}?mode=ro", uri=True)
    clip_db = con.execute(
        "SELECT COUNT(*) FROM features WHERE clip_embedding IS NOT NULL AND length(clip_embedding)>0"
    ).fetchone()[0]
    dino_db = con.execute(
        "SELECT COUNT(*) FROM features WHERE dino_embedding IS NOT NULL AND length(dino_embedding)>0"
    ).fetchone()[0]
    print("db clip_embedding", clip_db, "dino_embedding", dino_db)

    if not HAS_FAISS or faiss is None:
        print("no faiss")
        return
    clip_index = faiss.read_index(str(clip_idx)) if clip_idx.exists() else None
    dino_index = faiss.read_index(str(dino_idx)) if dino_idx.exists() else None
    print("clip ntotal", getattr(clip_index, "ntotal", None), "d", getattr(clip_index, "d", None))
    print("dino ntotal", getattr(dino_index, "ntotal", None), "d", getattr(dino_index, "d", None))

    dino_ids = []
    try:
        dino_ids = load_faiss_id_map(dino_map)
        print("dino_ids", len(dino_ids), "unique", len(set(dino_ids)), "head", dino_ids[:5], "tail", dino_ids[-5:])
    except Exception as exc:
        print("dino map load fail", exc)

    # Sample: does FAISS clip[i] match DB embedding for dino_ids[i]?
    if clip_index is None or not dino_ids:
        return
    n = int(clip_index.ntotal)
    print("len_dino_ids vs clip_ntotal", len(dino_ids), n)
    sample_i = [0, 1, 2, n // 2, n - 1] if n > 5 else list(range(n))
    sample_i = [i for i in sample_i if i < n and i < len(dino_ids)]
    hits = 0
    checked = 0
    for i in sample_i:
        fid = int(dino_ids[i])
        row = con.execute(
            "SELECT clip_embedding FROM features WHERE file_id=?", (fid,)
        ).fetchone()
        if not row or not row[0]:
            print("i", i, "fid", fid, "no db clip")
            continue
        dbv = np.frombuffer(row[0], dtype=np.float32)
        rec = clip_index.reconstruct(i).astype(np.float32)
        if dbv.size != rec.size:
            print("i", i, "fid", fid, "dim mismatch", dbv.size, rec.size)
            continue
        dbn = dbv / max(np.linalg.norm(dbv), 1e-8)
        recn = rec / max(np.linalg.norm(rec), 1e-8)
        sim = float(np.dot(dbn, recn))
        checked += 1
        ok = sim >= 0.995
        hits += int(ok)
        print("i", i, "fid", fid, "cosine", round(sim, 6), "match", ok)
    print("sample_hits", hits, "/", checked)

    # Also check if CLIP FAISS order matches DB clip rows order by file_id
    ids_by_file = [
        int(r[0])
        for r in con.execute(
            "SELECT file_id FROM features WHERE clip_embedding IS NOT NULL AND length(clip_embedding)>0 ORDER BY file_id"
        )
    ]
    print("db clip ids ordered by file_id", len(ids_by_file), "eq dino map", ids_by_file[:5] == dino_ids[:5] if dino_ids else None)
    con.close()


if __name__ == "__main__":
    main()
