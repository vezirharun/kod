"""Isolated engine A/B: worker_count=1 vs 4 on 58 real preview files.

Writes only to a temp DB. Production jobs/FAISS/settings are not modified.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _copy_subset(prod_db: Path, dest: Path, file_ids: list[int], source_ids: list[int]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    from core.db import Database

    Database(dest)  # schema
    src = sqlite3.connect(str(prod_db))
    dst = sqlite3.connect(str(dest))
    dst.execute(f"ATTACH DATABASE '{prod_db.as_posix()}' AS prod")
    sids = ",".join(str(i) for i in source_ids)
    dst.execute(
        f"INSERT OR IGNORE INTO sources SELECT * FROM prod.sources WHERE id IN ({sids})"
    )
    ids = ",".join(str(i) for i in file_ids)
    dst.execute(f"INSERT INTO files SELECT * FROM prod.files WHERE id IN ({ids})")
    dst.commit()
    dst.execute("DETACH DATABASE prod")
    dst.close()
    src.close()


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from core.db import Database
    from core.index_v3 import IndexEngineV3, Mode, assess_file
    from core.index_v3.real_processor import RealArtifactProcessor
    from core.index_v3.types import AI_FINAL_REQUIRED
    from core.index_v3.worker import ArtifactProcessor
    from core.settings import AppSettings

    settings = AppSettings.load()
    n_total = 60
    db0 = Database(settings.db_path)
    rows = []
    with db0.connect() as conn:
        cur = conn.execute(
            """
            SELECT id, source_id, feature_preview_path
            FROM files
            WHERE source_id IN (8,9)
              AND COALESCE(physical_preview_ready,0)=1
              AND COALESCE(feature_preview_path,'')!=''
            ORDER BY id
            """
        )
        for rec in cur:
            prev = str(rec["feature_preview_path"] or "")
            if not prev or not Path(prev).is_file():
                continue
            rows.append((int(rec["id"]), int(rec["source_id"])))
            if len(rows) >= n_total:
                break
    if len(rows) < 10:
        raise SystemExit(f"not enough preview files: {len(rows)}")
    warmup_ids = [r[0] for r in rows[:2]]
    work_ids = [r[0] for r in rows[2:]]
    source_ids = sorted({r[1] for r in rows})
    print(f"batch {len(work_ids)} warmup {len(warmup_ids)}", flush=True)

    tmp = ROOT / "data" / "reports" / "heavy_stage_bench_tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    proc = RealArtifactProcessor(settings)
    wrapped = ArtifactProcessor(process_fn=proc.process)

    def _run(label: str, worker_count: int, fids: list[int]) -> dict:
        dest = tmp / f"{label}.db"
        _copy_subset(Path(settings.db_path), dest, fids, source_ids)
        s = replace(
            settings,
            worker_count=worker_count,
            db_path=str(dest),
            faiss_dino_path=str(tmp / f"{label}_dino.index"),
            faiss_clip_path=str(tmp / f"{label}_clip.index"),
        )
        db = Database(dest)
        inflight = [0, 0]
        lock = threading.Lock()
        cpu_samples: list[float] = []
        try:
            import psutil

            p = psutil.Process()
            p.cpu_percent(None)
        except Exception:
            p = None

        def cb(info=None):
            if not isinstance(info, dict):
                return
            phase = str(info.get("phase") or "")
            with lock:
                if phase == "claimed":
                    inflight[0] += 1
                    inflight[1] = max(inflight[1], inflight[0])
                    if p is not None:
                        try:
                            cpu_samples.append(float(p.cpu_percent(None)))
                        except Exception:
                            pass
                elif phase == "idle":
                    inflight[0] = max(0, inflight[0] - 1)

        t0 = time.perf_counter()
        eng = IndexEngineV3(
            db,
            job_db_path=tmp / f"{label}.v3jobs.db",
            processor=wrapped,
            settings=s,
            use_real_extractors=False,
        )
        sources = [{"id": sid, "root_path": ""} for sid in source_ids]
        eng.run(
            mode=Mode.GENERAL_AI,
            sources=sources,
            walk_disk=False,
            progress_callback=cb,
        )
        wall = time.perf_counter() - t0
        ready = 0
        ai_final = 0
        for fid in fids:
            rep = assess_file(db, fid, require_disk=False)
            if all(rep.ready(a) for a in AI_FINAL_REQUIRED):
                ai_final += 1
            ready += 1
        return {
            "wall_sec": round(wall, 3),
            "wall_min": round(wall / 60.0, 3),
            "files": len(fids),
            "ai_final": ai_final,
            "max_concurrent": inflight[1],
            "throughput_per_min": round(len(fids) / max(wall, 1e-6) * 60.0, 2),
            "cpu_avg": round(sum(cpu_samples) / len(cpu_samples), 2) if cpu_samples else None,
        }

    print("warmup...", flush=True)
    _run("warmup", 1, warmup_ids)
    print("serial worker_count=1...", flush=True)
    a = _run("serial", 1, work_ids)
    print("serial", a, flush=True)
    print("stage worker_count=4...", flush=True)
    b = _run("stage", 4, work_ids)
    print("stage", b, flush=True)
    speedup = (a["wall_sec"] / b["wall_sec"] - 1.0) * 100.0 if b["wall_sec"] else 0
    out = {
        "baseline_isolated_script_sec": 414.141,
        "engine_serial": a,
        "engine_stage": b,
        "speedup_vs_engine_serial_pct": round(speedup, 1),
        "speedup_vs_script_baseline_pct": round(
            (414.141 / b["wall_sec"] - 1.0) * 100.0, 1
        )
        if b["wall_sec"]
        else None,
    }
    path = ROOT / "data" / "reports" / "general_ai_stage_engine_bench.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
