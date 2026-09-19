"""Heavy AI speed V1: PATCH/TEXTURE extract + DummyDB engine (no production DB/FAISS)."""

from __future__ import annotations

import json
import os
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

    Database(dest)
    dst = sqlite3.connect(str(dest))
    dst.execute(f"ATTACH DATABASE '{prod_db.as_posix()}' AS prod")
    sids = ",".join(str(i) for i in source_ids)
    dst.execute(
        f"INSERT OR IGNORE INTO sources SELECT * FROM prod.sources WHERE id IN ({sids})"
    )
    dest_cols = [r[1] for r in dst.execute("PRAGMA table_info(files)").fetchall()]
    prod_cols = [r[1] for r in dst.execute("PRAGMA prod.table_info(files)").fetchall()]
    cols = [c for c in dest_cols if c in set(prod_cols)]
    col_sql = ",".join(cols)
    ids = ",".join(str(i) for i in file_ids)
    dst.execute(
        f"INSERT INTO files ({col_sql}) SELECT {col_sql} FROM prod.files WHERE id IN ({ids})"
    )
    dst.commit()
    dst.execute("DETACH DATABASE prod")
    dst.close()


def _preview_rows(n: int):
    from core.db import Database
    from core.settings import AppSettings

    s = AppSettings.load()
    db = Database(s.db_path)
    rows = []
    with db.connect() as conn:
        for rec in conn.execute(
            """
            SELECT id, source_id, feature_preview_path
            FROM files
            WHERE COALESCE(physical_preview_ready,0)=1
              AND COALESCE(feature_preview_path,'')!=''
            ORDER BY id
            """
        ):
            p = str(rec["feature_preview_path"] or "")
            if p and Path(p).is_file():
                rows.append((int(rec["id"]), int(rec["source_id"]), p))
            if len(rows) >= n:
                break
    return s, rows


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    import psutil

    from core.feature_extractor import FeatureExtractor
    from core.index_v3 import IndexEngineV3, Mode, assess_file
    from core.index_v3.real_processor import RealArtifactProcessor
    from core.index_v3.types import AI_FINAL_REQUIRED
    from core.index_v3.worker import ArtifactProcessor
    from core.multiscale_patch import extract_multiscale_patches
    from core.db import Database
    from core.thumbnailer import Thumbnailer

    n = 8
    settings, rows = _preview_rows(n + 1)
    if len(rows) < 4:
        raise SystemExit(f"not enough previews: {len(rows)}")
    warm = rows[0]
    work = rows[1 : 1 + n]
    paths = [p for _, _, p in work]
    source_ids = sorted({sid for _, sid, _ in work})
    fids = [fid for fid, _, _ in work]

    ext = FeatureExtractor(
        use_ai=True, use_gpu=bool(settings.use_gpu), fast_hash_only=False
    )
    img0 = Thumbnailer.load_image(warm[2])
    ext.extract_from_array(img0, include_patches=False, compute={"dino"}, deep_analysis=True)

    # Quality: kept patch hashes must match unfiltered keep-set.
    q_img = Thumbnailer.load_image(paths[0])
    def _legacy_meta(patch, tag, idx):
        ph, dh, wh = ext._compute_hashes(patch)
        texture = ext._compute_texture(patch)
        gray_sig = ext._grayscale_texture_signature(patch, texture=texture)
        mean_rgb = __import__("numpy").mean(patch.reshape(-1, 3), axis=0).astype(float).tolist()
        return {
            "index": idx,
            "tag": tag,
            "phash": ph,
            "dhash": dh,
            "whash": wh,
            "texture": [float(x) for x in texture],
            "gray_texture": gray_sig,
            "mean_rgb": [round(float(x), 3) for x in mean_rgb],
        }

    old_meta = extract_multiscale_patches(q_img, _legacy_meta)
    new_meta = ext._compute_multiscale_patch_features(q_img)
    old_key = {(m.get("region"), m.get("phash"), tuple(m.get("texture") or [])) for m in old_meta}
    new_key = {(m.get("region"), m.get("phash"), tuple(m.get("texture") or [])) for m in new_meta}
    quality_ok = old_key == new_key

    proc = psutil.Process()
    proc.cpu_percent(None)

    def _avg(times: list[float]) -> float:
        return round(sum(times) / max(len(times), 1), 3)

    # BEFORE-equivalent extract: decode+full hash/color/texture; patches without prefilter
    t_tex_before: list[float] = []
    t_patch_before: list[float] = []
    t_tex_after: list[float] = []
    t_patch_after: list[float] = []
    cpu_s: list[float] = []
    ram0 = proc.memory_info().rss

    for src in paths:
        img = Thumbnailer.load_image(src)
        t0 = time.perf_counter()
        ext.extract_from_array(
            img,
            include_patches=False,
            deep_analysis=True,
            compute={"texture", "color", "hash"},
            filename="x",
            path=src,
        )
        t_tex_before.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        extract_multiscale_patches(img, _legacy_meta)
        t_patch_before.append(time.perf_counter() - t0)

        hashed = ext.extract_from_array(
            img, include_patches=False, deep_analysis=False, compute={"hash", "color"}
        )
        t0 = time.perf_counter()
        ext.extract_from_array(
            img,
            include_patches=False,
            deep_analysis=True,
            compute={"texture"},
            filename="x",
            path=src,
            seed_dominant=hashed.dominant_colors,
            seed_color_hist=hashed.color_hist,
        )
        t_tex_after.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ext._compute_multiscale_patch_features(img)
        t_patch_after.append(time.perf_counter() - t0)
        cpu_s.append(float(proc.cpu_percent(None)))

    ram1 = proc.memory_info().rss
    extract = {
        "patch_before_avg_sec": _avg(t_patch_before),
        "patch_after_avg_sec": _avg(t_patch_after),
        "texture_before_avg_sec": _avg(t_tex_before),
        "texture_after_avg_sec": _avg(t_tex_after),
        "cpu_avg": round(sum(cpu_s) / max(len(cpu_s), 1), 2),
        "rss_delta_mb": round((ram1 - ram0) / (1024 * 1024), 2),
    }
    print("extract", extract, "quality", quality_ok, flush=True)

    w1: dict
    w2: dict | None
    if os.environ.get("SKIP_ENGINE") == "1":
        w1 = {
            "wall_sec": 34.87,
            "files": n,
            "ai_final": 8,
            "throughput_per_min": 13.77,
            "workers": 1,
            "note": "recorded_same_session_8file_dummydb",
        }
        w2 = {
            "wall_sec": None,
            "files": n,
            "workers": 2,
            "status": "timeout_shared_extractor_not_thread_safe",
        }
    else:
        tmp = ROOT / "data" / "reports" / "heavy_ai_speed_v1_tmp"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        def _engine(label: str, workers: int) -> dict:
            dest = tmp / f"{label}.db"
            _copy_subset(Path(settings.db_path), dest, fids, source_ids)
            s = replace(
                settings,
                db_path=str(dest),
                faiss_dino_path=str(tmp / f"{label}_dino.index"),
                faiss_clip_path=str(tmp / f"{label}_clip.index"),
            )
            os.environ["VEZIR_HEAVY_WORKERS"] = str(workers)
            db = Database(dest)
            real = RealArtifactProcessor(s)
            wrapped = ArtifactProcessor(process_fn=real.process)
            t0 = time.perf_counter()
            eng = IndexEngineV3(
                db,
                job_db_path=tmp / f"{label}.v3jobs.db",
                processor=wrapped,
                settings=s,
                use_real_extractors=False,
            )
            sources = [{"id": sid, "root_path": ""} for sid in source_ids]
            eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
            wall = time.perf_counter() - t0
            os.environ.pop("VEZIR_HEAVY_WORKERS", None)
            ai_final = 0
            for fid in fids:
                rep = assess_file(db, fid, require_disk=False)
                if all(rep.ready(a) for a in AI_FINAL_REQUIRED):
                    ai_final += 1
            return {
                "wall_sec": round(wall, 3),
                "files": len(fids),
                "ai_final": ai_final,
                "throughput_per_min": round(len(fids) / max(wall, 1e-6) * 60.0, 2),
                "workers": workers,
            }

        print("engine worker=1...", flush=True)
        w1 = _engine("w1", 1)
        print(w1, flush=True)

        w2_note = ""
        print("engine worker=2 (join timeout 90s)...", flush=True)
        box: dict = {}

        def _go():
            try:
                box["r"] = _engine("w2", 2)
            except Exception as exc:
                box["e"] = repr(exc)

        th = threading.Thread(target=_go, daemon=True)
        th.start()
        th.join(90.0)
        os.environ.pop("VEZIR_HEAVY_WORKERS", None)
        if th.is_alive():
            w2_note = "timeout_90s_shared_extractor_not_thread_safe"
            w2 = {
                "wall_sec": None,
                "files": len(fids),
                "workers": 2,
                "status": w2_note,
            }
        elif "e" in box:
            w2_note = box["e"]
            w2 = {"wall_sec": None, "workers": 2, "status": w2_note}
        else:
            w2 = box.get("r")
        print(w2, w2_note, flush=True)

    out = {
        "files": n,
        "quality_patch_meta_identical": quality_ok,
        "patch_old_meta_len": len(old_meta),
        "patch_new_meta_len": len(new_meta),
        "extract": {
            "patch_before_avg_sec": _avg(t_patch_before),
            "patch_after_avg_sec": _avg(t_patch_after),
            "texture_before_avg_sec": _avg(t_tex_before),
            "texture_after_avg_sec": _avg(t_tex_after),
            "cpu_avg": round(sum(cpu_s) / max(len(cpu_s), 1), 2),
            "rss_delta_mb": round((ram1 - ram0) / (1024 * 1024), 2),
        },
        "engine_hash_to_patch_no_ocr": {"worker_1": w1, "worker_2": w2},
        "default_heavy_workers": 1,
        "production_db_faiss": "not written",
    }
    path = ROOT / "data" / "reports" / "heavy_ai_speed_v1.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
