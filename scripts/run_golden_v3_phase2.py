"""Index Engine V3 Phase 2 — golden 100-file e2e with REAL extractors.

Isolated DB under data/golden_v3/. Does NOT touch production patterns.db.
Prefer project .venv (libvips via sitecustomize).
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode, count_progress
from core.settings import AppSettings

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden_v3"
FILES = GOLDEN / "files"
WORK = GOLDEN / "work"
REPORT = GOLDEN / "phase2_report.json"
REPORT_MD = ROOT / "data" / "reports" / "index_engine_v3_phase2.md"


def _settings(work: Path) -> AppSettings:
    s = AppSettings(
        cache_dir=str(work / "cache"),
        db_path=str(work / "golden.db"),
        ai_embedding_enabled=True,
        ocr_enabled=False,
        network_timeout_sec=180,
        use_gpu=False,
    )
    s.ensure_dirs()
    return s


def _prepare_db(settings: AppSettings, files_root: Path) -> Database:
    db = Database(settings.db_path)
    with db.connect() as conn:
        conn.execute("DELETE FROM sources")
        conn.execute(
            "INSERT INTO sources(id, name, root_path, is_active) VALUES (1,?,?,1)",
            ("golden_v3", str(files_root)),
        )
    return db


def _stage_counts(db: Database) -> dict:
    p = count_progress(db, source_ids=[1])
    return {
        "total": p.total,
        "light": p.light_complete,
        "ai_final": p.ai_final,
        "thumbnail": p.stage("thumbnail").completed,
        "preview": p.stage("preview").completed,
        "hash": p.stage("hash").completed,
        "metadata": p.stage("metadata").completed,
        "dino": p.stage("dino").completed,
        "clip": p.stage("clip").completed,
        "texture": p.stage("texture").completed,
        "semantic": p.stage("semantic").completed,
        "dna": p.stage("dna").completed,
        "patch": p.stage("patch").completed,
    }


def _break_artifacts(db: Database) -> dict:
    with db.connect() as conn:
        ids = [int(r[0]) for r in conn.execute("SELECT id FROM files ORDER BY id")]
    plan = {
        "dino": ids[0:2],
        "patch": ids[2:5],
        "semantic": ids[5:7],
        "dna": ids[7:8],
        "preview": ids[8:10],
    }
    with db.connect() as conn:
        for fid in plan["dino"]:
            conn.execute(
                "UPDATE features SET dino_embedding=NULL WHERE file_id=?", (fid,)
            )
        for fid in plan["patch"]:
            conn.execute(
                "UPDATE features SET patch_embeddings='[]' WHERE file_id=?", (fid,)
            )
        for fid in plan["semantic"]:
            row = conn.execute(
                "SELECT texture_map FROM features WHERE file_id=?", (fid,)
            ).fetchone()
            tm = json.loads(row[0] or "{}") if row else {}
            tm.pop("semantic_tags", None)
            conn.execute(
                "UPDATE features SET texture_map=? WHERE file_id=?",
                (json.dumps(tm), fid),
            )
        for fid in plan["dna"]:
            row = conn.execute(
                "SELECT texture_map FROM features WHERE file_id=?", (fid,)
            ).fetchone()
            tm = json.loads(row[0] or "{}") if row else {}
            tm.pop("pattern_dna", None)
            conn.execute(
                "UPDATE features SET texture_map=? WHERE file_id=?",
                (json.dumps(tm), fid),
            )
        for fid in plan["preview"]:
            conn.execute(
                "UPDATE files SET feature_preview_path='', physical_preview_ready=0 WHERE id=?",
                (fid,),
            )
    return {k: len(v) for k, v in plan.items()}


def _write_md(report: dict, manifest: dict) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Index Engine V3 — Phase 2 Report",
        "",
        f"**PASS:** `{report.get('pass')}`",
        "",
        "## Golden dataset",
        f"- total: {manifest.get('total')}",
        f"- by_kind: `{manifest.get('by_kind')}`",
        f"- copied/synthetic: {manifest.get('copied')}/{manifest.get('synthetic')}",
        f"- large_tiff_ge_5mb: {manifest.get('large_tiff_ge_5mb')}",
        "",
        "## Timings (sec)",
        "```json",
        json.dumps(report.get("timings_sec") or {}, indent=2),
        "```",
        "",
        "## Phase results",
    ]
    for name, phase in (report.get("phases") or {}).items():
        lines.append(f"### {name}")
        lines.append(f"- ok: `{phase.get('ok')}`")
        if "counts" in phase:
            lines.append(f"- counts: `{phase['counts']}`")
        if "worker" in phase:
            w = phase["worker"]
            lines.append(
                f"- processed/completed/failed/timeouts: "
                f"{w.get('processed')}/{w.get('completed')}/{w.get('failed')}/{w.get('timeouts')}"
            )
            lines.append(f"- errors_by_ext: `{w.get('errors_by_ext')}`")
            lines.append(f"- model_calls: `{w.get('model_calls')}`")
        if "model_calls" in phase and "worker" not in phase:
            lines.append(f"- model_calls: `{phase.get('model_calls')}`")
        if "broken" in phase:
            lines.append(f"- broken: `{phase.get('broken')}`")
        lines.append("")
    if report.get("errors"):
        lines.append("## Errors")
        lines.append("```")
        lines.extend(str(e) for e in report["errors"])
        lines.append("```")
    lines.append("")
    lines.append("## Decoder")
    lines.append(f"- HAS_VIPS / pyvips used in this run: `{report.get('has_vips')}`")
    lines.append(f"- pyvips/libvips: `{report.get('vips_version')}`")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not FILES.is_dir() or not any(FILES.iterdir()):
        raise SystemExit("Golden files missing — run scripts/build_golden_v3.py first")

    manifest = {}
    man_path = GOLDEN / "manifest.json"
    if man_path.is_file():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))

    if WORK.exists():
        for _ in range(5):
            try:
                shutil.rmtree(WORK)
                break
            except OSError:
                time.sleep(1.0)
        if WORK.exists():
            # last resort: rename aside
            junk = WORK.with_name(f"work_old_{int(time.time())}")
            try:
                WORK.rename(junk)
            except OSError as exc:
                raise SystemExit(f"Cannot clear work dir: {exc}") from exc
    WORK.mkdir(parents=True, exist_ok=True)
    cache_files = WORK / "files"
    shutil.copytree(FILES, cache_files)

    settings = _settings(WORK)
    db = _prepare_db(settings, cache_files)
    engine = IndexEngineV3(
        db,
        job_db_path=WORK / "jobs.db",
        settings=settings,
        use_real_extractors=True,
    )
    sources = [{"id": 1, "root_path": str(cache_files)}]
    report: dict = {
        "phases": {},
        "errors": [],
        "timings_sec": {},
        "pass": False,
        "has_vips": False,
        "vips_version": "",
    }
    try:
        from core.thumbnailer import HAS_VIPS
        import pyvips

        report["has_vips"] = bool(HAS_VIPS)
        report["vips_version"] = (
            f"pyvips={pyvips.__version__} libvips="
            f"{pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)}"
        )
    except Exception as exc:
        report["vips_version"] = f"unavailable:{exc!r}"

    try:
        t0 = time.perf_counter()
        r1 = engine.run(mode=Mode.FAST, sources=sources)
        report["timings_sec"]["fast"] = round(time.perf_counter() - t0, 2)
        c1 = _stage_counts(db)
        report["phases"]["fast"] = {
            "counts": c1,
            "worker": r1.worker,
            "ok": c1["light"] == c1["total"] == 40
            and c1["hash"] == 40
            and c1["thumbnail"] == 40
            and c1["preview"] == 40
            and c1["metadata"] == 40,
        }
        if not report["phases"]["fast"]["ok"]:
            raise RuntimeError(
                f"FAST gate failed (need hash/preview/light 40/40): {c1}"
            )

        t0 = time.perf_counter()
        r2 = engine.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
        report["timings_sec"]["general_ai"] = round(time.perf_counter() - t0, 2)
        c2 = _stage_counts(db)
        report["phases"]["general_ai"] = {
            "counts": c2,
            "worker": r2.worker,
            "ok": (
                c2["ai_final"] == 40
                and c2["dino"] == 40
                and c2["clip"] == 40
                and c2["texture"] == 40
                and c2["semantic"] == 40
                and c2["dna"] == 40
                and c2["patch"] == 40
            ),
        }

        broken = _break_artifacts(db)
        t0 = time.perf_counter()
        engine_r = IndexEngineV3(
            db,
            job_db_path=WORK / "jobs_repair.db",
            settings=settings,
            use_real_extractors=True,
        )
        r3 = engine_r.run(mode=Mode.REPAIR, sources=sources, walk_disk=False)
        report["timings_sec"]["repair"] = round(time.perf_counter() - t0, 2)
        c3 = _stage_counts(db)
        calls = r3.worker.get("model_calls") or {}
        bad = {k: v for k, v in calls.items() if k in ("clip", "texture") and v}
        report["phases"]["repair"] = {
            "broken": broken,
            "counts": c3,
            "worker": r3.worker,
            "model_calls": calls,
            "forbidden_reruns": bad,
            "ok": c3["ai_final"] == 40 and not bad,
        }

        t0 = time.perf_counter()
        r4 = engine_r.run(mode=Mode.REPAIR, sources=sources, walk_disk=False)
        report["timings_sec"]["repair2"] = round(time.perf_counter() - t0, 2)
        report["phases"]["repair2"] = {
            "worker": r4.worker,
            "model_calls": r4.worker.get("model_calls") or {},
            "processed": r4.worker.get("processed", 0),
            "ok": (r4.worker.get("processed", 0) == 0)
            or not (r4.worker.get("model_calls") or {}),
        }

        from PIL import Image

        for i in range(10):
            Image.new("RGB", (320, 240), color=(i * 20, 50, 90)).save(
                cache_files / f"extra_{i:02d}.jpg", "JPEG"
            )
        t0 = time.perf_counter()
        r5 = engine.run(mode=Mode.COMPLETE, sources=sources)
        report["timings_sec"]["new10"] = round(time.perf_counter() - t0, 2)
        c5 = _stage_counts(db)
        report["phases"]["new10"] = {
            "counts": c5,
            "worker": r5.worker,
            "ok": c5["total"] == 50 and c5["ai_final"] == 50,
        }

        extras = sorted(cache_files.glob("extra_*.jpg"))[:5]
        for p in extras:
            p.unlink()
        t0 = time.perf_counter()
        r6 = engine.run(mode=Mode.REPAIR, sources=sources, walk_disk=True)
        report["timings_sec"]["delete5"] = round(time.perf_counter() - t0, 2)
        c6 = _stage_counts(db)
        report["phases"]["delete5"] = {
            "counts": c6,
            "worker": r6.worker,
            "ok": c6["total"] == 45,
        }

        engine.reset_stale()
        before = _stage_counts(db)
        t0 = time.perf_counter()
        r7 = engine.run(mode=Mode.REPAIR, sources=sources, walk_disk=False)
        report["timings_sec"]["restart"] = round(time.perf_counter() - t0, 2)
        after = _stage_counts(db)
        report["phases"]["restart"] = {
            "before": before,
            "after": after,
            "worker": r7.worker,
            "model_calls": r7.worker.get("model_calls") or {},
            "ok": after["ai_final"] == before["ai_final"]
            and (
                r7.worker.get("processed", 0) == 0
                or not (r7.worker.get("model_calls") or {})
            ),
        }

        report["pass"] = all(
            bool(report["phases"][k].get("ok"))
            for k in (
                "fast",
                "general_ai",
                "repair",
                "repair2",
                "new10",
                "delete5",
                "restart",
            )
        )
    except Exception as exc:
        report["errors"].append(repr(exc))
        report["errors"].append(traceback.format_exc())
        report["pass"] = False

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(report, manifest)
    summary = {
        "pass": report["pass"],
        "has_vips": report.get("has_vips"),
        "vips_version": report.get("vips_version"),
        "timings_sec": report["timings_sec"],
        "phases": {k: v.get("ok") for k, v in report["phases"].items()},
    }
    print(json.dumps(summary, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
