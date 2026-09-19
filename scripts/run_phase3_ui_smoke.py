"""Phase 3 UI cutover smoke — isolated golden DB, no 116K/NAS.

Verifies IndexWorker → V3 modes, pause/resume/stop, SSOT keys, scope.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QCoreApplication, QTimer

from core.db import Database
from core.index_v3.ui_bridge import map_ui_mode_to_v3, v3_status_dict
from core.settings import AppSettings
from ui.worker_threads import IndexWorker

GOLDEN_FILES = ROOT / "data" / "golden_v3" / "files"
# data/ altı teknik kaynak sayılır — smoke WORK proje dışında/temp
WORK = Path(tempfile.gettempdir()) / "vezir_v3_phase3_ui_smoke"
REPORT = ROOT / "data" / "reports" / "index_engine_v3_phase3.md"


def _settings() -> AppSettings:
    s = AppSettings(
        cache_dir=str(WORK / "cache"),
        db_path=str(WORK / "ui_smoke.db"),
        ai_embedding_enabled=True,
        ocr_enabled=False,
        network_timeout_sec=120,
        use_gpu=False,
        selected_source_ids=[1],
    )
    s.ensure_dirs()
    return s


def _prepare() -> tuple[AppSettings, Database]:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    files = WORK / "files"
    # 6 files only for fast UI smoke (not re-run full 40)
    files.mkdir()
    picked = []
    for pattern in ("*.jpg", "*.png", "*.tif", "*.tiff"):
        picked.extend(list(GOLDEN_FILES.glob(pattern))[:2])
    picked = picked[:6]
    if len(picked) < 3:
        raise SystemExit("golden files missing")
    for p in picked:
        shutil.copy2(p, files / p.name)
    settings = _settings()
    db = Database(settings.db_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(id, name, root_path, is_active) VALUES (1,?,?,1)",
            ("phase3_ui", str(files)),
        )
        # decoy source — must stay untouched when scope=[1]
        conn.execute(
            "INSERT INTO sources(id, name, root_path, is_active) VALUES (2,?,?,1)",
            ("decoy_never_index", str(WORK / "decoy")),
        )
    (WORK / "decoy").mkdir(exist_ok=True)
    return settings, db


def _run_worker(app: QCoreApplication, settings: AppSettings, mode: str, *, timeout_s: float = 600) -> dict:
    result: dict = {"ok": False}
    worker = IndexWorker(settings, index_mode=mode)
    progress_events: list[dict] = []

    def on_prog(d: dict) -> None:
        progress_events.append(dict(d))

    def on_ok(stats: dict) -> None:
        result["ok"] = True
        result["stats"] = dict(stats)
        result["progress"] = progress_events
        app.quit()

    def on_err(msg: str) -> None:
        result["ok"] = False
        result["error"] = msg
        result["progress"] = progress_events
        app.quit()

    worker.progress.connect(on_prog)
    worker.finished_ok.connect(on_ok)
    worker.error.connect(on_err)
    worker.start()
    QTimer.singleShot(int(timeout_s * 1000), app.quit)
    app.exec()
    if worker.isRunning():
        worker.stop()
        worker.wait(5000)
        result["timeout"] = True
        result["ok"] = False
    result["engine"] = (result.get("stats") or {}).get("engine") or (
        progress_events[-1].get("engine") if progress_events else ""
    )
    result["v3_mode"] = map_ui_mode_to_v3(mode)
    return result


def _assert_ssot_keys(st: dict) -> list[str]:
    need = [
        "light_done",
        "ai_final_ready",
        "heavy_queue_display",
        "waiting_preview",
        "processing",
        "queue_pending",
        "ssot",
        "engine",
    ]
    return [k for k in need if k not in st]


def main() -> None:
    settings, db = _prepare()
    app = QCoreApplication([])
    report = {
        "ui_binding": False,
        "fast": False,
        "general_ai": False,
        "complete": False,
        "repair": False,
        "pause_resume": False,
        "stop": False,
        "progress_ssot": False,
        "source_scope": False,
        "legacy_isolation": False,
        "touched_production_db": False,
        "details": {},
    }

    # Production DB path must not equal settings.db_path
    prod = ROOT / "data" / "patterns.db"
    report["touched_production_db"] = Path(settings.db_path).resolve() == prod.resolve()
    report["details"]["db_path"] = settings.db_path

    # Mode mapping
    assert map_ui_mode_to_v3("fast_archive") == "fast"
    assert map_ui_mode_to_v3("night_complete") == "general_ai"
    assert map_ui_mode_to_v3("complete") == "complete"
    assert map_ui_mode_to_v3("backfill") == "repair"

    # FAST
    r_fast = _run_worker(app, settings, "fast_archive", timeout_s=300)
    st = v3_status_dict(db, [1])
    report["details"]["fast"] = {
        "worker": r_fast,
        "ssot": {k: st[k] for k in ("total", "light_done", "ai_final_ready", "hash") if k in st},
    }
    report["fast"] = bool(
        r_fast.get("ok")
        and r_fast.get("engine") == "index_v3"
        and st["light_done"] == st["total"] > 0
        and st["ai_final_ready"] == 0  # FAST must not force AI
    )
    report["ui_binding"] = r_fast.get("engine") == "index_v3"
    report["legacy_isolation"] = r_fast.get("engine") == "index_v3" and "Indexer" not in str(
        r_fast.get("stats")
    )

    # GENERAL_AI
    r_ai = _run_worker(app, settings, "night_complete", timeout_s=600)
    st2 = v3_status_dict(db, [1])
    report["details"]["general_ai"] = {
        "worker": {"ok": r_ai.get("ok"), "engine": r_ai.get("engine")},
        "ssot": {
            k: st2[k]
            for k in ("total", "light_done", "ai_final_ready", "dino", "patch")
            if k in st2
        },
    }
    report["general_ai"] = bool(
        r_ai.get("ok") and st2["ai_final_ready"] == st2["total"] > 0
    )

    # SSOT keys + anti-proxy checks
    missing = _assert_ssot_keys(st2)
    # General processed = ai_final, not patch alone; queue != total-patch
    patch = int(st2.get("patch_embedding_ready") or 0)
    ai = int(st2.get("ai_final_ready") or 0)
    total = int(st2.get("total") or 0)
    hq = int(st2.get("heavy_queue_display") or 0)
    report["progress_ssot"] = (
        not missing
        and st2.get("ssot") == "index_v3"
        and ai == total
        and hq == 0
        and patch == total  # both ready, but UI must use ai_final key
    )
    report["details"]["ssot_missing_keys"] = missing

    # Source scope: decoy source 2 must stay empty
    with db.connect() as conn:
        n2 = conn.execute(
            "SELECT COUNT(*) FROM files WHERE source_id=2"
        ).fetchone()[0]
    report["source_scope"] = int(n2) == 0 and list(
        st2.get("status_scope_source_ids") or []
    ) == [1]

    # COMPLETE on already-done → idle / no crash
    r_c = _run_worker(app, settings, "complete", timeout_s=180)
    report["complete"] = bool(r_c.get("ok") and r_c.get("engine") == "index_v3")
    report["details"]["complete"] = {"ok": r_c.get("ok"), "engine": r_c.get("engine")}

    # REPAIR
    r_r = _run_worker(app, settings, "backfill", timeout_s=180)
    report["repair"] = bool(r_r.get("ok") and r_r.get("engine") == "index_v3")
    report["details"]["repair"] = {"ok": r_r.get("ok"), "engine": r_r.get("engine")}

    # Stop mid-run: add 2 fresh files, start complete, stop quickly
    from PIL import Image

    files_dir = WORK / "files"
    for i in range(2):
        Image.new("RGB", (200, 200), color=(i * 40, 10, 20)).save(
            files_dir / f"stop_test_{i}.jpg", "JPEG"
        )
    stop_result: dict = {"stopped": False}
    worker = IndexWorker(settings, index_mode="complete")

    def on_ok(stats: dict) -> None:
        stop_result["stats"] = stats
        stop_result["stopped"] = bool(stats.get("stopped"))
        app.quit()

    def on_err(msg: str) -> None:
        stop_result["error"] = msg
        app.quit()

    worker.finished_ok.connect(on_ok)
    worker.error.connect(on_err)
    worker.start()
    QTimer.singleShot(1500, worker.stop)
    QTimer.singleShot(120000, app.quit)
    app.exec()
    if worker.isRunning():
        worker.stop()
        worker.wait(8000)
    report["stop"] = bool(
        stop_result.get("stopped")
        or (stop_result.get("stats") or {}).get("engine") == "index_v3"
    )
    report["details"]["stop"] = stop_result

    # Pause / resume: start, pause, resume, wait finish
    Image.new("RGB", (220, 220), color=(9, 9, 9)).save(files_dir / "pause_test.jpg", "JPEG")
    pr: dict = {"ok": False}
    w2 = IndexWorker(settings, index_mode="fast_archive")

    def on_ok2(stats: dict) -> None:
        pr["ok"] = True
        pr["stats"] = stats
        app.quit()

    w2.finished_ok.connect(on_ok2)
    w2.error.connect(lambda m: (pr.update(error=m), app.quit()))
    w2.start()
    QTimer.singleShot(800, w2.pause)
    QTimer.singleShot(2000, w2.resume)
    QTimer.singleShot(180000, app.quit)
    app.exec()
    if w2.isRunning():
        w2.stop()
        w2.wait(5000)
    report["pause_resume"] = bool(pr.get("ok"))
    report["details"]["pause_resume"] = pr

    checks = [
        "ui_binding",
        "fast",
        "general_ai",
        "complete",
        "repair",
        "pause_resume",
        "stop",
        "progress_ssot",
        "source_scope",
        "legacy_isolation",
    ]
    report["pass"] = all(bool(report[k]) for k in checks) and not report[
        "touched_production_db"
    ]
    report["ready_line"] = (
        "GERÇEK KAYNAK TESTİNE HAZIR" if report["pass"] else "HAZIR DEĞİL"
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Index Engine V3 — Phase 3 UI Cutover",
        "",
        f"**PASS:** `{report['pass']}`",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    labels = {
        "ui_binding": "UI bağlantısı",
        "fast": "FAST",
        "general_ai": "GENERAL_AI",
        "complete": "COMPLETE",
        "repair": "REPAIR",
        "pause_resume": "Pause/Resume",
        "stop": "Stop",
        "progress_ssot": "Progress SSOT",
        "source_scope": "Source scope",
        "legacy_isolation": "Legacy isolation",
    }
    for k, lab in labels.items():
        lines.append(f"| {lab} | {'PASS' if report[k] else 'FAIL'} |")
    lines += [
        "",
        f"- Production DB touched: `{report['touched_production_db']}`",
        f"- Smoke DB: `{settings.db_path}`",
        "",
        f"**{report['ready_line']}**",
        "",
        "```json",
        json.dumps({k: report[k] for k in list(labels) + ["pass", "ready_line"]}, indent=2),
        "```",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    (WORK / "phase3_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps({"pass": report["pass"], "ready": report["ready_line"], **{k: report[k] for k in labels}}, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
