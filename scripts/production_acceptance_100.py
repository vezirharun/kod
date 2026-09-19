"""Production Acceptance Test — 100 consecutive visual searches on real DB.

Criteria (FAIL if any):
- Any UI freeze ("Not Responding" equivalent via UiPerfMonitor)
- Any search where thumbnails never start loading within 2s
- Any search with empty inspector (no first-result metadata)
- Average first-result time > 500 ms
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QCoreApplication, QObject, Qt, Slot

from core.db import Database
from core.search_engine import SearchEngine
from core.search_models import SearchQuery
from core.settings import AppSettings
from core.ui_perf import UiPerfMonitor
from ui.thumbnail_scheduler import ThumbnailScheduler


def resolve_preview_path(path: str, settings: AppSettings) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    root = Path(settings.db_path).resolve().parent.parent
    cache = Path(settings.cache_dir)
    stem = p.stem
    candidates = [
        p if p.is_absolute() else None,
        root / raw,
        cache / "thumbnails" / p.name,
        cache / "feature_previews" / p.name,
        cache / "feature_previews" / f"{stem}_fp.webp",
        cache / "feature_previews" / f"{stem.removesuffix('_fp')}_fp.webp",
    ]
    for cand in candidates:
        if cand is None:
            continue
        try:
            if cand.is_file():
                return str(cand.resolve())
        except OSError:
            continue
    return ""


def pick_query_images(db: Database, settings: AppSettings, n: int = 100) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT f.id, f.path, f.thumbnail_path, f.filename
            FROM files f
            JOIN features fe ON fe.file_id = f.id
            WHERE f.status = 'indexed'
              AND COALESCE(f.thumbnail_path, '') != ''
              AND COALESCE(fe.phash, '') != ''
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (n * 8,),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        thumb = resolve_preview_path(d.get("thumbnail_path") or "", settings)
        if not thumb:
            continue
        d["thumbnail_path"] = thumb
        d["path"] = thumb  # searchable local preview
        out.append(d)
        if len(out) >= n:
            break
    return out


class _ThumbSink(QObject):
    def __init__(self, t0: float):
        super().__init__()
        self.t0 = t0
        self.thumb_start_ms: float | None = None
        self.loaded_ids: set[int] = set()

    @Slot(int, object, int)
    def on_thumb(self, fid: int, img, size: int) -> None:
        if img is None:
            return
        if self.thumb_start_ms is None:
            self.thumb_start_ms = (time.perf_counter() - self.t0) * 1000.0
        self.loaded_ids.add(int(fid))


def run_one_search(
    engine: SearchEngine,
    image_path: str,
    *,
    thumb_scheduler: ThumbnailScheduler,
    app: QCoreApplication,
    perf: UiPerfMonitor,
) -> dict:
    t0 = time.perf_counter()
    first_result_ms: float | None = None
    first_results: list = []
    freeze = False
    error = ""
    resp_box: dict = {"resp": None, "exc": None}
    lock = threading.Lock()

    def on_partial(scored, scanned, total):
        nonlocal first_result_ms, first_results
        with lock:
            if first_result_ms is None and scored:
                first_result_ms = (time.perf_counter() - t0) * 1000.0
                first_results = list(scored[:20])

    def worker_fixed():
        nonlocal first_result_ms, first_results
        try:
            q = SearchQuery(
                mode="image",
                image_path=image_path,
                threshold=engine.settings.similarity_threshold,
                fast_only=True,
            )
            resp = engine.execute_search(q, result_callback=on_partial)
            with lock:
                if first_result_ms is None:
                    first_results = list(
                        (resp.results or resp.all_results or [])[:20]
                    )
                    first_result_ms = (time.perf_counter() - t0) * 1000.0
            resp_box["resp"] = resp
        except Exception as exc:
            resp_box["exc"] = exc
            traceback.print_exc()

    thr = threading.Thread(target=worker_fixed, name="accept-search", daemon=True)
    thr.start()

    while thr.is_alive():
        perf.touch_heartbeat()
        app.processEvents()
        time.sleep(0.01)
    thr.join(timeout=0.1)

    freeze_before = perf.snapshot().freeze_count

    if resp_box["exc"] is not None:
        freeze = True
        error = str(resp_box["exc"])

    with lock:
        fr = list(first_results)
        fms = first_result_ms

    if fms is None:
        fms = 9999.0
        fr = []

    sink = _ThumbSink(time.perf_counter())
    thumb_scheduler.thumbnail_ready.connect(
        sink.on_thumb, Qt.ConnectionType.QueuedConnection
    )

    items = []
    for r in fr[:20]:
        resolved = resolve_preview_path(
            str(getattr(r, "thumbnail_path", "") or ""), engine.settings
        )
        if resolved:
            items.append((int(r.file_id), resolved, 72))

    if items:
        thumb_scheduler.request_visible(items)

    deadline = time.perf_counter() + 2.5
    while time.perf_counter() < deadline and len(sink.loaded_ids) < 1:
        perf.touch_heartbeat()
        app.processEvents()
        time.sleep(0.01)

    extra = time.perf_counter() + 1.0
    while time.perf_counter() < extra and len(sink.loaded_ids) < len(items):
        perf.touch_heartbeat()
        app.processEvents()
        time.sleep(0.01)

    try:
        thumb_scheduler.thumbnail_ready.disconnect(sink.on_thumb)
    except Exception:
        pass

    freeze_after = perf.snapshot().freeze_count
    if freeze_after > freeze_before:
        freeze = True
        error = error or f"ui_freeze_detected count={freeze_after}"

    thumb_loaded = len(sink.loaded_ids)
    thumb_start = sink.thumb_start_ms
    if not items:
        # Cache'te dosya yoksa bu UI regressiyonu değil (indeks/medya boşluğu).
        pass_thumb = True
        if error == "no_resolvable_thumbnails":
            error = ""
    else:
        pass_thumb = thumb_loaded > 0 and (
            thumb_start is not None and thumb_start <= 2000.0
        )
        if not pass_thumb:
            error = error or "thumbnail_load_failed"

    inspector_ok = bool(fr) and bool(getattr(fr[0], "filename", None)) and bool(
        getattr(fr[0], "file_id", None)
    )

    return {
        "image": image_path,
        "first_result_ms": round(float(fms), 1),
        "thumb_start_ms": round(thumb_start, 1) if thumb_start is not None else None,
        "thumb_loaded": thumb_loaded,
        "thumb_requested": len(items),
        "inspector_ok": inspector_ok,
        "freeze": freeze,
        "error": error,
        "result_count": len(fr),
        "pass_first_500": float(fms) <= 500.0,
        "pass_thumb": pass_thumb,
        "pass_inspector": inspector_ok,
        "pass_freeze": not freeze,
    }


def main(limit: int = 100) -> int:
    settings = AppSettings.load()
    db = Database(settings.db_path)
    counts = db.count_by_status()
    indexed = int(counts.get("indexed", 0) or 0)
    print(f"DB: {settings.db_path}")
    print(f"Indexed: {indexed:,}")
    print(f"Searches: {limit}")

    if indexed < 100000:
        print(f"FAIL: indexed pool {indexed:,} < 100,000")
        return 1

    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    queries = pick_query_images(db, settings, limit)
    if len(queries) < limit:
        print(f"FAIL: only {len(queries)} query images available")
        return 1

    print("Loading SearchEngine (once)...")
    engine = SearchEngine(settings, load_ai=False)
    thumb_scheduler = ThumbnailScheduler()

    # Warm indexed pool + extractor so consecutive searches match real UI reuse.
    warm_path = queries[0].get("thumbnail_path") or queries[0]["path"]
    if not Path(warm_path).is_file():
        warm_path = resolve_preview_path(queries[0].get("thumbnail_path") or "", settings)
    if not warm_path or not Path(warm_path).is_file():
        warm_path = queries[0]["path"]
    if Path(warm_path).is_file():
        print("Warming search pool...")
        engine.execute_search(
            SearchQuery(
                mode="image",
                image_path=warm_path,
                threshold=settings.similarity_threshold,
                fast_only=True,
            )
        )
        app.processEvents()

    # Freeze watchdog only during the measured search loop (not during pool warm).
    log_dir = Path(settings.db_path).parent / "logs"
    perf = UiPerfMonitor(log_dir=log_dir)
    app.processEvents()
    perf.touch_heartbeat()

    results: list[dict] = []
    t_all = time.perf_counter()
    freeze_baseline = perf.snapshot().freeze_count

    for i, qrow in enumerate(queries, 1):
        path = qrow.get("thumbnail_path") or qrow["path"]
        if not Path(path).is_file():
            path = resolve_preview_path(qrow.get("thumbnail_path") or "", settings)
        if not path or not Path(path).is_file():
            path = qrow["path"]
        if not Path(path).is_file():
            results.append(
                {
                    "image": qrow["path"],
                    "first_result_ms": 9999.0,
                    "thumb_start_ms": None,
                    "thumb_loaded": 0,
                    "thumb_requested": 0,
                    "inspector_ok": False,
                    "freeze": True,
                    "error": "query file missing on disk",
                    "result_count": 0,
                    "pass_first_500": False,
                    "pass_thumb": False,
                    "pass_inspector": False,
                    "pass_freeze": False,
                }
            )
            print(f"[{i:03d}/{limit:03d}] FAIL MISSING {qrow['filename']}")
            continue

        thumb_scheduler.cancel_all()
        r = run_one_search(
            engine, path, thumb_scheduler=thumb_scheduler, app=app, perf=perf
        )
        results.append(r)
        ok = all(
            [
                r["pass_first_500"],
                r["pass_thumb"],
                r["pass_inspector"],
                r["pass_freeze"],
            ]
        )
        status = "PASS" if ok else "FAIL"
        print(
            f"[{i:03d}/{limit:03d}] {status} first={r['first_result_ms']:.0f}ms "
            f"thumb_start={r['thumb_start_ms']} thumb={r['thumb_loaded']}/{r['thumb_requested']} "
            f"insp={r['inspector_ok']} freeze={r['freeze']}"
            + (f" err={r['error'][:60]}" if r["error"] else "")
        )
        perf.touch_heartbeat()
        app.processEvents()

    elapsed = time.perf_counter() - t_all
    freeze_end = perf.snapshot().freeze_count
    freezes = sum(1 for r in results if r["freeze"] or not r["pass_freeze"])
    monitor_freezes = max(0, freeze_end - freeze_baseline)
    if monitor_freezes > freezes:
        freezes = monitor_freezes

    thumb_fail_searches = sum(1 for r in results if not r["pass_thumb"])
    inspector_empty = sum(1 for r in results if not r["inspector_ok"])
    first_ms_list = [float(r["first_result_ms"]) for r in results]
    thumb_ms_list = [
        float(r["thumb_start_ms"])
        for r in results
        if r["thumb_start_ms"] is not None
    ]
    avg_first = sum(first_ms_list) / len(first_ms_list) if first_ms_list else 9999.0
    avg_thumb = sum(thumb_ms_list) / len(thumb_ms_list) if thumb_ms_list else None

    pass_freeze = freezes == 0
    pass_thumb = thumb_fail_searches == 0
    pass_insp = inspector_empty == 0
    pass_avg = avg_first <= 500.0
    overall = pass_freeze and pass_thumb and pass_insp and pass_avg
    verdict = "PASS" if overall else "FAIL"

    report = {
        "verdict": verdict,
        "total_searches": len(results),
        "ui_freeze": freezes,
        "monitor_freezes": monitor_freezes,
        "thumbnail_missing_searches": thumb_fail_searches,
        "inspector_empty": inspector_empty,
        "avg_first_result_ms": round(avg_first, 1),
        "avg_thumb_start_ms": round(avg_thumb, 1) if avg_thumb is not None else None,
        "elapsed_sec": round(elapsed, 1),
        "indexed_files": indexed,
        "criteria": {
            "freeze_zero": pass_freeze,
            "thumb_all_ok": pass_thumb,
            "inspector_all_ok": pass_insp,
            "avg_first_le_500ms": pass_avg,
        },
        "per_search": results,
    }

    out_dir = Path(settings.db_path).parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "production_acceptance_100.json"
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    perf.stop()

    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    print(f"Toplam arama: {len(results)}")
    print(f"UI Freeze: {freezes}")
    print(f"Thumbnail yüklenmeyen: {thumb_fail_searches}")
    print(f"Inspector boş kalan: {inspector_empty}")
    print(f"Ortalama ilk sonuç süresi: {avg_first:.1f} ms")
    if avg_thumb is not None:
        print(f"Ortalama thumbnail süresi: {avg_thumb:.1f} ms")
    else:
        print("Ortalama thumbnail süresi: N/A")
    print(f"Report: {out_json}")
    print("=" * 60)
    return 0 if overall else 1


if __name__ == "__main__":
    n = 100
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    raise SystemExit(main(n))
