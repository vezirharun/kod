"""Isolated General AI pipeline A/B benchmark.

Does NOT modify Index Engine V3, jobs DB, files DB, FAISS, or settings.
Loads the same extractors as production (CPU/GPU from settings) and times
serial per-file vs pipeline-parallel stage workers on local feature previews.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]


def _now() -> float:
    return time.perf_counter()


def _nvidia_snapshot() -> dict[str, float] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    parts = [p.strip() for p in out.split(",")[:3]]
    try:
        return {
            "gpu_util": float(parts[0]),
            "mem_util": float(parts[1]),
            "mem_mb": float(parts[2]),
        }
    except (ValueError, IndexError):
        return None


class Sampler:
    def __init__(self, proc: psutil.Process) -> None:
        self.proc = proc
        self.cpu: list[float] = []
        self.gpu: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.proc.cpu_percent(None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.4):
            try:
                self.cpu.append(float(self.proc.cpu_percent(None)))
            except Exception:
                pass
            snap = _nvidia_snapshot()
            if snap is not None:
                self.gpu.append(snap)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        cpu = self.cpu or [0.0]
        gpu_u = [g["gpu_util"] for g in self.gpu]
        return {
            "cpu_avg": round(sum(cpu) / len(cpu), 2),
            "cpu_max": round(max(cpu), 2),
            "cpu_samples": len(cpu),
            "gpu_avg": round(sum(gpu_u) / len(gpu_u), 2) if gpu_u else None,
            "gpu_max": round(max(gpu_u), 2) if gpu_u else None,
            "gpu_samples": len(gpu_u),
            "gpu_present": bool(self.gpu) or _nvidia_snapshot() is not None,
        }


@dataclass
class Event:
    t: float
    file_id: int
    stage: str
    kind: str
    dt: float = 0.0


@dataclass
class BenchState:
    events: list[Event] = field(default_factory=list)
    queue_wait: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    idle: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    busy: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    io: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    lock = threading.Lock()
    inflight = 0
    inflight_max = 0
    inflight_hist: list[int] = field(default_factory=list)

    def add(self, ev: Event) -> None:
        with self.lock:
            self.events.append(ev)
            if ev.kind == "start":
                self.inflight += 1
                self.inflight_max = max(self.inflight_max, self.inflight)
                self.inflight_hist.append(self.inflight)
            elif ev.kind == "end":
                self.inflight = max(0, self.inflight - 1)
                self.inflight_hist.append(self.inflight)
                self.busy[ev.stage] += ev.dt


def inspect_engine(settings: Any) -> dict[str, Any]:
    from core.index_v3.planner import plan_jobs_for_file
    from core.index_v3.types import Artifact, ArtifactStatus, FileArtifactReport, Mode, QueueKind
    from core.network_index_throttle import is_network_path

    jobs_path = Path(settings.db_path).with_name("patterns.v3jobs.db")
    con = sqlite3.connect(str(jobs_path))
    con.row_factory = sqlite3.Row
    pending = con.execute(
        """
        SELECT id, file_id, artifact, queue, state
        FROM index_v3_jobs
        WHERE queue='heavy'
        ORDER BY id
        LIMIT 40
        """
    ).fetchall()
    done = con.execute(
        """
        SELECT file_id, artifact, state, updated_at
        FROM index_v3_jobs
        WHERE artifact IN ('dino','clip','texture','semantic','dna','patch','ocr')
          AND state='done'
        ORDER BY updated_at
        LIMIT 40
        """
    ).fetchall()
    by_art = con.execute(
        """
        SELECT artifact, state, COUNT(*) n
        FROM index_v3_jobs
        WHERE queue IN ('heavy','repair')
        GROUP BY artifact, state
        """
    ).fetchall()
    con.close()

    dummy = FileArtifactReport(file_id=1, source_id=8, path="x.jpg")
    dummy.status[Artifact.PREVIEW] = ArtifactStatus.READY
    dummy.status[Artifact.THUMBNAIL] = ArtifactStatus.READY
    planned = [j.artifact.value for j in plan_jobs_for_file(dummy, Mode.GENERAL_AI, ocr_enabled=True)]

    claim_sql = (
        "ORDER BY id LIMIT 1  "
        "(JobStore.claim) + engine has a single v3-heavy worker"
    )
    return {
        "heavy_workers_in_engine": 1,
        "claim_order": claim_sql,
        "planner_heavy_order": planned,
        "queues": [q.value for q in QueueKind],
        "ai_final": [a.value for a in (
            Artifact.DINO, Artifact.CLIP, Artifact.TEXTURE,
            Artifact.SEMANTIC, Artifact.DNA, Artifact.PATCH,
        )],
        "ocr_in_ai_final": False,
        "settings_use_gpu": bool(getattr(settings, "use_gpu", False)),
        "settings_worker_count": int(getattr(settings, "worker_count", 0) or 0),
        "pending_job_sample": [dict(r) for r in pending],
        "done_job_sample": [dict(r) for r in done],
        "heavy_job_counts": [dict(r) for r in by_art],
        "is_network_helper": is_network_path.__name__,
    }


def pick_files(db: Any, n: int) -> list[dict[str, Any]]:
    from core.network_index_throttle import is_network_path

    rows: list[dict[str, Any]] = []
    with db.connect() as conn:
        cur = conn.execute(
            """
            SELECT id, filename, path, feature_preview_path, thumbnail_path
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
            src = str(rec["path"] or "")
            rows.append(
                {
                    "file_id": int(rec["id"]),
                    "filename": rec["filename"],
                    "path": src,
                    "preview": prev,
                    "network_source": bool(is_network_path(src))
                    if src
                    else False,
                }
            )
            if len(rows) >= n:
                break
    return rows


def summarize(state: BenchState, files: list[dict[str, Any]], wall: float) -> dict[str, Any]:
    n = max(1, len(files))
    by_file: dict[int, dict[str, Any]] = defaultdict(lambda: {"start": None, "end": None, "stages": {}})
    for ev in state.events:
        rec = by_file[ev.file_id]
        if ev.kind == "start":
            rec["start"] = ev.t if rec["start"] is None else min(rec["start"], ev.t)
        elif ev.kind == "end":
            rec["end"] = ev.t if rec["end"] is None else max(rec["end"], ev.t)
            rec["stages"][ev.stage] = rec["stages"].get(ev.stage, 0.0) + ev.dt

    file_walls = []
    for rec in by_file.values():
        if rec["start"] is not None and rec["end"] is not None:
            file_walls.append(rec["end"] - rec["start"])

    def thr(stage: str) -> dict[str, Any]:
        times = [ev.dt for ev in state.events if ev.stage == stage and ev.kind == "end"]
        waits = list(state.queue_wait.get(stage) or [])
        busy = float(state.busy.get(stage) or 0.0)
        return {
            "count": len(times),
            "busy_sec": round(busy, 3),
            "throughput_per_min": round(len(times) / max(wall, 1e-6) * 60.0, 2),
            "avg_sec": round(sum(times) / len(times), 3) if times else None,
            "max_sec": round(max(times), 3) if times else None,
            "queue_wait_avg_sec": round(sum(waits) / len(waits), 3) if waits else 0.0,
            "idle_sec": round(float(state.idle.get(stage) or 0.0), 3),
        }

    # Overlap proof: after file-1 dino end, is file-2 dino started before file-1 clip end?
    proof = _pipeline_proof(state.events, [f["file_id"] for f in files])
    io_preview = state.io.get("preview") or [0.0]
    io_src = state.io.get("source") or []
    return {
        "files": n,
        "wall_sec": round(wall, 3),
        "wall_min": round(wall / 60.0, 3),
        "per_file_avg_sec": round(sum(file_walls) / len(file_walls), 3) if file_walls else None,
        "per_file_max_sec": round(max(file_walls), 3) if file_walls else None,
        "max_concurrent_artifacts": state.inflight_max,
        "avg_concurrent_artifacts": round(
            sum(state.inflight_hist) / len(state.inflight_hist), 3
        )
        if state.inflight_hist
        else 0,
        "throughput": {s: thr(s) for s in (
            "dino", "clip", "texture", "semantic", "dna", "patch", "ocr"
        )},
        "preview_read_avg_ms": round(1000 * sum(io_preview) / len(io_preview), 2),
        "nas_source_stat_samples": len(io_src),
        "nas_source_stat_avg_ms": round(1000 * sum(io_src) / len(io_src), 2) if io_src else 0.0,
        "proof": proof,
    }


def _pipeline_proof(events: list[Event], file_ids: list[int]) -> dict[str, Any]:
    if len(file_ids) < 2:
        return {"ok": False, "reason": "need_2_files"}
    f1, f2 = file_ids[0], file_ids[1]
    marks: dict[tuple[int, str, str], float] = {}
    for ev in events:
        marks[(ev.file_id, ev.stage, ev.kind)] = ev.t
    d1_end = marks.get((f1, "dino", "end"))
    d2_start = marks.get((f2, "dino", "start"))
    c1_end = marks.get((f1, "clip", "end"))
    c1_start = marks.get((f1, "clip", "start"))
    last_f1 = None
    first_f2 = None
    for ev in events:
        if ev.kind == "end" and ev.file_id == f1:
            last_f1 = ev.t if last_f1 is None else max(last_f1, ev.t)
        if ev.kind == "start" and ev.file_id == f2:
            first_f2 = ev.t if first_f2 is None else min(first_f2, ev.t)
    return {
        "file1": f1,
        "file2": f2,
        "file1_dino_end": d1_end,
        "file2_dino_start": d2_start,
        "file1_clip_start": c1_start,
        "file1_clip_end": c1_end,
        "file2_any_start_before_file1_chain_done": bool(
            first_f2 is not None and last_f1 is not None and first_f2 < last_f1
        ),
        "file2_dino_starts_after_file1_dino": bool(
            d1_end is not None and d2_start is not None and d2_start >= d1_end - 1e-6
        ),
        "file2_dino_overlaps_file1_clip": bool(
            d2_start is not None
            and c1_start is not None
            and c1_end is not None
            and d2_start < c1_end
        ),
    }


def time_read(path: str) -> float:
    t0 = _now()
    try:
        Path(path).stat()
        with open(path, "rb") as fh:
            fh.read(64 * 1024)
    except OSError:
        pass
    return _now() - t0


def run_model_a(
    files: list[dict[str, Any]],
    ctx: dict[str, Any],
) -> tuple[BenchState, float]:
    state = BenchState()
    t_batch = _now()
    for rec in files:
        _run_chain_serial(rec, ctx, state)
    return state, _now() - t_batch


def _run_chain_serial(rec: dict[str, Any], ctx: dict[str, Any], state: BenchState) -> None:
    fid = rec["file_id"]
    tm = None
    for stage in ("dino", "clip", "texture", "semantic", "dna", "patch", "ocr"):
        t_wait = _now()
        state.queue_wait[stage].append(0.0)
        state.add(Event(t_wait, fid, stage, "start"))
        try:
            dt, tm = _exec_stage(stage, rec, ctx, tm)
        except Exception as exc:
            print(f"stage_fail {stage} {fid}: {exc}", flush=True)
            dt = 0.0
        state.add(Event(_now(), fid, stage, "end", dt=dt))


def run_model_b(
    files: list[dict[str, Any]],
    ctx: dict[str, Any],
) -> tuple[BenchState, float]:
    """One worker per stage. Independent stages start immediately; semantic waits
    on texture; DNA waits on semantic. DINO/CLIP/Patch share model locks.
    """
    state = BenchState()
    dino_q: Queue = Queue()
    clip_q: Queue = Queue()
    texture_q: Queue = Queue()
    patch_q: Queue = Queue()
    ocr_q: Queue = Queue()
    semantic_q: Queue = Queue()
    dna_q: Queue = Queue()
    stop = object()
    texture_store: dict[int, Any] = {}
    store_lock = threading.Lock()

    for rec in files:
        dino_q.put(rec)
        clip_q.put(rec)
        texture_q.put(rec)
        patch_q.put(rec)
        ocr_q.put(rec)
    n = len(files)

    def worker(stage: str, in_q: Queue, n_expected: int) -> None:
        idle_start = _now()
        seen = 0
        while seen < n_expected:
            t_block = _now()
            rec = in_q.get()
            if rec is stop:
                break
            wait = _now() - t_block
            state.idle[stage] += max(0.0, wait)
            state.queue_wait[stage].append(wait)
            fid = rec["file_id"]
            tm = None
            if stage in ("semantic", "dna"):
                with store_lock:
                    tm = texture_store.get(fid)
            state.add(Event(_now(), fid, stage, "start"))
            try:
                dt, tm_out = _exec_stage(stage, rec, ctx, tm)
            except Exception as exc:
                print(f"stage_fail {stage} {fid}: {exc}", flush=True)
                dt, tm_out = 0.0, tm
            state.add(Event(_now(), fid, stage, "end", dt=dt))
            if stage == "texture" and tm_out is not None:
                with store_lock:
                    texture_store[fid] = tm_out
                semantic_q.put(rec)
            if stage == "semantic":
                with store_lock:
                    if tm_out is not None:
                        texture_store[fid] = tm_out
                dna_q.put(rec)
            seen += 1
        state.idle[stage] += max(0.0, _now() - idle_start) * 0.0

    t_batch = _now()
    threads = [
        threading.Thread(target=worker, args=("dino", dino_q, n), daemon=True),
        threading.Thread(target=worker, args=("clip", clip_q, n), daemon=True),
        threading.Thread(target=worker, args=("texture", texture_q, n), daemon=True),
        threading.Thread(target=worker, args=("patch", patch_q, n), daemon=True),
        threading.Thread(target=worker, args=("ocr", ocr_q, n), daemon=True),
        threading.Thread(target=worker, args=("semantic", semantic_q, n), daemon=True),
        threading.Thread(target=worker, args=("dna", dna_q, n), daemon=True),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return state, _now() - t_batch


def _exec_stage(
    stage: str,
    rec: dict[str, Any],
    ctx: dict[str, Any],
    tm: Any,
) -> tuple[float, Any]:
    preview = rec["preview"]
    src_path = rec["path"]
    fname = rec["filename"]
    ext = ctx["extractor"]
    ocr = ctx["ocr"]
    t0 = _now()
    io_dt = time_read(preview)
    ctx["state_io"]["preview"].append(io_dt)
    if rec.get("network_source"):
        t_n = _now()
        try:
            Path(src_path).stat()
        except OSError:
            pass
        ctx["state_io"]["source"].append(_now() - t_n)

    if stage in ("dino", "clip", "texture", "patch"):
        compute = {stage}
        if stage == "texture":
            compute.update({"texture", "hash", "color"})
        include_patches = stage == "patch"
        lock = ctx["locks"]["dino"] if stage in ("dino", "patch") else (
            ctx["locks"]["clip"] if stage == "clip" else ctx["locks"]["cpu"]
        )
        with lock:
            features = ext.extract_from_path(
                preview,
                source_path=src_path,
                include_patches=include_patches,
                deep_analysis=True,
                compute=compute,
            )
        if stage == "texture":
            tm = dict(features.texture_map or {})
        return _now() - t0, tm

    if stage == "semantic":
        from core.semantic_tags import build_semantic_tags

        tm = dict(tm or {})
        tm["semantic_tags"] = build_semantic_tags(
            tm,
            category_path="",
            ocr_text="",
            filename=fname,
        )
        return _now() - t0, tm

    if stage == "dna":
        from core.pattern_dna import apply_dna_to_texture_map, build_pattern_dna

        tm = dict(tm or {})
        dna = build_pattern_dna(
            tm,
            semantic_tags=tm.get("semantic_tags"),
            category_path="",
            source="bench",
        )
        tm = apply_dna_to_texture_map(tm, dna)
        return _now() - t0, tm

    if stage == "ocr":
        with ctx["locks"]["ocr"]:
            try:
                ocr.extract_text(preview)
            except Exception:
                pass
        return _now() - t0, tm

    raise RuntimeError(stage)


def gpu_overlap_probe(files: list[dict[str, Any]], n: int = 12) -> dict[str, Any] | None:
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    from core.feature_extractor import FeatureExtractor

    sample = files[: min(n, len(files))]
    if len(sample) < 4:
        return {"cuda": True, "skipped": "too_few_files"}
    ext = FeatureExtractor(use_ai=True, use_gpu=True, fast_hash_only=False)
    if ext._dino_model is None or ext._clip_model is None:
        return {"cuda": True, "skipped": "models_missing"}

    dino_lock = threading.Lock()
    clip_lock = threading.Lock()

    def one(stage: str, rec: dict[str, Any]) -> float:
        lock = dino_lock if stage == "dino" else clip_lock
        t0 = _now()
        with lock:
            ext.extract_from_path(
                rec["preview"],
                source_path=rec["path"],
                include_patches=False,
                deep_analysis=False,
                compute={stage},
            )
        return _now() - t0

    # Serial: file dino then clip
    serial_times: list[float] = []
    t_s = _now()
    for rec in sample:
        serial_times.append(one("dino", rec) + one("clip", rec))
    serial_wall = _now() - t_s

    # Overlap: DINO and CLIP workers
    q_d: Queue = Queue()
    q_c: Queue = Queue()
    for rec in sample:
        q_d.put(rec)
        q_c.put(rec)

    def drain(stage: str, q: Queue) -> None:
        while True:
            try:
                rec = q.get_nowait()
            except Empty:
                break
            one(stage, rec)

    t_p = _now()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(drain, "dino", q_d)
        f2 = pool.submit(drain, "clip", q_c)
        f1.result()
        f2.result()
    parallel_wall = _now() - t_p
    gpu = _nvidia_snapshot()
    speedup = (serial_wall / parallel_wall - 1.0) * 100.0 if parallel_wall else 0.0
    return {
        "cuda": True,
        "files": len(sample),
        "serial_dino_then_clip_sec": round(serial_wall, 3),
        "overlapped_dino_plus_clip_sec": round(parallel_wall, 3),
        "speedup_pct": round(speedup, 1),
        "faster": parallel_wall < serial_wall,
        "nvidia": gpu,
        "note": "Same FeatureExtractor, two CUDA modules, two threads.",
    }


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from core.db import Database
    from core.feature_extractor import FeatureExtractor
    from core.ocr_engine import OCREngine
    from core.settings import AppSettings

    n = 60
    settings = AppSettings.load()
    db = Database(settings.db_path)
    out_dir = ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "general_ai_pipeline_bench.json"

    print("=== inspect engine (no extract) ===", flush=True)
    engine = inspect_engine(settings)
    pending_files: dict[int, list[str]] = defaultdict(list)
    for row in engine["pending_job_sample"]:
        pending_files[int(row["file_id"])].append(row["artifact"])
    serial_like = True
    prev_fid = None
    switched_before_full = 0
    arts_of: dict[int, list[str]] = {}
    for row in engine["pending_job_sample"]:
        fid = int(row["file_id"])
        arts_of.setdefault(fid, [])
        if prev_fid is not None and fid != prev_fid and len(arts_of.get(prev_fid, [])) < 6:
            # next file started before previous file had 6+ heavy arts in this window
            switched_before_full += 1
        arts_of[fid].append(row["artifact"])
        prev_fid = fid
    engine["claim_window_looks_per_file_serial"] = switched_before_full == 0
    engine["files_in_pending_sample"] = len(pending_files)

    files = pick_files(db, n)
    print(f"selected {len(files)} preview-ready files", flush=True)
    if len(files) < 8:
        report = {
            "error": "not_enough_preview_files",
            "engine": engine,
            "selected": len(files),
        }
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    use_gpu = bool(getattr(settings, "use_gpu", False))
    print(f"loading extractors use_gpu={use_gpu} ...", flush=True)
    extractor = FeatureExtractor(
        use_ai=True, use_gpu=use_gpu, fast_hash_only=False
    )
    ocr = OCREngine(
        enabled=bool(getattr(settings, "ocr_enabled", False)),
        languages=list(getattr(settings, "ocr_languages", None) or ["en", "tr"]),
    )
    ctx = {
        "extractor": extractor,
        "ocr": ocr,
        "locks": {
            "dino": threading.Lock(),
            "clip": threading.Lock(),
            "cpu": threading.Lock(),
            "ocr": threading.Lock(),
        },
        "state_io": defaultdict(list),
    }

    warmup = files[:2]
    work = files[2:]
    print(f"warmup {len(warmup)} files, batch {len(work)}", flush=True)
    for rec in warmup:
        _run_chain_serial(rec, ctx, BenchState())
    print("warmup done", flush=True)

    proc = psutil.Process()
    sampler_a = Sampler(proc)
    ctx["state_io"] = defaultdict(list)
    print("=== Model A serial ===", flush=True)
    sampler_a.start()
    state_a, wall_a = run_model_a(work, ctx)
    res_a = sampler_a.stop()
    state_a.io = ctx["state_io"]
    sum_a = summarize(state_a, work, wall_a)
    sum_a["resources"] = res_a
    print(
        f"A wall={wall_a:.1f}s concurrent_max={state_a.inflight_max} "
        f"file2_before_file1_done={sum_a['proof']['file2_any_start_before_file1_chain_done']}",
        flush=True,
    )

    sampler_b = Sampler(proc)
    ctx["state_io"] = defaultdict(list)
    print("=== Model B pipeline-parallel ===", flush=True)
    sampler_b.start()
    state_b, wall_b = run_model_b(work, ctx)
    res_b = sampler_b.stop()
    state_b.io = ctx["state_io"]
    sum_b = summarize(state_b, work, wall_b)
    sum_b["resources"] = res_b
    print(
        f"B wall={wall_b:.1f}s concurrent_max={state_b.inflight_max} "
        f"f2_dino_overlaps_f1_clip={sum_b['proof']['file2_dino_overlaps_file1_clip']}",
        flush=True,
    )

    print("=== GPU overlap probe ===", flush=True)
    gpu = gpu_overlap_probe(work, n=12)
    speedup = (wall_a / wall_b - 1.0) * 100.0 if wall_b else 0.0
    if wall_b < wall_a:
        winner = "pipeline-parallel"
        rec_conc = (
            "Pipeline-parallel stage workers. Keep DINO and Patch on the same "
            "model mutex. Do not start a second DINO/CLIP copy unless GPU VRAM allows."
        )
    else:
        winner = "serial"
        rec_conc = (
            "Keep current single heavy worker (serial artifact drain). "
            "Pipeline-parallel did not reduce wall time."
        )

    bottleneck = []
    for name, spec in sum_a["throughput"].items():
        bottleneck.append((spec.get("busy_sec") or 0.0, name))
    bottleneck.sort(reverse=True)
    cpu_a = res_a.get("cpu_avg") or 0
    gpu_a = res_a.get("gpu_avg")
    if gpu_a in (None, 0) and not use_gpu:
        hw = "CPU (settings.use_gpu=false); GPU idle"
    elif (gpu_a or 0) >= 80:
        hw = "GPU"
    elif cpu_a >= 85:
        hw = "CPU"
    else:
        hw = "mixed / stage imbalance"
    if any(f["network_source"] for f in work):
        nas_note = "some originals on NAS; heavy used local feature_preview"
    else:
        nas_note = "heavy path used local cache previews (NAS not on critical path)"

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "did_not_modify_engine": True,
        "did_not_write_jobs_or_db": True,
        "batch_files": len(work),
        "warmup_files": len(warmup),
        "use_gpu": use_gpu,
        "engine_current": engine,
        "model_a_serial": sum_a,
        "model_b_pipeline": sum_b,
        "gpu_overlap_probe": gpu,
        "decision": {
            "serial_minutes": round(wall_a / 60.0, 3),
            "pipeline_minutes": round(wall_b / 60.0, 3),
            "speedup_pct": round(speedup, 1),
            "winner": winner,
            "bottleneck": hw,
            "heaviest_stages_serial": [x[1] for x in bottleneck[:4]],
            "nas": nas_note,
            "recommended_concurrency": rec_conc,
        },
        "where_to_apply_if_b_wins": {
            "today": (
                "IndexEngineV3 GENERAL_AI: one v3-heavy worker, claim_fair "
                "limit=1, JobStore.claim ORDER BY id. Planner emits "
                "DINO,CLIP,TEXTURE,SEMANTIC,DNA,PATCH[,OCR] per file, "
                "files in id order → Model A."
            ),
            "apply": (
                "Optional heavy lane: N stage workers claiming by artifact "
                "(WHERE artifact=? AND state=pending) OR extra heavy workers "
                "with GPU/DINO mutex. Semantic after texture READY; DNA after "
                "semantic READY. Do not change FAST, Repair verify, FAISS, Search."
            ),
        },
    }
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
