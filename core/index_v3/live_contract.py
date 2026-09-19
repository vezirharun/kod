"""Index Engine V3 — live progress contract (pool SSOT, no event counters).

Tamamlanan = physical READY pool
İşleniyor  = worker claim (0|1)
Session+   = current_pool - session_start_pool  (+/-)
Kalan      = TOTAL - READY
Executable = JobStore pending (ayrı kavram)
Speed/ETA  = READY Δ / zaman  + JobStore recent completes (UI)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

POOL_KEYS: tuple[str, ...] = (
    "thumbnail",
    "preview",
    "dino",
    "clip",
    "texture",
    "semantic",
    "dna",
    "patch",
    "ai_final",
    "light_coverage",
)

# Worker drain sırasında v3_progress SSOT YASAK. Sayaç UI 1 Hz timer'dadır.
_WORKER_SSOT_STAGES = frozenset(
    {"index_starting", "index_ready", "index_done"}
)


def worker_progress_should_compute_ssot(stage: str) -> bool:
    """Index worker thread: tam count_v3_ssot yalnız oturum başı/sonu.

    Aşama callback (v3_progress / claimed / idle / engine drain) SSOT çekmez;
    aksi halde HASH→DINO arasında ~8 sn UI sayacı oluşur.
    """
    return str(stage or "").strip().lower() in _WORKER_SSOT_STAGES



def session_plus_from_pools(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, int]:
    """session+ = current_real_pool - session_start_pool (event yok)."""
    out: dict[str, int] = {}
    keys = set(POOL_KEYS) | set(current) | set(baseline)
    skip = {
        "total",
        "legacy_thumbnail",
        "legacy_preview",
        "legacy_dino",
        "legacy_clip",
        "legacy_texture",
        "legacy_semantic",
        "legacy_dna",
        "legacy_patch",
        "thumbnail_db_path",
        "preview_db_path",
        "texture_db",
        "semantic_db",
        "dna_db",
        "patch_db",
        "light_queue",
        "waiting_preview",
        "light_complete",
    }
    for key in keys:
        if key in skip or key.startswith("legacy_"):
            continue
        try:
            out[str(key)] = int(current.get(key) or 0) - int(baseline.get(key) or 0)
        except (TypeError, ValueError):
            continue
    for key in POOL_KEYS:
        out.setdefault(key, int(current.get(key) or 0) - int(baseline.get(key) or 0))
    return out


def remaining_vs_executable(
    *,
    total: int,
    ready: int,
    executable_jobs: int,
) -> dict[str, int]:
    """Kalan = TOTAL-READY; Executable = çalıştırılabilir job (ayrı)."""
    total = max(0, int(total))
    ready = max(0, min(int(ready), total))
    return {
        "remaining": max(0, total - ready),
        "executable": max(0, int(executable_jobs)),
        "ready": ready,
        "total": total,
    }


def speed_eta_from_pool(
    *,
    samples: list[tuple[float, int]],
    remaining: int,
    min_span_sec: float = 5.0,
    min_delta: int = 2,
) -> dict[str, Any]:
    """Hız/ETA yalnız gerçek READY pool hareketinden.

    samples: [(monotonic_ts, pool_completed), ...] artan zaman.
    Event/callback sayısı kullanılmaz.
    """
    remaining = max(0, int(remaining))
    if len(samples) < 2:
        return {
            "speed_per_sec": None,
            "speed_per_min": None,
            "eta_sec": None,
            "speed_label": "hesaplanıyor…",
            "eta_label": "hesaplanıyor…",
        }
    t0, n0 = samples[0]
    t1, n1 = samples[-1]
    dt = float(t1) - float(t0)
    dn = int(n1) - int(n0)
    if dt < min_span_sec or dn < min_delta:
        return {
            "speed_per_sec": None,
            "speed_per_min": None,
            "eta_sec": None,
            "speed_label": "hesaplanıyor…",
            "eta_label": "hesaplanıyor…",
        }
    sps = dn / dt
    spm = sps * 60.0
    eta = (remaining / sps) if sps > 0 and remaining > 0 else None
    return {
        "speed_per_sec": sps,
        "speed_per_min": spm,
        "eta_sec": eta,
        "speed_label": f"{sps:.2f} dosya/sn",
        "eta_label": _fmt_eta(eta) if eta is not None else "—",
    }


def _fmt_eta(secs: float | None) -> str:
    if secs is None or secs < 0:
        return "—"
    s = int(secs)
    if s < 60:
        return f"{s} sn"
    if s < 3600:
        return f"{s // 60} dk {s % 60} sn"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h} sa {m} dk"


def build_live_progress(
    status: dict[str, Any],
    *,
    session_start: dict[str, Any] | None = None,
    claim: dict[str, Any] | None = None,
    stall_sec: float = 30.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Tek canlı progress payload — pool SSOT + claim runtime.

    Tamamlanan ≠ İşleniyor. Event sayacı yok.
    """
    claim = dict(claim or {})
    session_start = dict(session_start or {})
    pools = dict(status.get("artifact_pools") or {})
    if not pools:
        pools = {
            "total": int(status.get("total") or 0),
            "thumbnail": int(status.get("thumbnail_ready") or 0),
            "preview": int(status.get("preview_ready") or 0),
            "dino": int(status.get("db_dino_embeddings") or 0),
            "clip": int(status.get("db_clip_embeddings") or 0),
            "texture": int(status.get("texture_done") or 0),
            "semantic": int(status.get("semantic_tag_count") or 0),
            "dna": int(status.get("pattern_dna_count") or 0),
            "patch": int(status.get("patch_embedding_ready") or 0),
            "light_complete": int(
                status.get("light_complete_physical")
                or status.get("light_complete")
                or 0
            ),
            "light_coverage": int(
                status.get("light_done")
                or status.get("fast_completed")
                or status.get("light_coverage")
                or 0
            ),
            "ai_final": int(status.get("ai_final_ready") or 0),
        }
    total = int(pools.get("total") or status.get("total") or 0)

    # Normalize pool keys from count_v3_ssot baseline
    for k in POOL_KEYS:
        if k not in pools and k in session_start:
            pools[k] = int(status.get(k) or 0)
        pools.setdefault(k, int(status.get(k) or pools.get(k) or 0))

    # Prefer count_v3_ssot-shaped session_start keys
    base_pools = {k: int(session_start.get(k) or 0) for k in POOL_KEYS}
    plus = session_plus_from_pools(pools, base_pools)

    stages: dict[str, dict[str, Any]] = {}
    for key in POOL_KEYS:
        ready = int(pools.get(key) or 0)
        rem = max(0, total - ready)
        pct = (100.0 * ready / total) if total else 0.0
        stages[key] = {
            "ready": ready,
            "remaining": rem,
            "percent": pct,
            "total": total,
            "session_plus": int(plus.get(key) or 0),
        }

    phase = str(claim.get("phase") or "")
    claim_fresh = status.get("claimed_fresh_jobs")
    if claim_fresh is None:
        claim_fresh = status.get("claim_fresh", 1 if phase == "claimed" else 0)

    fname = str(claim.get("filename") or "").strip()
    if not fname and claim.get("path"):
        fname = Path(str(claim.get("path"))).name
    stage = str(
        claim.get("stage")
        or claim.get("artifact")
        or status.get("current_stage")
        or ""
    ).strip()
    worker = str(claim.get("worker") or status.get("current_worker") or "").strip()
    source_id = int(claim.get("source_id") or 0)

    tnow = float(now if now is not None else time.monotonic())
    started = claim.get("claim_started")
    elapsed = 0.0
    if started is not None:
        try:
            elapsed = max(0.0, tnow - float(started))
        except (TypeError, ValueError):
            elapsed = float(claim.get("elapsed") or 0)
    else:
        elapsed = float(claim.get("elapsed") or status.get("current_elapsed_sec") or 0)

    # İşleniyor: aktif claim. claimed_fresh=0 tek başına sıfırlamaz (uzun PATCH).
    # Ama claim_started yok + fresh=0 → stale/dead worker → 0.
    status_proc = max(
        0,
        int(status.get("processing") or 0),
        int(status.get("light_processing") or 0),
        int(status.get("heavy_processing") or 0),
        int(status.get("claimed_light_jobs") or 0),
        int(status.get("claimed_heavy_jobs") or 0),
    )
    fresh_ok = int(claim_fresh or 0) > 0
    # Heartbeat 180s; UI grace uzun job'lar için ~10 dk
    claim_clock_ok = started is not None and elapsed < max(float(stall_sec) * 20.0, 600.0)
    if phase == "idle":
        processing = 0
    elif phase == "claimed":
        if fresh_ok or claim_clock_ok:
            processing = max(1, status_proc)
        else:
            processing = 0
    else:
        processing = status_proc

    stall = bool(processing and elapsed >= float(stall_sec))

    preview_ready = int(pools.get("preview") or 0)
    thumb_ready = int(pools.get("thumbnail") or status.get("thumbnail_ready") or 0)
    light_complete = int(pools.get("light_complete") or status.get("light_done") or 0)
    if light_complete <= 0:
        light_complete = (
            min(preview_ready, thumb_ready) if thumb_ready > 0 else preview_ready
        )
    ai_ready = int(pools.get("ai_final") or 0)
    terminal_error_files = max(0, int(status.get("terminal_error_files") or 0))
    exec_jobs = int(
        status.get("v3_executable_jobs")
        or status.get("pending_jobs")
        or 0
    )
    rve_fast = remaining_vs_executable(
        total=total,
        ready=min(total, light_complete),
        executable_jobs=int(
            status.get("pending_light_jobs")
            if status.get("pending_light_jobs") is not None
            else exec_jobs
        ),
    )
    rve_ai = remaining_vs_executable(
        total=total,
        ready=min(total, ai_ready + terminal_error_files),
        executable_jobs=int(
            status.get("pending_heavy_jobs")
            if status.get("pending_heavy_jobs") is not None
            else exec_jobs
        ),
    )

    light_stages = ("thumbnail", "preview")
    heavy_stages = (
        "hash",
        "metadata",
        "dino",
        "clip",
        "texture",
        "semantic",
        "dna",
        "patch",
        "ocr",
        "object_concept",
        "owlv2",
    )
    q = str(claim.get("queue") or "").lower()
    status_light = int(status.get("light_processing") or status.get("claimed_light_jobs") or 0)
    status_heavy = int(status.get("heavy_processing") or status.get("claimed_heavy_jobs") or 0)
    if processing <= 0 or phase == "idle":
        fast_proc = 0 if phase == "idle" else (status_light if phase != "claimed" else 0)
        heavy_proc = 0 if phase == "idle" else (status_heavy if phase != "claimed" else 0)
        if phase == "claimed" and processing <= 0:
            fast_proc = 0
            heavy_proc = 0
    else:
        live_light = q in ("light", "preview") or stage in light_stages
        live_heavy = (not live_light) and (
            q in ("heavy", "repair")
            or stage in heavy_stages
            or "heavy" in worker.lower()
        )
        fast_proc = max(
            status_light,
            1 if (phase == "claimed" and live_light) else 0,
        )
        heavy_proc = max(
            status_heavy,
            1 if (phase == "claimed" and (live_heavy or (not live_light))) else 0,
        )
        if phase == "claimed" and live_light:
            heavy_proc = status_heavy  # light claim → heavy'ye yazma
        if phase == "claimed" and live_heavy:
            fast_proc = status_light


    return {
        "ssot": "index_v3_physical_pool",
        "total": total,
        "pools": {k: int(pools.get(k) or 0) for k in ("total",) + POOL_KEYS},
        "stages": stages,
        "session_plus": {k: int(plus.get(k) or 0) for k in POOL_KEYS},
        "fast": {
            "completed": light_complete,
            "remaining": rve_fast["remaining"],
            "file_queue": rve_fast["remaining"],
            "percent": (100.0 * light_complete / total) if total else 0.0,
            "executable": rve_fast["executable"],
            "job_queue": rve_fast["executable"],
            "failed": int(status.get("light_failed") or 0),
            "processing": fast_proc,
        },
        "general_ai": {
            "completed": ai_ready,
            "remaining": rve_ai["remaining"],
            "file_queue": rve_ai["remaining"],
            "percent": stages["ai_final"]["percent"],
            "executable": rve_ai["executable"],
            "job_queue": rve_ai["executable"],
            "failed": int(status.get("heavy_failed") or 0),
            "processing": heavy_proc,
        },
        "processing": int(processing),
        "terminal_error_files": terminal_error_files,
        "claimed_jobs": int(status.get("claimed_jobs") or 0),
        "claimed_fresh_jobs": int(claim_fresh or 0),
        "current_filename": fname,
        "current_stage": stage,
        "current_worker": worker,
        "current_source_id": source_id,
        "current_path": str(claim.get("path") or ""),
        "elapsed_sec": elapsed,
        "stall": stall,
        "remaining": rve_fast["remaining"],
        "file_queue": rve_fast["remaining"],
        "executable": exec_jobs,
        "job_queue": exec_jobs,
    }


def lane_speed_eta(
    *,
    lane: str,
    samples: list[tuple[float, int]],
    remaining: int,
) -> dict[str, Any]:
    """Lane-ayrı hız/ETA — Fast ve Heavy sample listeleri karışmaz."""
    _ = lane
    return speed_eta_from_pool(samples=samples, remaining=remaining)
