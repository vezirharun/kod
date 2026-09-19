"""Archive Health — thin projection over existing V3 SSOT / JobStore / preview signals.

No repair engine, no schema changes, no indexer writes.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    MISSING = "MISSING"
    STALE = "STALE"
    BROKEN = "BROKEN"
    REPAIR_PENDING = "REPAIR_PENDING"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


# Deterministic priority (worst first).
PRIORITY: tuple[HealthState, ...] = (
    HealthState.FAILED,
    HealthState.BROKEN,
    HealthState.REPAIR_PENDING,
    HealthState.STALE,
    HealthState.MISSING,
    HealthState.UNKNOWN,
    HealthState.HEALTHY,
)

_PRIORITY_INDEX = {s: i for i, s in enumerate(PRIORITY)}

_SOURCE_MISSING = frozenset({"missing", "excluded_internal", "excluded"})


def merge_states(*states: HealthState | str | None) -> HealthState:
    """Return the worst HealthState by PRIORITY."""
    best_idx = len(PRIORITY) - 1  # HEALTHY
    found = False
    for raw in states:
        if raw is None:
            continue
        try:
            st = raw if isinstance(raw, HealthState) else HealthState(str(raw))
        except ValueError:
            st = HealthState.UNKNOWN
        found = True
        idx = _PRIORITY_INDEX.get(st, _PRIORITY_INDEX[HealthState.UNKNOWN])
        if idx < best_idx:
            best_idx = idx
    if not found:
        return HealthState.UNKNOWN
    return PRIORITY[best_idx]


def _as_state(value: Any) -> HealthState:
    if isinstance(value, HealthState):
        return value
    if value is None:
        return HealthState.UNKNOWN
    try:
        return HealthState(str(value))
    except ValueError:
        return HealthState.UNKNOWN


def _job_states_for_file(job_store: Any, file_id: int) -> set[str]:
    """Collect distinct JobStore states for one file (read-only)."""
    fid = int(file_id)
    if job_store is None or fid <= 0:
        return set()

    # Test / duck-typed hook
    hook = getattr(job_store, "file_job_states", None)
    if callable(hook):
        try:
            return {str(s) for s in (hook(fid) or ()) if s}
        except Exception:
            return set()

    out: set[str] = set()
    connect = getattr(job_store, "_connect", None)
    if callable(connect):
        try:
            with connect() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT state FROM index_v3_jobs WHERE file_id=?",
                    (fid,),
                ).fetchall()
            for row in rows:
                if hasattr(row, "keys"):
                    st = str(row["state"] or "")
                else:
                    st = str(row[0] or "")
                if st:
                    out.add(st)
            return out
        except Exception:
            out = set()

    # Fallbacks using public JobStore APIs
    try:
        pending = job_store.list_pending_artifacts(fid) or []
        if pending:
            out.add("pending")
    except Exception:
        pass
    try:
        # Sample claimed via job_state on known pending list is incomplete;
        # leave empty if no _connect.
        pass
    except Exception:
        pass
    return out


def _preview_status_state(preview_status: str) -> tuple[HealthState | None, str | None]:
    """Map files.preview_status → optional HealthState + reason cite.

    preview_suspicious alone is NOT BROKEN (mono may be OK via self-heal).
    Empty legacy preview_status → no signal (None).
    """
    ps = str(preview_status or "").strip()
    if not ps:
        return None, None
    low = ps.lower()
    if low.startswith("preview_invalid"):
        return HealthState.BROKEN, f"preview_status={ps}"
    if low == "preview_heal_failed" or low.startswith("preview_heal_failed"):
        return HealthState.FAILED, f"preview_status={ps}"
    if low == "preview_missing_physical" or low.startswith("preview_missing"):
        return HealthState.STALE, f"preview_status={ps}"
    # preview_ok / preview_suspicious / other → no forced bad state
    return None, None


def assess_file_health(
    db: Any,
    file_id: int,
    *,
    job_store: Any | None = None,
    require_disk: bool = False,
) -> dict[str, Any]:
    """Project per-file Archive Health from existing DB / assess / JobStore signals."""
    from core.index_v3.types import AI_FINAL_REQUIRED, Artifact, ArtifactStatus

    reasons: list[str] = []
    source_st = HealthState.UNKNOWN
    preview_st = HealthState.UNKNOWN
    ai_st = HealthState.UNKNOWN
    physical_st = HealthState.UNKNOWN
    job_st = HealthState.UNKNOWN

    fid = int(file_id)
    row: dict[str, Any] = {}
    try:
        row = db.get_file_by_id(fid) or {}
    except Exception as exc:
        reasons.append(f"get_file_by_id_error={type(exc).__name__}")
        return {
            "state": HealthState.UNKNOWN,
            "reasons": reasons,
            "preview": HealthState.UNKNOWN,
            "source": HealthState.UNKNOWN,
            "ai": HealthState.UNKNOWN,
            "physical": HealthState.UNKNOWN,
        }

    if not row:
        reasons.append("file_row_missing")
        return {
            "state": HealthState.UNKNOWN,
            "reasons": reasons,
            "preview": HealthState.UNKNOWN,
            "source": HealthState.UNKNOWN,
            "ai": HealthState.UNKNOWN,
            "physical": HealthState.UNKNOWN,
        }

    # --- source ---
    fstatus = str(row.get("status") or "").strip().lower()
    if fstatus in _SOURCE_MISSING:
        source_st = HealthState.MISSING
        reasons.append(f"files.status={fstatus}")
    else:
        source_st = HealthState.HEALTHY

    # --- preview_status content signals ---
    ps_state, ps_reason = _preview_status_state(str(row.get("preview_status") or ""))
    if ps_state is not None:
        preview_st = ps_state
        if ps_reason:
            reasons.append(ps_reason)

    # --- physical path vs flag ---
    fpp = str(row.get("feature_preview_path") or "").strip()
    phys = int(row.get("physical_preview_ready") or 0)
    if fpp and phys != 1:
        physical_st = HealthState.STALE
        reasons.append("feature_preview_path set but physical_preview_ready=0")
    elif fpp and phys == 1:
        physical_st = HealthState.HEALTHY
    else:
        # no path → physical gap (missing artifact on disk/db)
        physical_st = HealthState.MISSING

    # --- artifact_state.assess_file ---
    report = None
    try:
        from core.index_v3.artifact_state import assess_file as _assess_file

        report = _assess_file(db, fid, require_disk=require_disk)
    except Exception as exc:
        reasons.append(f"assess_file_unavailable={type(exc).__name__}")

    if report is not None:
        prev_art = report.get(Artifact.PREVIEW)
        if prev_art == ArtifactStatus.INVALID:
            preview_st = (
                HealthState.BROKEN
                if preview_st == HealthState.UNKNOWN
                else merge_states(preview_st, HealthState.BROKEN)
            )
            reasons.append("artifact.PREVIEW=INVALID")
        elif prev_art == ArtifactStatus.MISSING:
            preview_st = (
                HealthState.MISSING
                if preview_st == HealthState.UNKNOWN
                else merge_states(preview_st, HealthState.MISSING)
            )
            reasons.append("artifact.PREVIEW=MISSING")
        elif prev_art == ArtifactStatus.READY:
            # Do not let HEALTHY erase a worse preview_status signal.
            if preview_st == HealthState.UNKNOWN:
                preview_st = HealthState.HEALTHY
            else:
                preview_st = merge_states(preview_st, HealthState.HEALTHY)

        # Light thumbs contribute to preview/light picture
        thumb_art = report.get(Artifact.THUMBNAIL)
        if thumb_art == ArtifactStatus.MISSING and preview_st == HealthState.HEALTHY:
            preview_st = HealthState.MISSING
            reasons.append("artifact.THUMBNAIL=MISSING")

        missing_ai = [
            a.value
            for a in AI_FINAL_REQUIRED
            if report.get(a) != ArtifactStatus.READY
        ]
        invalid_ai = [
            a.value
            for a in AI_FINAL_REQUIRED
            if report.get(a) == ArtifactStatus.INVALID
        ]
        if invalid_ai:
            ai_st = HealthState.BROKEN
            reasons.append(f"ai_invalid={','.join(invalid_ai)}")
        elif missing_ai:
            ai_st = HealthState.MISSING
            reasons.append(f"ai_missing={','.join(missing_ai)}")
        else:
            ai_st = HealthState.HEALTHY

        # All light READY + no bad preview flags → healthy preview lane
        if report.light_complete and preview_st in (
            HealthState.UNKNOWN,
            HealthState.HEALTHY,
        ):
            if ps_state is None:
                preview_st = HealthState.HEALTHY
    else:
        # Without assess: infer preview from path/physical only
        if preview_st == HealthState.UNKNOWN:
            if physical_st == HealthState.HEALTHY:
                preview_st = HealthState.HEALTHY
            elif physical_st == HealthState.STALE:
                preview_st = HealthState.STALE
            elif physical_st == HealthState.MISSING:
                preview_st = HealthState.MISSING

    # --- JobStore ---
    if job_store is not None:
        jstates = _job_states_for_file(job_store, fid)
        if "failed_permanent" in jstates:
            job_st = HealthState.FAILED
            reasons.append("jobstore.state=failed_permanent")
        elif jstates & {"pending", "claimed"}:
            job_st = HealthState.REPAIR_PENDING
            hit = sorted(jstates & {"pending", "claimed"})
            reasons.append(f"jobstore.state={','.join(hit)}")

    # Only merge components that carried a real signal (skip unset UNKNOWN).
    parts: list[HealthState] = []
    for st in (source_st, preview_st, physical_st, ai_st, job_st):
        if st != HealthState.UNKNOWN:
            parts.append(st)
    overall = merge_states(*parts) if parts else HealthState.UNKNOWN

    # Row exists but nothing ready yet → MISSING rather than UNKNOWN.
    if (
        overall == HealthState.UNKNOWN
        and source_st == HealthState.HEALTHY
        and preview_st == HealthState.UNKNOWN
        and physical_st == HealthState.MISSING
        and ai_st == HealthState.UNKNOWN
        and job_st == HealthState.UNKNOWN
    ):
        overall = HealthState.MISSING
        if "preview_empty" not in reasons:
            reasons.append("preview_empty")

    return {
        "state": overall,
        "reasons": reasons,
        "preview": preview_st,
        "source": source_st,
        "ai": ai_st,
        "physical": physical_st,
    }


def _scope_where(source_ids: list[int] | None) -> tuple[str, list[Any]]:
    if source_ids is None:
        return "status NOT IN ('excluded_internal','missing')", []
    if not source_ids:
        return "1=0", []
    ph = ",".join("?" * len(source_ids))
    return (
        f"status NOT IN ('excluded_internal','missing') AND source_id IN ({ph})",
        [int(x) for x in source_ids],
    )


def _count_preview_invalid(db: Any, source_ids: list[int] | None) -> int:
    where, params = _scope_where(source_ids)
    sql = (
        f"SELECT COUNT(*) AS n FROM files WHERE {where} "
        f"AND ifnull(preview_status,'') LIKE 'preview_invalid%'"
    )
    try:
        with db.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        if row is None:
            return 0
        if hasattr(row, "keys"):
            return int(row["n"] or 0)
        return int(row[0] or 0)
    except Exception:
        return 0


def _jobstore_summary(
    job_store: Any | None,
    source_ids: list[int] | None,
) -> dict[str, int]:
    out = {
        "pending_files": 0,
        "pending_light_preview_repair_files": 0,
        "failed_permanent_files": 0,
        "claimed_files": 0,
    }
    if job_store is None:
        return out

    # Duck-typed override for tests
    hook = getattr(job_store, "archive_health_counts", None)
    if callable(hook):
        try:
            data = hook(source_ids=source_ids) or {}
            for k in out:
                if k in data:
                    out[k] = int(data[k] or 0)
            return out
        except Exception:
            pass

    try:
        from core.index_v3.types import QueueKind

        queues = (QueueKind.LIGHT, QueueKind.PREVIEW, QueueKind.REPAIR)
        if hasattr(job_store, "count_pending_files"):
            out["pending_light_preview_repair_files"] = int(
                job_store.count_pending_files(
                    queues=queues, source_ids=source_ids
                )
                or 0
            )
            out["pending_files"] = int(
                job_store.count_pending_files(source_ids=source_ids) or 0
            )
        if hasattr(job_store, "count_failed_permanent_files"):
            out["failed_permanent_files"] = int(
                job_store.count_failed_permanent_files(source_ids=source_ids)
                or 0
            )
        if hasattr(job_store, "count_claimed"):
            # claimed jobs (not distinct files) — best-effort detail
            try:
                claimed = 0
                for q in queues:
                    claimed += int(
                        job_store.count_claimed(q, source_ids=source_ids) or 0
                    )
                out["claimed_files"] = claimed  # approx (job count)
            except TypeError:
                try:
                    out["claimed_files"] = int(
                        job_store.count_claimed(source_ids=source_ids) or 0
                    )
                except Exception:
                    pass
    except Exception:
        pass
    return out


def summarize_archive_health(
    db: Any,
    *,
    job_store: Any | None = None,
    source_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Fast aggregate buckets for UI — no per-file full scan.

    Approximations (documented; buckets may overlap / not sum to total):
    - healthy ≈ ssot['preview'] (physical-ready preview baseline)
    - missing ≈ ssot['waiting_preview']
    - stale ≈ ssot['legacy_preview'] (+ legacy_thumbnail)
    - repair_pending ≈ JobStore pending files on light|preview|repair
    - broken ≈ COUNT preview_status LIKE 'preview_invalid%'
    - failed ≈ ssot['failed'] + JobStore failed_permanent_files
    - repair_needed = stale + repair_pending  (main 5-bucket UI)
    """
    from core.index_v3.ui_bridge import count_v3_ssot

    ssot = count_v3_ssot(db, source_ids)
    js = _jobstore_summary(job_store, source_ids)

    healthy = int(ssot.get("preview") or 0)
    missing = int(ssot.get("waiting_preview") or 0)
    stale = int(ssot.get("legacy_preview") or 0) + int(
        ssot.get("legacy_thumbnail") or 0
    )
    repair_pending = int(js.get("pending_light_preview_repair_files") or 0)
    broken = _count_preview_invalid(db, source_ids)
    failed = int(ssot.get("failed") or 0) + int(
        js.get("failed_permanent_files") or 0
    )
    repair_needed = stale + repair_pending
    unknown = 0
    total = int(ssot.get("total") or 0)

    return {
        "healthy": healthy,
        "missing": missing,
        "repair_needed": repair_needed,
        "broken": broken,
        "failed": failed,
        "stale": stale,
        "repair_pending": repair_pending,
        "unknown": unknown,
        "total": total,
        "ssot": {
            "preview": int(ssot.get("preview") or 0),
            "waiting_preview": int(ssot.get("waiting_preview") or 0),
            "legacy_preview": int(ssot.get("legacy_preview") or 0),
            "legacy_thumbnail": int(ssot.get("legacy_thumbnail") or 0),
            "failed": int(ssot.get("failed") or 0),
            "total": total,
            "light_complete": int(ssot.get("light_complete") or 0),
        },
        "jobstore": js,
    }


def format_archive_health_labels(summary: dict[str, Any]) -> dict[str, str]:
    """Turkish labels for the HealthPanel Arşiv Sağlığı group."""
    def _n(key: str) -> int:
        try:
            return int(summary.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "healthy": f"Sağlıklı: {_n('healthy'):,}",
        "missing": f"Eksik: {_n('missing'):,}",
        "repair_needed": f"Onarım gerekli: {_n('repair_needed'):,}",
        "broken": f"Bozuk: {_n('broken'):,}",
        "failed": f"Başarısız: {_n('failed'):,}",
    }


def job_store_path_for_db(db_path: str | Path) -> Path:
    """Same convention as app_status: <stem>.v3jobs.db beside patterns.db."""
    p = Path(db_path)
    return p.with_name(p.stem + ".v3jobs.db")


def open_job_store_for_settings(settings: Any) -> Any | None:
    """Lazily construct JobStore from settings.db_path; None on failure."""
    db_path = str(getattr(settings, "db_path", "") or "").strip()
    if not db_path:
        return None
    path = job_store_path_for_db(db_path)
    if not path.exists():
        return None
    try:
        from core.index_v3.queues import JobStore

        return JobStore(path)
    except Exception:
        return None


__all__ = [
    "HealthState",
    "PRIORITY",
    "merge_states",
    "assess_file_health",
    "summarize_archive_health",
    "format_archive_health_labels",
    "job_store_path_for_db",
    "open_job_store_for_settings",
]
