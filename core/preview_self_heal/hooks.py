"""Thin integration helpers for Modes A/B/C — no parallel preview system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.index_analysis_guards import local_artifact_exists
from core.logger import setup_logger
from core.preview_self_heal.repair import repair_preview_artifact
from core.preview_self_heal.validate import Verdict, ValidationResult, validate_preview

logger = setup_logger(__name__)

# Low-prio recheck markers (DB preview_status) — never rewrite Index algorithm.
STATUS_SUSPICIOUS = "preview_suspicious"
STATUS_HEAL_FAILED = "preview_heal_failed"
STATUS_OK = "preview_ok"


def gate_new_preview(
    artifact_path: str,
    source_path: str,
    *,
    compare_path: str | None = None,
    allow_source_compare: bool = True,
) -> ValidationResult:
    """Mode C — light gate after produce. Prefer local compare_path (no NAS).

    INVALID → caller must repair / not mark ready.
    SUSPICIOUS without compare → return suspicious (caller schedules recheck).
    """
    peer = compare_path
    # Avoid NAS for every valid preview: only compare when light says suspicious
    # and we have a local peer (render_path) or explicit allow + local source.
    light_first = validate_preview(
        artifact_path, source_path, allow_source_compare=False
    )
    if light_first.verdict != Verdict.SUSPICIOUS:
        return light_first
    if not allow_source_compare:
        return light_first
    if peer and local_artifact_exists(peer):
        return validate_preview(
            artifact_path,
            source_path,
            allow_source_compare=True,
            compare_path=peer,
        )
    # Source on local disk only — skip UNC/network
    try:
        from core.network_index_throttle import is_network_path

        if source_path and not is_network_path(source_path):
            return validate_preview(
                artifact_path,
                source_path,
                allow_source_compare=True,
                compare_path=source_path,
            )
    except Exception:
        pass
    return light_first


def invalidate_if_bad_preview(
    db: Any,
    report: Any,
    *,
    allow_source_compare: bool = True,
    settings: Any | None = None,
) -> Any:
    """Mode A — if PREVIEW looks READY but content is bad, mark MISSING/INVALID.

    VALID → no-op (skip re-render). Mutates report.status when repair needed.
    """
    from core.index_v3.types import Artifact, ArtifactStatus

    if report is None:
        return report
    if not report.ready(Artifact.PREVIEW):
        return report

    row = db.get_file_by_id(int(report.file_id)) or {}
    preview = str(row.get("feature_preview_path") or "").strip()
    path = str(report.path or row.get("path") or "").strip()
    if not preview:
        report.status[Artifact.PREVIEW] = ArtifactStatus.MISSING
        return report

    if not local_artifact_exists(preview):
        report.status[Artifact.PREVIEW] = ArtifactStatus.INVALID
        try:
            db.update_physical_readiness(
                int(report.file_id),
                thumbnail_ready=bool(
                    local_artifact_exists(str(row.get("thumbnail_path") or ""))
                ),
                preview_ready=False,
                requeue_missing=False,
            )
            db.upsert_file(
                {
                    "path": path,
                    "preview_status": "preview_missing_physical",
                }
            )
        except Exception as exc:
            logger.debug("invalidate missing physical: %s", exc)
        return report

    vr = validate_preview(
        preview,
        path,
        allow_source_compare=allow_source_compare,
        compare_path=None,
    )
    # EPS/AI: mono preview without proven mono source → reopen (white-page)
    ext = Path(path).suffix.lower()
    if (
        vr.verdict == Verdict.SUSPICIOUS
        and ext in {".eps", ".ai"}
        and "source_matches_mono" not in vr.reason
    ):
        from core.preview_self_heal.validate import ValidationResult

        vr = ValidationResult(
            Verdict.INVALID,
            f"blank_white_preview:{vr.reason}",
            artifact_path=preview,
        )
    if vr.verdict == Verdict.VALID:
        return report

    if vr.verdict == Verdict.SUSPICIOUS and not allow_source_compare:
        try:
            db.upsert_file({"path": path, "preview_status": STATUS_SUSPICIOUS})
        except Exception:
            pass
        return report

    if vr.verdict == Verdict.SUSPICIOUS:
        # Still ambiguous — mark soft status, do not force re-render all
        try:
            db.upsert_file({"path": path, "preview_status": STATUS_SUSPICIOUS})
        except Exception:
            pass
        return report

    # INVALID → clear ready so planner requeues Preview
    report.status[Artifact.PREVIEW] = ArtifactStatus.INVALID
    try:
        # Unlink bad artifact only (cache)
        from core.preview_self_heal.repair import _unlink_artifact

        _unlink_artifact(preview)
        db.update_physical_readiness(
            int(report.file_id),
            thumbnail_ready=bool(
                local_artifact_exists(str(row.get("thumbnail_path") or ""))
            ),
            preview_ready=False,
            requeue_missing=False,
        )
        db.upsert_file(
            {
                "path": path,
                "feature_preview_path": "",
                "preview_status": f"preview_invalid:{vr.reason}"[:120],
            }
        )
    except Exception as exc:
        logger.debug("invalidate bad preview: %s", exc)
    _ = settings
    return report


def heal_candidates_for_sources(
    db: Any,
    store: Any,
    source_ids: list[int],
    *,
    limit: int = 40,
    settings: Any | None = None,
    index_busy: bool = False,
) -> dict[str, Any]:
    """Mode B — candidate-only heal enqueue. Throttle when index busy.

    Candidates: DB path without file, physical not ready, heal-failed / suspicious
    with light INVALID after optional compare. Never walks whole archive images.
    """
    from core.index_v3.artifact_state import assess_file
    from core.index_v3.planner import plan_jobs_for_file
    from core.index_v3.types import Artifact, Mode

    stats = {
        "examined": 0,
        "enqueued": 0,
        "skipped_valid": 0,
        "suspicious": 0,
        "invalid": 0,
        "throttled": False,
    }
    if not source_ids:
        return stats
    if index_busy:
        limit = max(5, min(int(limit), 15))
        stats["throttled"] = True

    ph = ",".join("?" * len(source_ids))
    sql = f"""
        SELECT id FROM files
        WHERE source_id IN ({ph})
          AND status NOT IN ('excluded_internal','missing')
          AND (
            (ifnull(feature_preview_path,'')!='' AND ifnull(physical_preview_ready,0)!=1)
            OR (ifnull(physical_preview_ready,0)=1 AND ifnull(feature_preview_path,'')='')
            OR ifnull(preview_status,'') LIKE 'preview_invalid%'
            OR ifnull(preview_status,'') LIKE 'preview_missing%'
            OR ifnull(preview_status,'') = ?
            OR ifnull(preview_status,'') = ?
            OR ifnull(preview_status,'') LIKE '%blank%'
            OR ifnull(preview_status,'') LIKE '%fail%'
          )
        ORDER BY id
        LIMIT ?
    """
    params: list[Any] = list(int(x) for x in source_ids)
    params.extend([STATUS_SUSPICIOUS, STATUS_HEAL_FAILED, int(limit)])
    try:
        with db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
    except Exception as exc:
        logger.debug("heal candidate query failed: %s", exc)
        return stats

    for row in rows:
        fid = int(row["id"] if not isinstance(row, int) else row)
        stats["examined"] += 1
        try:
            report = assess_file(db, fid, require_disk=True)
            # Content check when still "ready" but flagged suspicious
            frow = db.get_file_by_id(fid) or {}
            prev = str(frow.get("feature_preview_path") or "").strip()
            path = str(frow.get("path") or report.path or "")
            status = str(frow.get("preview_status") or "")
            if report.ready(Artifact.PREVIEW) and prev:
                # Artifact-only first; source compare only if suspicious flag
                allow = status == STATUS_SUSPICIOUS
                report = invalidate_if_bad_preview(
                    db,
                    report,
                    allow_source_compare=allow,
                    settings=settings,
                )
            jobs = plan_jobs_for_file(report, Mode.REPAIR, repair=True)
            light = [j for j in jobs if j.artifact in (Artifact.PREVIEW, Artifact.THUMBNAIL)]
            if not light:
                stats["skipped_valid"] += 1
                continue
            if any(j.artifact == Artifact.PREVIEW for j in light):
                if "suspicious" in status:
                    stats["suspicious"] += 1
                else:
                    stats["invalid"] += 1
            added = int(
                store.enqueue(
                    light,
                    reopen_permanent=True,
                    reopen_done=True,
                )
                or 0
            )
            stats["enqueued"] += added
        except Exception as exc:
            logger.debug("heal candidate fid=%s: %s", fid, exc)
            continue
        if index_busy and stats["enqueued"] >= limit:
            break
    return stats


def apply_gate_or_repair(
    *,
    source_path: str,
    artifact_path: str,
    create_fn: Any,
    compare_path: str | None = None,
    settings: Any | None = None,
) -> tuple[bool, str, str]:
    """Mode C helper: validate; on INVALID attempt one repair. Returns (ok, path, reason)."""
    vr = gate_new_preview(
        artifact_path,
        source_path,
        compare_path=compare_path,
        allow_source_compare=True,
    )
    if vr.verdict == Verdict.VALID:
        return True, artifact_path, vr.reason
    if vr.verdict == Verdict.SUSPICIOUS:
        # Do not serve as hard-valid; caller may still persist with soft status
        return True, artifact_path, f"suspicious:{vr.reason}"
    outcome = repair_preview_artifact(
        source_path,
        artifact_path=artifact_path,
        create_fn=create_fn,
        settings=settings,
        reason=vr.reason,
        compare_path=compare_path,
    )
    if outcome.ok and outcome.preview_path:
        return True, outcome.preview_path, outcome.reason
    return False, artifact_path, outcome.reason
