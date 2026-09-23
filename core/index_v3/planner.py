"""Index Engine V3 — sparse job planner (artifact-level, not file-level heavy)."""

from __future__ import annotations

from core.index_v3.types import (
    Artifact,
    ArtifactStatus,
    FileArtifactReport,
    Job,
    Mode,
    QueueKind,
)


def queue_for_artifact(art: Artifact) -> QueueKind:
    if art == Artifact.PREVIEW:
        return QueueKind.PREVIEW
    if art == Artifact.THUMBNAIL:
        return QueueKind.LIGHT
    # Hash/Metadata artık Genel AI lane'indedir ve Preview kapısına tabidir.
    return QueueKind.HEAVY


def _maybe_post_ga_job(
    report: FileArtifactReport,
    art: Artifact,
    *,
    repair: bool,
    manual: bool,
) -> Job | None:
    """PATCH/OCR: yalnız AI_FINAL sonrası. Tamamlanan iş auto'da tekrarlanmaz."""
    if not manual and not report.ai_final:
        return None
    if report.ready(art) and not repair:
        return None
    if (
        report.get(art) == ArtifactStatus.INVALID
        and not repair
        and not manual
    ):
        return None
    q = QueueKind.REPAIR if repair else QueueKind.HEAVY
    return Job(report.file_id, art, q, report.source_id, report.path)


def plan_jobs_for_file(
    report: FileArtifactReport,
    mode: Mode,
    *,
    repair: bool = False,
    ocr_enabled: bool = False,
    patch_enabled: bool = True,
    manual_patch: bool = False,
    manual_ocr: bool = False,
) -> list[Job]:
    """Artifact-level plan: Preview + Thumbnail independent; GA gated on Preview.

    FAST: Preview and Thumbnail are planned independently. Thumbnail may be
    produced from source (256px) without requiring FeaturePreview (1024).
    When a Preview already exists, the processor prefers preview→thumb.

    GENERAL_AI: Preview repair when missing (unblocks heavy); no Thumbnail
    invent / no heavy until Preview is READY.

    COMPLETE/REPAIR: both lanes; heavy still respects the Preview gate.
    """
    jobs: list[Job] = []
    want_fast = mode in (Mode.FAST, Mode.COMPLETE, Mode.REPAIR) or repair
    want_general = mode in (Mode.GENERAL_AI, Mode.COMPLETE, Mode.REPAIR) or repair

    # FAST lane: Preview and Thumbnail are independent light artifacts.
    if want_fast:
        if not report.ready(Artifact.PREVIEW):
            jobs.append(Job(report.file_id, Artifact.PREVIEW, QueueKind.PREVIEW, report.source_id, report.path))
        if not report.ready(Artifact.THUMBNAIL):
            jobs.append(Job(report.file_id, Artifact.THUMBNAIL, QueueKind.LIGHT, report.source_id, report.path))
    elif want_general and not report.preview_ready:
        # Smallest GA unblock: repair Preview only; no Thumbnail / no heavy yet.
        jobs.append(
            Job(
                report.file_id,
                Artifact.PREVIEW,
                QueueKind.PREVIEW,
                report.source_id,
                report.path,
            )
        )

    # GENERAL AI lane: Preview is a hard gate. No heavy without Preview READY.
    if want_general and report.preview_ready:
        general_arts: list[Artifact] = [
            Artifact.HASH,
            Artifact.METADATA,
            Artifact.DINO,
            Artifact.CLIP,
            Artifact.TEXTURE,
            Artifact.SEMANTIC,
            Artifact.DNA,
            Artifact.OBJECT_CONCEPT,
            Artifact.OWLV2,
        ]
        for art in general_arts:
            if report.ready(art):
                continue
            q = QueueKind.REPAIR if repair else QueueKind.HEAVY
            jobs.append(Job(report.file_id, art, q, report.source_id, report.path))
        if patch_enabled:
            job = _maybe_post_ga_job(
                report, Artifact.PATCH, repair=repair, manual=False
            )
            if job is not None:
                jobs.append(job)
        if ocr_enabled:
            job = _maybe_post_ga_job(
                report, Artifact.OCR, repair=repair, manual=False
            )
            if job is not None:
                jobs.append(job)
    # Manuel PATCH/OCR: AI_FINAL kapısı yok (otomatik hâlâ yukarıda).
    if manual_patch:
        job = _maybe_post_ga_job(
            report, Artifact.PATCH, repair=repair, manual=True
        )
        if job is not None:
            jobs.append(job)
    if manual_ocr:
        job = _maybe_post_ga_job(
            report, Artifact.OCR, repair=repair, manual=True
        )
        if job is not None:
            jobs.append(job)
    return jobs


def enqueue_post_ga(
    db: object,
    store: object,
    source_ids: list[int],
    *,
    patch: bool = False,
    ocr: bool = False,
    reopen_done: bool = False,
) -> int:
    """Manuel PATCH/OCR. AI_FINAL beklemez. Tamamlanan iş reopen_done olmadan açılmaz."""
    from core.index_v3.artifact_state import assess_file

    if not patch and not ocr:
        return 0
    n = 0
    for sid in source_ids:
        with db.connect() as conn:
            rows = conn.execute(
                """
                SELECT id FROM files
                WHERE source_id=? AND status NOT IN ('excluded_internal','missing')
                """,
                (int(sid),),
            ).fetchall()
        for row in rows:
            fid = int(row[0] if not hasattr(row, "keys") else row["id"])
            report = assess_file(db, fid, require_disk=False)
            report.source_id = int(sid)
            jobs = plan_jobs_for_file(
                report,
                Mode.GENERAL_AI,
                repair=reopen_done,
                patch_enabled=False,
                ocr_enabled=False,
                manual_patch=bool(patch),
                manual_ocr=bool(ocr),
            )
            extra = [
                j
                for j in jobs
                if j.artifact
                in (
                    *( (Artifact.PATCH,) if patch else () ),
                    *( (Artifact.OCR,) if ocr else () ),
                )
            ]
            n += int(
                store.enqueue(
                    extra,
                    reopen_permanent=True,
                    reopen_done=bool(reopen_done),
                )
            )
    return n


def reconcile_stale_pendings(store: object, report: FileArtifactReport) -> int:
    """Complete-skip READY pendings; cancel work blocked by permanent Preview failure.

    Does not mass-delete learned/done rows — only pending rows that are obsolete
    after re-assess (gap scan / claim reconcile).
    """
    pending = store.list_pending_artifacts(int(report.file_id))
    if not pending:
        return 0
    preview_dead = (
        not report.preview_ready
        and store.job_state(int(report.file_id), Artifact.PREVIEW)
        == "failed_permanent"
    )
    n = 0
    for art in pending:
        if report.ready(art):
            store.complete(int(report.file_id), art)
            n += 1
            continue
        if preview_dead and art != Artifact.PREVIEW:
            n += int(store.cancel_pending_artifact(int(report.file_id), art) or 0)
    return n


def jobs_equal_sparse(expected_missing: set[str], jobs: list[Job]) -> bool:
    got = {j.artifact.value for j in jobs}
    return got == expected_missing
