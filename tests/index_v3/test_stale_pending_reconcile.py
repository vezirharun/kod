"""Stale JobStore pending: complete-skip when READY; cancel when preview permanently failed."""

from __future__ import annotations

from pathlib import Path

from core.index_v3.planner import reconcile_stale_pendings
from core.index_v3.queues import JobStore
from core.index_v3.types import (
    Artifact,
    ArtifactStatus,
    FileArtifactReport,
    Job,
    QueueKind,
)


def _report(**ready: bool) -> FileArtifactReport:
    r = FileArtifactReport(file_id=1, source_id=1, path="x.jpg")
    for art in Artifact:
        r.status[art] = (
            ArtifactStatus.READY if ready.get(art.value) else ArtifactStatus.MISSING
        )
    return r


def test_reconcile_completes_pending_preview_when_physical_ready(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue(
        [Job(1, Artifact.PREVIEW, QueueKind.PREVIEW, 1, "x.jpg")]
    )
    assert store.job_state(1, Artifact.PREVIEW) == "pending"

    n = reconcile_stale_pendings(store, _report(preview=True))
    assert n == 1
    assert store.job_state(1, Artifact.PREVIEW) == "done"


def test_reconcile_cancels_patch_ocr_when_preview_failed_permanent(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue(
        [
            Job(1, Artifact.PREVIEW, QueueKind.PREVIEW, 1, "x.jpg"),
            Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "x.jpg"),
            Job(1, Artifact.OCR, QueueKind.HEAVY, 1, "x.jpg"),
            Job(1, Artifact.OBJECT_CONCEPT, QueueKind.HEAVY, 1, "x.jpg"),
        ]
    )
    store.fail(1, Artifact.PREVIEW, error="decode_failed", permanent=True)
    assert store.job_state(1, Artifact.PREVIEW) == "failed_permanent"

    n = reconcile_stale_pendings(store, _report())
    assert n >= 3
    assert store.job_state(1, Artifact.PREVIEW) == "failed_permanent"
    assert store.job_state(1, Artifact.PATCH) == ""
    assert store.job_state(1, Artifact.OCR) == ""
    assert store.job_state(1, Artifact.OBJECT_CONCEPT) == ""


def test_reconcile_does_not_cancel_heavy_while_preview_still_pending(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue(
        [
            Job(1, Artifact.PREVIEW, QueueKind.PREVIEW, 1, "x.jpg"),
            Job(1, Artifact.DINO, QueueKind.HEAVY, 1, "x.jpg"),
        ]
    )
    n = reconcile_stale_pendings(store, _report())
    assert n == 0
    assert store.job_state(1, Artifact.PREVIEW) == "pending"
    assert store.job_state(1, Artifact.DINO) == "pending"
