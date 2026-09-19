"""GENERAL_AI may plan Preview repair when physical preview is missing."""

from __future__ import annotations

from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.types import Artifact, ArtifactStatus, FileArtifactReport, Mode, QueueKind


def _report(**ready: bool) -> FileArtifactReport:
    r = FileArtifactReport(file_id=1, source_id=1, path="x.jpg")
    for art in Artifact:
        r.status[art] = (
            ArtifactStatus.READY if ready.get(art.value) else ArtifactStatus.MISSING
        )
    return r


def test_general_ai_plans_preview_repair_when_missing():
    jobs = plan_jobs_for_file(_report(), Mode.GENERAL_AI)
    assert [j.artifact for j in jobs] == [Artifact.PREVIEW]
    assert jobs[0].queue == QueueKind.PREVIEW
    # No heavy work until Preview exists.
    assert Artifact.DINO not in {j.artifact for j in jobs}
    assert Artifact.HASH not in {j.artifact for j in jobs}


def test_general_ai_plans_heavy_only_when_preview_ready():
    jobs = plan_jobs_for_file(_report(preview=True), Mode.GENERAL_AI)
    arts = {j.artifact for j in jobs}
    assert Artifact.PREVIEW not in arts
    assert Artifact.HASH in arts
    assert Artifact.DINO in arts
