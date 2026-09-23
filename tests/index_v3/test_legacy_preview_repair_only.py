"""Legacy repair: missing physical Preview only — never re-DINO/CLIP/DNA."""

from __future__ import annotations

from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.types import Artifact, ArtifactStatus, FileArtifactReport, Mode


def _legacy_dino_preview_missing() -> FileArtifactReport:
    """Example B: DINO present + physical preview missing → only preview job."""
    r = FileArtifactReport(file_id=1, source_id=1, path="legacy.jpg")
    for art in Artifact:
        r.status[art] = ArtifactStatus.MISSING
    r.status[Artifact.DINO] = ArtifactStatus.READY
    r.status[Artifact.CLIP] = ArtifactStatus.READY
    r.status[Artifact.HASH] = ArtifactStatus.READY
    r.status[Artifact.METADATA] = ArtifactStatus.READY
    r.status[Artifact.TEXTURE] = ArtifactStatus.READY
    r.status[Artifact.SEMANTIC] = ArtifactStatus.READY
    r.status[Artifact.DNA] = ArtifactStatus.READY
    # Preview / Thumbnail physically missing
    return r


def test_example_b_repair_only_plans_preview():
    jobs = plan_jobs_for_file(_legacy_dino_preview_missing(), Mode.REPAIR, repair=True)
    assert {j.artifact for j in jobs} == {Artifact.PREVIEW, Artifact.THUMBNAIL}


def test_example_b_complete_only_plans_preview():
    jobs = plan_jobs_for_file(_legacy_dino_preview_missing(), Mode.COMPLETE)
    assert {j.artifact for j in jobs} == {Artifact.PREVIEW, Artifact.THUMBNAIL}


def test_example_b_general_ai_only_plans_preview_repair():
    jobs = plan_jobs_for_file(_legacy_dino_preview_missing(), Mode.GENERAL_AI)
    assert [j.artifact for j in jobs] == [Artifact.PREVIEW]


def test_example_a_complete_file_plans_empty():
    r = _legacy_dino_preview_missing()
    r.status[Artifact.PREVIEW] = ArtifactStatus.READY
    r.status[Artifact.THUMBNAIL] = ArtifactStatus.READY
    r.status[Artifact.OBJECT_CONCEPT] = ArtifactStatus.READY
    r.status[Artifact.OWLV2] = ArtifactStatus.READY
    r.status[Artifact.PATCH] = ArtifactStatus.READY
    r.status[Artifact.OCR] = ArtifactStatus.READY
    assert plan_jobs_for_file(r, Mode.COMPLETE) == []
    # REPAIR may re-touch post-GA with repair=True; must not re-DINO/CLIP/DNA/Preview.
    arts = {j.artifact for j in plan_jobs_for_file(r, Mode.REPAIR, repair=True)}
    assert Artifact.PREVIEW not in arts
    assert Artifact.DINO not in arts
    assert Artifact.CLIP not in arts
    assert Artifact.DNA not in arts
    assert Artifact.TEXTURE not in arts
    assert Artifact.SEMANTIC not in arts
