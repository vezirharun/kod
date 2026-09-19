from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.types import Artifact, ArtifactStatus, FileArtifactReport, Mode, QueueKind


def report(preview=False, thumb=False, hash=False, metadata=False, dino=False):
    r=FileArtifactReport(file_id=1, source_id=1, path='x')
    for a,v in [(Artifact.PREVIEW,preview),(Artifact.THUMBNAIL,thumb),(Artifact.HASH,hash),(Artifact.METADATA,metadata),(Artifact.DINO,dino)]:
        r.status[a]=ArtifactStatus.READY if v else ArtifactStatus.MISSING
    return r


def test_fast_preview_first_then_thumbnail():
    jobs=plan_jobs_for_file(report(), Mode.FAST)
    assert [j.artifact for j in jobs] == [Artifact.PREVIEW]
    r=report(preview=True)
    jobs=plan_jobs_for_file(r, Mode.FAST)
    assert [j.artifact for j in jobs] == [Artifact.THUMBNAIL]
    assert jobs[0].queue == QueueKind.LIGHT


def test_general_ai_repairs_preview_then_plans_heavy():
    jobs=plan_jobs_for_file(report(), Mode.GENERAL_AI)
    assert [j.artifact for j in jobs] == [Artifact.PREVIEW]
    assert jobs[0].queue == QueueKind.PREVIEW
    jobs=plan_jobs_for_file(report(preview=True), Mode.GENERAL_AI)
    arts=[j.artifact for j in jobs]
    assert Artifact.PREVIEW not in arts
    assert Artifact.HASH in arts
    assert Artifact.METADATA in arts
    assert Artifact.DINO in arts
    assert Artifact.OBJECT_CONCEPT in arts
    assert Artifact.OWLV2 in arts
    assert all(j.queue == QueueKind.HEAVY for j in jobs)


def test_object_index_waits_for_preview_and_is_not_fast_index():
    jobs = plan_jobs_for_file(report(), Mode.GENERAL_AI)
    assert {j.artifact for j in jobs} == {Artifact.PREVIEW}
    assert Artifact.OBJECT_CONCEPT not in {j.artifact for j in jobs}
    fast = {j.artifact for j in plan_jobs_for_file(report(preview=True), Mode.FAST)}
    assert Artifact.OBJECT_CONCEPT not in fast
    assert Artifact.OWLV2 not in fast
    assert Artifact.PREVIEW not in fast


def test_fast_is_complete_without_hash_metadata():
    r=report(preview=True, thumb=True, hash=False, metadata=False)
    assert r.light_complete
    jobs=plan_jobs_for_file(r, Mode.FAST)
    assert jobs == []
    assert not r.ready(Artifact.HASH)
    assert not r.ready(Artifact.METADATA)
