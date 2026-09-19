from core.index_v3.live_contract import build_live_progress
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, QueueKind


def test_fast_completion_requires_all_light_procedures():
    status = {
        "artifact_pools": {
            "total": 10,
            "thumbnail": 10,
            "preview": 10,
            "hash": 10,
            "metadata": 9,
            "light_complete": 9,
            "ai_final": 0,
        },
        "pending_light_jobs": 1,
        "pending_heavy_jobs": 0,
        "claimed_fresh_jobs": 0,
    }
    live = build_live_progress(status, claim={"phase": "idle"})
    assert live["fast"]["completed"] == 9
    assert live["fast"]["remaining"] == 1


def test_retry_count_is_distinct_files_and_retry_goes_to_end(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    jobs = [
        Job(1, Artifact.PREVIEW, QueueKind.PREVIEW, path="/a/one.tif"),
        Job(2, Artifact.PREVIEW, QueueKind.PREVIEW, path="/a/two.tif"),
        Job(3, Artifact.PREVIEW, QueueKind.PREVIEW, path="/a/three.tif"),
    ]
    store.enqueue(jobs)
    store.fail(3, Artifact.PREVIEW, error="decode error", max_attempts=3)
    assert store.count_pending_retry() == 1
    assert store.list_pending_retry_files()[0]["path"].endswith("three.tif")
    assert store.claim(QueueKind.PREVIEW, "w")[0].file_id == 1
    assert store.claim(QueueKind.PREVIEW, "w")[0].file_id == 2
    assert store.claim(QueueKind.PREVIEW, "w")[0].file_id == 3
