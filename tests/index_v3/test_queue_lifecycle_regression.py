from __future__ import annotations

import threading
import time
from pathlib import Path

from core.index_v3 import Artifact
from core.index_v3.queues import JobStore
from core.index_v3.types import Job, QueueKind
from core.index_v3.worker import ArtifactProcessor, Worker
from core.index_v3 import worker as worker_module


class _Db:
    def get_file_by_id(self, file_id):
        return {"id": file_id, "source_id": 1, "path": f"/x/{file_id}.jpg"}

    def get_features(self, file_id):
        return {}


class _ReadyReport:
    preview_ready = True

    @staticmethod
    def ready(_artifact):
        return True


def test_worker_waits_for_jobs_arriving_during_scan(tmp_path: Path, monkeypatch):
    """Tarama sırasında kuyruk kısa süre boşalırsa worker kapanmamalı."""
    store = JobStore(tmp_path / "jobs.db")
    first = Job(1, Artifact.DINO, QueueKind.HEAVY, 1, "/x/1.jpg")
    second = Job(2, Artifact.DINO, QueueKind.HEAVY, 1, "/x/2.jpg")
    store.enqueue([first])

    monkeypatch.setattr(worker_module, "IDLE_GRACE_SEC", 0.5)
    monkeypatch.setattr(worker_module, "IDLE_POLL_SEC", 0.02)
    monkeypatch.setattr(worker_module, "assess_file", lambda *a, **k: _ReadyReport())

    worker = Worker(_Db(), store, processor=ArtifactProcessor(process_fn=lambda *_: True))

    def add_late_job():
        time.sleep(0.08)
        store.enqueue([second])

    t = threading.Thread(target=add_late_job)
    t.start()
    stats = worker.run_queue(QueueKind.HEAVY, [1])
    t.join()

    assert stats.completed == 2
    assert store.count_pending(source_ids=[1]) == 0
    assert store.count_claimed() == 0


def test_worker_releases_own_claims_on_loop_exit(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    job = Job(1, Artifact.DINO, QueueKind.HEAVY, 1, "/x/1.jpg")
    store.enqueue([job])
    claimed = store.claim(QueueKind.HEAVY, "v3-test")
    assert claimed
    assert store.requeue_claims_for_worker("v3-test") == 1
    assert store.count_claimed() == 0
    assert store.count_pending(source_ids=[1]) == 1


def test_live_remaining_excludes_terminal_error_files():
    from core.index_v3.live_contract import build_live_progress

    status = {
        "artifact_pools": {
            "total": 100,
            "thumbnail": 100,
            "preview": 100,
            "dino": 80,
            "clip": 80,
            "texture": 80,
            "semantic": 80,
            "dna": 80,
            "patch": 80,
            "ai_final": 80,
        },
        "pending_light_jobs": 0,
        "pending_heavy_jobs": 20,
        "claimed_fresh_jobs": 0,
        "terminal_error_files": 5,
    }
    live = build_live_progress(status)
    assert live["general_ai"]["completed"] == 80
    assert live["general_ai"]["remaining"] == 15
    assert live["terminal_error_files"] == 5


def test_run_with_timeout_does_not_wait_for_hung_thread():
    """fut.result timeout sonrası executor shutdown hung işi beklememeli."""
    from core.index_v3.safe_decode import run_with_timeout

    def _hang():
        time.sleep(30)
        return "done"

    t0 = time.monotonic()
    try:
        run_with_timeout(_hang, timeout_sec=0.6, retries=0)
        raised = False
    except TimeoutError:
        raised = True
    elapsed = time.monotonic() - t0
    assert raised
    assert elapsed < 3.0, elapsed


def test_timeout_failure_is_permanent_not_retry_spin(tmp_path: Path, monkeypatch):
    """Timeout iş pending retry'de dönmesin; kalıcı hata olsun, kuyruk boşalsın."""
    store = JobStore(tmp_path / "jobs.db")
    job = Job(1, Artifact.PREVIEW, QueueKind.PREVIEW, 1, "/x/1.psd")
    store.enqueue([job])

    class _Missing:
        preview_ready = False

        @staticmethod
        def ready(_artifact):
            return False

    monkeypatch.setattr(worker_module, "assess_file", lambda *a, **k: _Missing())
    monkeypatch.setattr(worker_module, "IDLE_GRACE_SEC", 0.15)
    monkeypatch.setattr(worker_module, "IDLE_POLL_SEC", 0.02)

    def _timeout_proc(db, job, stats):
        raise TimeoutError("v3_decode_timeout:8.0s")

    worker = Worker(
        _Db(),
        store,
        processor=ArtifactProcessor(process_fn=_timeout_proc),
        worker_id="v3-test",
        max_attempts=2,
    )
    stats = worker.run_queue(QueueKind.PREVIEW, [1])
    assert stats.failed >= 1
    assert store.count_pending(QueueKind.PREVIEW, source_ids=[1]) == 0
    assert store.count_pending_retry(QueueKind.PREVIEW, source_ids=[1]) == 0
    assert store.count_failed_permanent(source_ids=[1]) >= 1


def test_discovery_skips_macos_sidecar_files(tmp_path):
    from core.index_v3 import discovery
    root = tmp_path / "src"
    root.mkdir()
    (root / "real.jpg").write_bytes(b"x")
    (root / "._real.jpg").write_bytes(b"x")
    (root / ".DS_Store").write_bytes(b"x")
    rows = list(discovery._iter_files(str(root)))
    names = [Path(p).name for p, _, _ in rows]
    assert names == ["real.jpg"]


def test_failed_job_goes_to_end_of_queue_and_retries_after_normal_jobs(tmp_path: Path):
    """Hata alan iş ilk turun başına dönmez; normal işler bittikten sonra retry edilir."""
    store = JobStore(tmp_path / "order.db")
    jobs = [
        Job(1, Artifact.THUMBNAIL, QueueKind.LIGHT, 1, "/x/1.jpg"),
        Job(2, Artifact.THUMBNAIL, QueueKind.LIGHT, 1, "/x/2.jpg"),
        Job(3, Artifact.THUMBNAIL, QueueKind.LIGHT, 1, "/x/3.jpg"),
    ]
    store.enqueue(jobs)

    first = store.claim(QueueKind.LIGHT, "w")[0]
    assert first.file_id == 1
    store.fail(first.file_id, first.artifact, error="temporary", max_attempts=3)

    # Normal işler önce: 2 ve 3. 1 hemen tekrar claim edilmez.
    nxt = store.claim(QueueKind.LIGHT, "w", limit=2)
    assert [j.file_id for j in nxt] == [2, 3]
    for j in nxt:
        store.complete(j.file_id, j.artifact)

    retry = store.claim(QueueKind.LIGHT, "w")[0]
    assert retry.file_id == 1
    store.fail(retry.file_id, retry.artifact, error="temporary", max_attempts=3)

    later = [
        Job(4, Artifact.THUMBNAIL, QueueKind.LIGHT, 1, "/x/4.jpg"),
        Job(5, Artifact.THUMBNAIL, QueueKind.LIGHT, 1, "/x/5.jpg"),
    ]
    store.enqueue(later)
    nxt2 = store.claim(QueueKind.LIGHT, "w", limit=2)
    assert [j.file_id for j in nxt2] == [4, 5]
    for j in nxt2:
        store.complete(j.file_id, j.artifact)

    retry2 = store.claim(QueueKind.LIGHT, "w")[0]
    assert retry2.file_id == 1
    store.fail(retry2.file_id, retry2.artifact, error="temporary", max_attempts=3)
    assert store.job_state(1, Artifact.THUMBNAIL) == "failed_permanent"
    assert store.claim(QueueKind.LIGHT, "w") == []


def test_retry_counter_excludes_dependency_wait(tmp_path: Path):
    """dep_wait gerçek hata değildir; retry sayacını şişirmemeli."""
    store = JobStore(tmp_path / "counter.db")
    jobs = [
        Job(1, Artifact.PREVIEW, QueueKind.PREVIEW, 1, "/x/1.jpg"),
        Job(2, Artifact.PREVIEW, QueueKind.PREVIEW, 1, "/x/2.jpg"),
    ]
    store.enqueue(jobs)
    a = store.claim(QueueKind.PREVIEW, "w")[0]
    b = store.claim(QueueKind.PREVIEW, "w")[0]
    store.release_dep_wait(a.file_id, a.artifact)
    store.fail(b.file_id, b.artifact, error="temporary", max_attempts=2)
    assert store.count_pending_retry(QueueKind.PREVIEW, source_ids=[1]) == 1
