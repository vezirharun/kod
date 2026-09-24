"""Two-phase queue: normal FAST jobs before jumbo TIFF (fast_tif_defer_mb)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.index_v3.jumbo_phase import (
    JUMBO_AVAILABLE_AT_FAR,
    JUMBO_DEFER_MARKER,
    is_jumbo_for_queue,
)
from core.index_v3.queues import JobStore
from core.index_v3.scheduler import claim_fair
from core.index_v3.types import Artifact, Job, QueueKind


class _Settings:
    fast_tif_defer_mb = 64


def _job(
    fid: int,
    *,
    jumbo: bool,
    artifact: Artifact = Artifact.PREVIEW,
    source_id: int = 1,
) -> Job:
    if jumbo:
        path = f"\\\\server\\share\\big_{fid}.tif"
        size = 65 * 1024 * 1024
    else:
        path = f"\\\\server\\share\\ok_{fid}.jpg"
        size = 1024 * 1024
    return Job(
        fid,
        artifact,
        QueueKind.PREVIEW if artifact == Artifact.PREVIEW else QueueKind.LIGHT,
        source_id,
        path,
        file_size=size,
    )


def test_is_jumbo_uses_fast_tif_defer_mb_not_network_jpg():
    s = _Settings()
    assert is_jumbo_for_queue(r"\\server\a.tif", 65 * 1024 * 1024, s) is True
    assert is_jumbo_for_queue(r"\\server\a.tif", 10 * 1024 * 1024, s) is False
    assert is_jumbo_for_queue(r"\\server\a.jpg", 200 * 1024 * 1024, s) is False
    assert is_jumbo_for_queue(r"C:\local\small.tif", 65 * 1024 * 1024, s) is True


def test_100_normal_then_10_jumbo_claim_order(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    normals = [_job(i, jumbo=False) for i in range(1, 101)]
    jumbos = [_job(1000 + i, jumbo=True) for i in range(1, 11)]
    # Interleave enqueue order on purpose — phase must still prefer normals.
    mixed = []
    for i in range(10):
        mixed.extend(normals[i * 10 : (i + 1) * 10])
        mixed.append(jumbos[i])
    store.enqueue(mixed, settings=_Settings())

    assert store.count_jumbo_deferred(source_ids=[1]) == 10
    assert store.count_claimable_fast_pending(source_ids=[1]) == 100

    claimed_ids: list[int] = []
    for _ in range(100):
        batch = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=1)
        assert batch, "expected normal job"
        assert batch[0].file_id <= 100
        store.complete(batch[0].file_id, batch[0].artifact)
        claimed_ids.append(batch[0].file_id)

    assert claimed_ids == list(range(1, 101))
    assert store.count_claimable_fast_pending(source_ids=[1]) == 0

    # Still deferred until release
    assert claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=1) == []
    n = store.release_jumbo_deferred(source_ids=[1])
    assert n == 10
    assert store.count_jumbo_deferred(source_ids=[1]) == 0

    jumbo_ids: list[int] = []
    for _ in range(10):
        batch = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=1)
        assert batch
        jumbo_ids.append(batch[0].file_id)
        store.complete(batch[0].file_id, batch[0].artifact)
    assert jumbo_ids == list(range(1001, 1011))


def test_9800_normal_200_jumbo_simulation(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    settings = _Settings()
    jobs = [_job(i, jumbo=False) for i in range(1, 9801)]
    jobs.extend(_job(20_000 + i, jumbo=True) for i in range(1, 201))
    store.enqueue(jobs, settings=settings)

    assert store.count_claimable_fast_pending(source_ids=[1]) == 9800
    assert store.count_jumbo_deferred(source_ids=[1]) == 200

    # Sample: first 50 claims must all be normal; no jumbo id (>=20000)
    for _ in range(50):
        batch = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=1)
        assert batch and batch[0].file_id < 20_000
        store.complete(batch[0].file_id, batch[0].artifact)

    # Force phase transition without completing all 9800 (complete remaining normals via SQL)
    with store._lock, store._connect() as conn:
        conn.execute(
            """
            UPDATE index_v3_jobs SET state='done', available_at=0
            WHERE state='pending' AND available_at <= strftime('%s','now')
            """
        )
        conn.commit()

    assert store.count_claimable_fast_pending(source_ids=[1]) == 0
    assert store.count_jumbo_deferred(source_ids=[1]) == 200
    assert store.release_jumbo_deferred(source_ids=[1]) == 200

    batch = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=5)
    assert len(batch) == 5
    assert all(j.file_id >= 20_000 for j in batch)


def test_restart_preserves_phase1_deferral(tmp_path: Path):
    path = tmp_path / "jobs.db"
    store = JobStore(path)
    store.enqueue(
        [_job(1, jumbo=False), _job(2, jumbo=True)],
        settings=_Settings(),
    )
    assert store.count_jumbo_deferred(source_ids=[1]) == 1

    store2 = JobStore(path)
    assert store2.count_jumbo_deferred(source_ids=[1]) == 1
    assert store2.count_claimable_fast_pending(source_ids=[1]) == 1
    batch = claim_fair(store2, QueueKind.PREVIEW, "t", [1], limit=2)
    assert len(batch) == 1
    assert batch[0].file_id == 1


def test_phase2_release_then_claim(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([_job(1, jumbo=True), _job(2, jumbo=True)], settings=_Settings())
    assert store.count_claimable_fast_pending(source_ids=[1]) == 0
    assert store.release_jumbo_deferred(source_ids=[1]) == 2
    batch = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=2)
    assert {j.file_id for j in batch} == {1, 2}


def test_jumbo_fail_retry_stays_claimable_in_phase2(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([_job(1, jumbo=True)], settings=_Settings())
    store.release_jumbo_deferred(source_ids=[1])
    batch = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=1)
    assert batch
    store.fail(batch[0].file_id, batch[0].artifact, error="boom", max_attempts=3)
    # Soft retry pending with short available_at backoff
    assert store.count_pending(QueueKind.PREVIEW, source_ids=[1]) == 1
    # After backoff window, claimable (not re-FAR'd)
    import time

    time.sleep(1.05)
    again = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=1)
    assert again and again[0].file_id == 1


def test_normal_fail_retry_not_deferred_as_jumbo(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([_job(1, jumbo=False), _job(2, jumbo=True)], settings=_Settings())
    batch = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=1)
    assert batch[0].file_id == 1
    store.fail(batch[0].file_id, batch[0].artifact, error="tmp", max_attempts=3)
    import time

    time.sleep(1.05)
    # Jumbo still deferred; normal soft-retry claimable
    batch2 = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=1)
    assert batch2 and batch2[0].file_id == 1
    assert store.count_jumbo_deferred(source_ids=[1]) == 1


def test_new_normal_during_phase2_is_claimable(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([_job(10, jumbo=True)], settings=_Settings())
    store.release_jumbo_deferred(source_ids=[1])
    assert store.count_claimable_fast_pending(source_ids=[1]) == 1
    store.enqueue([_job(2, jumbo=False)], settings=_Settings())
    # New normal is immediately claimable (available_at=0); does not stay deferred.
    assert store.count_claimable_fast_pending(source_ids=[1]) == 2
    assert store.count_jumbo_deferred(source_ids=[1]) == 0
    seen: set[int] = set()
    for _ in range(2):
        batch = claim_fair(store, QueueKind.PREVIEW, "t", [1], limit=1)
        assert batch
        seen.add(batch[0].file_id)
        store.complete(batch[0].file_id, batch[0].artifact)
    assert seen == {2, 10}


def test_jobstore_marker_integrity(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([_job(1, jumbo=True)], settings=_Settings())
    with store._lock, store._connect() as conn:
        row = conn.execute(
            "SELECT available_at, error_msg FROM index_v3_jobs WHERE file_id=1"
        ).fetchone()
    assert int(row["available_at"]) >= JUMBO_AVAILABLE_AT_FAR
    assert row["error_msg"] == JUMBO_DEFER_MARKER
