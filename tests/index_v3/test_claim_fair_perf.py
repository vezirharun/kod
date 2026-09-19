"""claim_fair must not load all pending rows (perf)."""

from pathlib import Path

from core.db import Database
from core.index_v3.queues import JobStore
from core.index_v3.scheduler import claim_fair, measure_backlogs
from core.index_v3.types import Artifact, Job, QueueKind


def test_measure_backlogs_grouped_not_full_scan(tmp_path: Path):
    store = JobStore(tmp_path / "j.db")
    jobs = [
        Job(
            file_id=i,
            artifact=Artifact.THUMBNAIL,
            queue=QueueKind.LIGHT,
            source_id=1 if i < 50 else 2,
            path=f"/x/{i}.jpg",
        )
        for i in range(100)
    ]
    # poison other source
    jobs += [
        Job(
            file_id=10_000 + i,
            artifact=Artifact.DINO,
            queue=QueueKind.HEAVY,
            source_id=99,
            path=f"/o/{i}.jpg",
        )
        for i in range(200)
    ]
    store.enqueue(jobs)
    backs = measure_backlogs(store, [1, 2])
    assert len(backs) == 2
    assert backs[0].source_id == 1 and backs[0].light == 50
    assert backs[1].source_id == 2 and backs[1].light == 50
    assert all(b.heavy == 0 for b in backs)

    claimed = claim_fair(store, QueueKind.LIGHT, "w", [1, 2], limit=3)
    assert len(claimed) == 3
    assert all(j.source_id in (1, 2) for j in claimed)
    # source 99 asla
    assert claim_fair(store, QueueKind.HEAVY, "w", [1, 2], limit=1) == []
