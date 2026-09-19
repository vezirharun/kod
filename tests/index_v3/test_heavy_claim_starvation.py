"""HEAVY claim must not starve large sources behind small-source dep_wait."""

from __future__ import annotations

from pathlib import Path

from core.index_v3.queues import JobStore
from core.index_v3.scheduler import claim_fair
from core.index_v3.types import Artifact, Job, QueueKind


def test_claim_fair_skips_dep_wait_small_source_for_real_heavy(tmp_path: Path):
    """Canlı regresyon: source=1 yalnızca dep_wait PATCH/OCR (küçük backlog),
    source=4'te HASH pending → claim source=4 HASH almalı (önceki davranış
    sonsuza dek source=1 dep_wait claim ederdi)."""
    store = JobStore(tmp_path / "jobs.db")
    # Küçük source — fair_claim_order bunu öne alır
    for i, art in enumerate((Artifact.PATCH, Artifact.OCR)):
        store.enqueue(
            [
                Job(
                    file_id=1000 + i,
                    artifact=art,
                    queue=QueueKind.HEAVY,
                    source_id=1,
                    path=f"s1_{art.value}.jpg",
                )
            ]
        )
        store.claim(
            QueueKind.HEAVY, "seed", source_id=1, limit=1, artifacts=(art,)
        )
        store.release_dep_wait(1000 + i, art)

    store.enqueue(
        [
            Job(
                file_id=4001,
                artifact=Artifact.HASH,
                queue=QueueKind.HEAVY,
                source_id=4,
                path="s4_hash.jpg",
            )
        ]
    )

    batch = claim_fair(
        store, QueueKind.HEAVY, "v3-heavy", [1, 4], limit=1
    )
    assert batch, "must claim a HEAVY job"
    assert batch[0].source_id == 4
    assert batch[0].artifact == Artifact.HASH


def test_claim_fair_dep_wait_fallback_when_no_ready_work(tmp_path: Path):
    """Hazır iş yoksa dep_wait hâlâ claim edilir (ileride preview/AI Final gelince)."""
    import time

    store = JobStore(tmp_path / "jobs.db")
    store.enqueue(
        [
            Job(
                file_id=1,
                artifact=Artifact.PATCH,
                queue=QueueKind.HEAVY,
                source_id=1,
                path="p.jpg",
            )
        ]
    )
    store.claim(QueueKind.HEAVY, "seed", source_id=1, limit=1)
    store.release_dep_wait(1, Artifact.PATCH)
    time.sleep(1.05)

    batch = claim_fair(store, QueueKind.HEAVY, "v3-heavy", [1], limit=1)
    assert len(batch) == 1
    assert batch[0].artifact == Artifact.PATCH


def test_complete_session_source_filter_does_not_drop_scoped_heavy(tmp_path: Path):
    """COMPLETE scope [1,4] iken source=4 pending HEAVY claim_fair ile görünür."""
    store = JobStore(tmp_path / "j.db")
    store.enqueue(
        [
            Job(1, Artifact.HASH, QueueKind.HEAVY, 4, "x.jpg"),
            Job(2, Artifact.DINO, QueueKind.HEAVY, 4, "y.jpg"),
        ]
    )
    assert store.count_pending(QueueKind.HEAVY, source_ids=[1, 4]) == 2
    batch = claim_fair(store, QueueKind.HEAVY, "v3-heavy", [1, 4], limit=1)
    assert batch[0].source_id == 4
