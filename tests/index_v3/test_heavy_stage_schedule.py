"""Heavy stage scheduler: deps, no duplicate claim, AI Final gate."""

from __future__ import annotations

import threading
import time
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from core.db import Database
from core.index_v3 import ARTIFACT_DEPENDENCIES, Artifact, IndexEngineV3, Mode
from core.index_v3.types import HEAVY_ARTIFACTS
from core.index_v3.artifact_state import assess_file
from core.index_v3.queues import JobStore
from core.index_v3.scheduler import claim_fair, heavy_artifact_lanes
from core.index_v3.types import Job, QueueKind
from core.index_v3.worker import ArtifactProcessor
from core.index_v3 import worker as worker_module


def _imgs(folder: Path, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (32, 32), color=(i % 180, 40, 90)).save(
            folder / f"f_{i:03d}.jpg", "JPEG"
        )


def test_dependency_graph_from_code():
    assert ARTIFACT_DEPENDENCIES[Artifact.DINO] == ()
    assert ARTIFACT_DEPENDENCIES[Artifact.CLIP] == ()
    assert ARTIFACT_DEPENDENCIES[Artifact.TEXTURE] == ()
    assert ARTIFACT_DEPENDENCIES[Artifact.OCR] == ()
    assert ARTIFACT_DEPENDENCIES[Artifact.PATCH] == ()
    assert ARTIFACT_DEPENDENCIES[Artifact.SEMANTIC] == (Artifact.TEXTURE,)
    assert ARTIFACT_DEPENDENCIES[Artifact.DNA] == (Artifact.SEMANTIC,)


def test_heavy_lanes_use_worker_count():
    assert heavy_artifact_lanes(1) == [("v3-heavy", None)]
    lanes4 = heavy_artifact_lanes(4)
    assert len(lanes4) == 4
    names = [n for n, _ in lanes4]
    assert "v3-heavy-dino" in names
    dino_arts = dict(lanes4)["v3-heavy-dino"]
    assert Artifact.DINO in dino_arts and Artifact.PATCH in dino_arts
    assert Artifact.CLIP not in dino_arts


def test_claim_unique_under_contention(tmp_path: Path):
    store = JobStore(tmp_path / "j.db")
    store.enqueue(
        [
            Job(
                file_id=i,
                artifact=Artifact.DINO,
                queue=QueueKind.HEAVY,
                source_id=1,
                path=f"/x/{i}.jpg",
            )
            for i in range(40)
        ]
    )
    got: list[tuple[int, str]] = []
    lock = threading.Lock()

    def _worker(wid: str) -> None:
        for _ in range(20):
            batch = claim_fair(
                store,
                QueueKind.HEAVY,
                wid,
                [1],
                artifacts=(Artifact.DINO,),
            )
            if not batch:
                return
            with lock:
                got.append((batch[0].file_id, batch[0].artifact.value))
            store.complete(batch[0].file_id, batch[0].artifact)

    threads = [threading.Thread(target=_worker, args=(f"w{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 40
    assert len(set(got)) == 40


def test_worker_continues_after_preflight_exception(tmp_path: Path, monkeypatch):
    store = JobStore(tmp_path / "j.db")
    store.enqueue(
        [
            Job(
                file_id=i,
                artifact=Artifact.DINO,
                queue=QueueKind.HEAVY,
                source_id=1,
                path=f"/x/{i}.jpg",
            )
            for i in (1, 2)
        ]
    )

    class _Db:
        def get_file_by_id(self, file_id):
            return {"id": file_id, "source_id": 1, "path": f"/x/{file_id}.jpg"}

        def get_features(self, file_id):
            return {}

    class _Report:
        preview_ready = True

        @staticmethod
        def ready(_artifact):
            return True

    calls = {"count": 0}

    def _assess(_db, _file_id, *, require_disk=False):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("preflight boom")
        return _Report()

    monkeypatch.setattr(worker_module, "assess_file", _assess)
    worker = worker_module.Worker(
        _Db(),
        store,
        processor=ArtifactProcessor(process_fn=lambda *_args: True),
    )

    stats = worker.run_queue(QueueKind.HEAVY, [1])

    assert stats.failed == 1
    assert stats.completed == 2
    assert stats.processed == 3
    assert store.count_claimed() == 0
    assert store.count_pending() == 0


def test_940_queue_reclaims_stale_claim_and_preserves_retry(tmp_path: Path):
    store = JobStore(tmp_path / "j.db")
    jobs = [
        Job(
            file_id=i,
            artifact=Artifact.PREVIEW,
            queue=QueueKind.PREVIEW,
            source_id=1,
            path=f"/x/{i}.jpg",
        )
        for i in range(1, 941)
    ]
    assert store.enqueue(jobs) == 940
    first = store.claim(QueueKind.PREVIEW, "w1", source_id=1)
    assert len(first) == 1
    store.complete(first[0].file_id, first[0].artifact)
    assert store.enqueue([jobs[0]]) == 0
    assert store.count_pending(QueueKind.PREVIEW, source_ids=[1]) == 939
    claimed = store.claim(QueueKind.PREVIEW, "dead-worker", source_id=1)
    assert len(claimed) == 1
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE index_v3_jobs SET heartbeat_at=1, claimed_at=1 "
            "WHERE file_id=? AND artifact=?",
            (claimed[0].file_id, claimed[0].artifact.value),
        )
        conn.commit()

    assert store.count_claimed_fresh(timeout_sec=30) == 0
    assert store.requeue_stale_claims(timeout_sec=30) == 1
    assert store.count_claimed() == 0
    assert store.count_pending(QueueKind.PREVIEW, source_ids=[1]) == 939

    recovered = store.claim(QueueKind.PREVIEW, "w2", source_id=1)
    assert len(recovered) == 1
    store.complete(recovered[0].file_id, recovered[0].artifact)
    assert store.count_claimed() == 0
    assert store.count_pending(QueueKind.PREVIEW, source_ids=[1]) == 938

    retry_store = JobStore(tmp_path / "retry.db")
    retry_job = Job(
        file_id=10001,
        artifact=Artifact.PREVIEW,
        queue=QueueKind.PREVIEW,
        source_id=1,
        path="/x/retry.jpg",
    )
    retry_store.enqueue([retry_job])
    claimed_retry = retry_store.claim(QueueKind.PREVIEW, "w3", source_id=1)[0]
    retry_store.fail(
        claimed_retry.file_id,
        claimed_retry.artifact,
        error="temporary",
        max_attempts=3,
    )
    with sqlite3.connect(retry_store.db_path) as conn:
        attempts = conn.execute(
            "SELECT attempts FROM index_v3_jobs WHERE file_id=? AND artifact=?",
            (claimed_retry.file_id, claimed_retry.artifact.value),
        ).fetchone()[0]
    assert attempts == 1
    retry_store.enqueue([retry_job])
    with sqlite3.connect(retry_store.db_path) as conn:
        assert conn.execute(
            "SELECT attempts FROM index_v3_jobs WHERE file_id=? AND artifact=?",
            (claimed_retry.file_id, claimed_retry.artifact.value),
        ).fetchone()[0] == 1


def test_dead_worker_requeues_fresh_claim(tmp_path: Path):
    store = JobStore(tmp_path / "dead.db")
    job = Job(
        file_id=1,
        artifact=Artifact.THUMBNAIL,
        queue=QueueKind.LIGHT,
        source_id=1,
        path="/x/dead.jpg",
    )
    store.enqueue([job])
    assert store.claim(QueueKind.LIGHT, "dead-worker", source_id=1)
    assert store.requeue_claims_for_worker("dead-worker") == 1
    assert store.count_claimed() == 0
    assert store.count_pending(QueueKind.LIGHT, source_ids=[1]) == 1


def test_semantic_after_texture_and_ai_final(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    root = tmp_path / "src"
    _imgs(root, 8)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(root)),
        )
    order: list[tuple[int, str]] = []
    lock = threading.Lock()
    inner = ArtifactProcessor()

    def _process(db, job, stats):
        if job.artifact == Artifact.TEXTURE:
            time.sleep(0.04)
        with lock:
            order.append((job.file_id, job.artifact.value))
        return inner.process(db, job, stats)

    settings = SimpleNamespace(worker_count=4, ocr_enabled=False)
    eng = IndexEngineV3(
        db,
        job_db_path=tmp_path / "jobs.db",
        processor=ArtifactProcessor(process_fn=_process),
        settings=settings,
    )
    sources = [{"id": 1, "root_path": str(root)}]
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    by_file: dict[int, list[str]] = {}
    for fid, art in order:
        by_file.setdefault(fid, []).append(art)
    for fid, arts in by_file.items():
        if "semantic" in arts and "texture" in arts:
            assert arts.index("texture") < arts.index("semantic")
        if "dna" in arts and "semantic" in arts:
            assert arts.index("semantic") < arts.index("dna")
        report = assess_file(db, fid, require_disk=False)
        assert report.ai_final
        assert report.ready(Artifact.DINO)
        assert report.ready(Artifact.CLIP)
        assert report.ready(Artifact.TEXTURE)
        assert report.ready(Artifact.SEMANTIC)
        assert report.ready(Artifact.DNA)
        assert report.ready(Artifact.PATCH)


def test_claim_dino_before_fresh_patch(tmp_path: Path):
    store = JobStore(tmp_path / "prio.db")
    store.enqueue(
        [
            Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(1, Artifact.DINO, QueueKind.HEAVY, 1, "a.jpg"),
        ]
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE index_v3_jobs SET error_msg='dino_empty' WHERE artifact='dino'"
        )
        conn.commit()
    batch = store.claim(QueueKind.HEAVY, "w", source_id=1, limit=1)
    assert batch and batch[0].artifact == Artifact.DINO


def test_same_file_chain_before_other_file_hash(tmp_path: Path):
    from core.index_v3.worker import Worker

    class DB:
        def __init__(self):
            self.rows = {
                1: {"id": 1, "path": "a.jpg", "source_id": 1, "physical_preview_ready": 1, "feature_preview_path": "p.webp", "physical_thumbnail_ready": 1, "thumbnail_path": "t.webp", "format_metadata": "{}", "width": 8},
                2: {"id": 2, "path": "b.jpg", "source_id": 1, "physical_preview_ready": 1, "feature_preview_path": "p2.webp", "physical_thumbnail_ready": 1, "thumbnail_path": "t2.webp", "format_metadata": "{}", "width": 8},
            }
            self.feat = {1: {"phash": "", "texture_features": [], "texture_map": {}},
                         2: {"phash": "", "texture_features": [], "texture_map": {}}}

        def get_file_by_id(self, fid):
            return dict(self.rows[int(fid)])

        def get_features(self, fid):
            return dict(self.feat.get(int(fid)) or {})

        def upsert_features(self, fid, payload):
            self.feat.setdefault(int(fid), {}).update(payload)

        def upsert_file(self, payload):
            return None

        def update_physical_readiness(self, *a, **k):
            return None

        def connect(self):
            class C:
                def __enter__(self_inner):
                    return self_inner
                def __exit__(self_inner, *a):
                    return False
                def execute(self_inner, *a, **k):
                    return self_inner
            return C()

    store = JobStore(tmp_path / "chain.db")
    store.enqueue(
        [
            Job(1, Artifact.HASH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(1, Artifact.DINO, QueueKind.HEAVY, 1, "a.jpg"),
            Job(2, Artifact.HASH, QueueKind.HEAVY, 1, "b.jpg"),
        ]
    )
    order: list[tuple[int, str]] = []

    def proc(db, job, stats):
        order.append((job.file_id, job.artifact.value))
        if job.artifact == Artifact.HASH:
            db.feat[job.file_id]["phash"] = "aabb"
        if job.artifact == Artifact.DINO:
            db.feat[job.file_id]["dino_embedding"] = b"\x00\x00\x00\x01" * 4
        return True

    w = Worker(DB(), store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=2)
    assert order[0] == (1, "hash")
    assert order[1] == (1, "dino")
    assert store.job_state(1, Artifact.DINO) == "done"


def test_ocr_claimed_after_patch(tmp_path: Path):
    store = JobStore(tmp_path / "ocrlast.db")
    store.enqueue(
        [
            Job(1, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg"),
            Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(1, Artifact.SEMANTIC, QueueKind.HEAVY, 1, "a.jpg"),
        ]
    )
    batch = store.claim(QueueKind.HEAVY, "w", source_id=1, limit=1)
    assert batch and batch[0].artifact == Artifact.SEMANTIC


def test_general_ai_drain_reaches_repair_dino_while_heavy_has_patch(
    tmp_path: Path, monkeypatch
):
    """HEAVY'de PATCH varken REPAIR DINO claim edilmeli (Genel AI drain)."""
    monkeypatch.setattr(worker_module, "IDLE_GRACE_SEC", 0.02)
    from core.index_v3.worker import Worker

    class DB:
        def __init__(self):
            self.rows = {
                1: {
                    "id": 1,
                    "path": "a.jpg",
                    "source_id": 1,
                    "physical_preview_ready": 1,
                    "feature_preview_path": "p.webp",
                    "physical_thumbnail_ready": 1,
                    "thumbnail_path": "t.webp",
                    "format_metadata": "{}",
                    "width": 8,
                },
            }
            self.feat = {
                1: {
                    "phash": "aabb",
                    "texture_features": [1.0],
                    "texture_map": {},
                }
            }

        def get_file_by_id(self, fid):
            return dict(self.rows[int(fid)])

        def get_features(self, fid):
            return dict(self.feat.get(int(fid)) or {})

        def upsert_features(self, fid, payload):
            self.feat.setdefault(int(fid), {}).update(payload)

        def upsert_file(self, payload):
            return None

        def update_physical_readiness(self, *a, **k):
            return None

        def connect(self):
            class C:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *a):
                    return False

                def execute(self_inner, *a, **k):
                    return self_inner

            return C()

    store = JobStore(tmp_path / "ga_repair_dino.db")
    store.enqueue(
        [
            Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(1, Artifact.DINO, QueueKind.REPAIR, 1, "a.jpg"),
        ]
    )
    order: list[str] = []

    def proc(db, job, stats):
        order.append(job.artifact.value)
        if job.artifact == Artifact.DINO:
            db.feat[job.file_id]["dino_embedding"] = b"\x00\x00\x00\x01" * 4
        return True

    w = Worker(DB(), store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], artifacts=HEAVY_ARTIFACTS)
    w.run_queue(QueueKind.REPAIR, [1], artifacts=HEAVY_ARTIFACTS)
    assert order == ["dino"]
    assert store.job_state(1, Artifact.DINO) == "done"
    assert store.job_state(1, Artifact.PATCH) == "pending"


def test_general_ai_gap_does_not_reopen_done_dino(tmp_path, monkeypatch):
    """Heavy done stays done; reopen_done only applies to Preview/Thumbnail."""
    from core.index_v3.discovery import enqueue_existing_gaps

    db_path = tmp_path / "g.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE files(id INTEGER PRIMARY KEY, source_id INTEGER, status TEXT, path TEXT)"
    )
    conn.execute(
        "INSERT INTO files(id,source_id,status,path) VALUES(1,1,'pending','/x/1.jpg')"
    )
    conn.commit()
    conn.close()

    class _Db:
        def __init__(self, path):
            self.path = path

        def connect(self):
            c = sqlite3.connect(self.path)
            c.row_factory = sqlite3.Row
            return c

    class Report:
        file_id = 1
        source_id = 1
        path = "/x/1.jpg"
        preview_ready = True
        light_complete = True
        ai_final = False

        @staticmethod
        def ready(artifact):
            return artifact != Artifact.DINO

        @staticmethod
        def get(artifact):
            from core.index_v3.types import ArtifactStatus

            return (
                ArtifactStatus.MISSING
                if artifact == Artifact.DINO
                else ArtifactStatus.READY
            )

    def _report(_db, fid, require_disk=False):
        return Report()

    monkeypatch.setattr("core.index_v3.discovery.assess_file", _report)
    store = JobStore(tmp_path / "jobs.sqlite")
    store.enqueue([Job(1, Artifact.DINO, QueueKind.HEAVY, 1, "/x/1.jpg")])
    with store._connect() as c:
        c.execute("UPDATE index_v3_jobs SET state='done' WHERE artifact='dino'")
        c.commit()
    enqueue_existing_gaps(_Db(db_path), store, source_id=1, mode=Mode.GENERAL_AI)
    assert store.job_state(1, Artifact.DINO) == "done"


def test_claim_dep_wait_hash_does_not_starve_object_concept(tmp_path: Path):
    store = JobStore(tmp_path / "starve.db")
    store.enqueue(
        [
            Job(1, Artifact.HASH, QueueKind.HEAVY, 1, "blocked.jpg"),
            Job(2, Artifact.OBJECT_CONCEPT, QueueKind.HEAVY, 1, "ready.jpg"),
        ]
    )
    first = store.claim(QueueKind.HEAVY, "w", source_id=1, limit=1)
    assert first and first[0].artifact == Artifact.HASH
    store.release_dep_wait(first[0].file_id, first[0].artifact)
    nxt = store.claim(
        QueueKind.HEAVY,
        "w",
        source_id=1,
        limit=1,
        artifacts=(Artifact.HASH, Artifact.OBJECT_CONCEPT),
    )
    assert nxt and nxt[0].artifact == Artifact.OBJECT_CONCEPT


def test_claim_dino_retry_still_before_patch(tmp_path: Path):
    store = JobStore(tmp_path / "prio.db")
    store.enqueue(
        [
            Job(1, Artifact.DINO, QueueKind.HEAVY, 1, "a.jpg"),
            Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
        ]
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE index_v3_jobs SET error_msg='timeout' WHERE artifact='dino'"
        )
        conn.commit()
    nxt = store.claim(QueueKind.HEAVY, "w", source_id=1, limit=1)
    assert nxt and nxt[0].artifact == Artifact.DINO


def test_claim_ready_owl_before_dep_wait_patch_ocr(tmp_path: Path):
    store = JobStore(tmp_path / "owl_prio.db")
    store.enqueue(
        [
            Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "p.jpg"),
            Job(2, Artifact.OCR, QueueKind.HEAVY, 1, "o.jpg"),
            Job(3, Artifact.OWLV2, QueueKind.HEAVY, 1, "owl.jpg"),
            Job(4, Artifact.OWLV2, QueueKind.HEAVY, 1, "owl2.jpg"),
            Job(5, Artifact.OWLV2, QueueKind.HEAVY, 1, "owl3.jpg"),
            Job(6, Artifact.OWLV2, QueueKind.HEAVY, 1, "owl4.jpg"),
            Job(7, Artifact.OWLV2, QueueKind.HEAVY, 1, "owl5.jpg"),
        ]
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE index_v3_jobs SET error_msg='dep_wait' WHERE artifact IN ('patch','ocr')"
        )
        conn.commit()
    batch = store.claim(QueueKind.HEAVY, "w", source_id=1, limit=1)
    assert batch and batch[0].artifact == Artifact.OWLV2


def test_claim_dep_wait_owl_before_dep_wait_patch(tmp_path: Path):
    store = JobStore(tmp_path / "owl_dep.db")
    store.enqueue(
        [
            Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "p.jpg"),
            Job(2, Artifact.OCR, QueueKind.HEAVY, 1, "o.jpg"),
            Job(3, Artifact.OWLV2, QueueKind.HEAVY, 1, "owl.jpg"),
        ]
    )
    with store._connect() as conn:
        conn.execute("UPDATE index_v3_jobs SET error_msg='dep_wait'")
        conn.commit()
    batch = store.claim(QueueKind.HEAVY, "w", source_id=1, limit=1)
    assert batch and batch[0].artifact == Artifact.OWLV2


def test_heavy_dep_wait_patch_ocr_yields_to_repair_owl(tmp_path: Path, monkeypatch):
    """Patch/OCR dep_wait HEAVY'yi kilitlemesin; REPAIR READY OWL claim edilsin."""
    monkeypatch.setattr(worker_module, "IDLE_GRACE_SEC", 0.02)
    from core.index_v3.worker import Worker

    processed: set[int] = set()

    class Report:
        def __init__(self, fid: int):
            self.file_id = int(fid)
            self.preview_ready = True
            self.light_complete = True

        def ready(self, artifact):
            return artifact == Artifact.OWLV2 and self.file_id in processed

    monkeypatch.setattr(
        worker_module, "assess_file", lambda db, fid, require_disk=False: Report(fid)
    )

    class DB:
        def get_file_by_id(self, fid):
            return {
                "id": int(fid),
                "path": f"{fid}.jpg",
                "source_id": 1,
                "physical_preview_ready": 1,
                "feature_preview_path": "p.webp",
                "physical_thumbnail_ready": 1,
                "thumbnail_path": "t.webp",
                "format_metadata": "{}",
                "width": 8,
            }

        def get_features(self, fid):
            return {"phash": "aabb", "texture_map": {}}

        def upsert_features(self, fid, payload):
            return None

        def upsert_file(self, payload):
            return None

        def update_physical_readiness(self, *a, **k):
            return None

        def connect(self):
            class C:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *a):
                    return False

                def execute(self_inner, *a, **k):
                    return self_inner

            return C()

    store = JobStore(tmp_path / "owl_yield.db")
    jobs = [
        Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "p.jpg"),
        Job(2, Artifact.OCR, QueueKind.HEAVY, 1, "o.jpg"),
        Job(3, Artifact.PATCH, QueueKind.HEAVY, 1, "p2.jpg"),
    ]
    for i in range(5):
        jobs.append(Job(10 + i, Artifact.OWLV2, QueueKind.REPAIR, 1, f"owl{i}.jpg"))
    store.enqueue(jobs)
    order: list[str] = []

    def proc(db, job, stats):
        order.append(job.artifact.value)
        if job.artifact in (Artifact.PATCH, Artifact.OCR):
            raise RuntimeError("preview_required")
        processed.add(int(job.file_id))
        return True

    arts = (Artifact.PATCH, Artifact.OCR, Artifact.OWLV2)
    w = Worker(DB(), store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], artifacts=arts)
    w.run_queue(QueueKind.REPAIR, [1], artifacts=arts)
    assert "owlv2" in order
    assert store.job_state(10, Artifact.OWLV2) == "done"


def test_heavy_lane_does_not_exit_while_owl_side_pending():
    import inspect
    from core.index_v3.engine import IndexEngineV3

    src = inspect.getsource(IndexEngineV3._run_inner)
    assert "_ga_side_pending" in src
    assert "GA_SIDE_ARTIFACTS" in src
    assert "_ga_side_pending() == 0" in src
    assert "_ga_side_pending() > 0" in src

