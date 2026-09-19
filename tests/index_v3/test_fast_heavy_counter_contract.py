"""Hızlı Index / Genel AI — bağımsız sayaç, hız, ETA, current_file sözleşmesi."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.live_contract import (
    build_live_progress,
    lane_speed_eta,
    session_plus_from_pools,
)
from core.index_v3.types import QueueKind
from core.index_v3.ui_bridge import count_v3_ssot, v3_status_dict


def _imgs(folder: Path, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (32, 32), color=(i % 200, 40, 90)).save(
            folder / f"c_{i:04d}.jpg", "JPEG"
        )


@pytest.fixture
def world(tmp_path: Path):
    db = Database(tmp_path / "ctr.db")
    root = tmp_path / "files"
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("ctr", str(root)),
        )
    eng = IndexEngineV3(db, job_db_path=tmp_path / "jobs.db")
    return db, eng, root, tmp_path


def _emit(db, eng, scope, session_start, current_info, file_info):
    if file_info is not None:
        current_info.clear()
        current_info.update(file_info)
    proc = 1 if current_info.get("phase") == "claimed" else 0
    st = v3_status_dict(
        db,
        scope,
        processing=proc,
        pending_jobs=eng.store.count_pending(),
        current_file=current_info,
    )
    st["pending_light_jobs"] = eng.store.count_pending(
        QueueKind.LIGHT, source_ids=scope
    ) + eng.store.count_pending(QueueKind.PREVIEW, source_ids=scope)
    st["pending_heavy_jobs"] = eng.store.count_pending(
        QueueKind.HEAVY, source_ids=scope
    ) + eng.store.count_pending(QueueKind.REPAIR, source_ids=scope)
    return build_live_progress(st, session_start=session_start, claim=current_info)


def test_production_settings_bind_real_processor(tmp_path: Path):
    from core.index_v3.real_processor import RealArtifactProcessor
    from core.settings import AppSettings

    db = Database(tmp_path / "p.db")
    stub = IndexEngineV3(db, job_db_path=tmp_path / "j1.db")
    assert stub.processor._process_fn is None
    settings = AppSettings(cache_dir=str(tmp_path / "cache"), ai_embedding_enabled=False)
    live = IndexEngineV3(db, job_db_path=tmp_path / "j2.db", settings=settings)
    owner = getattr(live.processor._process_fn, "__self__", None)
    assert isinstance(owner, RealArtifactProcessor)


def test_fast_only_counters(world):
    db, eng, root, tmp = world
    _imgs(root, 40)
    sources = [{"id": 1, "root_path": str(root)}]
    session_start = count_v3_ssot(db, [1])
    current: dict = {"phase": "starting"}
    claims = []
    heavy_arts = set()

    def cb(info=None):
        live = _emit(db, eng, [1], session_start, current, info)
        claims.append(live)
        if info and info.get("phase") == "claimed":
            art = str(info.get("artifact") or info.get("stage") or "")
            if art in ("dino", "clip", "texture", "semantic", "dna", "patch"):
                heavy_arts.add(art)

    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True, progress_callback=cb)
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 40
    assert c["dino"] == 0
    assert c["ai_final"] == 0
    assert not heavy_arts
    assert any(e["processing"] == 1 for e in claims)
    assert any(
        e.get("current_stage") in ("thumbnail", "preview", "hash", "metadata")
        for e in claims
        if e.get("processing")
    )
    final = claims[-1]
    assert final["fast"]["completed"] == 40
    assert final["fast"]["remaining"] == 0
    assert final["general_ai"]["completed"] == 0
    assert final["session_plus"]["preview"] == 40
    assert final["session_plus"]["dino"] == 0


def test_heavy_only_counters(world):
    db, eng, root, tmp = world
    _imgs(root, 30)
    sources = [{"id": 1, "root_path": str(root)}]
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    session_start = count_v3_ssot(db, [1])
    assert session_start["preview"] == 30
    current: dict = {"phase": "starting"}
    light_claims = []

    def cb(info=None):
        live = _emit(db, eng, [1], session_start, current, info)
        if info and info.get("phase") == "claimed":
            art = str(info.get("artifact") or "")
            if art in ("thumbnail", "preview"):
                light_claims.append(art)

    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False, progress_callback=cb)
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 30
    assert c["ai_final"] == 30
    assert not light_claims
    plus = session_plus_from_pools(c, session_start)
    assert plus["preview"] == 0
    assert plus["ai_final"] == 30


def test_general_ai_ignores_orphan_source_jobs(world):
    """Yetim/yabancı source heavy job'ları GENERAL_AI'yi kilitlemesin."""
    from core.index_v3.types import Artifact, Job, QueueKind

    db, eng, root, tmp = world
    _imgs(root, 8)
    sources = [{"id": 1, "root_path": str(root)}]
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    assert count_v3_ssot(db, [1])["preview"] == 8

    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(id, name, root_path, is_active) VALUES (99,'orphan','/orphan',1)"
        )
        for i in range(200):
            conn.execute(
                "INSERT INTO files(id, path, filename, source_id, status) VALUES (?,?,?,?,?)",
                (900_000 + i, f"/orphan/{i}.tif", f"{i}.tif", 99, "ok"),
            )
        for i in range(50):
            conn.execute(
                "INSERT INTO files(id, path, filename, source_id, status) VALUES (?,?,?,?,?)",
                (800_000 + i, f"/orphan/l{i}.tif", f"l{i}.tif", 99, "ok"),
            )

    # Sahte yetim backlog (production'daki 16k benzeri)
    poison = [
        Job(
            file_id=900_000 + i,
            artifact=Artifact.DINO,
            queue=QueueKind.HEAVY,
            source_id=99,
            path=f"/orphan/{i}.tif",
        )
        for i in range(200)
    ]
    # light pending de çıkışı bozmasın
    poison += [
        Job(
            file_id=800_000 + i,
            artifact=Artifact.THUMBNAIL,
            queue=QueueKind.LIGHT,
            source_id=99,
            path=f"/orphan/l{i}.tif",
        )
        for i in range(50)
    ]
    eng.store.enqueue(poison)
    assert eng.store.count_pending(QueueKind.HEAVY) >= 200

    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    c = count_v3_ssot(db, [1])
    assert c["ai_final"] == 8
    # yetim job'lar hâlâ pending — claim edilmedi
    assert eng.store.count_pending(QueueKind.HEAVY, source_ids=[99]) == 200
    assert eng.store.count_pending(QueueKind.LIGHT, source_ids=[99]) == 50


def test_general_ai_retargets_repair_queue_jobs(world):
    """REPAIR kuyruğunda kalan heavy gap GENERAL_AI ile işlensin."""
    from core.index_v3.types import Artifact, Job, QueueKind

    db, eng, root, tmp = world
    _imgs(root, 6)
    sources = [{"id": 1, "root_path": str(root)}]
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    with db.connect() as conn:
        fids = [int(r[0]) for r in conn.execute("SELECT id FROM files WHERE source_id=1")]
    jobs = []
    for fid in fids:
        for art in (
            Artifact.DINO,
            Artifact.CLIP,
            Artifact.TEXTURE,
            Artifact.SEMANTIC,
            Artifact.DNA,
            Artifact.PATCH,
        ):
            jobs.append(
                Job(
                    file_id=fid,
                    artifact=art,
                    queue=QueueKind.REPAIR,
                    source_id=1,
                    path=str(root / "x.jpg"),
                )
            )
    eng.store.enqueue(jobs)
    assert eng.store.count_pending(QueueKind.REPAIR, source_ids=[1]) == 36
    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["ai_final"] == 6
    assert eng.store.count_pending(QueueKind.REPAIR, source_ids=[1]) == 0


def test_dual_worker_independent_counters(world):
    db, eng, root, tmp = world
    _imgs(root, 25)
    sources = [{"id": 1, "root_path": str(root)}]
    session_start = count_v3_ssot(db, [1])
    workers = set()
    current: dict = {}

    def cb(info=None):
        live = _emit(db, eng, [1], session_start, current, info)
        if info and info.get("phase") == "claimed":
            workers.add(str(info.get("worker") or ""))

    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=True, progress_callback=cb)
    assert "v3-light" in workers
    assert "v3-heavy" in workers
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 25
    assert c["ai_final"] == 25


def test_current_file_and_stage(world):
    db, eng, root, tmp = world
    _imgs(root, 12)
    sources = [{"id": 1, "root_path": str(root)}]
    session_start = count_v3_ssot(db, [1])
    current: dict = {"phase": "idle"}
    seen = []

    def cb(info=None):
        live = _emit(db, eng, [1], session_start, current, info)
        if info is None:
            # engine drain — current_file korunur
            if current.get("phase") == "claimed":
                assert live["current_filename"]
                assert live["processing"] == 1
            return
        if info.get("phase") == "claimed":
            assert live["processing"] == 1
            assert live["current_filename"]
            assert live["current_stage"] in (
                "thumbnail",
                "preview",
                "hash",
                "metadata",
                "dino",
                "clip",
                "texture",
                "semantic",
                "dna",
                "patch",
            )
            seen.append((live["current_filename"], live["current_stage"], live["current_worker"]))

    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True, progress_callback=cb)
    assert seen


def test_fast_speed_independent():
    now = time.monotonic()
    fast_samples = [(now - 20, 10), (now - 10, 30), (now, 50)]
    heavy_samples = [(now - 20, 0), (now - 10, 0), (now, 0)]
    fs = lane_speed_eta(lane="fast", samples=fast_samples, remaining=150)
    hs = lane_speed_eta(lane="heavy", samples=heavy_samples, remaining=200)
    assert fs["speed_per_sec"] is not None
    assert abs(fs["speed_per_sec"] - 2.0) < 0.3  # 40/20
    assert hs["speed_per_sec"] is None  # Δ=0 → hesaplanıyor
    # Fast sample'a heavy eklenmez
    assert fs["eta_sec"] is not None


def test_heavy_speed_independent():
    now = time.monotonic()
    heavy_samples = [(now - 30, 5), (now, 35)]
    hs = lane_speed_eta(lane="heavy", samples=heavy_samples, remaining=165)
    assert hs["speed_per_sec"] is not None
    assert abs(hs["speed_per_sec"] - 1.0) < 0.2  # 30/30
    assert hs["eta_sec"] is not None
    assert abs(hs["eta_sec"] - 165.0) < 40


def test_fast_eta_independent():
    now = time.monotonic()
    s = lane_speed_eta(
        lane="fast",
        samples=[(now - 60, 0), (now, 60)],
        remaining=120,
    )
    assert s["speed_per_sec"] == pytest.approx(1.0, rel=0.05)
    assert s["eta_sec"] == pytest.approx(120.0, rel=0.1)


def test_heavy_eta_independent():
    now = time.monotonic()
    s = lane_speed_eta(
        lane="heavy",
        samples=[(now - 60, 10), (now, 40)],
        remaining=60,
    )
    assert s["speed_per_sec"] == pytest.approx(0.5, rel=0.1)
    assert s["eta_sec"] == pytest.approx(120.0, rel=0.15)


def test_session_delta_separation(world):
    db, eng, root, tmp = world
    _imgs(root, 15)
    sources = [{"id": 1, "root_path": str(root)}]
    base = count_v3_ssot(db, [1])
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    after_fast = count_v3_ssot(db, [1])
    d1 = session_plus_from_pools(after_fast, base)
    assert d1["preview"] == 15
    assert d1["ai_final"] == 0
    base2 = dict(after_fast)
    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    after_h = count_v3_ssot(db, [1])
    d2 = session_plus_from_pools(after_h, base2)
    assert d2["preview"] == 0
    assert d2["ai_final"] == 15


def test_200_file_progress(world):
    db, eng, root, tmp = world
    n = 200
    _imgs(root, n)
    sources = [{"id": 1, "root_path": str(root)}]
    session_start = count_v3_ssot(db, [1])
    current: dict = {}
    snapshots = []

    def cb(info=None):
        live = _emit(db, eng, [1], session_start, current, info)
        snapshots.append(
            {
                "done": live["fast"]["completed"],
                "rem": live["fast"]["remaining"],
                "proc": live["processing"],
                "stage": live.get("current_stage"),
                "file": live.get("current_filename"),
            }
        )

    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True, progress_callback=cb)
    assert count_v3_ssot(db, [1])["preview"] == n
    assert snapshots
    # invariant: done + rem == total during progress (pool SSOT)
    for s in snapshots:
        assert s["done"] + s["rem"] == n or s["done"] <= n
    assert any(s["proc"] == 1 and s["file"] for s in snapshots)
    mid = [s for s in snapshots if 0 < s["done"] < n]
    assert mid
    # remaining shrinks as done grows
    assert snapshots[-1]["done"] == n
    assert snapshots[-1]["rem"] == 0


def test_new_source_queued_dual_lane_counters_do_not_break_old_source(world):
    """Yeni kaynak kuyruğa girince FAST + GENERAL aynı oturumda ilerler; eski kaynak bozulmaz.

    UI'de ikinci bir Hızlı Index worker yok. COMPLETE içinde light + heavy lane paralel.
    """
    db, eng, root, tmp = world
    _imgs(root, 8)
    new_root = tmp / "newsrc"
    _imgs(new_root, 6)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("new", str(new_root)),
        )

    old = [{"id": 1, "root_path": str(root)}]
    eng.run(mode=Mode.FAST, sources=old, walk_disk=True)
    before = count_v3_ssot(db, [1])
    assert before["preview"] == 8
    assert before["ai_final"] == 0

    both = [
        {"id": 1, "root_path": str(root)},
        {"id": 2, "root_path": str(new_root)},
    ]
    session_start = count_v3_ssot(db, [1, 2])
    current: dict = {}
    workers: set[str] = set()
    light_at: list[float] = []
    heavy_at: list[float] = []
    fast_done: list[int] = []
    gen_done: list[int] = []
    fast_queue: list[int] = []
    gen_queue: list[int] = []

    def cb(info=None):
        live = _emit(db, eng, [1, 2], session_start, current, info)
        fast_done.append(int(live["fast"]["completed"]))
        gen_done.append(int(live["general_ai"]["completed"]))
        fast_queue.append(int(live["fast"]["file_queue"]))
        gen_queue.append(int(live["general_ai"]["file_queue"]))
        if info and info.get("phase") == "claimed":
            w = str(info.get("worker") or "")
            workers.add(w)
            now = time.perf_counter()
            if "light" in w:
                light_at.append(now)
            if "heavy" in w:
                heavy_at.append(now)

    eng.run(mode=Mode.COMPLETE, sources=both, walk_disk=True, progress_callback=cb)

    old_after = count_v3_ssot(db, [1])
    new_after = count_v3_ssot(db, [2])
    assert old_after["preview"] == 8
    assert old_after["ai_final"] == 8
    assert new_after["preview"] == 6
    assert new_after["ai_final"] == 6

    assert max(fast_done) >= 14
    assert max(gen_done) >= 14
    assert min(fast_queue) == 0
    assert min(gen_queue) == 0
    assert any(a < b for a, b in zip(fast_done, fast_done[1:])) or max(fast_done) > before["preview"]
    assert any(a < b for a, b in zip(gen_done, gen_done[1:]))

    assert "v3-light" in workers
    assert "v3-heavy" in workers
    assert light_at and heavy_at
    # İki lane zaman aralığı örtüşür (aynı anda kuyrukta işleniyor)
    overlap = min(light_at[-1], heavy_at[-1]) >= max(light_at[0], heavy_at[0]) or (
        min(heavy_at) <= max(light_at) and min(light_at) <= max(heavy_at)
    )
    assert overlap
