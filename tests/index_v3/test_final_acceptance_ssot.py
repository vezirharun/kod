"""FINAL ACCEPTANCE — Index Engine V3 + Live Progress + Counter SSOT (tmp, 40 dosya)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.progress import count_progress, session_delta
from core.index_v3.scope import resolve_index_scope
from core.index_v3.ui_bridge import count_v3_ssot, v3_status_dict
from core.index_v3.live_contract import (
    build_live_progress,
    remaining_vs_executable,
    speed_eta_from_pool,
    session_plus_from_pools,
)

N = 40
ARTIFACTS = (
    "thumbnail",
    "preview",
    "dino",
    "clip",
    "texture",
    "semantic",
    "dna",
    "patch",
    "ai_final",
)


def _imgs(folder: Path, n: int, prefix: str = "fa") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (48, 48), color=(i % 200, 25, 80)).save(
            folder / f"{prefix}_{i:03d}.jpg", "JPEG"
        )


@pytest.fixture
def world(tmp_path: Path):
    db = Database(tmp_path / "final40.db")
    root = tmp_path / "files"
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("final40", str(root)),
        )
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("other", str(tmp_path / "other")),
        )
    _imgs(root, N)
    _imgs(tmp_path / "other", 5, prefix="oth")
    eng = IndexEngineV3(db, job_db_path=tmp_path / "jobs.db")
    sources = [{"id": 1, "root_path": str(root)}]
    return db, eng, sources, root, tmp_path


def _pools(db: Database, sids: list[int] | None = None) -> dict:
    return count_v3_ssot(db, sids or [1])


def test_artifact_ssot_and_live_counters(world):
    db, eng, sources, root, tmp = world
    session_start = {
        "preview": 0,
        "thumbnail": 0,
        "dino": 0,
        "clip": 0,
        "texture": 0,
        "semantic": 0,
        "dna": 0,
        "patch": 0,
        "ai_final": 0,
    }
    claims: list[dict] = []

    def on_prog(info=None):
        info = info or {}
        st = v3_status_dict(
            db,
            [1],
            processing=1 if info.get("phase") == "claimed" else 0,
            current_file=info,
            pending_jobs=eng.store.count_pending(),
        )
        claims.append(build_live_progress(st, session_start=session_start, claim=info))

    eng.run(mode=Mode.FAST, sources=sources, progress_callback=on_prog)
    after_fast = _pools(db)
    assert after_fast["total"] == N
    assert after_fast["preview"] == N
    assert after_fast["dino"] == 0

    session_start = dict(after_fast)
    claims.clear()

    def on_heavy(info=None):
        info = info or {}
        st = v3_status_dict(
            db,
            [1],
            processing=1 if info.get("phase") == "claimed" else 0,
            current_file=info,
            pending_jobs=eng.store.count_pending(),
        )
        claims.append(build_live_progress(st, session_start=session_start, claim=info))

    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=False, progress_callback=on_heavy)
    done = _pools(db)
    for a in ARTIFACTS:
        assert done["clip" if a == "clip" else a] == N, a
    assert max(c["pools"]["texture"] for c in claims) == N


def test_session_plus_and_minus(world):
    db, eng, sources, root, tmp = world
    eng.run(mode=Mode.COMPLETE, sources=sources)
    base = _pools(db)
    assert all(v == 0 for v in session_plus_from_pools(base, base).values())
    with db.connect() as conn:
        fid = int(conn.execute("SELECT id FROM files ORDER BY id LIMIT 1").fetchone()[0])
        conn.execute(
            "UPDATE files SET feature_preview_path='', physical_preview_ready=0 WHERE id=?",
            (fid,),
        )
    delta = session_plus_from_pools(_pools(db), base)
    assert delta["preview"] == -1
    eng.run(mode=Mode.REPAIR, sources=sources, walk_disk=False)
    assert _pools(db)["preview"] == N


def test_scope_ui_worker_same_total(world):
    db, eng, sources, root, tmp = world
    eng.run(mode=Mode.FAST, sources=sources)
    eng.run(mode=Mode.FAST, sources=[{"id": 2, "root_path": str(tmp / "other")}])
    scope1 = resolve_index_scope(db, [1])
    assert scope1.source_ids == [1]
    assert count_v3_ssot(db, scope1.source_ids)["total"] == N


def test_speed_eta_from_real_pool_delta():
    assert speed_eta_from_pool(samples=[], remaining=100)["speed_per_sec"] is None
    now = time.monotonic()
    s2 = speed_eta_from_pool(
        samples=[(now - 60.0, 100), (now - 30.0, 160), (now, 220)],
        remaining=780,
    )
    assert 1.5 <= s2["speed_per_sec"] <= 3.0


def test_remaining_vs_executable_separate():
    r = remaining_vs_executable(total=100, ready=40, executable_jobs=12)
    assert r["remaining"] == 60
    assert r["executable"] == 12
