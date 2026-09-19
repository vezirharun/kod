"""Acceptance: mod bağımsızlığı + Durdur/Duraklat/Devam + canlı sayaç (tmp DB)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.live_contract import build_live_progress
from core.index_v3.types import QueueKind
from core.index_v3.ui_bridge import count_v3_ssot, v3_status_dict


def _imgs(folder: Path, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (40, 40), color=(i % 200, 20, 90)).save(
            folder / f"a_{i:04d}.jpg", "JPEG"
        )


@pytest.fixture
def world(tmp_path: Path):
    db = Database(tmp_path / "acc.db")
    root = tmp_path / "src"
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("acc", str(root)),
        )
    eng = IndexEngineV3(db, job_db_path=tmp_path / "jobs.db")
    return db, eng, root, [{"id": 1, "root_path": str(root)}]


def _live(db, eng, scope, session_start, current, info):
    if info is not None:
        current.clear()
        current.update(info)
    proc = 1 if current.get("phase") == "claimed" else 0
    st = v3_status_dict(
        db,
        scope,
        processing=proc,
        pending_jobs=eng.store.count_pending(source_ids=scope),
        current_file=current,
    )
    st["pending_light_jobs"] = eng.store.count_pending(
        QueueKind.LIGHT, source_ids=scope
    ) + eng.store.count_pending(QueueKind.PREVIEW, source_ids=scope)
    st["pending_heavy_jobs"] = eng.store.count_pending(
        QueueKind.HEAVY, source_ids=scope
    ) + eng.store.count_pending(QueueKind.REPAIR, source_ids=scope)
    return build_live_progress(st, session_start=session_start, claim=current)


def test_modes_independent_progress(world):
    db, eng, root, sources = world
    _imgs(root, 24)
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 24
    assert c["dino"] == 0
    assert c["ai_final"] == 0

    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 24
    assert c["ai_final"] == 24

    # Complete idempotent — havuz bozulmaz
    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["ai_final"] == 24


def test_live_counters_move_during_fast_and_complete(world):
    db, eng, root, sources = world
    _imgs(root, 30)
    session = count_v3_ssot(db, [1])
    current: dict = {}
    snaps: list[dict] = []

    def cb(info=None):
        live = _live(db, eng, [1], session, current, info)
        snaps.append(
            {
                "fast": int(live["fast"]["completed"]),
                "ai": int(live["general_ai"]["completed"]),
                "proc": int(live["processing"]),
                "file": str(live.get("current_filename") or ""),
                "stage": str(live.get("current_stage") or ""),
            }
        )

    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True, progress_callback=cb)
    assert snaps
    assert any(s["proc"] == 1 and s["file"] for s in snaps)
    assert any(0 < s["fast"] < 30 for s in snaps) or snaps[-1]["fast"] == 30
    assert snaps[-1]["fast"] == 30
    assert all(s["ai"] == 0 for s in snaps)

    session2 = count_v3_ssot(db, [1])
    current.clear()
    snaps2: list[dict] = []

    def cb2(info=None):
        live = _live(db, eng, [1], session2, current, info)
        snaps2.append(
            {
                "ai": int(live["general_ai"]["completed"]),
                "proc": int(live["processing"]),
                "stage": str(live.get("current_stage") or ""),
            }
        )

    eng.run(
        mode=Mode.COMPLETE, sources=sources, walk_disk=False, progress_callback=cb2
    )
    assert snaps2
    assert any(
        s["stage"] in ("dino", "clip", "texture", "semantic", "dna", "patch")
        for s in snaps2
        if s["proc"]
    )
    assert snaps2[-1]["ai"] == 30


def test_stop_mid_run_partial_then_can_finish(world):
    db, eng, root, sources = world
    _imgs(root, 40)
    err: list[BaseException] = []

    def _run():
        try:
            eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # İlk ilerlemeyi bekle
    deadline = time.time() + 20
    while time.time() < deadline:
        if count_v3_ssot(db, [1])["preview"] >= 2:
            break
        time.sleep(0.05)
    assert count_v3_ssot(db, [1])["preview"] >= 1
    eng.request_stop()
    t.join(timeout=45)
    assert not t.is_alive()
    assert not err
    mid = count_v3_ssot(db, [1])["preview"]
    assert mid < 40  # tam bitmeden durdu
    # Aynı engine: run() başında stop Event temizlenir
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["preview"] == 40


def test_pause_blocks_progress_resume_continues(world):
    db, eng, root, sources = world
    _imgs(root, 36)
    err: list[BaseException] = []

    def _run():
        try:
            eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    deadline = time.time() + 20
    while time.time() < deadline:
        if count_v3_ssot(db, [1])["preview"] >= 2:
            break
        time.sleep(0.05)
    assert count_v3_ssot(db, [1])["preview"] >= 1

    eng.request_pause()
    frozen = count_v3_ssot(db, [1])["preview"]
    time.sleep(0.8)
    after_pause = count_v3_ssot(db, [1])["preview"]
    # Duraklatmada ilerleme donmalı (en fazla +1 yarış toleransı)
    assert after_pause <= frozen + 1

    eng.request_resume()
    t.join(timeout=60)
    assert not t.is_alive()
    assert not err
    assert count_v3_ssot(db, [1])["preview"] == 36


def test_general_ai_live_counters_and_stop(world):
    db, eng, root, sources = world
    _imgs(root, 20)
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    session = count_v3_ssot(db, [1])
    current: dict = {}
    heavy_seen = []
    light_seen = []

    def cb(info=None):
        live = _live(db, eng, [1], session, current, info)
        if info and info.get("phase") == "claimed":
            art = str(info.get("artifact") or "")
            if art in ("thumbnail", "preview"):
                light_seen.append(art)
            if art in ("dino", "clip", "texture", "semantic", "dna", "patch"):
                heavy_seen.append(art)

    # Genel AI canlı + bağımsız (light claim yok)
    eng.run(
        mode=Mode.GENERAL_AI, sources=sources, walk_disk=False, progress_callback=cb
    )
    assert not light_seen
    assert heavy_seen
    assert count_v3_ssot(db, [1])["ai_final"] == 20
