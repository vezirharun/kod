"""Index zinciri ↔ canlı sayaç senkronu (tmp DB, production yok).

Zincir:
  discover → enqueue → claim → process → verify → complete
       ↓
  count_v3_ssot / v3_status_dict / build_live_progress
       ↓
  emit payload (processing, pools, session+, remaining, executable)

Kural: emit Tamamlanan = physical pool; İşleniyor = claim 0|1; event sayacı yok.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.live_contract import build_live_progress, worker_progress_should_compute_ssot
from core.index_v3.ui_bridge import count_v3_ssot, v3_status_dict


def _imgs(folder: Path, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (40, 40), color=(i % 160, 40, 90)).save(
            folder / f"sync_{i:03d}.jpg", "JPEG"
        )


@pytest.fixture
def sync_world(tmp_path: Path):
    db = Database(tmp_path / "sync.db")
    root = tmp_path / "files"
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("sync", str(root)),
        )
    _imgs(root, 20)
    eng = IndexEngineV3(db, job_db_path=tmp_path / "jobs.db")
    sources = [{"id": 1, "root_path": str(root)}]
    return db, eng, sources


def _emit_like_index_worker(
    db: Database,
    eng: IndexEngineV3,
    *,
    scope_ids: list[int],
    session_start: dict,
    current_info: dict,
    file_info: dict | None,
    stage: str = "v3_progress",
) -> tuple[dict, dict]:
    """IndexWorker._on_worker_progress sözleşmesi: drain'de SSOT yok."""
    if file_info is not None:
        current_info.clear()
        current_info.update(file_info)
    processing_n = 1 if current_info.get("phase") == "claimed" else 0
    if not worker_progress_should_compute_ssot(stage):
        fname = str(current_info.get("filename") or "")
        art = str(
            current_info.get("stage") or current_info.get("artifact") or ""
        )
        return {}, {
            "processing": processing_n,
            "current_filename": fname,
            "current_stage": art,
            "pools": {},
            "session_plus": {},
        }
    pending_n = int(eng.store.count_pending() or 0)
    st = v3_status_dict(
        db,
        scope_ids,
        processing=processing_n,
        claimed=processing_n,
        pending_jobs=pending_n,
        current_file=current_info,
    )
    live = build_live_progress(st, session_start=session_start, claim=current_info)
    return st, live


def test_live_counter_sync_with_index_pipeline(sync_world):
    db, eng, sources = sync_world
    scope = [1]
    session_start = dict(count_v3_ssot(db, scope))
    current_info: dict = {"phase": "starting"}
    emits: list[dict] = []

    def cb(info=None):
        st, live = _emit_like_index_worker(
            db,
            eng,
            scope_ids=scope,
            session_start=session_start,
            current_info=current_info,
            file_info=info,
        )
        if info and info.get("phase") == "claimed":
            assert live["processing"] == 1
            assert live["current_filename"] or info.get("filename") or info.get("path")
            assert live["current_stage"] or info.get("artifact")
        if info and info.get("phase") == "idle":
            assert live["processing"] == 0
        if info is None:
            assert live["processing"] in (0, 1)
            if current_info.get("phase") == "claimed":
                assert live["processing"] == 1
        emits.append(live)

    eng.run(mode=Mode.FAST, sources=sources, progress_callback=cb)
    assert emits
    assert any(e["processing"] == 1 for e in emits)
    assert any(e["processing"] == 0 for e in emits)
    pools = count_v3_ssot(db, scope)
    assert int(pools["preview"]) == 20
    plus = build_live_progress(
        v3_status_dict(db, scope, pending_jobs=0),
        session_start=session_start,
    )
    assert plus["session_plus"]["preview"] == 20
    assert plus["fast"]["remaining"] == 0
    assert "executable" in plus

    # COMPLETE — texture / ai_final senkron
    session_start = dict(count_v3_ssot(db, scope))
    current_info.clear()
    current_info["phase"] = "starting"
    emits.clear()

    def cb2(info=None):
        _, live = _emit_like_index_worker(
            db,
            eng,
            scope_ids=scope,
            session_start=session_start,
            current_info=current_info,
            file_info=info,
        )
        emits.append(live)

    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=False, progress_callback=cb2)
    after = count_v3_ssot(db, scope)
    assert int(after["texture"]) == 20
    assert int(after["ai_final"]) == 20
    live_end = build_live_progress(
        v3_status_dict(db, scope, pending_jobs=0),
        session_start=session_start,
    )
    assert live_end["session_plus"]["texture"] == 20
    assert live_end["session_plus"]["ai_final"] == 20


def test_engine_none_callback_does_not_wipe_claim(sync_world):
    db, eng, sources = sync_world
    eng.run(mode=Mode.FAST, sources=sources)
    session_start = count_v3_ssot(db, [1])
    current_info = {
        "phase": "claimed",
        "filename": "keep_me.jpg",
        "stage": "texture",
        "artifact": "texture",
        "worker": "v3-main",
        "path": "x/keep_me.jpg",
    }
    # Engine drain tarzı None
    _, live = _emit_like_index_worker(
        db,
        eng,
        scope_ids=[1],
        session_start=session_start,
        current_info=current_info,
        file_info=None,
    )
    assert current_info["filename"] == "keep_me.jpg"
    assert live["processing"] == 1
    assert live["current_filename"] == "keep_me.jpg"
    assert live["current_stage"] == "texture"

    # Boş dict eski bug — artık caller None kullanır; idle açık gelmeli
    _, live_idle = _emit_like_index_worker(
        db,
        eng,
        scope_ids=[1],
        session_start=session_start,
        current_info=current_info,
        file_info={"phase": "idle", "worker": "v3-main"},
    )
    assert live_idle["processing"] == 0


def test_processing_never_uses_jobstore_claim_count(sync_world):
    db, eng, sources = sync_world
    eng.run(mode=Mode.FAST, sources=sources)
    # Sahte stale claim
    with eng.store._connect() as conn:
        row = conn.execute(
            "SELECT file_id, artifact FROM index_v3_jobs LIMIT 1"
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE index_v3_jobs SET state='claimed', worker='dead' "
                "WHERE file_id=? AND artifact=?",
                (row["file_id"], row["artifact"]),
            )
            conn.commit()
    assert eng.store.count_claimed() >= 1
    current_info = {"phase": "idle"}
    _, live = _emit_like_index_worker(
        db,
        eng,
        scope_ids=[1],
        session_start=count_v3_ssot(db, [1]),
        current_info=current_info,
        file_info=None,
    )
    # Stale claim olsa bile processing=0
    assert live["processing"] == 0


def test_stale_claim_is_not_reported_as_processing():
    status = {
        "total": 940,
        "artifact_pools": {"total": 940, "preview": 1, "ai_final": 0},
        "pending_jobs": 939,
        "pending_light_jobs": 939,
        "pending_heavy_jobs": 0,
        "processing": 1,
        "claimed_jobs": 1,
        "claimed_fresh_jobs": 0,
    }
    claim = {
        "phase": "claimed",
        "filename": "stale.psd",
        "stage": "thumbnail",
        "worker": "dead-worker",
    }
    stale = build_live_progress(status, claim=claim)
    assert stale["processing"] == 0
    assert stale["fast"]["processing"] == 0

    status["claimed_fresh_jobs"] = 1
    fresh = build_live_progress(status, claim=claim)
    assert fresh["processing"] == 1
    assert fresh["fast"]["processing"] == 1
