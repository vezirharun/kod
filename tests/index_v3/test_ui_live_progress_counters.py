"""UI canlı sayaç — processing / light-heavy / throughput / race / coverage."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication

from core.index_v3.live_contract import build_live_progress
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, QueueKind


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_processing_live_without_claimed_fresh():
    """phase=claimed + claim_started varken claimed_fresh=0 olsa bile İşleniyor=1."""
    status = {
        "total": 100,
        "artifact_pools": {"total": 100, "preview": 10, "ai_final": 5},
        "light_done": 10,
        "ai_final_ready": 5,
        "processing": 0,
        "claimed_fresh_jobs": 0,
        "claimed_jobs": 1,
        "heavy_processing": 1,
        "claimed_heavy_jobs": 1,
    }
    live = build_live_progress(
        status,
        claim={
            "phase": "claimed",
            "artifact": "patch",
            "queue": "heavy",
            "filename": "shutterstock_2237082289.jpg",
            "worker": "v3-heavy",
            "claim_started": time.monotonic() - 5,
        },
    )
    assert live["processing"] >= 1
    assert live["general_ai"]["processing"] >= 1
    assert live["fast"]["processing"] == 0
    assert live["current_filename"].startswith("shutterstock")


def test_light_heavy_lane_independence_in_live_contract():
    status = {
        "total": 50,
        "artifact_pools": {"total": 50, "preview": 20, "ai_final": 5},
        "light_done": 20,
        "ai_final_ready": 5,
        "light_processing": 1,
        "heavy_processing": 0,
        "claimed_light_jobs": 1,
        "claimed_heavy_jobs": 0,
        "claimed_fresh_jobs": 1,
    }
    live = build_live_progress(
        status,
        claim={
            "phase": "claimed",
            "artifact": "preview",
            "queue": "preview",
            "filename": "a.jpg",
            "worker": "v3-light",
            "claim_started": time.monotonic() - 1,
        },
    )
    assert live["fast"]["processing"] >= 1
    assert live["general_ai"]["processing"] == 0


def test_jobstore_throughput_and_claimed_by_queue(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue(
        [
            Job(1, Artifact.PREVIEW, QueueKind.PREVIEW, source_id=1, path="/a/1.tif"),
            Job(2, Artifact.DINO, QueueKind.HEAVY, source_id=1, path="/a/2.tif"),
            Job(3, Artifact.PATCH, QueueKind.HEAVY, source_id=1, path="/a/3.tif"),
        ]
    )
    claimed = store.claim(QueueKind.HEAVY, "v3-heavy", limit=1)
    assert claimed
    assert (
        store.count_claimed(
            queues=(QueueKind.HEAVY, QueueKind.REPAIR), source_ids=[1]
        )
        >= 1
    )
    assert (
        store.count_claimed(
            queues=(QueueKind.LIGHT, QueueKind.PREVIEW), source_ids=[1]
        )
        == 0
    )
    store.complete(claimed[0].file_id, claimed[0].artifact)
    n = store.count_completed_recent(
        60.0, source_ids=[1], queues=(QueueKind.HEAVY, QueueKind.REPAIR)
    )
    assert n >= 1


def test_progress_panel_eta_measuring_not_dash(qapp):
    from ui.progress_panel import ProgressPanel

    panel = ProgressPanel()
    panel._session_active = True
    panel._update_lane_card(
        summary=panel.lbl_general_summary,
        perf=panel.lbl_general_perf,
        bar=panel.bar_general,
        done=100,
        total=1000,
        speed_per_min=0.0,
        processing=1,
        queue=900,
        title="GENEL AI",
        executable_pending=50,
        claimed_jobs=1,
    )
    txt = panel.lbl_general_perf.text()
    assert "Ölçülüyor" in txt or "Bekleniyor" in txt
    assert "İşleniyor   : 1" in panel.lbl_general_summary.text()


def test_progress_panel_speed_from_job_throughput(qapp):
    from ui.progress_panel import ProgressPanel

    panel = ProgressPanel()
    spm = panel._lane_speed_per_min(100, "heavy", job_speed_pm=12.0)
    assert spm >= 12.0
    panel._update_lane_card(
        summary=panel.lbl_general_summary,
        perf=panel.lbl_general_perf,
        bar=panel.bar_general,
        done=100,
        total=1000,
        speed_per_min=spm,
        processing=1,
        queue=800,
        title="GENEL AI",
        eta_queue=800,
        executable_pending=40,
    )
    assert "dosya" in panel.lbl_general_perf.text()
    assert "—" not in panel.lbl_general_perf.text().split("\n")[0]


def test_progress_panel_does_not_zero_live_claim(qapp):
    from ui.progress_panel import ProgressPanel

    panel = ProgressPanel()
    panel.begin_session()
    panel._db_status_snapshot = {
        "total": 100,
        "light_done": 10,
        "ai_final_ready": 5,
        "light_processing": 0,
        "heavy_processing": 1,
        "current_filename": "x.jpg",
    }
    panel.update_progress(
        {
            "engine": "index_v3",
            "total": 100,
            "light_done": 10,
            "ai_final_ready": 5,
            "processing": 0,
            "current_stage": "patch",
            "current_filename": "shutterstock.jpg",
            "current_file_info": {
                "phase": "claimed",
                "artifact": "patch",
                "queue": "heavy",
            },
        }
    )
    snap = panel._db_status_snapshot or {}
    assert int(snap.get("heavy_processing") or 0) >= 1
    assert int(snap.get("light_processing") or 0) == 0


def test_status_worker_zero_does_not_wipe_other_lane(qapp):
    from ui.progress_panel import ProgressPanel

    panel = ProgressPanel()
    panel._session_active = True
    panel._progress_frozen = False
    panel._db_status_snapshot = {
        "total": 100,
        "light_done": 40,
        "ai_final_ready": 10,
        "light_processing": 1,
        "heavy_processing": 1,
        "current_filename": "a.jpg",
        "claimed_light_jobs": 1,
        "claimed_heavy_jobs": 1,
        "current_file_info": {"phase": "claimed"},
    }
    panel.update_status(
        {
            "total": 100,
            "light_done": 40,
            "ai_final_ready": 10,
            "light_processing": 0,
            "heavy_processing": 1,
            "claimed_light_jobs": 0,
            "claimed_heavy_jobs": 1,
            "current_filename": "a.jpg",
            "current_file_info": {"phase": "claimed", "artifact": "dino"},
            "artifact_pools": {"total": 100, "ai_final": 10},
            "pipeline_counts_exact": True,
        }
    )
    assert int(panel._db_status_snapshot.get("heavy_processing") or 0) >= 1


def test_past_coverage_not_reset_by_live_claim():
    status = {
        "total": 50,
        "artifact_pools": {
            "total": 50,
            "light_complete": 20,
            "preview": 25,
            "ai_final": 12,
        },
        "light_done": 30,
        "fast_completed": 30,
        "ai_final_ready": 12,
        "light_processing": 0,
        "heavy_processing": 1,
    }
    live = build_live_progress(
        status,
        claim={
            "phase": "claimed",
            "artifact": "dino",
            "queue": "heavy",
            "claim_started": time.monotonic() - 2,
        },
    )
    assert live["general_ai"]["completed"] == 12
    assert live["fast"]["completed"] >= 20
