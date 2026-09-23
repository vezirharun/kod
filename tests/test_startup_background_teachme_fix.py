"""BackgroundTask shutdown + TeachMe right-click multi-select contract."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


def test_background_task_cleanup_waits_until_finished(qapp):
    """Regression: wait 500ms + clear destroyed running BackgroundTask."""
    from ui.worker_threads import BackgroundTask

    hold_s = 1.0

    def slow():
        t0 = time.time()
        while time.time() - t0 < hold_s:
            time.sleep(0.05)
        return "ok"

    task = BackgroundTask(slow)
    task.setObjectName("BackgroundTask")
    task.start()
    assert task.wait(50) is False
    task.request_stop()
    assert task.wait_until_finished(5_000) is True
    assert not task.isRunning()
    task.deleteLater()
    qapp.processEvents()


def test_background_task_cleanup_source_no_short_clear():
    import inspect

    from ui.main_window import MainWindow

    src = inspect.getsource(MainWindow._cleanup_workers)
    assert "30_000" in src or "30000" in src
    assert "setParent(None)" in src
    assert "wait_until_finished(500)" not in src


def test_teachme_right_click_preserves_multiselect(qapp, tmp_path):
    from core.autonomous_learn import upsert_review
    from core.db import Database
    from core.settings import AppSettings
    from ui.teach_me_panel import TeachMePanel, _CandidateCardWidget, _quick_pick_candidates

    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("qa", str(tmp_path)),
        )
    path = str(tmp_path / "patterns.db")
    rivals = [["floral", 0.2], ["leaf", 0.18], ["nature", 0.15]]
    for i in range(12):
        img = tmp_path / f"s_{i:03d}.jpg"
        img.write_bytes(b"x")
        fid = int(
            db.upsert_file(
                {
                    "path": str(img),
                    "filename": img.name,
                    "source_id": 1,
                    "status": "indexed",
                }
            )
        )
        upsert_review(
            path,
            fid,
            lane="suspicious",
            suggested="floral",
            confidence=0.2,
            rivals=rivals,
            reason="qa",
        )

    settings = AppSettings()
    settings.db_path = path
    settings.cache_dir = str(tmp_path / "cache")
    panel = TeachMePanel(settings)
    panel.show()
    panel.reload(blocking=True)
    qapp.processEvents()
    panel.tabs.setCurrentIndex(1)
    qapp.processEvents()

    lw = panel.list_suspicious
    assert lw.count() >= 3
    for i in range(3):
        lw.item(i).setSelected(True)
    qapp.processEvents()
    before = list(panel.selected_ids())
    assert len(before) >= 2

    w = lw.itemWidget(lw.item(1))
    assert isinstance(w, _CandidateCardWidget)
    panel._focus_card_widget_for_context(w)
    qapp.processEvents()
    assert set(panel.selected_ids()) == set(before)

    cards = [panel._card_by_id(i) for i in before]
    picks = _quick_pick_candidates([c for c in cards if c])
    names = {n for n, _ in picks}
    assert "floral" in names
    assert len(picks) >= 3, f"CANDIDATE UI GAP: rivals common={picks}"
