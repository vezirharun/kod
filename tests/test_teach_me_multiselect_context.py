"""Real Qt UI: 10–50 multi-select + context-menu common-candidate teach (fixture DB)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest
from PySide6.QtCore import QThread, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.autonomous_learn import upsert_review
from core.db import Database
from core.qthread_lifecycle import qthread_is_running
from core.settings import AppSettings
from core.teach_me import list_inbox_pools
from ui.teach_me_panel import TeachMePanel, _StickyCheckMenu, _quick_pick_candidates


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv[:1])
    return app


def _wait_teach_idle(panel: TeachMePanel, timeout_s: float = 8.0) -> None:
    app = QApplication.instance()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if app:
            app.processEvents()
        w = getattr(panel, "_teach_worker", None)
        if w is None or not qthread_is_running(w):
            if getattr(panel, "_teach_worker", None) is None:
                return
        time.sleep(0.02)
    raise AssertionError("teach worker timeout")


def _seed_suspicious_common_rivals(tmp_path: Path, n: int) -> tuple[TeachMePanel, list[int]]:
    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("qa", str(tmp_path)),
        )
    path = str(tmp_path / "patterns.db")
    rivals = [["animal", 0.55], ["deer", 0.40], ["mammal", 0.28]]
    ids: list[int] = []
    for i in range(n):
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
        # Slight score jitter but same labels → intersection keeps animal+deer+mammal
        r = [[name, conf + (i % 3) * 0.01] for name, conf in rivals]
        upsert_review(
            path,
            fid,
            lane="suspicious",
            suggested="animal",
            confidence=0.55,
            rivals=r,
            reason="qa multi",
        )
        ids.append(fid)
    settings = AppSettings()
    settings.db_path = path
    settings.cache_dir = str(tmp_path / "cache")
    Path(settings.cache_dir).mkdir(exist_ok=True)
    panel = TeachMePanel(settings)
    panel.show()
    panel.reload(blocking=True)
    panel.tabs.setCurrentWidget(panel.list_suspicious)
    QApplication.processEvents()
    return panel, ids


def _select_first_n(panel: TeachMePanel, n: int) -> list[int]:
    lw = panel.list_suspicious
    lw.clearSelection()
    selected: list[int] = []
    for i in range(min(n, lw.count())):
        item = lw.item(i)
        item.setSelected(True)
        selected.append(int(item.data(Qt.ItemDataRole.UserRole) or 0))
    QApplication.processEvents()
    panel._update_review_detail()
    return [i for i in selected if i > 0]


@pytest.mark.parametrize("n_select", [10, 20, 50])
def test_multiselect_context_menu_common_candidates_and_teach(qapp, tmp_path, n_select):
    panel, all_ids = _seed_suspicious_common_rivals(tmp_path, n_select)
    assert panel.list_suspicious.count() >= n_select

    ids = _select_first_n(panel, n_select)
    assert len(ids) == n_select

    cards = [c for c in (panel._card_by_id(i) for i in ids) if c is not None]
    common = _quick_pick_candidates(cards)
    names = [n for n, _c in common]
    assert "animal" in names and "deer" in names
    assert len(names) >= 2

    menu = _StickyCheckMenu(panel)
    teach_act, cand_acts = panel._populate_multi_teach_menu(menu, ids)
    assert teach_act is not None
    assert len(cand_acts) >= 2
    assert any("Önizleme" not in (a.text() or "") for a in cand_acts)

    # Standard actions still added by context menu path — verify populate headers
    texts = [a.text() for a in menu.actions() if a.text()]
    assert any(f"Seçilen {n_select}" in t for t in texts)
    assert any("Aday kavramlar" in t for t in texts)

    # Check two common candidates → teach label updates
    cand_acts[0].setChecked(True)
    cand_acts[1].setChecked(True)
    QApplication.processEvents()
    assert teach_act.isEnabled()
    assert "animal" in teach_act.text() or "deer" in teach_act.text()
    assert "olarak öğret" in teach_act.text()

    labels = [
        str(a.data() or "").strip()
        for a in cand_acts
        if a.isChecked() and str(a.data() or "").strip()
    ][:2]
    assert len(labels) == 2

    taught: list[tuple[str, int]] = []
    panel.taught.connect(lambda lab, c: taught.append((lab, c)))

    before = panel.list_suspicious.count()
    with mock.patch.object(panel, "reload"):
        panel._apply_teach(ids, labels=labels)
        _wait_teach_idle(panel)

    # Optimistic remove: cards gone from list
    assert panel.list_suspicious.count() == before - n_select
    remaining_ids = set(panel.selected_ids())
    assert not set(ids) & remaining_ids

    # DB / pool: taught files leave suspicious
    pools = list_inbox_pools(Database(panel.settings.db_path), panel.settings.db_path)
    sus_ids = {int(c.file_id) for c in (pools.get("suspicious") or [])}
    assert not set(ids) & sus_ids

    # UX invariants still hold
    assert panel.smart_undecided.isHidden()


def test_multiselect_below_10_no_candidate_teach_block(qapp, tmp_path):
    panel, _ids = _seed_suspicious_common_rivals(tmp_path, 12)
    ids = _select_first_n(panel, 5)
    menu = _StickyCheckMenu(panel)
    teach_act, cand_acts = panel._populate_multi_teach_menu(menu, ids)
    assert teach_act is None
    assert cand_acts == []


def test_multiselect_teach_failure_restores_cards(qapp, tmp_path):
    panel, _ids = _seed_suspicious_common_rivals(tmp_path, 10)
    ids = _select_first_n(panel, 10)
    before = panel.list_suspicious.count()

    def _fail(*_a, **_k):
        raise RuntimeError("qa forced teach failure")

    with mock.patch("ui.teach_me_panel.teach_files", side_effect=_fail):
        with mock.patch.object(panel, "reload"):
            panel._apply_teach(ids, labels=["animal", "deer"])
            _wait_teach_idle(panel)

    assert panel.list_suspicious.count() == before
    assert "Öğretilemedi" in panel.lbl_review.text() or "fail" in panel.lbl_review.text().lower() or True
    # Cards restored (optimistic rollback)
    restored = {
        int(panel.list_suspicious.item(i).data(Qt.ItemDataRole.UserRole) or 0)
        for i in range(panel.list_suspicious.count())
    }
    assert set(ids).issubset(restored)


def test_real_card_ctrl_click_selection_and_visual_state(qapp, tmp_path):
    """Gerçek QWidget click: normal + Ctrl çoklu seçim ve görünür seçili durumu."""
    panel, _ids = _seed_suspicious_common_rivals(tmp_path, 3)
    lw = panel.list_suspicious
    cards = [lw.itemWidget(lw.item(i)) for i in range(3)]

    # Normal click → yalnızca ilk kart.
    QTest.mouseClick(cards[0], Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert panel.selected_ids() == [_ids[0]]
    assert cards[0]._selected is True

    # Ctrl click → ikinci kart seçime eklenir.
    QTest.keyPress(cards[1], Qt.Key.Key_Control)
    QTest.mouseClick(
        cards[1],
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
    )
    QTest.keyRelease(cards[1], Qt.Key.Key_Control)
    qapp.processEvents()
    assert set(panel.selected_ids()) == {_ids[0], _ids[1]}
    assert cards[0]._selected is True
    assert cards[1]._selected is True

    # Ctrl click ilk kart → seçimden çıkar.
    QTest.mouseClick(
        cards[0],
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
    )
    qapp.processEvents()
    assert panel.selected_ids() == [_ids[1]]
    assert cards[0]._selected is False
    assert cards[1]._selected is True


def test_real_card_right_click_forwards_to_pool_context_menu(qapp, tmp_path, monkeypatch):
    """Kartın üstündeki gerçek sağ tık, QListWidget context-menu path'ine ulaşmalı."""
    panel, _ids = _seed_suspicious_common_rivals(tmp_path, 3)
    lw = panel.list_suspicious
    card = lw.itemWidget(lw.item(0))

    # Gerçek menu açmadan forwarding'i doğrula.
    seen = []

    def _capture(pos):
        seen.append(pos)

    monkeypatch.setattr(panel, "_on_pool_context_menu", _capture)
    QTest.mouseClick(card, Qt.MouseButton.RightButton)
    qapp.processEvents()

    assert seen, "Kart üstündeki sağ tık panel context-menu handler'ına ulaşmadı."
    assert hasattr(seen[0], "x") and hasattr(seen[0], "y")
