"""Thumbnail Durumu paneli kompakt kalmalı; sonuç listesini şişirmemeli."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication

from ui.results_panel import ResultsPanel


@dataclass
class _Snap:
    total: int = 50
    loaded: int = 15
    missing: int = 20
    decode_error: int = 5
    cache_miss: int = 10
    file_missing: int = 5
    pending: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_thumb_health_stays_compact_with_many_fails(qapp):
    panel = ResultsPanel()
    fails = [
        {
            "filename": f"zebra_{i}.eps",
            "reason": (
                "INDEX_FROZEN_WRITE_BLOCKED thumbnail.create "
                f"caller=scheduler path=C:/very/long/path/to/file_{i}.eps "
                + ("x" * 180)
            ),
        }
        for i in range(25)
    ]
    panel._on_thumb_health(_Snap(failures=fails))

    assert not panel.thumb_health_box.isHidden()
    assert panel.thumb_health_box.maximumHeight() <= 110
    text = panel.lbl_thumb_health.text()
    assert "Toplam: 50" in text
    assert "Hazır: 15" in text
    assert "Eksik: 20" in text
    assert "Decode: 5" in text
    assert "INDEX_FROZEN" not in text  # uzun fail özet satırında değil
    assert not panel.btn_thumb_fails.isHidden()
    assert not panel.btn_thumb_fails.isChecked()
    assert panel.txt_thumb_fails.isHidden()

    panel.btn_thumb_fails.setChecked(True)
    assert not panel.txt_thumb_fails.isHidden()
    assert panel.txt_thumb_fails.maximumHeight() <= 72
    body = panel.txt_thumb_fails.toPlainText()
    assert "zebra_0.eps" in body
    assert "INDEX_FROZEN" in body
    assert panel.thumb_health_box.maximumHeight() <= 110


def test_thumb_health_hidden_when_empty(qapp):
    panel = ResultsPanel()
    panel._on_thumb_health(_Snap(total=0))
    assert panel.thumb_health_box.isHidden()


def test_ai_pipeline_stays_single_line(qapp):
    panel = ResultsPanel()
    panel.set_pipeline_health(
        {
            "ai_active": True,
            "checklist": {
                "Embedding": "✓",
                "FAISS": "✓",
                "Pattern DNA": "✓",
                "Knowledge Graph": "✓",
                "Re-ranker": "✓",
                "Family Gate": "✓",
            },
            "stages": [
                {"name": "Hızlı", "count": 10},
                {"name": "Tam", "count": 40},
            ],
        }
    )
    html = panel.lbl_ai_health.text()
    assert not panel.lbl_ai_health.isHidden()
    assert "<br>" not in html
    assert panel.lbl_ai_health.maximumHeight() <= 28
    assert "AI ✓" in html
