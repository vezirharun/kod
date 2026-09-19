"""Startup / Edit softness contracts — UI thread must not do heavy DB work eagerly."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.settings import AppSettings
from core.sources import SourceManager


def test_source_manager_defer_db_skips_open(tmp_path: Path):
    settings = AppSettings(db_path=str(tmp_path / "patterns.db"), cache_dir=str(tmp_path / "cache"))
    sm = SourceManager(settings, run_maintenance=False, defer_db=True)
    assert sm._db is None
    # First access opens DB
    sm.list_sources()
    assert sm._db is not None


def test_main_window_uses_deferred_source_manager():
    src = (Path(__file__).resolve().parents[1] / "ui" / "main_window.py").read_text(
        encoding="utf-8"
    )
    assert "defer_db=True" in src
    assert "_warmup_sources_async" in src
    assert "Hesaplanıyor" in src
    assert "Hazırlanıyor" in src
    # Status refresh deferred via QTimer, not sync in __init__ body before show
    assert "QTimer.singleShot(100, self._refresh_status)" in src


def test_edit_dialog_loads_options_async():
    src = (Path(__file__).resolve().parents[1] / "ui" / "result_metadata_dialog.py").read_text(
        encoding="utf-8"
    )
    assert "_start_options_load" in src
    assert "_OptionsWorker" in src
    assert "Listeler hazırlanıyor" in src
    # Sync fill helpers must not run inside _build_ui
    build_idx = src.index("def _build_ui")
    next_def = src.index("\n    def ", build_idx + 1)
    build_body = src[build_idx:next_def]
    assert "_fill_parents" not in build_body
    assert "parent_categories(" not in build_body
    assert "brand_names(" not in build_body


def test_edit_open_does_not_sync_overlay_before_dialog():
    src = (Path(__file__).resolve().parents[1] / "ui" / "main_window.py").read_text(
        encoding="utf-8"
    )
    fn = src[src.index("def _on_edit_result_metadata") : src.index("def _on_ai_prediction_action")]
    assert "ResultMetadataDialog(" in fn
    assert "def load_overlay():" in fn
    assert "build_common_edit_overlay" in fn
    # Dialog constructed before background overlay load body runs
    dlg_pos = fn.index("dlg = ResultMetadataDialog(")
    load_def = fn.index("def load_overlay():")
    assert dlg_pos < load_def
    assert "build_common_edit_overlay" in fn[load_def:]
