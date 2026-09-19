"""Face index is off by default; scan only on explicit command."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.face_auto_indexer import FaceAutoIndexer
from core.face_background import FaceBackgroundIndexer
from core.face_scanner import FaceIndexScanner
from core.settings import AppSettings


def test_face_index_enabled_defaults_false():
    assert AppSettings().face_index_enabled is False


def test_load_omitted_key_is_disabled(tmp_path):
    cfg = tmp_path / "settings.json"
    cfg.write_text("{}", encoding="utf-8")
    loaded = AppSettings.load(cfg)
    assert loaded.face_index_enabled is False


def test_auto_indexer_does_not_start_when_disabled():
    idx = FaceAutoIndexer(SimpleNamespace(face_index_enabled=False, db_path="", face_db_path=""))
    idx.start()
    assert idx._thread is None


def test_background_indexer_does_not_start_when_disabled():
    bg = FaceBackgroundIndexer(SimpleNamespace(face_index_enabled=False))
    assert bg.start() is False
    assert bg.thread is None


def test_launch_does_not_auto_start_face_indexer():
    src = (Path(__file__).resolve().parents[1] / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "window._face_auto_indexer.start()" not in src
    assert "FaceAutoIndexer(settings)" in src


def test_explicit_start_runs_scanner_on_fixture_not_archive(tmp_path):
    from core.db import Database
    from PIL import Image

    patterns = tmp_path / "patterns.db"
    face_db = tmp_path / "face_index.db"
    img = tmp_path / "fixture.jpg"
    Image.new("RGB", (64, 64), (30, 40, 50)).save(img, "JPEG")
    db = Database(str(patterns))
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES(?,?,1)",
            ("fixture", str(tmp_path)),
        )
    db.upsert_file(
        {
            "path": str(img),
            "filename": img.name,
            "source_id": 1,
            "status": "indexed",
            "file_size": img.stat().st_size,
            "mtime": img.stat().st_mtime,
            "width": 64,
            "height": 64,
            "feature_preview_path": str(img),
        }
    )
    settings = SimpleNamespace(
        face_index_enabled=True,
        db_path=str(patterns),
        face_db_path=str(face_db),
        face_identity_threshold=0.62,
        face_identity_min_margin=0.05,
    )
    scanner = FaceIndexScanner(
        settings.db_path,
        settings.face_db_path,
        allow_vision_fallback=False,
    )
    assert scanner.engine.allow_vision_fallback is False
    result = scanner.scan_incremental(batch_size=8)
    assert "scanned" in result
    assert Path(scanner.store.path).resolve() == face_db.resolve()
    idx = FaceAutoIndexer(settings)
    idx.start()
    assert idx._thread is not None and idx._thread.is_alive()
    idx.stop()
