"""Sorgu görseli hedef indeks koleksiyonu değildir."""
from __future__ import annotations

from pathlib import Path

from core.db import Database
from core.on_demand_scan import (
    collect_preflight_paths,
    should_prompt_query_folder_index,
)
from core.settings import AppSettings


def _settings(tmp: Path) -> AppSettings:
    cache = tmp / "cache"
    data = tmp / "data"
    cache.mkdir()
    data.mkdir()
    return AppSettings(
        db_path=str(data / "patterns.db"),
        cache_dir=str(cache),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        face_db_path=str(data / "face_index.db"),
        face_index_enabled=False,
        ai_embedding_enabled=False,
        ocr_enabled=False,
    )


def test_query_folder_never_prompts_index(tmp_path: Path):
    desktop = tmp_path / "Desktop" / "musteri.jpg"
    desktop.parent.mkdir()
    desktop.write_bytes(b"preview")
    assert should_prompt_query_folder_index(str(desktop)) is False


def test_preflight_does_not_queue_external_query_folder(tmp_path: Path):
    settings = _settings(tmp_path)
    Database(settings.db_path)
    customer = tmp_path / "unindexed_client" / "p1.jpg"
    customer.parent.mkdir()
    customer.write_bytes(b"preview")
    assert collect_preflight_paths(settings, str(customer), query_only=True) == []
    assert collect_preflight_paths(settings, str(customer), query_only=False) == []


def test_set_query_image_source_skips_folder_index_prompt():
    from pathlib import Path as P

    text = (P(__file__).resolve().parents[1] / "ui" / "main_window.py").read_text(
        encoding="utf-8"
    )
    start = text.find("def _set_query_image")
    end = text.find("def _pick_query_image")
    body = text[start:end]
    assert "_maybe_index_before_search" not in body
    assert "Index Gerekli" not in body
