"""Bana Öğret önizleme: review kartları thumbnail yollarını taşımalı."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.autonomous_learn import pending_review_cards, upsert_review
from core.db import Database
from core.teach_me import TeachMeCard, best_preview_path


def _db(tmp_path):
    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db, str(tmp_path / "patterns.db")


def _add_with_thumbs(db, tmp_path, name, *, thumb: Path, fp: Path):
    img = tmp_path / name
    img.write_bytes(b"x")
    thumb.write_bytes(b"fake-thumb")
    fp.write_bytes(b"fake-fp")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": name,
                "source_id": 1,
                "status": "indexed",
                "thumbnail_path": str(thumb),
                "feature_preview_path": str(fp),
            }
        )
    )
    return fid


def test_pending_review_cards_include_thumbnail_paths(tmp_path):
    db, path = _db(tmp_path)
    cache = tmp_path / "cache"
    (cache / "thumbnails").mkdir(parents=True)
    (cache / "feature_previews").mkdir(parents=True)
    thumb = cache / "thumbnails" / "a.webp"
    fp = cache / "feature_previews" / "a_fp.webp"
    fid = _add_with_thumbs(db, tmp_path, "rose.png", thumb=thumb, fp=fp)
    upsert_review(
        path,
        fid,
        lane="low",
        suggested="GÜL",
        confidence=0.8,
        status="pending",
    )
    pools = pending_review_cards(db, path)
    cards = pools.get("suspicious") or []
    assert cards, "pending review card missing"
    card = cards[0]
    assert card.file_id == fid
    assert card.filename == "rose.png"
    assert card.preview_path == str(thumb)
    assert card.feature_preview_path == str(fp)
    assert Path(card.preview_path).is_file()


def test_thumb_path_prefers_existing_thumbnail(tmp_path):
    from ui.teach_me_panel import _thumb_path

    thumb = tmp_path / "t.webp"
    fp = tmp_path / "f.webp"
    thumb.write_bytes(b"x")
    fp.write_bytes(b"y")
    card = TeachMeCard(
        file_id=1,
        filename="a.png",
        path=str(tmp_path / "a.png"),
        preview_path=str(thumb),
        pool="suspicious",
        reason="",
        guess="x",
        confidence=0.5,
        feature_preview_path=str(fp),
    )
    got = _thumb_path(card, cache_dir=str(tmp_path))
    assert Path(got).is_file()
    assert Path(got).name == thumb.name


def test_thumb_path_falls_back_to_preview(tmp_path):
    from ui.teach_me_panel import _thumb_path

    fp = tmp_path / "only_fp.webp"
    fp.write_bytes(b"y")
    card = TeachMeCard(
        file_id=1,
        filename="a.png",
        path=str(tmp_path / "a.png"),
        preview_path="",
        pool="suspicious",
        reason="",
        guess="x",
        confidence=0.5,
        feature_preview_path=str(fp),
    )
    assert Path(_thumb_path(card, cache_dir=str(tmp_path))).is_file()


def test_thumb_path_empty_when_missing(tmp_path):
    from ui.teach_me_panel import _thumb_path

    card = TeachMeCard(
        file_id=1,
        filename="a.tif",
        path=str(tmp_path / "a.tif"),
        preview_path="",
        pool="undefined",
        reason="",
        guess="x",
        confidence=0.1,
        feature_preview_path="",
    )
    assert _thumb_path(card, cache_dir=str(tmp_path)) == ""


def test_best_preview_path_uses_cache_not_tif_when_present(tmp_path):
    src = tmp_path / "a.tif"
    src.write_bytes(b"tif")
    thumb = tmp_path / "t.webp"
    thumb.write_bytes(b"x")
    card = TeachMeCard(
        file_id=1,
        filename="a.tif",
        path=str(src),
        preview_path=str(thumb),
        pool="undefined",
        reason="",
        guess="x",
        confidence=0.1,
        feature_preview_path="",
    )
    assert best_preview_path(card) == str(thumb)


def test_preview_pix_placeholder():
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication.instance() or QApplication(sys.argv[:1])
    assert app is not None
    from ui.teach_me_panel import _preview_pix

    pix = _preview_pix("")
    assert not pix.isNull()
    assert pix.width() == 160
