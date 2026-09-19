
from pathlib import Path

from core.db import Database
from core.index_v3.physical_reconcile import reconcile_stale_physical_flags
from core.index_v3.ui_bridge import count_v3_ssot
from core.utils import file_id_from_path


def test_cache_first_relinks_existing_artifacts(tmp_path: Path):
    db = Database(str(tmp_path / "patterns.db"))
    source = tmp_path / "source" / "leopard.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")

    cache = tmp_path / "cache"
    (cache / "thumbnails").mkdir(parents=True)
    (cache / "feature_previews").mkdir(parents=True)

    fid = str(file_id_from_path(str(source)))
    thumb = cache / "thumbnails" / f"{fid}.webp"
    prev = cache / "feature_previews" / f"{fid}_fp.webp"
    thumb.write_bytes(b"thumb")
    prev.write_bytes(b"preview")

    file_id = db.upsert_file({
        "path": str(source),
        "filename": source.name,
        "source_id": 1,
        "status": "indexed",
        "physical_thumbnail_ready": 0,
        "physical_preview_ready": 0,
        "thumbnail_path": "",
        "feature_preview_path": "",
    })

    before = count_v3_ssot(db, [1])
    assert before["thumbnail"] == 0
    assert before["preview"] == 0

    reconcile_stale_physical_flags(db, [1], cache_dir=str(cache))

    row = db.get_file_by_id(file_id)
    assert row["thumbnail_path"] == str(thumb)
    assert row["feature_preview_path"] == str(prev)
    assert row["physical_thumbnail_ready"] == 1
    assert row["physical_preview_ready"] == 1

    after = count_v3_ssot(db, [1])
    assert after["thumbnail"] == 1
    assert after["preview"] == 1


def test_relinks_from_project_cache_when_settings_cache_empty(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "patterns.db"))
    source = tmp_path / "source" / "keep.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"keep")

    project_cache = tmp_path / "project_cache"
    (project_cache / "thumbnails").mkdir(parents=True)
    (project_cache / "feature_previews").mkdir(parents=True)
    fid = str(file_id_from_path(str(source)))
    thumb = project_cache / "thumbnails" / f"{fid}.webp"
    prev = project_cache / "feature_previews" / f"{fid}_fp.webp"
    thumb.write_bytes(b"thumb")
    prev.write_bytes(b"preview")

    monkeypatch.setattr(
        "core.index_v3.physical_reconcile.DEFAULT_CACHE_DIR", project_cache
    )

    file_id = db.upsert_file({
        "path": str(source),
        "filename": source.name,
        "source_id": 1,
        "status": "indexed",
        "physical_thumbnail_ready": 0,
        "physical_preview_ready": 0,
        "thumbnail_path": "",
        "feature_preview_path": "",
    })
    empty = tmp_path / "settings_cache"
    empty.mkdir()
    reconcile_stale_physical_flags(db, [1], cache_dir=str(empty))
    row = db.get_file_by_id(file_id)
    assert row["feature_preview_path"] == str(prev)
    assert row["physical_preview_ready"] == 1
    assert count_v3_ssot(db, [1])["preview"] == 1
