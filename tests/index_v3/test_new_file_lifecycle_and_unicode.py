from pathlib import Path
import inspect

import cv2
import numpy as np

from core.db import Database
from core.index_v3.discovery import discover_source
from core.index_v3.queues import JobStore
from core.index_v3.types import Mode
from core.settings import AppSettings
from core.thumbnailer import Thumbnailer


def setup_source(tmp_path):
    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(tmp_path / "cache")
    settings.ensure_dirs()
    root = tmp_path / "karşıdan yüklemeler"
    root.mkdir()
    with db.connect() as con:
        con.execute("INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)", ("test", str(root)))
        sid = int(con.execute("SELECT id FROM sources ORDER BY id DESC LIMIT 1").fetchone()[0])
    return db, settings, JobStore(tmp_path / "jobs.db"), sid, root


def write_image(path: Path):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    assert cv2.imencode(path.suffix, img)[0] is not None
    ok, buf = cv2.imencode(path.suffix, img)
    assert ok
    path.write_bytes(buf.tobytes())


def test_new_file_is_added_to_total_and_queue(tmp_path):
    db, settings, jobs, sid, root = setup_source(tmp_path)
    write_image(root / "ilk.jpg")
    first = discover_source(db, jobs, source_id=sid, root_path=str(root), mode=Mode.FAST, settings=settings)
    assert first.inserted == 1
    assert jobs.count_pending(source_ids=[sid]) == 1

    write_image(root / "yeni_şık_üç.jpg")
    second = discover_source(db, jobs, source_id=sid, root_path=str(root), mode=Mode.FAST, settings=settings)
    assert second.inserted == 1
    assert second.unchanged == 1
    with db.connect() as con:
        total = int(con.execute("SELECT COUNT(*) FROM files WHERE source_id=? AND status!='missing'", (sid,)).fetchone()[0])
    assert total == 2
    assert jobs.count_pending(source_ids=[sid]) == 2


def test_rescan_does_not_duplicate_already_indexed_jobs(tmp_path):
    db, settings, jobs, sid, root = setup_source(tmp_path)
    write_image(root / "a.jpg")
    discover_source(db, jobs, source_id=sid, root_path=str(root), mode=Mode.FAST, settings=settings)
    # Mark the two planned light jobs and their physical artifacts ready, then
    # rescan: an actually indexed/ready file must not reopen its DONE jobs.
    with db.connect() as con:
        row = con.execute("SELECT id FROM files WHERE source_id=?", (sid,)).fetchone()
        fid = int(row[0])
        con.execute(
            "UPDATE files SET thumbnail_path=?, feature_preview_path=?, physical_thumbnail_ready=1, physical_preview_ready=1, format_metadata='{}', width=32, height=32 WHERE id=?",
            (str(tmp_path / "thumb.webp"), str(tmp_path / "prev.webp"), fid),
        )
        con.execute(
            "INSERT OR REPLACE INTO features(file_id, phash, texture_features) VALUES (?,?,?)",
            (fid, "ready", "[0.1]"),
        )
    (tmp_path / "thumb.webp").write_bytes(b"thumb")
    (tmp_path / "prev.webp").write_bytes(b"prev")
    with jobs._connect() as con:
        con.execute("UPDATE index_v3_jobs SET state='done' WHERE source_id=?", (sid,))
        con.commit()
    second = discover_source(db, jobs, source_id=sid, root_path=str(root), mode=Mode.FAST, settings=settings)
    assert second.inserted == 0
    assert second.changed == 0
    assert second.unchanged == 1
    assert jobs.count_pending(source_ids=[sid]) == 0


def test_unicode_image_loader_reads_turkish_path(tmp_path):
    db, settings, jobs, sid, root = setup_source(tmp_path)
    path = root / "yüz_şekli_ışık.jpg"
    write_image(path)
    image = Thumbnailer.load_image(str(path))
    assert image is not None
    assert image.shape[:2] == (32, 32)


def test_face_scanner_no_direct_cv2_imread():
    import core.face_scanner as mod
    src = inspect.getsource(mod.FaceIndexScanner)
    assert "cv2.imread" not in src


def test_quick_index_walks_source_for_new_files():
    src = Path(__file__).resolve().parents[2].joinpath("ui", "worker_threads.py").read_text(encoding="utf-8")
    assert "walk_disk = False" in src
    assert "if n_existing == 0" in src
    assert "walk_disk = True" in src
