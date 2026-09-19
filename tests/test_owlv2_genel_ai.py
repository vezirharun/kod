"""OWLv2 Genel AI progress SSOT. Search stays closed."""
from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

from PIL import Image

from core.index_freeze import allow_index_writes
from core.object_index import ObjectIndexStore
from core.ovd_index import index_open_vocab_image, owlv2_progress
from core.search_engine import SearchEngine
from core.settings import AppSettings

ROOT = Path(__file__).resolve().parents[1]


class _FakeOwl:
    calls = 0

    def detect(self, image, prompts, threshold):
        _FakeOwl.calls += 1
        w, h = image.size
        return [{
            "label": "crow",
            "confidence": 0.82,
            "xyxy": [w * 0.1, h * 0.1, w * 0.4, h * 0.4],
        }]


def test_flag_off_and_query_time_owl_zero():
    assert AppSettings().open_vocab_object_enabled is False
    src = inspect.getsource(SearchEngine.search_by_text)
    assert "get_owlv2_backend" not in src
    assert "Owlv2ForObjectDetection" not in inspect.getsource(SearchEngine)


def test_progress_live_schema_without_status_106_of_11818(tmp_path):
    db = tmp_path / "live_like.db"
    con = sqlite3.connect(str(db))
    con.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT DEFAULT '');
        CREATE TABLE ovd_file_scans(
          file_id INTEGER PRIMARY KEY,
          mtime REAL DEFAULT 0,
          file_size INTEGER DEFAULT 0,
          vocab_version TEXT DEFAULT '',
          model TEXT DEFAULT '',
          threshold REAL DEFAULT 0,
          scanned_at TEXT DEFAULT '',
          box_count INTEGER DEFAULT 0
        );
        """
    )
    con.executemany(
        """INSERT INTO ovd_file_scans(
             file_id,mtime,file_size,vocab_version,model,threshold,scanned_at,box_count)
           VALUES (?,?,?,?,?,?,?,?)""",
        [
            (i, 0, 0, "owlv2-ovd-v1", "owlv2/base-patch16", 0.1, "2026-01-01", 0)
            for i in range(1, 107)
        ],
    )
    con.commit()
    con.close()
    cols = {
        r[1]
        for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(ovd_file_scans)")
    }
    assert "status" not in cols
    p = ObjectIndexStore(db, readonly=True).owlv2_scan_progress(total=11818)
    assert p["completed"] == 106
    assert p["pending"] == 11818 - 106
    assert p["failed"] == 0
    assert p["retry"] == 0
    assert p["percent"] == round(100.0 * 106 / 11818, 1)


def test_progress_counts_resume_retry_no_duplicate(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (80, 80), (2, 2, 2)).save(img)
    store = ObjectIndexStore(tmp_path / "obj.db")
    fake = _FakeOwl()
    _FakeOwl.calls = 0
    with allow_index_writes():
        index_open_vocab_image(file_id=1, image_path=str(img), store=store, backend=fake)
        index_open_vocab_image(file_id=1, image_path=str(img), store=store, backend=fake)
    assert _FakeOwl.calls == 1
    p = owlv2_progress(store, total=50)
    assert p["total"] == 50
    assert p["completed"] == 1
    assert p["pending"] == 49
    assert p["failed"] == 0
    assert p["retry"] == 0
    assert round(p["percent"], 0) == 2.0

    store.upsert_open_vocab_objects(
        2, str(tmp_path / "fail.png"), [], vocab_version="owlv2-ovd-v1",
        model="owlv2/base-patch16", scan_status="failed", scan_error="boom",
    )
    store.upsert_open_vocab_objects(
        3, str(tmp_path / "retry.png"), [], vocab_version="owlv2-ovd-v1",
        model="owlv2/base-patch16", scan_status="retry", scan_error="tmp",
    )
    p2 = owlv2_progress(store, total=50)
    assert p2["completed"] == 3
    assert p2["failed"] == 1
    assert p2["retry"] == 1
    assert p2["pending"] == 47


def test_pdf_does_not_stop_queue(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    img = tmp_path / "b.png"
    Image.new("RGB", (40, 40), (3, 3, 3)).save(img)
    store = ObjectIndexStore(tmp_path / "obj2.db")
    fake = _FakeOwl()
    _FakeOwl.calls = 0
    with allow_index_writes():
        a = index_open_vocab_image(file_id=8, image_path=str(pdf), store=store, backend=fake)
        b = index_open_vocab_image(file_id=9, image_path=str(img), store=store, backend=fake)
    assert a["skipped"] == "non_raster"
    assert b["ok"] is True
    assert _FakeOwl.calls == 1
    p = owlv2_progress(store, total=2)
    assert p["completed"] == 2
    assert p["failed"] >= 1


def test_ui_defines_owlv2_outside_genel_ai_percent():
    text = (ROOT / "ui" / "progress_panel.py").read_text(encoding="utf-8")
    assert '("OWLv2", "owlv2_ready")' in text
    ai_ready = text.split("_AI_READY_KEYS")[1].split(")")[0]
    assert "owlv2" not in ai_ready
    health = text.split("_HEALTH_WEIGHTS")[1].split("}")[0]
    assert "owlv2" not in health
    assert "Bekleyen:" in text
    assert 'ov.get("percent")' in text or "percent" in text


def test_probe_progress_example_31_of_50(tmp_path):
    store = ObjectIndexStore(tmp_path / "obj4.db")
    img = tmp_path / "c.png"
    Image.new("RGB", (20, 20), (1, 1, 1)).save(img)
    for i in range(31):
        store.upsert_open_vocab_objects(
            i + 1, str(tmp_path / f"c{i}.png"), [], vocab_version="owlv2-ovd-v1",
            model="owlv2/base-patch16", scan_status="done",
        )
    p = owlv2_progress(store, total=50)
    assert p["completed"] == 31
    assert p["pending"] == 19
    assert p["percent"] == 62.0


def test_rtdetr_untouched_when_owl_progress_written(tmp_path):
    img = tmp_path / "c.png"
    Image.new("RGB", (40, 40), (4, 4, 4)).save(img)
    store = ObjectIndexStore(tmp_path / "obj3.db")
    store.replace_file_objects(
        7, str(img),
        [{"label": "bird", "confidence": 0.9, "bbox": (1, 2, 10, 20), "area_ratio": 0.1}],
        detector="ultralytics_rtdetr_l_coco",
    )
    n = store.rtdetr_count(7)
    fake = _FakeOwl()
    with allow_index_writes():
        index_open_vocab_image(file_id=7, image_path=str(img), store=store, backend=fake)
    assert store.rtdetr_count(7) == n
    assert store.search_label(["bird"]) == [7]
