
from __future__ import annotations

import sqlite3
from pathlib import Path

from core.index_v3 import Artifact, Mode
from core.index_v3.discovery import enqueue_existing_gaps
from core.index_v3.queues import JobStore


class _Db:
    def __init__(self, path: Path):
        self.path = path

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

def _setup(db_path: Path, n: int = 600):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE files(
            id INTEGER PRIMARY KEY,
            source_id INTEGER,
            status TEXT,
            light_status TEXT,
            heavy_status TEXT,
            path TEXT,
            filename TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO files(id,source_id,status,light_status,heavy_status,path,filename) "
        "VALUES(?,?,?,?,?,?,?)",
        [
            (i, 1, "pending", "pending", "pending", f"/x/{i}.jpg", f"{i}.jpg")
            for i in range(1, n + 1)
        ],
    )
    conn.commit()
    conn.close()

def test_gap_planner_moves_past_completed_first_chunk(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    _setup(db_path, 600)
    store = JobStore(tmp_path / "jobs.sqlite")
    db = _Db(db_path)

    class Report:
        file_id = 0
        source_id = 1
        path = ""
        preview_ready = False
        light_complete = False
        @staticmethod
        def ready(_artifact):
            return False

    def _report(_db, fid, require_disk=False):
        r = Report()
        r.file_id = int(fid)
        r.path = f"/x/{fid}.jpg"
        return r

    monkeypatch.setattr("core.index_v3.discovery.assess_file", _report)

    # First chunk: 300 files, each gets light jobs.
    n1, last1 = enqueue_existing_gaps(
        db, store, source_id=1, mode=Mode.FAST, limit=300
    )
    assert n1 > 0
    assert last1 == 300

    # Simulate that the first chunk is fully completed.
    with store._connect() as conn:
        conn.execute("UPDATE index_v3_jobs SET state='done' WHERE file_id <= 300")
        conn.commit()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE files SET light_status='done' WHERE id <= 300"
        )
        conn.commit()

    # The next planner call MUST select 301..600, not reopen 1..300.
    n2, last2 = enqueue_existing_gaps(
        db, store, source_id=1, mode=Mode.FAST, limit=300
    )
    assert n2 > 0
    assert last2 == 600

    with store._connect() as conn:
        ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT file_id FROM index_v3_jobs WHERE state='pending' ORDER BY file_id"
        ).fetchall()]
    assert ids
    assert min(ids) == 301
    assert max(ids) == 600
    assert not set(ids) & set(range(1, 301))


def test_gap_planner_stops_when_pending_already_exists(tmp_path, monkeypatch):
    """limit=N: added=0 olsa bile ~2N dosyadan fazla taranmaz (full-table yok)."""
    db_path = tmp_path / "db.sqlite"
    _setup(db_path, 500)
    store = JobStore(tmp_path / "jobs.sqlite")
    db = _Db(db_path)

    class Report:
        file_id = 0
        source_id = 1
        path = ""
        preview_ready = False
        light_complete = False

        @staticmethod
        def ready(_artifact):
            return False

    def _report(_db, fid, require_disk=False):
        r = Report()
        r.file_id = int(fid)
        r.path = f"/x/{fid}.jpg"
        return r

    monkeypatch.setattr("core.index_v3.discovery.assess_file", _report)

    # Pre-fill pending for all files (idempotent enqueue → added=0).
    n0, _ = enqueue_existing_gaps(
        db, store, source_id=1, mode=Mode.FAST, limit=500
    )
    assert n0 > 0
    assessed = {"n": 0}

    def _counting_report(_db, fid, require_disk=False):
        assessed["n"] += 1
        return _report(_db, fid, require_disk=require_disk)

    monkeypatch.setattr("core.index_v3.discovery.assess_file", _counting_report)
    n1, last1 = enqueue_existing_gaps(
        db, store, source_id=1, mode=Mode.FAST, limit=50
    )
    assert n1 == 0
    # 2×limit examine cap — asla 500'ün tamamı taranmamalı.
    assert assessed["n"] <= 100
    assert assessed["n"] >= 50
    assert last1 > 0
    assert last1 <= 100
