"""OCR/PATCH files columns: idempotent ALTER when schema_version already 17/19."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.db import Database, SCHEMA_VERSION
from core.index_v3.ui_bridge import count_v3_ssot


def _seed_missing_ocr_error(db_path: Path, meta_version: str = "17") -> None:
    db = Database(db_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(db_path.parent / "src")),
        )
        conn.execute(
            """
            INSERT INTO files(path, filename, source_id, status, ocr_text,
                ocr_processed, partial_hash, full_hash)
            VALUES ('a.jpg','a.jpg',1,'indexed','HELLO',1,'ph','fh')
            """
        )
        conn.execute(
            "INSERT INTO features(file_id, phash) VALUES (1, 'pp')"
        )
        conn.execute("ALTER TABLE files DROP COLUMN ocr_error")
        conn.execute("ALTER TABLE files DROP COLUMN patch_error")
        conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'",
            (meta_version,),
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
        assert "ocr_error" not in cols
        assert "patch_error" not in cols


def _assert_repaired(db_path: Path) -> None:
    db = Database(db_path)
    with db.connect() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
        ver = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0]
        row = conn.execute(
            "SELECT ocr_text, ocr_processed, partial_hash, full_hash, "
            "ifnull(ocr_error,'') AS ocr_error, ifnull(patch_error,'') AS patch_error "
            "FROM files WHERE id=1"
        ).fetchone()
        feat = conn.execute(
            "SELECT phash FROM features WHERE file_id=1"
        ).fetchone()
        nfiles = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    assert "ocr_error" in cols
    assert "patch_error" in cols
    assert "ocr_processed" in cols
    assert int(ver) == SCHEMA_VERSION
    assert nfiles == 1
    assert row["ocr_text"] == "HELLO"
    assert int(row["ocr_processed"]) == 1
    assert row["partial_hash"] == "ph"
    assert row["full_hash"] == "fh"
    assert row["ocr_error"] == ""
    assert row["patch_error"] == ""
    assert feat["phash"] == "pp"
    pools = count_v3_ssot(db, [1])
    assert "ocr_error" in pools
    assert "patch_error" in pools
    assert pools["ocr_error"] == 0
    assert pools["patch_error"] == 0


def test_v17_db_gains_ocr_error_and_patch_error(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    _seed_missing_ocr_error(db_path, "17")
    _assert_repaired(db_path)


def test_schema_19_missing_ocr_error_is_repaired(tmp_path: Path):
    """Production shape: meta=19, patch_error present path; ocr_error missing."""
    db_path = tmp_path / "prod_shape.db"
    _seed_missing_ocr_error(db_path, "19")
    _assert_repaired(db_path)


def test_ensure_columns_idempotent(tmp_path: Path):
    db_path = tmp_path / "twice.db"
    _seed_missing_ocr_error(db_path, "19")
    Database(db_path)
    Database(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()]
    assert cols.count("ocr_error") == 1
    assert cols.count("patch_error") == 1
