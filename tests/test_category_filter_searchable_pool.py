"""Category filter uses same processing-ready pool as search."""
from __future__ import annotations

from pathlib import Path

from core.category_memory import register_category, register_root_category
from core.category_tree import child_categories, parent_categories
from core.db import Database, sql_processing_ready_clause
from core.format_audit import audit_format_counts


def _db(tmp_path: Path) -> Database:
    return Database(tmp_path / "patterns.db")


def _set_category(db: Database, file_id: int, category_path: str) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE files
            SET category_path=?, manual_category_path=?
            WHERE id=?
            """,
            (category_path, category_path, int(file_id)),
        )


def _add_visual_ready(
    db: Database,
    tmp_path: Path,
    name: str,
    *,
    status: str = "pending",
    category_path: str = "",
) -> int:
    img = tmp_path / name
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": name,
                "status": status,
            }
        )
    )
    db.upsert_features(
        fid,
        {
            "phash": "abc123",
            "texture_map": {"pattern_family": "animal_print"},
            "clip_embedding": b"\x00\x01\x02\x03",
        },
    )
    if category_path:
        _set_category(db, fid, category_path)
    return fid


def test_processing_ready_pending_tiger_in_filter(tmp_path):
    db = _db(tmp_path)
    tiger = _add_visual_ready(
        db, tmp_path, "t.tif", category_path="Animal Print/Tiger"
    )
    leo = _add_visual_ready(
        db, tmp_path, "l.tif", category_path="Animal Print/Leopard"
    )
    bare = tmp_path / "bare.jpg"
    bare.write_bytes(b"y")
    bare_id = int(
        db.upsert_file(
            {
                "path": str(bare),
                "filename": "bare.jpg",
                "status": "pending",
            }
        )
    )
    _set_category(db, bare_id, "Animal Print/Tiger")

    tiger_ids = db.list_file_ids_by_category_path("Animal Print/Tiger")
    leo_ids = db.list_file_ids_by_category_path("Animal Print/Leopard")
    parent_ids = db.list_file_ids_by_category_path("Animal Print")

    assert tiger in tiger_ids
    assert bare_id not in tiger_ids
    assert leo in leo_ids
    assert tiger not in leo_ids
    assert leo not in tiger_ids
    assert tiger in parent_ids and leo in parent_ids
    assert set(tiger_ids) <= set(parent_ids)
    assert set(leo_ids) <= set(parent_ids)


def test_child_only_does_not_include_sibling(tmp_path):
    db = _db(tmp_path)
    tiger = _add_visual_ready(db, tmp_path, "t.jpg", category_path="Animal Print/Tiger")
    leo = _add_visual_ready(db, tmp_path, "l.jpg", category_path="Animal Print/Leopard")
    ids = db.list_file_ids_by_category_path("Animal Print/Tiger")
    assert tiger in ids
    assert leo not in ids


def test_format_searchable_matches_processing_ready(tmp_path):
    db = _db(tmp_path)
    _add_visual_ready(db, tmp_path, "a.jpg", status="pending")
    _add_visual_ready(db, tmp_path, "b.jpg", status="indexed")
    bare = tmp_path / "c.jpg"
    bare.write_bytes(b"z")
    db.upsert_file({"path": str(bare), "filename": "c.jpg", "status": "pending"})

    rows = {r["format"]: r for r in audit_format_counts(str(db.db_path))}
    jpg = rows.get("jpg") or {}
    assert jpg.get("found", 0) >= 3
    assert jpg.get("searchable", 0) == 2
    assert jpg.get("indexed", 0) == 1

    ready = sql_processing_ready_clause()
    with db.connect() as conn:
        n = conn.execute(
            f"""
            SELECT COUNT(*) FROM files f
            LEFT JOIN features fe ON fe.file_id=f.id
            WHERE {ready}
              AND lower(substr(f.filename, instr(f.filename, '.') + 1))='jpg'
            """
        ).fetchone()[0]
    assert int(n) == int(jpg["searchable"])


def test_dynamic_taught_category_appears_in_tree(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    register_root_category(db_path, "YeniAnaGrup")
    register_category(db_path, "YeniAnaGrup", "YeniAlt")
    parents = parent_categories(db_path)
    assert any(p == "YeniAnaGrup" for p in parents)
    kids = child_categories("YeniAnaGrup", db_path)
    assert "YeniAlt" in kids
    assert "Animal Print" in parents
    assert "Tiger" in child_categories("Animal Print", db_path)
    assert "Leopard" in child_categories("Animal Print", db_path)
