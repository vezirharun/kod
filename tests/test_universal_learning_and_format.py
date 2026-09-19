"""Stage 3 universal learning + format polish."""
from __future__ import annotations

import json
from pathlib import Path

from core.category_tree import child_categories, parent_categories
from core.concept_registry import concepts
from core.db import Database
from core.format_audit import audit_format_counts
from core.teach_me import teach_files


def _db(tmp_path: Path) -> tuple[Database, str]:
    path = str(tmp_path / "patterns.db")
    db = Database(path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db, path


def _concept_row(db_path: str, name: str) -> dict:
    key = name.casefold()
    for row in concepts(db_path):
        if str(row.get("canonical") or "").casefold() == key:
            return row
    return {}


def test_3_teach_adds_tr_en_aliases_and_category(tmp_path):
    db, path = _db(tmp_path)
    img = tmp_path / "m.jpg"
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": "m.jpg",
                "source_id": 1,
                "status": "indexed",
            }
        )
    )
    teach_files(
        db,
        path,
        [fid],
        "Mickey",
        overlay={"parent": "Cartoon", "child": "Mickey"},
    )
    row = _concept_row(path, "Mickey")
    assert row
    aliases = row.get("aliases") or []
    if isinstance(aliases, str):
        aliases = json.loads(aliases)
    assert any(str(a).lower() == "mickey" for a in aliases)
    parents = parent_categories(path)
    assert any(p == "Cartoon" for p in parents)
    kids = child_categories("Cartoon", path)
    assert any(k == "Mickey" for k in kids)


def test_3_teach_tiger_gets_kaplan_alias(tmp_path):
    db, path = _db(tmp_path)
    img = tmp_path / "t.jpg"
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {"path": str(img), "filename": "t.jpg", "source_id": 1, "status": "indexed"}
        )
    )
    teach_files(db, path, [fid], "Tiger", overlay={"parent": "Animal Print"})
    row = _concept_row(path, "Tiger")
    aliases = row.get("aliases") or []
    if isinstance(aliases, str):
        aliases = json.loads(aliases)
    joined = " ".join(str(a).lower() for a in aliases)
    assert "kaplan" in joined or "tiger" in joined


def test_format_merges_jpg_jpeg_and_reports_missing(tmp_path):
    db = Database(tmp_path / "patterns.db")
    for name, status in (("a.jpg", "pending"), ("b.jpeg", "pending"), ("c.jpg", "missing")):
        p = tmp_path / name
        p.write_bytes(b"x")
        fid = int(db.upsert_file({"path": str(p), "filename": name, "status": status}))
        if status != "missing":
            db.upsert_features(
                fid,
                {
                    "phash": "x",
                    "texture_map": {"pattern_family": "x"},
                    "clip_embedding": b"\x00\x01",
                },
            )
        else:
            with db.connect() as conn:
                conn.execute("UPDATE files SET status='missing' WHERE id=?", (fid,))
    rows = {r["format"]: r for r in audit_format_counts(str(db.db_path))}
    assert "jpeg" not in rows  # merged into jpg
    assert rows["jpg"]["found"] == 3
    assert rows["jpg"]["missing"] == 1
    assert rows["jpg"]["searchable"] == 2
