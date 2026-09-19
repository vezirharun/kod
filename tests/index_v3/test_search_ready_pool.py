from __future__ import annotations

import json
from pathlib import Path

from core.db import Database


def _insert_file(
    db: Database,
    *,
    file_id: int,
    root: Path,
    status: str,
    phash: str = "a" * 16,
    texture_map: dict | None = None,
    dino: bytes | None = b"1",
    clip: bytes | None = b"2",
) -> None:
    path = root / f"{file_id}.jpg"
    path.write_bytes(b"x")
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO files(id, path, filename, status, source_id)
            VALUES (?, ?, ?, ?, 1)
            """,
            (file_id, str(path), path.name, status),
        )
        conn.execute(
            """
            INSERT INTO features(file_id, phash, texture_map, dino_embedding, clip_embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                file_id,
                phash,
                json.dumps(texture_map or {"pattern_family": "animal_print"}),
                dino,
                clip,
            ),
        )


def test_search_pool_includes_v3_ready_pending_files(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    root = tmp_path / "src"
    root.mkdir()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(id, name, root_path, is_active) VALUES (1, ?, ?, 1)",
            ("src", str(root)),
        )

    _insert_file(db, file_id=1, root=root, status="pending")
    _insert_file(db, file_id=2, root=root, status="pending", dino=None, clip=None)
    _insert_file(db, file_id=3, root=root, status="missing")
    _insert_file(db, file_id=4, root=root, status="excluded_internal")

    rows = db.get_indexed_files(include_processing_ready=True)
    assert [r["id"] for r in rows] == [1]

    detailed = db.get_indexed_files_by_ids([1, 2, 3, 4], include_processing_ready=True)
    assert [r["id"] for r in detailed] == [1]

    indexed_only = db.get_indexed_files()
    assert indexed_only == []
