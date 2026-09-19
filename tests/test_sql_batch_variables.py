from __future__ import annotations

from pathlib import Path

import pytest

from core.db import Database, iter_sql_batches, sql_in_chunk_size, sqlite_max_variables


def _records(n: int, root: Path, source_id: int = 1) -> list[dict]:
    return [
        {
            "path": str(root / f"f{i:05d}.jpg"),
            "filename": f"f{i:05d}.jpg",
            "source_id": source_id,
            "file_size": 10,
            "mtime": 1.0 + i,
        }
        for i in range(n)
    ]


def test_sql_chunk_accounts_for_params_per_item():
    size = sql_in_chunk_size(extra_params=1, params_per_item=2, max_vars=999)
    assert size == (999 - 1 - 16) // 2
    chunks = list(
        iter_sql_batches(list(range(10_500)), extra_params=1, params_per_item=1, max_vars=999)
    )
    assert chunks
    assert all(len(c) <= 999 - 1 - 16 for c in chunks)
    assert sum(len(c) for c in chunks) == 10_500


@pytest.mark.parametrize("n", [20, 500, 1000, 10_000])
def test_bulk_insert_source_files_batches(tmp_path: Path, n: int):
    db = Database(tmp_path / "patterns.db")
    root = tmp_path / "src"
    recs = _records(n, root)
    first = db.bulk_insert_source_files(recs)
    assert len(first) == n
    second = db.bulk_insert_source_files(recs)
    assert second == first
    with db.connect() as conn:
        count = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
    assert count == n
    assert sqlite_max_variables() >= 1


def test_legacy_style_in_clause_with_extra_bind(tmp_path: Path):
    db = Database(tmp_path / "patterns.db")
    root = tmp_path / "src"
    recs = _records(1200, root, source_id=1)
    ids = list(db.bulk_insert_source_files(recs).values())
    with db.connect() as conn:
        n = 0
        for part in iter_sql_batches(ids, extra_params=1, conn=conn):
            ph = ",".join("?" * len(part))
            cur = conn.execute(
                f"UPDATE files SET source_id=? WHERE id IN ({ph})",
                [2, *part],
            )
            n += int(cur.rowcount or 0)
        conn.commit()
        left = int(conn.execute("SELECT COUNT(*) FROM files WHERE source_id=2").fetchone()[0])
    assert n == 1200
    assert left == 1200
