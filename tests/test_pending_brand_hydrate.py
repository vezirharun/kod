from pathlib import Path

from core.db import Database


def test_get_indexed_files_by_ids_include_pending(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES ('s','/x',1)"
        )
        sid = int(conn.execute("SELECT id FROM sources").fetchone()[0])
        conn.execute(
            "INSERT INTO files(path, filename, source_id, status) VALUES (?,?,?,?)",
            ("/x/dolce.tif", "dolce.tif", sid, "pending"),
        )
        fid = int(conn.execute("SELECT id FROM files").fetchone()[0])
        conn.commit()
    gated = db.get_indexed_files_by_ids([fid], include_processing_ready=True, lightweight=True)
    assert gated == []
    pending = db.get_indexed_files_by_ids(
        [fid], include_processing_ready=True, include_pending=True, lightweight=True
    )
    assert pending and pending[0]["filename"] == "dolce.tif"
