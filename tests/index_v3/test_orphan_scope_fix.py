"""Orphan scope: Tüm Arşiv orphan içermez; yetim ayrı diagnostic."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.scope import resolve_index_scope
from core.index_v3.ui_bridge import count_v3_ssot, v3_status_dict
from core.sources import SourceManager


def _imgs(folder: Path, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (32, 32), color=(i * 40 % 200, 20, 70)).save(
            folder / f"o_{i}.jpg", "JPEG"
        )


def test_remove_source_keep_index_orphan_not_in_archive(tmp_path: Path):
    db = Database(tmp_path / "orph.db")
    root = tmp_path / "files"
    _imgs(root, 3)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("src", str(root)),
        )
    eng = IndexEngineV3(db, job_db_path=tmp_path / "j.db")
    eng.run(mode=Mode.FAST, sources=[{"id": 1, "root_path": str(root)}])
    assert count_v3_ssot(db, [1])["total"] == 3

    # Kaynağı kaldır, index koru
    db.delete_source(1)
    scope = resolve_index_scope(db, [])
    assert scope.mode == "all_archive"
    assert scope.source_ids == []
    assert scope.active_registry_ids == []
    assert scope.orphan_file_total == 3
    assert scope.orphan_source_ids == [1]

    # Tüm arşiv / Index = 0 (boş source_ids → 1=0 clause)
    c = count_v3_ssot(db, scope.source_ids)
    assert c["total"] == 0
    st = v3_status_dict(db, scope.source_ids, scope=scope)
    assert st["total"] == 0
    assert st["ssot_total"] == 0
    assert st["light_queue_display"] == 0
    assert st["heavy_queue_display"] == 0
    assert st["orphan_file_total"] == 3

    # Index kuyruğu orphan'ı almaz
    eng2 = IndexEngineV3(db, job_db_path=tmp_path / "j2.db")
    # sources listesi boş (aktif yok)
    eng2.run(mode=Mode.COMPLETE, sources=[], walk_disk=False)
    assert count_v3_ssot(db, [])["total"] == 0
    # orphan files hâlâ DB'de
    with db.connect() as conn:
        n = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
    assert n == 3


def test_purge_orphans_clears_ghost_files(tmp_path: Path):
    db = Database(tmp_path / "purg.db")
    root = tmp_path / "files"
    _imgs(root, 3)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("src", str(root)),
        )
    eng = IndexEngineV3(db, job_db_path=tmp_path / "j.db")
    eng.run(mode=Mode.FAST, sources=[{"id": 1, "root_path": str(root)}])
    db.delete_source(1)
    assert resolve_index_scope(db, []).orphan_file_total == 3

    class _S:
        pass

    mgr = _S()
    mgr.db = db
    mgr.settings = None
    result = SourceManager.purge_orphan_indexes(
        mgr, str(tmp_path / "cache")  # type: ignore[arg-type]
    )
    assert int(result["files_removed"]) == 3
    assert resolve_index_scope(db, []).orphan_file_total == 0
    assert count_v3_ssot(db, [])["total"] == 0

def test_empty_scope_clause_not_all_files(tmp_path: Path):
    db = Database(tmp_path / "all.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("gone", str(tmp_path)),
        )
        # file without going through discovery — after delete becomes orphan
        sid = 1
        conn.execute(
            "INSERT INTO files(path, filename, source_id, status) VALUES (?,?,?,?)",
            (str(tmp_path / "x.jpg"), "x.jpg", sid, "indexed"),
        )
        conn.execute("DELETE FROM sources WHERE id=?", (sid,))
    # [] must be 0, not all-files
    assert count_v3_ssot(db, [])["total"] == 0
    # None still unscoped diagnostic
    assert count_v3_ssot(db, None)["total"] == 1
