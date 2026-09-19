from pathlib import Path
import sqlite3

from core.db import Database
from core.legacy_index_recovery import recover_orphan_source_registry, recover_previous_database
from core.settings import AppSettings


def _db(path: Path, files):
    db=Database(path)
    with db.connect() as con:
        for p,sid in files:
            con.execute("INSERT INTO files(path, filename, source_id, status) VALUES (?,?,?,?)",(str(p),Path(p).name,sid,"indexed"))
    return db


def test_recover_orphan_source_registry(tmp_path):
    root=tmp_path/"archive"/"patterns"
    root.mkdir(parents=True)
    dbp=tmp_path/"data"/"patterns.db"
    db=_db(dbp,[root/"a.jpg",1,]) if False else _db(dbp,[(root/"a.jpg",1),(root/"b.jpg",1)])
    s=AppSettings(db_path=str(dbp), cache_dir=str(tmp_path/"cache"))
    out=recover_orphan_source_registry(s)
    assert out["recovered"] and out["sources"]==1
    with db.connect() as con:
        sid=con.execute("SELECT id FROM sources LIMIT 1").fetchone()[0]
        assert con.execute("SELECT COUNT(*) FROM files WHERE source_id=?",(sid,)).fetchone()[0] == 2


def test_recover_previous_database_does_not_overwrite(tmp_path):
    old=tmp_path/"old"/"data"/"patterns.db"
    old.parent.mkdir(parents=True)
    _db(old,[(tmp_path/"archive"/"a.jpg",1)])
    current=tmp_path/"new"/"data"/"patterns.db"
    s=AppSettings(db_path=str(current), cache_dir=str(tmp_path/"new"/"cache"))
    out=recover_previous_database(s)
    assert out["recovered"]
    assert Path(s.db_path).resolve()==old.resolve()
    assert sqlite3.connect(old).execute("SELECT COUNT(*) FROM files").fetchone()[0]==1


def test_recover_legacy_source_id_zero_makes_v3_archive_searchable(tmp_path):
    """Pre-source-schema indexes commonly have source_id=0."""
    from core.index_v3.scope import resolve_index_scope
    from core.index_v3.ui_bridge import count_v3_ssot, v3_status_dict

    root = tmp_path / "archive" / "patterns"
    root.mkdir(parents=True)
    paths = [root / f"p{i}.jpg" for i in range(3)]
    dbp = tmp_path / "data" / "patterns.db"
    db = _db(dbp, [(p, 0) for p in paths])
    s = AppSettings(db_path=str(dbp), cache_dir=str(tmp_path / "cache"))

    out = recover_orphan_source_registry(s)

    assert out["recovered"]
    assert out["files_reassigned"] == 3
    scope = resolve_index_scope(db, [])
    assert scope.source_ids
    assert scope.orphan_file_total == 0
    assert count_v3_ssot(db, scope.source_ids)["total"] == 3
    assert v3_status_dict(db, scope.source_ids, scope=scope)["total"] == 3


def test_recover_orphan_source_even_when_other_sources_exist(tmp_path):
    """A valid legacy bucket must not be ignored just because one source exists."""
    root = tmp_path / "legacy" / "patterns"
    root.mkdir(parents=True)
    dbp = tmp_path / "data" / "patterns.db"
    db = _db(dbp, [(root / "a.jpg", 0), (root / "b.jpg", 0)])
    with db.connect() as con:
        con.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("current", str(tmp_path / "current"),),
        )
    s = AppSettings(db_path=str(dbp), cache_dir=str(tmp_path / "cache"))

    out = recover_orphan_source_registry(s)

    assert out["recovered"]
    assert out["files_reassigned"] == 2
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM files WHERE source_id=0").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 2
