from pathlib import Path

from core.path_safety import remap_legacy_storage_path
from core.settings import PROJECT_ROOT


def test_remap_mnt_and_windows_legacy_roots():
    cache = remap_legacy_storage_path("/mnt/data/work_v12424/cache")
    db = remap_legacy_storage_path(r"C:\mnt\data\work_v12424\data\patterns.db")
    assert Path(cache) == (PROJECT_ROOT / "cache").resolve()
    assert Path(db) == (PROJECT_ROOT / "data" / "patterns.db").resolve()
    assert remap_legacy_storage_path(str(PROJECT_ROOT / "cache")) == str(
        PROJECT_ROOT / "cache"
    )
