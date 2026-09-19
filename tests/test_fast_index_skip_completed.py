
from pathlib import Path

from core.indexer import Indexer


class _DB:
    def __init__(self):
        self.updated = []
    def get_features(self, fid):
        return {"phash": "abc"}
    def update_physical_readiness(self, *args, **kwargs):
        self.updated.append((args, kwargs))
    def connect(self):
        class Ctx:
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def execute(self,*a): return []
        return Ctx()


def test_completed_light_artifacts_skip_even_if_legacy_status_is_stale(tmp_path):
    t = tmp_path / "thumb.webp"; t.write_bytes(b"x")
    p = tmp_path / "preview.webp"; p.write_bytes(b"x")
    src = tmp_path / "a.jpg"; src.write_bytes(b"x")
    obj = Indexer.__new__(Indexer)
    obj.db = _DB()
    obj.settings = type("S", (), {"cache_dir": str(tmp_path / "cache")})()
    row = {
        "id": 1, "path": str(src), "file_size": src.stat().st_size,
        "mtime": src.stat().st_mtime,
        "status": "processing", "index_stage": "pending_light",
        "light_status": "pending", "needs_medium_preview": 1,
        "thumbnail_path": str(t), "feature_preview_path": str(p),
        "physical_thumbnail_ready": 0, "physical_preview_ready": 0,
        "width": 100, "format_metadata": "{}",
    }
    assert obj._is_fast_db_skip(row, row["file_size"], row["mtime"]) is True
