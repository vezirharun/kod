from pathlib import Path

import pytest
from PIL import Image

from core.index_ssot import archive_content_changed
from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.real_processor import RealArtifactProcessor
from core.index_v3.types import Artifact, ArtifactStatus, FileArtifactReport, Job, Mode, QueueKind
from core.index_v3.worker import WorkerStats
from core.preview_cache import FeaturePreviewCache
from core.settings import AppSettings


class FakeDB:
    def __init__(self, row, feat=None):
        self.row = dict(row)
        self.feat = dict(feat or {})
        self.upserts = []

    def get_file_by_id(self, _):
        return dict(self.row)

    def get_features(self, _):
        return dict(self.feat)

    def upsert_features(self, fid, payload):
        self.upserts.append((fid, dict(payload)))
        self.feat.update(payload)


def _proc(tmp_path):
    settings = AppSettings(cache_dir=str(tmp_path / "cache"), ai_embedding_enabled=False)
    return RealArtifactProcessor(settings, retries=0, enable_ai=False)


def _job(path, fid=1):
    return Job(file_id=fid, artifact=Artifact.HASH, queue=QueueKind.HEAVY, source_id=1, path=path)


def test_existing_hash_is_not_recomputed(tmp_path, monkeypatch):
    proc = _proc(tmp_path)
    db = FakeDB({"path": str(tmp_path / "a.jpg"), "mtime": 1}, {"phash": "abc", "dhash": "d"})
    called = []

    def boom(*a, **k):
        called.append(1)
        raise AssertionError("hash yeniden hesaplanmamalı")

    monkeypatch.setattr(proc, "_feature_source", boom)
    monkeypatch.setattr(proc, "_get_extractor", boom)
    assert proc._process_hash(db, _job(db.row["path"]), db.row, db.row["path"]) is True
    assert called == []
    assert db.upserts == []


def test_missing_hash_uses_preview_not_source(tmp_path, monkeypatch):
    source = tmp_path / "kaynak.tif"
    source.write_bytes(b"not-an-image")
    cache = tmp_path / "cache"
    preview = FeaturePreviewCache(str(cache), 128, "webp")
    preview_path = Path(preview.preview_path_for(str(source)))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (10, 20, 30)).save(preview_path, "WEBP")

    proc = _proc(tmp_path)
    db = FakeDB(
        {"path": str(source), "feature_preview_path": str(preview_path), "mtime": 1},
        {},
    )
    seen = []

    class Ext:
        fast_hash_only = False

        def extract_from_path(self, src, **kwargs):
            seen.append(src)
            assert src == str(preview_path)
            assert Path(src).read_bytes() != source.read_bytes()
            return type("F", (), {"phash": "p", "dhash": "d", "whash": "w", "color_hist": None, "dominant_colors": []})()

    monkeypatch.setattr(proc, "_get_extractor", lambda need_ai=False: Ext())
    assert proc._process_hash(db, _job(str(source)), db.row, str(source)) is True
    assert seen == [str(preview_path)]
    assert db.feat["phash"] == "p"


def test_missing_hash_without_preview_waits(tmp_path):
    proc = _proc(tmp_path)
    db = FakeDB({"path": r"C:\arsiv\x.jpg", "feature_preview_path": "", "thumbnail_path": ""}, {})
    with pytest.raises(RuntimeError, match="preview_required"):
        proc._process_hash(db, _job(db.row["path"]), db.row, db.row["path"])
    assert db.upserts == []


def test_unchanged_file_keeps_hash():
    existing = {"status": "ok", "file_size": 10, "mtime": 100.0}
    assert archive_content_changed(existing, 10, 100.0) is False


def test_changed_file_invalidates_hash_plan():
    existing = {"status": "ok", "file_size": 10, "mtime": 100.0}
    assert archive_content_changed(existing, 11, 100.0) is True
    r = FileArtifactReport(file_id=1, source_id=1, path="x.jpg")
    r.status[Artifact.PREVIEW] = ArtifactStatus.READY
    r.status[Artifact.HASH] = ArtifactStatus.MISSING
    jobs = plan_jobs_for_file(r, Mode.GENERAL_AI)
    assert Artifact.HASH in [j.artifact for j in jobs]


def test_existing_hash_not_reenqueued():
    r = FileArtifactReport(file_id=1, source_id=1, path="x.jpg")
    r.status[Artifact.PREVIEW] = ArtifactStatus.READY
    r.status[Artifact.HASH] = ArtifactStatus.READY
    jobs = plan_jobs_for_file(r, Mode.GENERAL_AI)
    assert Artifact.HASH not in [j.artifact for j in jobs]


def test_turkish_path_hash_ok(tmp_path, monkeypatch):
    source = tmp_path / "şablon_örnek.jpg"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(source, "JPEG")
    cache = tmp_path / "cache"
    preview = FeaturePreviewCache(str(cache), 64, "webp")
    preview_path = Path(preview.preview_path_for(str(source)))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (4, 5, 6)).save(preview_path, "WEBP")

    proc = _proc(tmp_path)
    db = FakeDB({"path": str(source), "feature_preview_path": str(preview_path)}, {})

    class Ext:
        fast_hash_only = False

        def extract_from_path(self, src, **kwargs):
            Image.open(src).verify()
            return type("F", (), {"phash": "tr", "dhash": "tr", "whash": "tr", "color_hist": None, "dominant_colors": []})()

    monkeypatch.setattr(proc, "_get_extractor", lambda need_ai=False: Ext())
    assert proc._process_hash(db, _job(str(source)), db.row, str(source)) is True
    assert db.feat["phash"] == "tr"
