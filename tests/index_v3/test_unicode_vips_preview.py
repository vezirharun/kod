"""Unicode Windows path + bozuk JPEG: Preview fail etmeli, lane durmamalı."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.network_index_throttle import is_network_path
from core.preview_cache import FeaturePreviewCache
from core.thumbnailer import HAS_VIPS, Thumbnailer, vips_new_from_file
from core.utils import fs_access_path, normalize_path


def test_extended_local_path_is_not_network():
    assert is_network_path(r"\\?\F:\karşıdan yüklemeler\a.jpg") is False
    assert is_network_path(r"F:\karşıdan yüklemeler\a.jpg") is False
    assert is_network_path(r"\\server\share\a.jpg") is True


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), (12, 80, 160)).save(path, "JPEG", quality=90)


def test_unicode_folder_preview_and_thumbnail(tmp_path):
    folder = tmp_path / "karşıdan yüklemeler"
    src = folder / "steptodown.com498374.jpg"
    _write_jpeg(src)
    cache = tmp_path / "cache"
    preview = FeaturePreviewCache(str(cache), max_edge=64, fmt="webp")
    result = preview.create(str(src))
    assert result.success, result.error
    assert Path(result.preview_path).is_file()
    thumb = Thumbnailer(str(cache), max_edge=32, fmt="webp")
    tr = thumb.create_from_existing_preview(str(src), result.preview_path)
    assert tr.success, tr.error
    assert Path(tr.thumbnail_path).is_file()


def test_vips_rejects_extended_prefix_but_helper_opens(tmp_path):
    if not HAS_VIPS:
        return
    folder = tmp_path / "karşıdan yüklemeler"
    src = folder / "ok.jpg"
    _write_jpeg(src)
    canonical = normalize_path(str(src))
    extended = fs_access_path(canonical)
    assert extended.startswith("\\\\?\\")
    img = vips_new_from_file(extended, access="sequential", page=0)
    assert int(img.width) > 0


def test_corrupt_jpeg_fails_and_next_succeeds(tmp_path):
    folder = tmp_path / "karşıdan yüklemeler"
    bad = folder / "broken.jpg"
    good = folder / "good.jpg"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not-a-jpeg-at-all")
    _write_jpeg(good)
    cache = FeaturePreviewCache(str(tmp_path / "cache"), max_edge=64, fmt="webp")
    bad_r = cache.create(str(bad))
    good_r = cache.create(str(good))
    assert bad_r.success is False
    assert good_r.success is True
    assert Path(good_r.preview_path).is_file()


def test_ten_files_one_corrupt_nine_ok(tmp_path):
    folder = tmp_path / "karşıdan yüklemeler"
    cache = FeaturePreviewCache(str(tmp_path / "cache"), max_edge=64, fmt="webp")
    ok = 0
    failed = 0
    for i in range(10):
        p = folder / f"f{i}.jpg"
        if i == 3:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00\x01\x02notjpeg")
        else:
            _write_jpeg(p)
        r = cache.create(str(p))
        if r.success:
            ok += 1
        else:
            failed += 1
    assert failed == 1
    assert ok == 9


def test_worker_marks_vips_format_error_permanent(tmp_path):
    from core.index_v3.queues import JobStore
    from core.index_v3.types import Artifact, Job, QueueKind
    from core.index_v3.worker import ArtifactProcessor, Worker, WorkerStats

    class DB:
        def get_file_by_id(self, fid):
            return {"id": fid, "path": "x.jpg", "source_id": 1}

        def get_features(self, fid):
            return {}

        def upsert_file(self, payload):
            return None

        def update_physical_readiness(self, *a, **k):
            return None

    def boom(db, job, stats: WorkerStats):
        raise RuntimeError(
            'VipsForeignLoad: "\\\\?\\F:\\karşıdan yüklemeler\\x.jpg" '
            "is not a known file format"
        )

    store = JobStore(tmp_path / "jobs.db")
    job = Job(1, Artifact.PREVIEW, QueueKind.PREVIEW, 1, "x.jpg")
    store.enqueue([job])
    worker = Worker(
        DB(),
        store,
        processor=ArtifactProcessor(process_fn=boom),
        max_attempts=1,
    )
    worker.run_queue(QueueKind.PREVIEW, [1], max_jobs=1)
    assert worker.stats.failed >= 1
    assert store.job_state(1, Artifact.PREVIEW) == "failed_permanent"
    assert store.count_pending(QueueKind.PREVIEW, source_ids=[1]) == 0


def test_live_steptodown_loaders_if_present():
    src = Path(r"F:\karşıdan yüklemeler\steptodown.com498374.jpg")
    if not src.is_file():
        return
    head = src.read_bytes()[:16]
    is_jpeg = head.startswith(b"\xff\xd8\xff")
    with src.open("rb") as fh:
        assert len(fh.read(8)) == 8
    pillow_ok = False
    try:
        with Image.open(src) as im:
            im.load()
            pillow_ok = im.size[0] > 0
    except Exception:
        pillow_ok = False
    if HAS_VIPS:
        from core.thumbnailer import pyvips

        helper_ok = True
        try:
            img = vips_new_from_file(str(src), access="sequential", page=0)
            assert int(img.width) > 0
        except Exception:
            helper_ok = False
        extended_ok = True
        try:
            pyvips.Image.new_from_file(
                fs_access_path(str(src)), access="sequential", page=0
            )
        except Exception:
            extended_ok = False
        if is_jpeg and pillow_ok:
            assert helper_ok is True
        assert extended_ok is False or helper_ok is True
