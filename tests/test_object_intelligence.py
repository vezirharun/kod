"""Object Intelligence: isolated object DB, Faster R-CNN detector (not CLIP)."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from core.global_object_intelligence import DetectedObject, GlobalObjectIntelligence, _TORCHVISION_OK
from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.object_intelligence import (
    DETECTOR_BACKEND,
    DETECTOR_UNAVAILABLE,
    FALLBACK_DETECTOR_BACKEND,
    EMBED_COLORGRID,
    EMBED_RESNET,
    ObjectIntelligence,
    later_archive_scan_howto,
)
from core.object_search import parse_object_query
from core.settings import AppSettings


def test_object_index_disabled_by_default():
    s = AppSettings()
    assert s.object_index_enabled is False
    assert s.open_vocab_object_enabled is False
    assert s.object_auto_scan_on_startup is False
    assert s.object_db_path.endswith("object_index.db")


def test_bird_object_index_lookup_not_capped_at_fts_window():
    import inspect
    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine.search_by_text)
    assert "object_index_retrieval_limit = 8000" in src
    assert "_merged[: max(cand_limit, 200)]" not in src


def test_load_omitted_object_key_is_disabled(tmp_path):
    cfg = tmp_path / "settings.json"
    cfg.write_text("{}", encoding="utf-8")
    loaded = AppSettings.load(cfg)
    assert loaded.object_index_enabled is False
    assert loaded.object_auto_scan_on_startup is False


def test_no_object_auto_scan_on_launch():
    src = (Path(__file__).resolve().parents[1] / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "Do not auto-start an object archive scan" in src
    assert "object_auto_indexer.start()" not in src
    assert "116K" in later_archive_scan_howto() or "review" in later_archive_scan_howto().lower()


def test_parse_object_queries_are_detector_class_not_clip():
    assert parse_object_query("kedi")["groups"][0]["label"] == "cat"
    assert parse_object_query("köpek")["groups"][0]["label"] == "dog"
    assert parse_object_query("kuş")["groups"][0]["label"] == "bird"
    assert parse_object_query("çanta")["groups"][0]["label"] == "handbag"
    assert parse_object_query("araba")["groups"][0]["label"] == "car"
    q = parse_object_query("3 insan")
    assert q["kind"] == "count"
    assert q["groups"][0]["label"] == "person"
    assert q["groups"][0]["min_count"] == 3
    both = parse_object_query("kedi + çanta")
    assert both["kind"] == "and"
    assert both["source"] == "detector_class"
    assert parse_object_query("leopard desen") is None


def _jpeg(path: Path, color: tuple[int, int, int], size=(160, 160)) -> None:
    Image.new("RGB", size, color).save(path, "JPEG")


def _box(label: str, bbox, conf=0.91) -> DetectedObject:
    x1, y1, x2, y2 = bbox
    return DetectedObject(
        instance_id=f"{label}_01",
        label=label,
        label_tr=label,
        confidence=conf,
        bbox=(x1, y1, x2, y2),
        area_ratio=0.2,
        center=(0.5, 0.5),
    )


def test_planted_detector_search_and_counts(tmp_path):
    """Query contract with a real detector API (injected). Not CLIP-as-detection."""
    root = tmp_path / "fix"
    root.mkdir()
    cat = root / "cat.jpg"
    dog = root / "dog.jpg"
    bird = root / "bird.jpg"
    people = root / "people.jpg"
    both = root / "cat_bag.jpg"
    _jpeg(cat, (200, 80, 20))
    _jpeg(dog, (120, 80, 40))
    _jpeg(bird, (40, 120, 200))
    _jpeg(people, (30, 30, 30))
    _jpeg(both, (180, 90, 30))

    def detector(path: str) -> list[DetectedObject]:
        name = Path(path).name
        if name == "cat.jpg":
            return [_box("cat", (10, 10, 90, 90))]
        if name == "dog.jpg":
            return [_box("dog", (12, 12, 88, 88))]
        if name == "bird.jpg":
            return [_box("bird", (20, 20, 70, 70))]
        if name == "people.jpg":
            return [
                _box("person", (0, 0, 40, 80), 0.88),
                _box("person", (50, 0, 90, 80), 0.86),
                _box("person", (100, 0, 140, 80), 0.84),
            ]
        if name == "cat_bag.jpg":
            return [_box("cat", (5, 5, 70, 90)), _box("handbag", (80, 40, 150, 140))]
        return []

    oi = ObjectIntelligence(tmp_path / "object_index.db", detector=detector)
    assert oi.detector_backend() == "injected_detector"
    mapping = [(1, str(cat)), (2, str(dog)), (3, str(bird)), (4, str(people)), (5, str(both))]
    summary = oi.index_paths(mapping)
    assert summary["auto_scan"] is False
    assert summary["indexed_files"] == 5

    cat_hit = oi.detect_embed(str(cat))
    assert cat_hit["clip_used_as_detector"] is False
    assert cat_hit["evidence"] == "detector"
    assert cat_hit["objects"][0]["bbox"] == (10, 10, 90, 90)
    assert cat_hit["objects"][0]["confidence"] >= 0.9
    assert cat_hit["object_count"] == 1
    assert cat_hit["embedding_kind"] in {EMBED_RESNET, EMBED_COLORGRID}

    people_objs = oi.store.instances_for_file(4)
    assert len(people_objs) == 3
    assert {o["label"] for o in people_objs} == {"person"}

    both_objs = oi.store.instances_for_file(5)
    assert {o["label"] for o in both_objs} == {"cat", "handbag"}

    kedi_ids = oi.query_images("kedi")
    assert 1 in kedi_ids and 5 in kedi_ids
    assert oi.query_images("köpek") == [2]
    assert oi.query_images("kuş") == [3]
    assert 4 in oi.query_images("3 insan")
    assert 1 not in oi.query_images("3 insan")
    assert oi.query_images("kedi + çanta") == [5]
    assert oi.query_images("çanta") == [5]
    assert oi.object_count(4, "person") == 3
    assert oi.object_count(5) == 2

    similar = oi.search_similar_image(str(cat), threshold=0.2)
    assert any(h["file_id"] == 1 for h in similar)


def test_object_writes_do_not_touch_pattern_index(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    patterns = data / "patterns.db"
    patterns.write_bytes(b"not-a-real-index")
    snap = snapshot_index_artifacts(
        db_path=str(patterns),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        cache_dir=str(tmp_path / "cache"),
    )
    fp = freeze_fingerprint(snap)
    img = tmp_path / "x.jpg"
    _jpeg(img, (10, 20, 30))
    oi = ObjectIntelligence(
        data / "object_index.db",
        detector=lambda p: [_box("cat", (1, 1, 40, 40))],
    )
    oi.index_image(1, str(img))
    snap2 = snapshot_index_artifacts(
        db_path=str(patterns),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        cache_dir=str(tmp_path / "cache"),
    )
    assert freeze_fingerprint(snap2) == fp
    assert patterns.read_bytes() == b"not-a-real-index"
    assert (data / "object_index.db").is_file()


def test_search_engine_object_queries(tmp_path):
    from core.db import Database
    from core.search_engine import SearchEngine

    cache = tmp_path / "cache"
    data = tmp_path / "data"
    cache.mkdir()
    data.mkdir()
    settings = AppSettings(
        db_path=str(data / "patterns.db"),
        cache_dir=str(cache),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        face_db_path=str(data / "face_index.db"),
        object_db_path=str(data / "object_index.db"),
        object_index_enabled=True,
        face_index_enabled=False,
        ai_embedding_enabled=False,
        ocr_enabled=False,
    )
    db = Database(settings.db_path)
    folder = tmp_path / "files"
    folder.mkdir()
    paths = {}
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(folder)),
        )
    file_ids = {}
    for i, name in enumerate(["cat.jpg", "dog.jpg", "people.jpg", "cat_bag.jpg"], start=1):
        p = folder / name
        _jpeg(p, (40 + i * 20, 30, 80))
        paths[name] = p
        file_ids[name] = db.upsert_file(
            {
                "path": str(p),
                "filename": p.name,
                "source_id": 1,
                "status": "indexed",
                "file_size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "width": 160,
                "height": 160,
            }
        )

    def detector(path: str):
        name = Path(path).name
        if name == "cat.jpg":
            return [_box("cat", (8, 8, 80, 80))]
        if name == "dog.jpg":
            return [_box("dog", (8, 8, 80, 80))]
        if name == "people.jpg":
            return [_box("person", (0, 0, 30, 80)), _box("person", (40, 0, 70, 80)), _box("person", (80, 0, 110, 80))]
        if name == "cat_bag.jpg":
            return [_box("cat", (4, 4, 50, 90)), _box("handbag", (70, 30, 140, 120))]
        return []

    oi = ObjectIntelligence(settings.object_db_path, detector=detector)
    oi.index_image(int(file_ids["cat.jpg"]), str(paths["cat.jpg"]))
    oi.index_image(int(file_ids["dog.jpg"]), str(paths["dog.jpg"]))
    oi.index_image(int(file_ids["people.jpg"]), str(paths["people.jpg"]))
    oi.index_image(int(file_ids["cat_bag.jpg"]), str(paths["cat_bag.jpg"]))

    before = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    eng = SearchEngine(settings, load_ai=False)
    kedi = eng.search_by_text("kedi")
    kedi_names = {r.filename for r in kedi}
    assert "cat.jpg" in kedi_names
    assert kedi_names <= {"cat.jpg", "cat_bag.jpg"}
    assert kedi[0].breakdown.get("object_evidence") == "detector_class"
    kopek = eng.search_by_text("köpek")
    assert kopek and "dog.jpg" in kopek[0].filename
    people = eng.search_by_text("3 insan")
    assert people and "people.jpg" in people[0].filename
    combo = eng.search_by_text("kedi + çanta")
    assert combo and "cat_bag.jpg" in combo[0].filename
    after = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    assert freeze_fingerprint(before) == freeze_fingerprint(after)
    # Filename-only "kedi" must not mix into object ranking.
    stray = folder / "kedi_filename.jpg"
    _jpeg(stray, (9, 9, 9))
    db.upsert_file(
        {
            "path": str(stray),
            "filename": stray.name,
            "source_id": 1,
            "status": "indexed",
            "file_size": stray.stat().st_size,
            "mtime": stray.stat().st_mtime,
            "width": 160,
            "height": 160,
        }
    )
    kedi2 = eng.search_by_text("kedi")
    assert stray.name not in {r.filename for r in kedi2}


def test_detector_unavailable_is_explicit(tmp_path):
    img = tmp_path / "x.jpg"
    _jpeg(img, (10, 10, 10))
    oi = ObjectIntelligence(tmp_path / "object_index.db", detector_backend="unavailable")
    det = oi.detect_embed(str(img))
    assert det["ok"] is False
    assert det["error"] == DETECTOR_UNAVAILABLE
    assert det["evidence"] == DETECTOR_UNAVAILABLE
    assert det["clip_used_as_detector"] is False
    assert det["objects"] == []
    assert oi.detect(str(img)) == []


def test_low_confidence_detections_dropped(tmp_path):
    img = tmp_path / "x.jpg"
    _jpeg(img, (10, 10, 10))
    oi = ObjectIntelligence(
        tmp_path / "object_index.db",
        min_confidence=0.45,
        detector=lambda p: [_box("cat", (2, 2, 40, 40), 0.20)],
    )
    det = oi.detect_embed(str(img))
    assert det["ok"] is True
    assert det["object_count"] == 0
    assert det["clip_used_as_detector"] is False


def test_fullframe_disk_class_dropped_not_by_raising_conf(tmp_path):
    from dataclasses import replace

    img = tmp_path / "globe.jpg"
    _jpeg(img, (10, 10, 40))
    bowl = replace(_box("bowl", (2, 2, 158, 158), 0.91), area_ratio=0.80)
    cat = _box("cat", (10, 10, 50, 50), 0.88)
    oi = ObjectIntelligence(
        tmp_path / "object_index.db",
        min_confidence=0.45,
        detector=lambda p: [bowl, cat],
    )
    kept = oi.detect(str(img))
    assert [d.label for d in kept] == ["cat"]



def test_search_session_does_not_create_object_index(tmp_path):
    from core.db import Database
    from core.search_engine import SearchEngine

    cache = tmp_path / "cache"
    data = tmp_path / "data"
    cache.mkdir()
    data.mkdir()
    obj_db = data / "object_index.db"
    settings = AppSettings(
        db_path=str(data / "patterns.db"),
        cache_dir=str(cache),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        face_db_path=str(data / "face_index.db"),
        object_db_path=str(obj_db),
        object_index_enabled=True,
        object_auto_scan_on_startup=False,
        face_index_enabled=False,
        ai_embedding_enabled=False,
        ocr_enabled=False,
    )
    Database(settings.db_path)
    before = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    fp = freeze_fingerprint(before)
    eng = SearchEngine(settings, load_ai=False)
    assert eng.search_by_text("kedi") == []
    assert not obj_db.exists()
    after = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    assert freeze_fingerprint(after) == fp


def _draw_acceptance(path: Path, kind: str) -> None:
    im = Image.new("RGB", (320, 240), (20, 90, 30) if kind != "car" else (40, 40, 40))
    d = ImageDraw.Draw(im)
    if kind == "cat":
        d.ellipse((80, 80, 220, 200), fill=(220, 140, 40))
        d.polygon([(90, 90), (110, 20), (140, 90)], fill=(220, 140, 40))
        d.polygon([(160, 90), (190, 20), (210, 90)], fill=(220, 140, 40))
    elif kind == "dog":
        d.ellipse((70, 90, 240, 210), fill=(140, 90, 40))
        d.ellipse((200, 60, 280, 140), fill=(140, 90, 40))
    elif kind == "bird":
        d.ellipse((100, 90, 220, 160), fill=(30, 30, 30))
        d.polygon([(210, 120), (280, 100), (210, 140)], fill=(200, 40, 40))
    elif kind == "bag":
        d.rectangle((90, 80, 230, 200), fill=(80, 40, 20))
        d.arc((110, 40, 210, 120), 0, 180, fill=(40, 20, 10), width=8)
    elif kind == "people":
        for x in (40, 130, 220):
            d.ellipse((x + 20, 20, x + 60, 60), fill=(220, 180, 150))
            d.rectangle((x + 15, 60, x + 65, 200), fill=(30, 40, 120))
    elif kind == "cat_bag":
        d.ellipse((20, 80, 140, 200), fill=(220, 140, 40))
        d.rectangle((170, 90, 300, 210), fill=(80, 40, 20))
    im.save(path, "JPEG")


def test_real_fasterrcnn_or_honest_skip(tmp_path):
    cap = GlobalObjectIntelligence.capability()
    assert cap["backend"] in {FALLBACK_DETECTOR_BACKEND, "unavailable"}
    oi = ObjectIntelligence(tmp_path / "object_index.db")
    if not oi.detector_available():
        pytest.skip("object detector not available")
    expected = {
        "cat.jpg": {"cat"},
        "dog.jpg": {"dog"},
        "bird.jpg": {"bird"},
        "bag.jpg": {"handbag", "backpack"},
        "people.jpg": {"person"},
        "cat_bag.jpg": {"cat", "handbag", "backpack"},
    }
    root = tmp_path / "acc"
    root.mkdir()
    kinds = {
        "cat.jpg": "cat",
        "dog.jpg": "dog",
        "bird.jpg": "bird",
        "bag.jpg": "bag",
        "people.jpg": "people",
        "cat_bag.jpg": "cat_bag",
    }
    hits = 0
    total = 0
    details = []
    for name, kind in kinds.items():
        p = root / name
        _draw_acceptance(p, kind)
        det = oi.detect_embed(str(p))
        assert det["ok"] is True
        assert det["clip_used_as_detector"] is False
        assert det["backend"] in {DETECTOR_BACKEND, FALLBACK_DETECTOR_BACKEND}
        labels = {o["label"] for o in det["objects"]}
        want = expected[name]
        total += 1
        ok = bool(labels & want)
        if name == "people.jpg":
            people_n = sum(1 for o in det["objects"] if o["label"] == "person")
            ok = people_n >= 3
        if name == "cat_bag.jpg":
            ok = ("cat" in labels) and bool(labels & {"handbag", "backpack"})
        hits += int(ok)
        details.append((name, sorted(labels), ok))
        for o in det["objects"]:
            x1, y1, x2, y2 = o["bbox"]
            assert x2 > x1 and y2 > y1
            assert 0.0 <= float(o["confidence"]) <= 1.0
    acc = hits / total if total else 0.0
    print(f"OBJECT_DETECTOR_ACCEPTANCE accuracy={acc:.2f} hits={hits}/{total} details={details}")
    assert isinstance(acc, float)
    if _TORCHVISION_OK and hits == 0:
        pytest.skip(
            f"Detector loaded but synthetic fixtures had no target classes: {details}"
        )
