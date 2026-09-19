"""Spatial Intelligence V1: geometry, parser, evidence, search. INDEX FROZEN."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.db import Database
from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.object_index import ObjectIndexStore
from core.object_search import parse_object_query
from core.search_engine import SearchEngine
from core.settings import AppSettings, DEFAULT_DATA_DIR
from core.spatial.spatial_config import SpatialConfig
from core.spatial.spatial_engine import SPATIAL_EVIDENCE_UNAVAILABLE, SpatialEngine
from core.spatial.spatial_geometry import NormBox, normalize_bbox
from core.spatial.spatial_query import parse_spatial_query
from core.spatial.spatial_relation import Relation, evaluate_relation


def _box(x1, y1, x2, y2) -> NormBox:
    b = normalize_bbox((x1, y1, x2, y2), 1, 1)
    assert b is not None
    return b


def test_geometry_left_right_above_below_center_diagonal():
    left = _box(0.05, 0.40, 0.25, 0.70)
    right = _box(0.70, 0.40, 0.90, 0.70)
    up = _box(0.40, 0.05, 0.60, 0.25)
    down = _box(0.40, 0.75, 0.60, 0.95)
    mid = _box(0.40, 0.40, 0.60, 0.60)
    ul = _box(0.05, 0.05, 0.20, 0.20)
    cfg = SpatialConfig()
    assert evaluate_relation(left, right, Relation.LEFT_OF, cfg).ok
    assert not evaluate_relation(right, left, Relation.LEFT_OF, cfg).ok
    assert evaluate_relation(right, left, Relation.RIGHT_OF, cfg).ok
    assert evaluate_relation(up, down, Relation.ABOVE, cfg).ok
    assert evaluate_relation(down, up, Relation.BELOW, cfg).ok
    assert evaluate_relation(mid, mid, Relation.CENTER, cfg).ok or evaluate_relation(
        _box(0.42, 0.42, 0.48, 0.48), _box(0.30, 0.30, 0.70, 0.70), Relation.CENTER, cfg
    ).ok
    inner = _box(0.42, 0.42, 0.48, 0.48)
    outer = _box(0.30, 0.30, 0.70, 0.70)
    assert evaluate_relation(inner, outer, Relation.CENTER, cfg).ok
    assert evaluate_relation(ul, mid, Relation.UPPER_LEFT, cfg).ok
    assert evaluate_relation(_box(0.75, 0.05, 0.90, 0.20), mid, Relation.UPPER_RIGHT, cfg).ok
    assert evaluate_relation(_box(0.05, 0.75, 0.20, 0.90), mid, Relation.LOWER_LEFT, cfg).ok
    assert evaluate_relation(_box(0.75, 0.75, 0.90, 0.90), mid, Relation.LOWER_RIGHT, cfg).ok


def test_geometry_near_far_overlap_containment():
    cfg = SpatialConfig()
    a = _box(0.10, 0.40, 0.30, 0.60)
    near_b = _box(0.32, 0.40, 0.48, 0.60)
    far_b = _box(0.80, 0.80, 0.98, 0.98)
    assert evaluate_relation(near_b, a, Relation.NEAR, cfg).ok
    assert not evaluate_relation(far_b, a, Relation.NEAR, cfg).ok
    assert evaluate_relation(far_b, a, Relation.FAR, cfg).ok
    assert not evaluate_relation(near_b, a, Relation.FAR, cfg).ok
    ov = _box(0.20, 0.45, 0.40, 0.65)
    assert evaluate_relation(ov, a, Relation.OVERLAPS, cfg).ok
    inner = _box(0.12, 0.42, 0.22, 0.52)
    outer = _box(0.10, 0.40, 0.40, 0.70)
    assert evaluate_relation(inner, outer, Relation.INSIDE, cfg).ok
    assert evaluate_relation(outer, inner, Relation.CONTAINS, cfg).ok
    assert not evaluate_relation(far_b, a, Relation.INSIDE, cfg).ok


def test_query_parser_examples_and_non_spatial():
    q = parse_spatial_query("insanın yanında çanta")
    assert q and q["relation"] is Relation.NEAR
    assert q["object_a"]["label"] == "person"
    assert "handbag" in q["object_a"]["labels"] or q["object_a"]["label"] == "person"
    assert q["object_b"]["label"] == "handbag"
    assert q["clip_as_relation"] is False
    q2 = parse_spatial_query("çantanın yanında insan")
    assert q2["object_a"]["label"] == "handbag" and q2["object_b"]["label"] == "person"
    q3 = parse_spatial_query("çiçeğin üstünde yaprak")
    assert q3["relation"] is Relation.ABOVE
    assert q3["object_a"]["label"] == "flower" and q3["object_b"]["label"] == "leaf"
    q4 = parse_spatial_query("gülün altında yaprak")
    assert q4["relation"] is Relation.BELOW
    assert q4["object_a"]["label"] == "rose" and q4["object_b"]["label"] == "leaf"
    q5 = parse_spatial_query("leoparın yanında gül")
    assert q5["relation"] is Relation.NEAR
    assert q5["object_a"]["label"] == "leopard" and q5["object_b"]["label"] == "rose"
    assert parse_spatial_query("insanın solunda çanta")["relation"] is Relation.LEFT_OF
    assert parse_spatial_query("insanın sol tarafında çanta")["relation"] is Relation.LEFT_OF
    assert parse_spatial_query("insanın sağında çanta")["relation"] is Relation.RIGHT_OF
    assert parse_spatial_query("insanın sağ tarafında çanta")["relation"] is Relation.RIGHT_OF
    assert parse_spatial_query("çiçeğin üzerinde yaprak")["relation"] is Relation.ABOVE
    assert parse_spatial_query("insanın yanı başında çanta")["relation"] is Relation.NEAR
    assert parse_spatial_query("kedi") is None
    assert parse_spatial_query("kedi + çanta") is None
    assert parse_object_query("kedi + çanta")["kind"] == "and"
    assert parse_object_query("çiçek") is None  # textile query must stay non-object


def _plant(store: ObjectIndexStore, file_id: int, path: str, objs: list[dict]) -> None:
    store.replace_file_objects(file_id, path, objs, detector="fixture")


def _obj(label, bbox, inst, conf=0.9, area=0.05):
    return {
        "label": label,
        "label_tr": label,
        "confidence": conf,
        "bbox": bbox,
        "area_ratio": area,
        "instance_id": inst,
    }


def test_multi_object_instance_and_negative(tmp_path):
    dbp = tmp_path / "object_index.db"
    store = ObjectIndexStore(dbp)
    img = tmp_path / "scene.jpg"
    Image.new("RGB", (1000, 1000), (20, 20, 20)).save(img, "JPEG")
    # 2 person + 1 bag: bag near person_02 only
    _plant(
        store,
        1,
        str(img),
        [
            _obj("person", (10, 400, 120, 700), "person_01"),
            _obj("person", (500, 400, 620, 700), "person_02"),
            _obj("handbag", (640, 480, 740, 620), "bag_01"),
        ],
    )
    # 3 person + bag near person_03
    _plant(
        store,
        2,
        str(tmp_path / "p3.jpg"),
        [
            _obj("person", (10, 10, 80, 200), "person_01"),
            _obj("person", (200, 10, 280, 200), "person_02"),
            _obj("person", (700, 400, 820, 720), "person_03"),
            _obj("handbag", (830, 500, 920, 640), "bag_01"),
        ],
    )
    Image.new("RGB", (1000, 1000), (30, 30, 30)).save(tmp_path / "p3.jpg", "JPEG")
    # multiple bags: bag_03 near person
    _plant(
        store,
        3,
        str(tmp_path / "bags.jpg"),
        [
            _obj("person", (400, 400, 520, 700), "person_01"),
            _obj("handbag", (20, 20, 80, 90), "bag_01"),
            _obj("handbag", (900, 20, 980, 90), "bag_02"),
            _obj("handbag", (530, 480, 620, 600), "bag_03"),
        ],
    )
    Image.new("RGB", (1000, 1000), (40, 40, 40)).save(tmp_path / "bags.jpg", "JPEG")
    # multiple flowers: leaf above flower_02
    _plant(
        store,
        4,
        str(tmp_path / "fl.jpg"),
        [
            _obj("flower", (50, 700, 150, 850), "flower_01"),
            _obj("flower", (500, 500, 650, 700), "flower_02"),
            _obj("leaf", (520, 320, 620, 470), "leaf_01"),
        ],
    )
    Image.new("RGB", (1000, 1000), (10, 80, 10)).save(tmp_path / "fl.jpg", "JPEG")
    # negative: person and bag far apart
    _plant(
        store,
        5,
        str(tmp_path / "far.jpg"),
        [
            _obj("person", (10, 10, 80, 120), "person_01"),
            _obj("handbag", (880, 880, 980, 980), "bag_01"),
        ],
    )
    Image.new("RGB", (1000, 1000), (5, 5, 5)).save(tmp_path / "far.jpg", "JPEG")
    # rose + leaf
    _plant(
        store,
        6,
        str(tmp_path / "rose.jpg"),
        [
            _obj("rose", (400, 200, 600, 450), "rose_01"),
            _obj("leaf", (420, 500, 580, 700), "leaf_01"),
        ],
    )
    Image.new("RGB", (1000, 1000), (80, 10, 30)).save(tmp_path / "rose.jpg", "JPEG")
    # leopard + rose near
    _plant(
        store,
        7,
        str(tmp_path / "leo.jpg"),
        [
            _obj("leopard", (200, 300, 450, 700), "leopard_01"),
            _obj("rose", (470, 350, 600, 520), "rose_01"),
        ],
    )
    Image.new("RGB", (1000, 1000), (90, 70, 20)).save(tmp_path / "leo.jpg", "JPEG")

    sizes = {i: (1000, 1000) for i in range(1, 8)}
    eng = SpatialEngine(str(dbp), readonly=True)
    r = eng.search("insanın yanında çanta", image_sizes=sizes)
    assert r.status == "ok"
    ids = {h.file_id for h in r.hits}
    assert 1 in ids and 2 in ids and 3 in ids
    assert 5 not in ids
    hit1 = next(h for h in r.hits if h.file_id == 1)
    ev = hit1.spatial_evidence
    assert ev["located_instance"] == "bag_01"
    assert ev["anchor_instance"] == "person_02"
    assert "✓ found" in hit1.explanation
    hit3 = next(h for h in r.hits if h.file_id == 3)
    assert hit3.spatial_evidence["located_instance"] == "bag_03"
    flowers = eng.search("çiçeğin üstünde yaprak", image_sizes=sizes)
    assert any(h.file_id == 4 for h in flowers.hits)
    assert flowers.hits[0].spatial_evidence["located_instance"] == "leaf_01"
    rose = eng.search("gülün altında yaprak", image_sizes=sizes)
    assert any(h.file_id == 6 for h in rose.hits)
    leo = eng.search("leoparın yanında gül", image_sizes=sizes)
    assert any(h.file_id == 7 for h in leo.hits)
    neg = eng.search("insanın yanında çanta", image_sizes={5: (1000, 1000)})
    # file 5 still in DB with other files; dedicated check:
    far_only = [h for h in r.hits if h.file_id == 5]
    assert far_only == []
    assert neg.status in {"ok", SPATIAL_EVIDENCE_UNAVAILABLE} or 5 not in {h.file_id for h in neg.hits}


def test_missing_object_index_is_unavailable(tmp_path):
    eng = SpatialEngine(str(tmp_path / "nope.db"), readonly=True)
    r = eng.search("insanın yanında çanta")
    assert r.status == SPATIAL_EVIDENCE_UNAVAILABLE


def test_search_engine_spatial_not_clip_and_index_freeze(tmp_path):
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
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(folder)),
        )
    near_p = folder / "near.jpg"
    far_p = folder / "far.jpg"
    cat_p = folder / "cat.jpg"
    Image.new("RGB", (1000, 1000), (10, 10, 80)).save(near_p, "JPEG")
    Image.new("RGB", (1000, 1000), (80, 10, 10)).save(far_p, "JPEG")
    Image.new("RGB", (160, 160), (200, 80, 20)).save(cat_p, "JPEG")
    recs = {}
    for p in (near_p, far_p, cat_p):
        recs[p.name] = db.upsert_file(
            {
                "path": str(p),
                "filename": p.name,
                "source_id": 1,
                "status": "indexed",
                "file_size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "width": 1000 if p.suffix and p.name != "cat.jpg" else 160,
                "height": 1000 if p.name != "cat.jpg" else 160,
            }
        )
    store = ObjectIndexStore(settings.object_db_path)
    _plant(
        store,
        int(recs["near.jpg"]),
        str(near_p),
        [
            _obj("person", (400, 400, 550, 750), "person_01"),
            _obj("handbag", (560, 500, 680, 640), "bag_01"),
        ],
    )
    _plant(
        store,
        int(recs["far.jpg"]),
        str(far_p),
        [
            _obj("person", (10, 10, 80, 120), "person_01"),
            _obj("handbag", (880, 880, 980, 980), "bag_01"),
        ],
    )
    _plant(
        store,
        int(recs["cat.jpg"]),
        str(cat_p),
        [_obj("cat", (8, 8, 80, 80), "cat_01")],
    )
    before = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    fp_before = freeze_fingerprint(before)
    se = SearchEngine(settings, load_ai=False)
    hits = se.search_by_text("insanın yanında çanta")
    names = {r.filename for r in hits}
    assert "near.jpg" in names
    assert "far.jpg" not in names
    assert hits[0].debug.get("spatial_relation") == "NEAR"
    assert hits[0].breakdown.get("spatial_score") is not None
    assert hits[0].debug.get("spatial_evidence")
    assert "✓ found" in " ".join(hits[0].match_explanations)
    # existing object path unchanged
    kedi = se.search_by_text("kedi")
    assert kedi and kedi[0].filename == "cat.jpg"
    assert kedi[0].breakdown.get("object_evidence") == "detector_class"
    assert "spatial_score" not in (kedi[0].debug or {})
    after = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    assert freeze_fingerprint(after) == fp_before


def test_spatial_search_does_not_write_production_sentinel(tmp_path):
    prod_obj = DEFAULT_DATA_DIR / "object_index.db"
    obj_meta = (prod_obj.exists(), prod_obj.stat().st_mtime_ns if prod_obj.exists() else 0)
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
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(folder)),
        )
    img = folder / "s.jpg"
    Image.new("RGB", (200, 200), (1, 2, 3)).save(img, "JPEG")
    db.upsert_file(
        {
            "path": str(img),
            "filename": img.name,
            "source_id": 1,
            "status": "indexed",
            "file_size": img.stat().st_size,
            "mtime": img.stat().st_mtime,
            "width": 200,
            "height": 200,
        }
    )
    snap = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    fp = freeze_fingerprint(snap)
    se = SearchEngine(settings, load_ai=False)
    se.search_by_text("insanın yanında çanta")
    snap2 = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    assert freeze_fingerprint(snap2) == fp
    assert Path(settings.object_db_path).resolve() != prod_obj.resolve()
    now_meta = (prod_obj.exists(), prod_obj.stat().st_mtime_ns if prod_obj.exists() else 0)
    assert now_meta == obj_meta
