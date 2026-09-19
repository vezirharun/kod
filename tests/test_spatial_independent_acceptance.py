"""Spatial V2 independent acceptance. Labels are not engine-derived.

INDEX FROZEN. tmp object_index only. No production object_index.db / patterns.db / FAISS.
Does not walk the 116K archive. CLIP is not used as a relation detector.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from core.db import Database
from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.object_index import ObjectIndexStore
from core.search_engine import SearchEngine
from core.settings import DEFAULT_DATA_DIR, AppSettings
from core.spatial.spatial_engine import SPATIAL_EVIDENCE_UNAVAILABLE, SpatialEngine

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "spatial_v2"
LABELS_PATH = FIXTURE_DIR / "labels.json"
REPORT = Path(__file__).resolve().parents[1] / "data" / "reports" / "spatial_independent_v2.json"

# COCO-80 has none of these textile-motif classes (potted plant ≠ flower/leaf/rose/leopard).
COCO80_MOTIF = ("flower", "leaf", "rose", "leopard")
_COLORS = {
    "flower": (220, 60, 120),
    "leaf": (40, 160, 60),
    "rose": (190, 20, 50),
    "leopard": (210, 160, 40),
    "person": (70, 90, 200),
    "handbag": (160, 90, 30),
}


def _load_labels() -> dict:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def _obj(item: dict) -> dict:
    box = [int(x) for x in item["bbox"]]
    x1, y1, x2, y2 = box
    return {
        "label": item["label"],
        "label_tr": item["label"],
        "confidence": 0.95,
        "bbox": box,
        "area_ratio": max(0, x2 - x1) * max(0, y2 - y1) / (400 * 400),
        "instance_id": item["instance_id"],
    }


def _render(case: dict, dest: Path, size: tuple[int, int]) -> None:
    im = Image.new("RGB", size, (24, 24, 28))
    dr = ImageDraw.Draw(im)
    for obj in case["objects"]:
        box = [int(x) for x in obj["bbox"]]
        dr.rectangle(box, fill=_COLORS.get(obj["label"], (180, 180, 180)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "PNG")


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r}


def test_spatial_independent_fixture_acceptance(tmp_path):
    prod_obj = DEFAULT_DATA_DIR / "object_index.db"
    obj_meta = (prod_obj.exists(), prod_obj.stat().st_mtime_ns if prod_obj.exists() else 0)

    spec = _load_labels()
    assert spec["authoring"]["circular"] is False
    w, h = spec["image_size"]
    cases = spec["cases"]

    store = ObjectIndexStore(tmp_path / "object_index.db")
    sizes: dict[int, tuple[int, int]] = {}
    by_id: dict[str, int] = {}
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i, case in enumerate(cases, start=1):
        dest = img_dir / case["file"]
        _render(case, dest, (w, h))
        store.replace_file_objects(
            i, str(dest), [_obj(o) for o in case["objects"]], detector="manual_independent_boxes"
        )
        sizes[i] = (w, h)
        by_id[case["id"]] = i

    eng = SpatialEngine(str(tmp_path / "object_index.db"), readonly=True)

    tp = fp = fn = 0
    rel_ok = rel_n = 0
    per_scenario: dict[str, int] = defaultdict(int)
    rows = []
    for case in cases:
        per_scenario[case["scenario"]] += 1
        fid = by_id[case["id"]]
        res = eng.search(case["query"], image_sizes=sizes)
        assert res.status != SPATIAL_EVIDENCE_UNAVAILABLE or case.get("expect_hit") is False
        hit_ids = {h.file_id for h in res.hits}
        predicted = fid in hit_ids
        expected = bool(case["expect_hit"])
        if expected and predicted:
            tp += 1
            rel_ok += 1
            rel_n += 1
            hit = next(h for h in res.hits if h.file_id == fid)
            assert hit.spatial_evidence.get("clip_as_relation") is False
            want_a = case.get("expected_anchor_instance")
            want_b = case.get("expected_located_instance")
            if want_a:
                assert hit.spatial_evidence.get("anchor_instance") == want_a
            if want_b:
                assert hit.spatial_evidence.get("located_instance") == want_b
        elif expected and not predicted:
            fn += 1
            rel_n += 1
        elif (not expected) and predicted:
            fp += 1
            rel_n += 1
        else:
            rel_ok += 1
            rel_n += 1
        rows.append(
            {
                "id": case["id"],
                "scenario": case["scenario"],
                "query": case["query"],
                "expected_relation": case["expected_relation"],
                "expect_hit": expected,
                "predicted_hit": predicted,
                "independent_label": case.get("independent_label"),
            }
        )

    far_fid = by_id["person_bag_far_not_near"]
    near_res = eng.search("insanın yanında çanta", image_sizes=sizes)
    assert far_fid not in {h.file_id for h in near_res.hits}

    metrics = _prf(tp, fp, fn)
    rel_acc = rel_ok / rel_n if rel_n else 0.0

    empty = ObjectIndexStore(tmp_path / "empty_objects.db")
    empty.replace_file_objects(
        1,
        str(img_dir / cases[0]["file"]),
        [_obj({"label": "person", "bbox" : [10, 10, 80, 80], "instance_id": "p1"})],
        detector="coco80_sim",
    )
    det_eng = SpatialEngine(str(tmp_path / "empty_objects.db"), readonly=True)
    motif_queries = {
        "flower+leaf": "çiçeğin üstünde yaprak",
        "rose+leaf": "gülün altında yaprak",
        "leopard+flower": "leoparın yanında gül",
    }
    motif_detector = {}
    for name, q in motif_queries.items():
        r = det_eng.search(q, image_sizes={1: (w, h)})
        motif_detector[name] = {
            "status": r.status,
            "evaluable": r.status != SPATIAL_EVIDENCE_UNAVAILABLE,
            "reason": SPATIAL_EVIDENCE_UNAVAILABLE
            if r.status == SPATIAL_EVIDENCE_UNAVAILABLE
            else r.explanation,
            "precision": 0.0,
            "recall": 0.0,
            "fp": 0,
            "fn": 0,
        }
    textile_detector_unavailable = all(not v["evaluable"] for v in motif_detector.values())
    proven = False

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "authoring": spec["authoring"],
                "image_counts_per_scenario": dict(per_scenario),
                "n_images": len(cases),
                "manual_box_geometry": {**metrics, "relation_accuracy": rel_acc},
                "cases": rows,
                "coco80_motif_classes_present": {c: False for c in COCO80_MOTIF},
                "detector_motif": motif_detector,
                "textile_motifs_detector_unavailable": textile_detector_unavailable,
                "spatial_quality_proven": proven,
                "clip_as_relation": False,
                "index_frozen": True,
                "production_object_index_exists": prod_obj.exists(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert Path(store.path).resolve() != prod_obj.resolve()
    now_meta = (prod_obj.exists(), prod_obj.stat().st_mtime_ns if prod_obj.exists() else 0)
    assert now_meta == obj_meta
    assert textile_detector_unavailable
    assert metrics["precision"] >= 0.99 and metrics["recall"] >= 0.99
    assert rel_acc >= 0.99
    assert proven is False


def test_independent_search_fingerprint_isolated_sentinel(tmp_path):
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
    spec = _load_labels()
    w, h = spec["image_size"]
    case = next(c for c in spec["cases"] if c["id"] == "person_bag_near")
    img = folder / case["file"]
    _render(case, img, (w, h))
    rec = db.upsert_file(
        {
            "path": str(img),
            "filename": img.name,
            "source_id": 1,
            "status": "indexed",
            "file_size": img.stat().st_size,
            "mtime": img.stat().st_mtime,
            "width": w,
            "height": h,
        }
    )
    store = ObjectIndexStore(settings.object_db_path)
    store.replace_file_objects(int(rec), str(img), [_obj(o) for o in case["objects"]], detector="fixture")
    before = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    fp_before = freeze_fingerprint(before)
    se = SearchEngine(settings, load_ai=False)
    hits = se.search_by_text("insanın yanında çanta")
    assert hits and hits[0].filename == case["file"]
    after = snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )
    assert freeze_fingerprint(after) == fp_before
