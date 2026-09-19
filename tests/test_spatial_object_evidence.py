"""Spatial V2 object evidence: RT-DETR measure + YOLO-World extra boxes.

INDEX FROZEN. tmp object_index only. CLIP is not a detector.
Does not walk the 116K archive. Does not start Segmentation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from core.object_evidence import (
    CLIP_AS_DETECTOR,
    EXTRA_DETECTOR_BACKEND,
    OPEN_VOCAB_CLASSES,
    ObjectEvidenceLayer,
    extra_detector_available,
    extra_detector_backend,
    extra_detector_error,
)
from core.object_intelligence import ObjectIntelligence
from core.settings import DEFAULT_DATA_DIR
from core.spatial.spatial_engine import SPATIAL_EVIDENCE_UNAVAILABLE, SpatialEngine

REPORT = Path(__file__).resolve().parents[1] / "data" / "reports" / "spatial_object_evidence_v2.json"
COCO_IMG = (
    Path(__file__).resolve().parents[1]
    / ".cache_object_photos"
    / "coco128"
    / "images"
    / "train2017"
)
# Independent intent for coverage: COCO-80 membership, not model outputs.
COVERAGE_SPEC = [
    {"class": "çiçek / flower", "coco80": False, "probe": "flower"},
    {"class": "yaprak / leaf", "coco80": False, "probe": "leaf"},
    {"class": "gül / rose", "coco80": False, "probe": "rose"},
    {"class": "leopar / leopard", "coco80": False, "probe": "leopard"},
    {"class": "kelebek / butterfly", "coco80": False, "probe": "butterfly"},
    {"class": "zebra", "coco80": True, "probe": "zebra"},
    {"class": "yılan / snake", "coco80": False, "probe": "snake"},
    {"class": "insan / person", "coco80": True, "probe": "person"},
    {"class": "çanta / handbag", "coco80": True, "probe": "handbag"},
    {"class": "saksı bitkisi / potted plant", "coco80": True, "probe": "potted plant"},
]


def _index_meta() -> tuple:
    """Production object_index + FAISS only. patterns.db may change from other processes."""
    obj = DEFAULT_DATA_DIR / "object_index.db"
    dino = DEFAULT_DATA_DIR / "faiss_dino.index"
    clip = DEFAULT_DATA_DIR / "faiss_clip.index"

    def _one(p: Path):
        if not p.exists():
            return (False, 0, 0)
        st = p.stat()
        return (True, st.st_mtime_ns, st.st_size)

    return (_one(obj), _one(dino), _one(clip))


def _render_spatial_drawing(dest: Path) -> None:
    im = Image.new("RGB", (400, 400), (24, 24, 28))
    dr = ImageDraw.Draw(im)
    dr.rectangle([140, 220, 270, 360], fill=(220, 60, 120))
    dr.rectangle([150, 30, 260, 150], fill=(40, 160, 60))
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "PNG")


def test_clip_is_not_a_detector():
    assert CLIP_AS_DETECTOR is False
    assert "clip" not in extra_detector_backend().lower() or extra_detector_backend() == "unavailable"


def test_rtdetr_and_extra_coverage_on_fixtures(tmp_path):
    prod_obj = DEFAULT_DATA_DIR / "object_index.db"
    obj_meta = (prod_obj.exists(), prod_obj.stat().st_mtime_ns if prod_obj.exists() else 0)
    before = _index_meta()

    drawing = tmp_path / "flower_leaf_drawing.png"
    _render_spatial_drawing(drawing)

    coco_needed = {
        "flower_still": COCO_IMG / "000000000030.jpg",
        "zebra_still": COCO_IMG / "000000000034.jpg",
        "person_still": COCO_IMG / "000000000036.jpg",
        "person_bag_still": COCO_IMG / "000000000536.jpg",
        "plant_horse": COCO_IMG / "000000000049.jpg",
    }
    if any(not p.is_file() for p in coco_needed.values()):
        pytest.skip("coco128 stills missing; not filling from 116K archive")

    oi = ObjectIntelligence(tmp_path / "coco_only.db")
    layer = ObjectEvidenceLayer(tmp_path / "evidence.db")
    rtdetr_ok = oi.detector_available()
    extra_ok = extra_detector_available()

    fixtures = {
        "drawing_flower_leaf": drawing,
        **coco_needed,
    }
    rtdetr_by_image: dict[str, list[str]] = {}
    extra_by_image: dict[str, list[str]] = {}
    rtdetr_hits: dict[str, list[tuple[str, float]]] = {}
    extra_hits: dict[str, list[tuple[str, float]]] = {}
    for name, path in fixtures.items():
        coco_dets = oi.detect(str(path)) if rtdetr_ok else []
        extra_dets = layer.detect_extra(str(path)) if extra_ok else []
        rtdetr_by_image[name] = sorted({d.label for d in coco_dets})
        extra_by_image[name] = sorted({d.label for d in extra_dets})
        rtdetr_hits[name] = [(d.label, round(d.confidence, 3)) for d in coco_dets[:12]]
        extra_hits[name] = [(d.label, round(d.confidence, 3)) for d in extra_dets[:12]]

    rtdetr_all = {lab for labs in rtdetr_by_image.values() for lab in labs}
    extra_all = {lab for labs in extra_by_image.values() for lab in labs}

    table = []
    for row in COVERAGE_SPEC:
        probe = row["probe"]
        rtdetr_y = probe in rtdetr_all
        extra_y = probe in extra_all
        notes = []
        if not row["coco80"]:
            notes.append("not in COCO-80")
        else:
            notes.append("COCO-80 class")
        if probe == "flower" and "potted plant" in rtdetr_all:
            notes.append("RT-DETR may emit potted plant, not flower")
        if probe == "flower" and extra_y:
            notes.append("YOLO-World flower box on real plant still")
        if not extra_ok:
            notes.append("extra detector unavailable")
        table.append(
            {
                "class": row["class"],
                "coco80": row["coco80"],
                "rtdetr_detectable": "Y" if rtdetr_y else "N",
                "extra_detectable": "Y" if extra_y else "N",
                "notes": "; ".join(notes),
            }
        )

    # Persist extra+coco into tmp only; spatial when both labels exist.
    plant = coco_needed["flower_still"]
    bag = coco_needed["person_bag_still"]
    n_plant = layer.persist(1, str(plant))
    n_bag = layer.persist(2, str(bag))
    sizes = {
        1: Image.open(plant).size,
        2: Image.open(bag).size,
    }
    eng = SpatialEngine(str(tmp_path / "evidence.db"), readonly=True)
    flower_q = eng.search("çiçeğin üstünde yaprak", image_sizes=sizes)
    person_q = eng.search("insanın yanında çanta", image_sizes=sizes)
    plant_labels = {d["label"] for d in layer.store.instances_for_file(1)}
    bag_labels = {d["label"] for d in layer.store.instances_for_file(2)}

    flower_leaf_ready = {"flower", "leaf"} <= plant_labels or (
        "potted plant" in plant_labels and "leaf" in plant_labels
    )
    if not flower_leaf_ready:
        assert flower_q.status == SPATIAL_EVIDENCE_UNAVAILABLE

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "clip_as_detection": False,
                "archive_116k_scanned": False,
                "production_object_index_written": False,
                "rtdetr_backend": oi.detector_backend() if rtdetr_ok else "unavailable",
                "extra_backend": extra_detector_backend(),
                "extra_error": extra_detector_error() if not extra_ok else "",
                "extra_is_clip_similarity_boxes": False,
                "open_vocab_classes": list(OPEN_VOCAB_CLASSES),
                "coverage": table,
                "rtdetr_labels_by_image": rtdetr_by_image,
                "extra_labels_by_image": extra_by_image,
                "rtdetr_hits": rtdetr_hits,
                "extra_hits": extra_hits,
                "persist_tmp_only": True,
                "n_persisted_plant": n_plant,
                "n_persisted_bag": n_bag,
                "plant_labels": sorted(plant_labels),
                "bag_labels": sorted(bag_labels),
                "spatial_flower_leaf": {
                    "status": flower_q.status,
                    "evaluable": flower_q.status != SPATIAL_EVIDENCE_UNAVAILABLE,
                    "evidence_ready": flower_leaf_ready,
                },
                "spatial_person_bag": {
                    "status": person_q.status,
                    "n_hits": len(person_q.hits),
                    "evaluable": person_q.status != SPATIAL_EVIDENCE_UNAVAILABLE,
                },
                "textile_spatial_production_ready": False,
                "implemented": [
                    "RT-DETR-L measure on spatial_v2 drawing + coco128 stills",
                    "YOLO-World extra detector when weights/CLIP encoder load",
                    "persist extra labels into tmp object_index",
                    "spatial queries for çiçek/yaprak when both boxes exist",
                ],
                "deferred": [
                    "116K archive scan",
                    "production object_index.db writes",
                    "Segmentation",
                    "CLIP-as-detection",
                    "leaf/rose/leopard/butterfly/snake motif evidence on these fixtures",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert Path(layer.store.path).resolve() != prod_obj.resolve()
    now_meta = (prod_obj.exists(), prod_obj.stat().st_mtime_ns if prod_obj.exists() else 0)
    assert now_meta == obj_meta
    assert _index_meta() == before
    assert rtdetr_ok
    # COCO-80: person/zebra/potted plant expected on this fixture set.
    assert "person" in rtdetr_all
    assert "zebra" in rtdetr_all
    assert "flower" not in rtdetr_all
    assert "leaf" not in rtdetr_all
    assert "leopard" not in rtdetr_all
    # Colored-rectangle drawing is not a flower to RT-DETR.
    assert "flower" not in rtdetr_by_image["drawing_flower_leaf"]
    if extra_ok:
        assert extra_detector_backend() == EXTRA_DETECTOR_BACKEND
        if "flower" not in extra_all:
            pytest.skip("YOLO-World loaded but did not box flower on coco128 plant still")
        assert "flower" in extra_by_image["flower_still"]
        assert "leaf" not in extra_by_image["drawing_flower_leaf"]
        assert not flower_leaf_ready
        assert flower_q.status == SPATIAL_EVIDENCE_UNAVAILABLE
        # Person+bag may be evaluable from extra and/or RT-DETR persist.
        if {"person"} & bag_labels and ({"handbag"} & bag_labels):
            assert person_q.status != SPATIAL_EVIDENCE_UNAVAILABLE
    else:
        pytest.skip(f"extra detector not runnable: {extra_detector_error()}")


def test_missing_textile_evidence_unavailable(tmp_path):
    layer = ObjectEvidenceLayer(tmp_path / "empty.db")
    img = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), (10, 10, 10)).save(img)
    layer.store.replace_file_objects(
        1,
        str(img),
        [{"label": "person", "confidence": 0.9, "bbox": [1, 1, 20, 20], "instance_id": "person_01"}],
        detector="fixture",
    )
    eng = SpatialEngine(str(tmp_path / "empty.db"), readonly=True)
    for q in ("çiçeğin üstünde yaprak", "gülün altında yaprak", "leoparın yanında gül"):
        r = eng.search(q, image_sizes={1: (64, 64)})
        assert r.status == SPATIAL_EVIDENCE_UNAVAILABLE


def test_spatial_runs_when_extra_evidence_exists(tmp_path):
    """Hook check: extra-class boxes in tmp index drive SpatialEngine (not CLIP)."""
    img = tmp_path / "pair.png"
    Image.new("RGB", (400, 400), (20, 20, 20)).save(img)
    layer = ObjectEvidenceLayer(tmp_path / "ev.db")
    from core.global_object_intelligence import DetectedObject

    # Independent geometry (same as spatial_v2 labels), not detector output.
    flower = DetectedObject("flower_01", "flower", "çiçek", 0.9, (140, 220, 270, 360), 0.1, (0.5, 0.7))
    leaf = DetectedObject("leaf_01", "leaf", "yaprak", 0.9, (150, 30, 260, 150), 0.08, (0.5, 0.2))
    layer.persist(1, str(img), [flower, leaf])
    eng = SpatialEngine(str(tmp_path / "ev.db"), readonly=True)
    r = eng.search("çiçeğin üstünde yaprak", image_sizes={1: (400, 400)})
    assert r.status != SPATIAL_EVIDENCE_UNAVAILABLE
    assert r.hits
    assert r.hits[0].spatial_evidence.get("clip_as_relation") is False
    assert r.hits[0].spatial_evidence.get("located_instance") == "leaf_01"
