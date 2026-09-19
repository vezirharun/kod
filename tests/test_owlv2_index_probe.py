"""OWLv2 index-time probe: sidecar isolation, resume, retry. No production writes."""
from __future__ import annotations

import inspect
from pathlib import Path

from PIL import Image

from core.owlv2_index_probe import (
    EXACT_CONCEPTS,
    OPEN_CONCEPTS,
    ProbeStore,
    canonicalize_label,
    judge_box,
    normalize_bbox,
    prepare_owl_image,
    process_file,
    process_queue,
    scale_xyxy,
    skip_owl_reason,
)
from core.search_engine import SearchEngine
from core.settings import AppSettings


class _FakeOwl:
    loads = 0
    calls = 0
    fail_ids: set[int] = set()

    def __init__(self, boxes=None):
        _FakeOwl.loads += 1
        self.boxes = boxes or [{
            "label": "crow",
            "confidence": 0.82,
            "xyxy": [10.0, 10.0, 40.0, 40.0],
        }]

    def detect(self, image, prompts, threshold):
        _FakeOwl.calls += 1
        return list(self.boxes)


def _clip_ok(im, xyxy, concept):
    return {
        "ok": True,
        "clip_target": 0.31,
        "clip_rival": "butterfly",
        "clip_margin": 0.09,
        "clip_textile": 0.12,
    }


def _clip_reject(im, xyxy, concept):
    return {
        "ok": False,
        "clip_target": 0.22,
        "clip_rival": "floral",
        "clip_margin": -0.04,
        "clip_textile": 0.30,
    }


def test_production_flag_stays_false():
    assert AppSettings().open_vocab_object_enabled is False


def test_query_time_owlv2_call_count_is_zero():
    src = inspect.getsource(SearchEngine.search_by_text)
    full = inspect.getsource(SearchEngine)
    assert "owlv2" not in src.lower()
    assert "Owlv2" not in full
    assert "apply_open_vocab_layer" not in src


def test_bbox_normalization_unit_range():
    n = normalize_bbox([50, 25, 150, 75], 200, 100)
    assert n[0] == 0.25
    assert n[1] == 0.25
    assert n[2] == 0.75
    assert n[3] == 0.75
    assert all(0.0 <= x <= 1.0 for x in n)
    assert canonicalize_label("a crow.") == "crow"


def test_evidence_separation_crow_vs_bird():
    rec = {"role": "crow", "pattern_family": ""}
    crow = judge_box(rec, "crow", {"confidence": 0.8, "xyxy_norm": [0.1, 0.1, 0.4, 0.4]}, {
        "ok": True, "clip_target": 0.3, "clip_rival": "butterfly",
        "clip_margin": 0.09, "clip_textile": 0.1,
    })
    bird = judge_box(rec, "bird", {"confidence": 0.9, "xyxy_norm": [0.1, 0.1, 0.5, 0.5]}, None)
    assert crow["evidence_type"] == "OPEN_VOCAB_OBJECT"
    assert bird["evidence_type"] == "EXACT_OBJECT"
    assert crow["final_verdict"] != bird["final_verdict"]
    assert "bird" in EXACT_CONCEPTS
    assert "crow" in OPEN_CONCEPTS


def test_sidecar_isolation_and_rtdetr_untouched(tmp_path):
    prod = tmp_path / "object_index.db"
    prod.write_bytes(b"rtdetr-sentinel")
    sidecar = tmp_path / "owlv2_index_probe.db"
    store = ProbeStore(sidecar)
    img = tmp_path / "crow.png"
    Image.new("RGB", (80, 80), (8, 8, 8)).save(img)
    rec = {
        "file_id": 541,
        "path": str(img),
        "filename": "crow.png",
        "role": "crow",
        "pattern_family": "",
        "texture_map": {},
    }
    owl = _FakeOwl()
    process_queue(store, [rec], owl.detect, _clip_ok)
    assert prod.read_bytes() == b"rtdetr-sentinel"
    assert sidecar.is_file()
    assert store.counts()["open_vocab"] == 1


def test_pattern_dna_skip_no_owl(tmp_path):
    _FakeOwl.calls = 0
    store = ProbeStore(tmp_path / "p.db")
    img = tmp_path / "leo.png"
    Image.new("RGB", (40, 40), (20, 20, 20)).save(img)
    rec = {
        "file_id": 9,
        "path": str(img),
        "filename": "leo.png",
        "role": "leopard",
        "pattern_family": "animal_print",
        "texture_map": {"pattern_dna": {"animal_print_type": "leopard"}},
    }
    owl = _FakeOwl()
    out = process_file(store, rec, owl.detect, _clip_ok)
    assert out["status"] == "done"
    assert _FakeOwl.calls == 0
    assert store.file_row(9)["owl_ran"] == 0
    assert store.detections(9) == []


def test_duplicate_protection_and_resume(tmp_path):
    _FakeOwl.calls = 0
    store = ProbeStore(tmp_path / "q.db")
    img = tmp_path / "a.png"
    Image.new("RGB", (80, 80), (1, 1, 1)).save(img)
    rec = {
        "file_id": 1,
        "path": str(img),
        "filename": "a.png",
        "role": "crow",
        "pattern_family": "",
        "texture_map": {},
    }
    owl = _FakeOwl()
    process_queue(store, [rec], owl.detect, _clip_ok, limit=1)
    assert _FakeOwl.calls == 1
    process_queue(store, [rec], owl.detect, _clip_ok)
    assert _FakeOwl.calls == 1
    assert store.counts()["detections"] == 1
    assert store.counts()["done"] == 1


def test_interrupt_resume_skips_completed(tmp_path):
    _FakeOwl.calls = 0
    store = ProbeStore(tmp_path / "r.db")
    recs = []
    owl = _FakeOwl()
    for i in (1, 2):
        img = tmp_path / f"{i}.png"
        Image.new("RGB", (80, 80), (i, i, i)).save(img)
        recs.append({
            "file_id": i,
            "path": str(img),
            "filename": f"{i}.png",
            "role": "crow",
            "pattern_family": "",
            "texture_map": {},
        })
    process_queue(store, recs, owl.detect, _clip_ok, limit=1)
    assert store.file_row(1)["status"] == "done"
    assert store.file_row(2) is None or store.file_row(2)["status"] == "pending"
    process_queue(store, recs, owl.detect, _clip_ok)
    assert store.file_row(2)["status"] == "done"
    assert _FakeOwl.calls == 2


def test_error_retry_does_not_stop_queue(tmp_path):
    store = ProbeStore(tmp_path / "e.db")
    recs = []
    for i in (1, 2):
        img = tmp_path / f"{i}.png"
        Image.new("RGB", (80, 80), (3, 3, 3)).save(img)
        recs.append({
            "file_id": i,
            "path": str(img),
            "filename": f"{i}.png",
            "role": "crow",
            "pattern_family": "",
            "texture_map": {},
        })

    def detect(image, prompts, threshold):
        name = Path(getattr(image, "filename", "") or "")
        # second call path: process uses Image.open; fail file 1 via size check
        return [{
            "label": "crow",
            "confidence": 0.8,
            "xyxy": [10.0, 10.0, 40.0, 40.0],
        }]

    boom = {"n": 0}

    def detect_fail(image, prompts, threshold):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("injected")
        return detect(image, prompts, threshold)

    out = process_queue(store, recs, detect_fail, _clip_ok)
    assert out[0]["status"] == "error"
    assert out[1]["status"] == "done"
    retry = process_file(store, recs[0], detect, _clip_ok)
    assert retry["status"] == "done"
    assert store.file_row(1)["status"] == "done"


def test_cherry_textile_and_jewelry_not_forced():
    floral = {"role": "floral", "pattern_family": "floral", "texture_map": {}}
    cherry = judge_box(floral, "cherry", {"confidence": 0.7, "xyxy_norm": [0.2, 0.2, 0.6, 0.6]}, {
        "ok": True, "clip_target": 0.29, "clip_rival": "butterfly",
        "clip_margin": 0.05, "clip_textile": 0.33,
    })
    assert cherry["final_verdict"] == "rejected_textile"
    jew = judge_box({"role": "jewelry", "pattern_family": ""}, "necklace", {
        "confidence": 0.13, "xyxy_norm": [0.1, 0.1, 0.3, 0.3],
    }, {
        "ok": False, "clip_target": 0.25, "clip_rival": "cherry",
        "clip_margin": 0.0, "clip_textile": 0.2,
    })
    assert jew["final_verdict"] == "rejected_clip"
    assert jew["evidence_type"] != "OPEN_VOCAB_OBJECT"


def test_lips_deferred():
    d = judge_box({"role": "lips", "pattern_family": ""}, "lips", {
        "confidence": 0.5, "xyxy_norm": [0.4, 0.4, 0.6, 0.5],
    }, None)
    assert d["evidence_type"] == "LIPS_DEFERRED"


def test_skip_owl_on_exact_jewelry_lips(tmp_path):
    _FakeOwl.calls = 0
    store = ProbeStore(tmp_path / "skip.db")
    owl = _FakeOwl()
    for i, role in enumerate(("car", "jewelry", "lips"), start=20):
        img = tmp_path / f"{role}.png"
        Image.new("RGB", (80, 80), (4, 4, 4)).save(img)
        rec = {
            "file_id": i,
            "path": str(img),
            "filename": img.name,
            "role": role,
            "pattern_family": "",
            "texture_map": {},
        }
        out = process_file(store, rec, owl.detect, _clip_ok)
        assert out["status"] == "done"
        assert skip_owl_reason(role)
        assert store.file_row(i)["owl_ran"] == 0
    assert _FakeOwl.calls == 0


def test_owl_resize_scales_boxes_to_original():
    im = Image.new("RGB", (1536, 768), (0, 0, 0))
    small, sx, sy = prepare_owl_image(im, max_side=768)
    assert small.size == (768, 384)
    assert abs(sx - 2.0) < 1e-6
    assert abs(sy - 2.0) < 1e-6
    xy = scale_xyxy([10.0, 20.0, 30.0, 40.0], sx, sy)
    assert xy == [20.0, 40.0, 60.0, 80.0]
