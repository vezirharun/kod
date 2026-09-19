"""Index-time OVD: no query-time GDino, RT-DETR rows stay, freeze honored."""
from __future__ import annotations

import inspect
from PIL import Image

from core.index_freeze import INDEX_FROZEN, allow_index_writes, search_session
from core.object_index import ObjectIndexStore
from core.ovd_index import (
    FLORAL_PENALTY,
    INDEX_VOCAB,
    candidate_priority,
    dry_run_queue,
    index_open_vocab_image,
    ovd_work_allowed,
    query_ovd_labels,
)
from core.search_engine import SearchEngine
from core.settings import AppSettings


class _FakeOVD:
    name = "grounding_dino"
    version = "tiny-v1"
    calls = 0
    prompts: list[list[str]] = []

    def detect(self, image, prompts, threshold):
        _FakeOVD.calls += 1
        _FakeOVD.prompts.append(list(prompts))
        w, h = image.size
        return [{
            "label": "crow",
            "confidence": 0.91,
            "xyxy": [w * 0.1, h * 0.1, w * 0.4, h * 0.4],
        }]


def test_production_flag_and_no_query_time_gdino():
    assert AppSettings().open_vocab_object_enabled is False
    src = inspect.getsource(SearchEngine.search_by_text)
    assert "apply_open_vocab_layer" not in src
    assert "get_grounding_dino_backend" not in src
    assert "get_owlv2_backend" not in src
    assert "index_open_vocab_image" not in src


def test_query_labels_jewelry_children():
    labs = query_ovd_labels("takı")
    assert "jewelry" in labs
    assert "necklace" in labs
    assert "brooch" in labs


def test_prefilter_does_not_eliminate_floral():
    assert FLORAL_PENALTY == 0.0
    floral = {"id": 1, "pattern_family": "floral"}
    plain = {"id": 2, "pattern_family": "unknown"}
    q = dry_run_queue([floral, plain], limit=10)
    assert len(q) == 2
    assert all(not r["eliminated"] for r in q)
    assert candidate_priority(floral) == candidate_priority(plain)


def test_frozen_skips_gdino_and_writes():
    """B: search_session blocks OVD; A (INDEX_FROZEN) alone does not."""
    assert INDEX_FROZEN is True
    with search_session():
        assert ovd_work_allowed() is False
    assert ovd_work_allowed() is True


def test_rtdetr_preserved_when_ovd_writes(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (80, 80), (10, 10, 10)).save(img)
    db = tmp_path / "obj.db"
    store = ObjectIndexStore(db)
    store.replace_file_objects(
        7,
        str(img),
        [{"label": "bird", "confidence": 0.92, "bbox": (1, 2, 10, 20), "area_ratio": 0.1}],
        detector="ultralytics_rtdetr_l_coco",
    )
    n_before = store.rtdetr_count(7)
    assert n_before == 1
    _FakeOVD.calls = 0

    def _pass_clip(im, xyxy, lab):
        return {
            "ok": True,
            "clip_target": 0.9,
            "clip_rival": 0.1,
            "clip_margin": 0.8,
            "clip_textile": 0.0,
        }

    with allow_index_writes():
        out = index_open_vocab_image(
            file_id=7,
            image_path=str(img),
            store=store,
            backend=_FakeOVD(),
            clip_fn=_pass_clip,
        )
    assert out["ok"] is True
    assert store.rtdetr_count(7) == n_before
    inst = store.instances_for_file(7)
    dets = {str(i.get("detector") or "") for i in inst}
    assert "ultralytics_rtdetr_l_coco" in dets or any(
        str(i.get("label")) == "bird" and str(i.get("detector") or "") != "grounding_dino"
        for i in inst
    )
    ovd = [i for i in inst if i.get("detector") == "owlv2"]
    assert ovd
    assert ovd[0]["evidence"] == "OPEN_VOCAB_OBJECT"
    bbox = ovd[0]["bbox"]
    assert all(0.0 <= float(x) <= 1.0 for x in bbox)
    assert store.search_label(["bird"]) == [7]
    assert store.search_label(["crow"]) == []  # GDino excluded from EXACT lookup
    assert store.search_ovd_labels(["crow"]) == [7]


def test_resume_skips_second_detect(tmp_path):
    img = tmp_path / "b.jpg"
    Image.new("RGB", (60, 60), (20, 20, 20)).save(img)
    store = ObjectIndexStore(tmp_path / "obj2.db")
    fake = _FakeOVD()
    _FakeOVD.calls = 0
    with allow_index_writes():
        index_open_vocab_image(file_id=3, image_path=str(img), store=store, backend=fake)
        n1 = _FakeOVD.calls
        index_open_vocab_image(file_id=3, image_path=str(img), store=store, backend=fake)
        n2 = _FakeOVD.calls
    assert n1 == 1
    assert n2 == 1
    assert fake.prompts[0] == list(INDEX_VOCAB)


def test_single_forward_closed_vocab(tmp_path):
    img = tmp_path / "c.jpg"
    Image.new("RGB", (40, 40), (30, 30, 30)).save(img)
    store = ObjectIndexStore(tmp_path / "obj3.db")
    fake = _FakeOVD()
    _FakeOVD.calls = 0
    with allow_index_writes():
        index_open_vocab_image(file_id=1, image_path=str(img), store=store, backend=fake)
    assert _FakeOVD.calls == 1
    assert "crow" in fake.prompts[0]
    assert "butterfly" in fake.prompts[0]
    assert "necklace" not in fake.prompts[0]


def test_textile_file_skips_owl(tmp_path):
    img = tmp_path / "d.jpg"
    Image.new("RGB", (40, 40), (40, 40, 40)).save(img)
    store = ObjectIndexStore(tmp_path / "obj4.db")
    fake = _FakeOVD()
    _FakeOVD.calls = 0
    with allow_index_writes():
        out = index_open_vocab_image(
            file_id=2,
            image_path=str(img),
            store=store,
            backend=fake,
            pattern_family="floral",
        )
    assert out["skipped"] == "textile_pattern_skip_owl"
    assert _FakeOVD.calls == 0


def test_owl_gate_skips_exact_car_runs_bird_parent():
    from core.ovd_index import owl_enrichment_decision, owl_queue_rank

    skip = owl_enrichment_decision(rtdetr_labels=["car", "person"])
    assert skip["decision"] == "SKIP"
    assert skip["reason"] == "exact_object_sufficient"
    run = owl_enrichment_decision(rtdetr_labels=["bird"])
    assert run["decision"] == "RUN"
    assert run["reason"] == "open_vocab_gap"
    assert owl_queue_rank(run, ["bird"]) == 1
    pat = owl_enrichment_decision(
        pattern_family="",
        texture_map={"pattern_dna": {"animal_print_type": "leopard"}},
        rtdetr_labels=["bird"],
    )
    assert pat["decision"] == "RUN"
    assert pat["reason"] == "open_vocab_gap"
    leopard = owl_enrichment_decision(
        texture_map={"animal_print_type": "leopard", "pattern_family": "animal_print"},
    )
    assert leopard["decision"] == "SKIP"
    crow = owl_enrichment_decision(filename="pngtree-crow-image.png", rtdetr_labels=["car"])
    assert crow["decision"] == "RUN"


def test_owl_gate_reads_dna_family_but_keeps_open_hint():
    from core.ovd_index import owl_enrichment_decision, owl_queue_rank

    floral = owl_enrichment_decision(
        pattern_family="",
        texture_map={
            "pattern_family": "floral",
            "pattern_dna": {"family": "floral", "main_family": "floral"},
            "semantic_tags": {"family": "floral"},
        },
    )
    assert floral["decision"] == "SKIP"
    assert floral["reason"] == "textile_pattern_skip_owl"
    assert owl_queue_rank(floral, []) == 0
    both = owl_enrichment_decision(
        filename="butterfly-print.jpg",
        texture_map={"pattern_dna": {"family": "floral"}},
    )
    assert both["decision"] == "RUN"
    assert both["reason"] == "open_vocab_gap"
    weak = owl_enrichment_decision(texture_map={"pattern_dna": {"family": "plaid_check"}})
    assert weak["decision"] == "RUN"
    assert weak["reason"] == "weak_evidence"
    assert owl_queue_rank(weak, []) == 4
    uncertain = owl_enrichment_decision(rtdetr_labels=["teddy bear"])
    assert uncertain["reason"] == "weak_evidence"
    assert owl_queue_rank(uncertain, ["teddy bear"]) == 3


def test_owl_gate_car_skips_detect(tmp_path):
    img = tmp_path / "car.jpg"
    Image.new("RGB", (40, 40), (50, 50, 50)).save(img)
    store = ObjectIndexStore(tmp_path / "obj5.db")
    store.replace_file_objects(
        9,
        str(img),
        [{"label": "car", "confidence": 0.9, "bbox": (1, 2, 10, 20), "area_ratio": 0.2}],
    )
    fake = _FakeOVD()
    _FakeOVD.calls = 0
    with allow_index_writes():
        out = index_open_vocab_image(
            file_id=9, image_path=str(img), store=store, backend=fake
        )
    assert out["owl_decision"] == "SKIP"
    assert out["skipped"] == "exact_object_sufficient"
    assert _FakeOVD.calls == 0


def test_stamp_from_index_not_exact(tmp_path):
    from types import SimpleNamespace
    from core.search_evidence_gate import apply_search_evidence_gate, has_hard_evidence_for_query

    row = SimpleNamespace(
        score=0.62,
        filename="x.jpg",
        path="",
        pattern_family="",
        breakdown={},
        debug={
            "open_vocab_object": True,
            "ovd_source": "object_index",
            "evidence_type": "OPEN_VOCAB_OBJECT",
            "ovd_confidence": 0.91,
        },
        file_id=1,
    )
    assert has_hard_evidence_for_query(row, "karga") is False
    out, _ = apply_search_evidence_gate([row], "karga")
    assert out
    assert out[0].debug.get("evidence_type") == "OPEN_VOCAB_OBJECT"
    assert out[0].debug.get("object_index_hit") in (None, False)
