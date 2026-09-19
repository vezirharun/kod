"""OPEN_VOCAB_OBJECT query-time layer — flag off, object-lane only."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from core.open_vocab_object import (
    OPEN_VOCAB_TYPE,
    apply_open_vocab_layer,
    cache_key,
    filter_box_area,
    nms_boxes,
    ovd_enabled,
    ovd_prompts,
    should_run_ovd,
    stamp_open_vocab,
    textile_blocks_ovd,
)
from core.search_evidence_gate import (
    ZERO_SHOT_TYPE,
    apply_search_evidence_gate,
    classify_accuracy_lane,
    has_hard_evidence_for_query,
    stamp_zero_shot,
)
from core.settings import AppSettings


class _FakeOVD:
    name = "fake_ovd"
    version = "test"

    def __init__(self, boxes=None):
        self.calls = 0
        self.prompts_seen: list[list[str]] = []
        self.boxes = boxes or [{
            "label": "crow",
            "confidence": 0.81,
            "xyxy": [10.0, 10.0, 80.0, 80.0],
        }]

    def detect(self, image, prompts, threshold):
        self.calls += 1
        self.prompts_seen.append(list(prompts))
        return list(self.boxes)


def _row(**kwargs):
    defaults = dict(
        filename="x.jpg",
        path="",
        color_family="",
        pattern_family="",
        score=0.50,
        score_percent=50.0,
        breakdown={},
        debug={},
        file_id=1,
        mtime=1.0,
        file_size=10,
        is_self_match=False,
    )
    defaults.update(kwargs)
    defaults["debug"] = dict(defaults["debug"])
    return SimpleNamespace(**defaults)


def test_production_flag_default_off():
    s = AppSettings()
    assert s.open_vocab_object_enabled is False
    assert ovd_enabled(s) is False
    assert s.ovd_candidate_limit == 12


def test_object_lane_allows_ovd_pattern_style_effect_do_not():
    assert classify_accuracy_lane("takı") == "object"
    assert classify_accuracy_lane("karga") == "object"
    assert classify_accuracy_lane("kolye") == "object"
    assert should_run_ovd("karga", enabled=True)
    assert should_run_ovd("takı", enabled=True)
    for q in (
        "leopar", "ekose", "barok", "suluboya", "etnik",
        "fırça etkisi", "çiçek", "gül", "paisley", "çizgili", "puantiye",
        "zebra deseni",
    ):
        assert classify_accuracy_lane(q) == "pattern", q
        assert should_run_ovd(q, enabled=True) is False, q


def test_pattern_lane_does_not_call_backend(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (100, 100), (20, 20, 20)).save(img)
    fake = _FakeOVD()
    row = _row(path=str(img), pattern_family="animal_print")
    apply_open_vocab_layer([row], "leopar", backend=fake)
    assert fake.calls == 0
    apply_open_vocab_layer([row], "barok", backend=fake)
    apply_open_vocab_layer([row], "suluboya", backend=fake)
    assert fake.calls == 0


def test_object_lane_calls_backend(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (120, 120), (10, 10, 10)).save(img)
    fake = _FakeOVD()
    row = _row(path=str(img))
    apply_open_vocab_layer([row], "karga", backend=fake)
    assert fake.calls == 1
    assert row.debug.get("evidence_type") == OPEN_VOCAB_TYPE
    assert row.debug.get("open_vocab_object") is True
    assert row.debug.get("ovd_source") == "grounding_dino"
    assert row.debug.get("ovd_query_concept") == "crow"


def test_exact_object_outranks_open_vocab():
    row = _row(debug={"object_index_hit": True, "evidence_type": "EXACT_OBJECT"})
    stamp_open_vocab(
        row,
        query_concept="bird",
        boxes=[{"label": "bird", "confidence": 0.88, "xyxy": [1, 1, 10, 10]}],
    )
    assert row.debug.get("evidence_type") == "EXACT_OBJECT"
    assert row.debug.get("ovd_supporting") is True
    assert has_hard_evidence_for_query(row, "kuş") is True
    ovd = _row(debug={
        "open_vocab_object": True,
        "ovd_confidence": 0.88,
        "evidence_type": OPEN_VOCAB_TYPE,
    })
    assert has_hard_evidence_for_query(ovd, "kuş") is False
    out, _ = apply_search_evidence_gate([ovd, row], "kuş")
    assert out[0].debug.get("object_index_hit") is True
    assert out[0].debug.get("evidence_type") == "EXACT_OBJECT"


def test_ovd_is_not_zero_shot():
    row = _row()
    stamp_open_vocab(
        row,
        query_concept="crow",
        boxes=[{"label": "crow", "confidence": 0.7, "xyxy": [1, 1, 8, 8]}],
    )
    stamp_zero_shot(row, {"concept": "crow"}, 0.30)
    assert row.debug.get("evidence_type") == OPEN_VOCAB_TYPE
    assert row.debug.get("zero_shot_visual") is True
    assert row.debug.get("evidence_type") != ZERO_SHOT_TYPE


def test_bbox_area_filter_and_nms():
    huge = {"label": "crow", "confidence": 0.9, "xyxy": [0.0, 0.0, 100.0, 100.0]}
    small = {"label": "crow", "confidence": 0.8, "xyxy": [10.0, 10.0, 40.0, 40.0]}
    kept = filter_box_area([huge, small], 100.0 * 100.0, 0.85)
    assert len(kept) == 1
    assert kept[0]["xyxy"] == small["xyxy"]
    a = {"label": "necklace", "confidence": 0.9, "xyxy": [0.0, 0.0, 50.0, 50.0]}
    b = {"label": "jewelry", "confidence": 0.8, "xyxy": [2.0, 2.0, 48.0, 48.0]}
    merged = nms_boxes([a, b], 0.5)
    assert len(merged) == 1
    assert merged[0]["label"] == "necklace"


def test_cache_isolation_from_zero_shot():
    k1 = cache_key(image_id="1", concept="crow", model="grounding_dino", version="tiny-v1", threshold=0.35)
    k2 = cache_key(image_id="1", concept="crow", model="openclip", version="tiny-v1", threshold=0.35)
    k3 = cache_key(image_id="1", concept="jewelry", model="grounding_dino", version="tiny-v1", threshold=0.35)
    assert k1 != k2
    assert k1 != k3


def test_textile_false_positive_rejection(tmp_path):
    img = tmp_path / "floral.jpg"
    Image.new("RGB", (80, 80), (200, 20, 40)).save(img)
    fake = _FakeOVD([{
        "label": "strawberry",
        "confidence": 0.77,
        "xyxy": [5.0, 5.0, 30.0, 30.0],
    }])
    row = _row(path=str(img), pattern_family="floral")
    apply_open_vocab_layer([row], "çilek", backend=fake)
    assert fake.calls == 1
    assert row.debug.get("ovd_rejected")
    assert row.debug.get("open_vocab_object") is False
    assert textile_blocks_ovd(row, "strawberry") is True
    croc = _row(pattern_family="animal_print")
    assert textile_blocks_ovd(croc, "crocodile") is True


def test_child_concept_preserved_on_parent_query(tmp_path):
    img = tmp_path / "j.jpg"
    Image.new("RGB", (90, 90), (30, 30, 30)).save(img)
    fake = _FakeOVD([{
        "label": "necklace",
        "confidence": 0.86,
        "xyxy": [8.0, 8.0, 40.0, 40.0],
    }])
    row = _row(path=str(img))
    parent, kids = ovd_prompts("takı")
    assert parent == "jewelry"
    assert "necklace" in kids
    apply_open_vocab_layer([row], "takı", backend=fake)
    assert row.debug.get("ovd_query_concept") == "jewelry"
    assert row.debug.get("ovd_matched_child") == "necklace"
    assert row.debug.get("ovd_parent_concept") == "jewelry"
    boxes = row.debug.get("ovd_boxes") or []
    assert boxes and boxes[0]["matched_child_concept"] == "necklace"


def test_disabled_settings_skip_without_backend(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (50, 50)).save(img)
    s = AppSettings()
    assert s.open_vocab_object_enabled is False
    row = _row(path=str(img))
    apply_open_vocab_layer([row], "karga", settings=s)
    assert row.debug.get("open_vocab_object") in (None, False)
