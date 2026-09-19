"""Görsel Anlam Testi — ölçer, raporlar, kod değiştirmez."""
from __future__ import annotations

from types import SimpleNamespace

from core.production.visual_meaning_test import (
    _classify_one,
    compile_query_needles,
    format_report,
    run_visual_meaning_test,
)


def _hit(*, filename="x.jpg", labels=None, motifs=None, clip=0.1, color="", face=False, score=0.7):
    tm = {
        "global_object_intelligence": {
            "objects": [{"label": x, "canonical_name": x} for x in (labels or [])]
        },
        "semantic_tags": {"motifs": list(motifs or [])},
        "visual_concept_dna": {"objects": [{"label": x} for x in (labels or [])], "concepts": []},
    }
    return SimpleNamespace(
        file_id=1,
        path="/t/" + filename,
        filename=filename,
        score=score,
        color_family=color,
        breakdown={"clip": clip},
        debug={
            "texture_map": tm,
            "ai_score": clip,
            "semantic_score": 0.4 if motifs else 0.0,
            "color_family": color,
            "face_gender_match": face,
            "human_semantic_score": 0.4 if face else 0.0,
        },
    )


def test_compile_needles_for_rose_and_car():
    rose = compile_query_needles("gül")
    assert "rose" in rose["needles"] or "gul" in rose["needles"]
    car = compile_query_needles("kırmızı araba")
    assert "red" in car["colors"]
    assert car["object_parse"] is not None or "car" in car["needles"]


def test_grounded_object_is_correct():
    spec = compile_query_needles("araba")
    row = _classify_one(spec, __import__("core.production.visual_meaning_test", fromlist=["_result_evidence"])._result_evidence(_hit(labels=["car"])))
    assert row["verdict"] == "doğru"
    assert row["motors"]["object"] is True


def test_clip_only_is_uncertain():
    spec = compile_query_needles("gül")
    from core.production.visual_meaning_test import _result_evidence

    row = _classify_one(spec, _result_evidence(_hit(labels=[], clip=0.55)))
    assert row["verdict"] == "belirsiz"
    assert row["motors"]["openclip"] is True


def test_competing_object_is_wrong():
    spec = compile_query_needles("gül")
    from core.production.visual_meaning_test import _result_evidence

    row = _classify_one(spec, _result_evidence(_hit(labels=["car"], clip=0.1)))
    assert row["verdict"] == "yanlış"


def test_run_does_not_mutate_and_formats_report(tmp_path):
    settings = SimpleNamespace(db_path=str(tmp_path / "patterns.db"))
    fake = {
        "araba": [_hit(labels=["car"])] * 18 + [_hit(labels=["cat"])] * 2,
        "gül": [_hit(labels=["rose"], motifs=["rose"])] * 19 + [_hit(clip=0.4)] * 1,
        "kadın yüzü": [_hit(face=True, labels=["person"])] * 11 + [_hit(labels=["car"])] * 9,
    }

    def search_fn(q, limit):
        return list(fake.get(q, []))[:limit]

    out = run_visual_meaning_test(
        settings,
        queries=["araba", "gül", "kadın yüzü"],
        search_fn=search_fn,
    )
    assert out["mutated_index"] is False
    assert out["mutated_weights"] is False
    assert out["concepts"][0]["correct"] == 18
    text = format_report(out)
    assert "GÖRSEL ANLAM TESTİ" in text
    assert "ZAYIF KAVRAMLAR" in text
    assert "kadın yüzü" in text
    assert (tmp_path / "reports").is_dir()
