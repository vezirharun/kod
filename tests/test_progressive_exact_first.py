"""Exact duplicate must stay #1 from the first progressive paint."""
from __future__ import annotations

from types import SimpleNamespace

from core.search_engine import (
    clear_query_features_cache,
    query_features_cache_stats,
)
from ui.worker_threads import _filter_progressive_results, _is_progressive_exact


def _hit(*, exact=False, **extra):
    dbg = extra.pop("debug", {})
    if exact:
        dbg = {
            "protected_exact": True,
            "result_layer": "same_files",
            "exact_search_score": 0.99,
            **dbg,
        }
    row = SimpleNamespace(
        score=0.99 if exact else 0.4,
        is_self_match=bool(exact),
        same_pattern_family=False,
        same_animal_family=False,
        debug=dbg,
        breakdown={"phash": 0.99} if exact else {},
        file_id=1 if exact else 2,
    )
    for key, val in extra.items():
        setattr(row, key, val)
    return row


def test_exact_stays_visible_on_later_progressive_stages():
    exact = _hit(exact=True)
    family = _hit(
        score=0.55,
        same_pattern_family=True,
        debug={"pattern_family_score": 0.7},
        file_id=3,
    )
    semantic = _hit(score=0.5, debug={"semantic_score": 0.6}, file_id=4)
    texture = _hit(score=0.45, debug={"texture_score": 0.4}, file_id=5)
    pool = [family, exact, semantic, texture]
    for stage in ("Exact", "Pattern Family", "Semantic", "Texture", "Deep Search"):
        shown = _filter_progressive_results(pool, stage)
        assert exact in shown, stage
        assert shown[0] is exact, stage
        assert _is_progressive_exact(shown[0])


def test_query_feature_cache_second_lookup_skips_extract(tmp_path, monkeypatch):
    from PIL import Image

    from core.db import Database
    from core.search_engine import SearchEngine
    from core.settings import AppSettings

    img = tmp_path / "q.jpg"
    Image.new("RGB", (32, 32), color=(12, 80, 40)).save(img)
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
        face_index_enabled=False,
        ai_embedding_enabled=False,
        ocr_enabled=False,
    )
    Database(settings.db_path)
    calls = {"n": 0}
    from core.feature_extractor import ExtractedFeatures, FeatureExtractor

    real = FeatureExtractor.extract_from_path

    def counted(self, path, source_path=""):
        calls["n"] += 1
        try:
            return real(self, path, source_path=source_path)
        except Exception:
            return ExtractedFeatures(phash="a" * 16, dhash="b" * 16, whash="c" * 16)

    monkeypatch.setattr(FeatureExtractor, "extract_from_path", counted)
    clear_query_features_cache()
    engine = SearchEngine(settings, load_ai=False)
    first = engine._query_features_from_image(str(img), fast_only=True)
    second = engine._query_features_from_image(str(img), fast_only=True)
    stats = query_features_cache_stats()
    assert first[0].phash == second[0].phash
    assert calls["n"] == 1
    assert stats["hits"] >= 1
    assert stats["size"] >= 1
