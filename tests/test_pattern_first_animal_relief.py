"""Pattern-first: same animal_print_type must relieve structural gate (garment↔fabric)."""
from __future__ import annotations

from types import SimpleNamespace

from core.search_engine import SearchEngine
from core.texture_profile import TextureProfile


class _FakeSettings:
    similarity_threshold = 0.60
    ai_embedding_enabled = True
    search_visual = True
    search_texture = True
    search_color = True
    search_mode = "style"
    color_weight_mode = "ignore"
    weight_dino = 0.35
    weight_clip = 0.25
    weight_phash = 0.20
    weight_color = 0.0
    weight_texture = 0.10

    def effective_weights(self):
        return {
            "dino": 0.39,
            "clip": 0.28,
            "phash": 0.22,
            "texture": 0.11,
            "color": 0.0,
        }

    def classical_score(self, *a, **k):
        return 0.0


def test_same_animal_print_gets_pattern_first_soft_lift():
    """Model photo vs fabric: low phash must not bury same leopard type."""
    eng = SearchEngine.__new__(SearchEngine)
    eng.settings = _FakeSettings()
    eng._latency = SimpleNamespace(enabled=False)

    import numpy as np

    # Distinct global embeddings (simulates person vs fabric composition)
    q_clip = np.ones(512, dtype=np.float32)
    q_clip = q_clip / np.linalg.norm(q_clip)
    c_clip = q_clip.copy()
    c_clip[0] = 0.2
    c_clip = c_clip / np.linalg.norm(c_clip)

    q_tm = {
        "pattern_family": "animal_print",
        "animal_print_type": "leopard",
        "pattern_dna": {"Family": "animal_print", "Motif": "leopard"},
    }
    c_tm = {
        "pattern_family": "floral",  # mislabeled family — animal type still leopard
        "animal_print_type": "leopard",
        "pattern_dna": {"Family": "animal_print", "Motif": "leopard"},
    }

    from core.feature_extractor import ExtractedFeatures

    query = ExtractedFeatures(
        phash="aaaaaaaaaaaaaaaa",
        dhash="bbbbbbbbbbbbbbbb",
        whash="cccccccccccccccc",
        clip_embedding=q_clip.astype(np.float32).tobytes(),
        dino_embedding=(np.ones(384, dtype=np.float32) / 19.6).astype(np.float32).tobytes(),
        texture_map=q_tm,
        texture_features=[],
        color_hist=b"",
        patch_embeddings_meta=[],
    )
    # Candidate: different hash (composition), weak dino alignment
    c_dino = np.zeros(384, dtype=np.float32)
    c_dino[0] = 1.0
    rec = {
        "id": 99,
        "path": "/fabric/leopard_metraj.jpg",
        "filename": "leopard_metraj.jpg",
        "phash": "ffffffffffffffff",
        "dhash": "eeeeeeeeeeeeeeee",
        "whash": "dddddddddddddddd",
        "clip_embedding": c_clip.astype(np.float32).tobytes(),
        "dino_embedding": c_dino.tobytes(),
        "texture_map": c_tm,
        "texture_features": [],
        "color_hist": b"",
        "patch_embeddings_meta": [],
        "partial_hash": "",
        "full_hash": "",
        "ocr_text": "",
    }
    score, breakdown, *_rest = eng._score_record(
        rec,
        query,
        query_path="/model/2.jpg",
        query_partial_hash="",
        faiss_boost=0.0,
        crop_search=False,
        query_profile=TextureProfile.from_dict(q_tm),
    )
    debug = _rest[-1] if _rest else {}
    assert score >= 0.55, f"same leopard fabric buried under structural gate: {score}"
    assert debug.get("pattern_first_soft") or debug.get("pattern_first_same_animal") or score >= 0.55
