from pathlib import Path

from core.query_intent_router import classify_query
from core.universal_visual_intel import parse_universal_query

_ROUTE_TABLE = [
    ("Amiri", "brand", "Marka"),
    ("Dolce", "brand", "Marka"),
    ("aksesuar", "category", "Aksesuar"),
    ("kolye", "category", "Aksesuar → kolye"),
    ("gold", "material_color", "material/color"),
    ("gümüş", "material_color", "material/color"),
    ("kiraz", "motif_object", "motif/object"),
    ("yaprak", "motif_object", "motif/object"),
    ("kelebek", "motif_object", "motif/object"),
    ("kanat", "motif_object", "motif/object"),
    ("maske", "motif_object", "motif/object"),
    ("kafatası", "motif_object", "motif/object"),
    ("grunge", "texture_style", "texture/style"),
    ("dantel", "texture_style", "pattern/texture"),
    ("popart", "texture_style", "style"),
]

def test_person_name_router():
    q = classify_query("Hülya Koçyiğit")
    assert q.kind == "person_name"
    assert "hulya" in q.tokens
    assert "kocyigit" in q.tokens

def test_gender_router():
    assert classify_query("kadın").value == "FEMALE"
    assert classify_query("erkek").value == "MALE"

def test_pattern_not_misrouted_as_person_name():
    assert classify_query("mavi elbise").kind == "pattern"
    assert classify_query("çiçek deseni").kind == "pattern"

def test_uvi_gender_nodes():
    assert parse_universal_query("kadın").node_id == "female_person"
    assert parse_universal_query("erkek").node_id == "male_person"


def test_concept_intelligence_route_table():
    for query, channel, label in _ROUTE_TABLE:
        q = classify_query(query)
        assert q.channel == channel, f"{query}: got {q.channel} expected {channel}"
        assert q.label == label, f"{query}: got {q.label} expected {label}"


def test_regression_leopard_leopar_dolce_louise():
    assert classify_query("leopard").channel == "pattern"
    assert classify_query("leopar").channel == "pattern"
    assert classify_query("dolce").channel == "brand"
    assert classify_query("louise").channel == "brand"
    assert classify_query("louise").label == "Marka"


def test_user_concept_persists_in_registry_not_index(tmp_path):
    from core.concept_registry import learn, find
    from core.db import Database
    from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
    from core.search_memory import memory_db_path

    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    before = freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=db_path,
            faiss_dino_path=str(tmp_path / "faiss_dino.index"),
            faiss_clip_path=str(tmp_path / "faiss_clip.index"),
        )
    )
    cid = learn(db_path, "kiraz-öğret", concept_type="motif_object")
    assert cid
    hits = find(db_path, "kiraz-öğret")
    assert hits and hits[0]["canonical"] == "kiraz-öğret"
    q = classify_query("kiraz-öğret", db_path=db_path)
    assert q.channel == "motif_object"
    after = freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=db_path,
            faiss_dino_path=str(tmp_path / "faiss_dino.index"),
            faiss_clip_path=str(tmp_path / "faiss_clip.index"),
        )
    )
    assert after == before
    assert Path(memory_db_path(db_path)).is_file()
