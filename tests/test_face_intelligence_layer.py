"""Face Intelligence: isolated face DB, real InsightFace when available."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.face_identity import FaceObservation
from core.face_index import FaceIndexStore
from core.face_intelligence import (
    BACKEND,
    GENDER_CONF_MIN,
    MATCH_THRESHOLD,
    MIN_DET_SCORE,
    MIN_MARGIN,
    FaceIntelligence,
    gender_tr,
    kisi_label,
    later_archive_scan_howto,
)
from core.face_search import parse_face_query
from core.index_freeze import snapshot_index_artifacts


def test_kisi_and_gender_labels():
    assert kisi_label("person_0017") == "Kişi 017"
    assert kisi_label("person_1") == "Kişi 001"
    assert gender_tr("FEMALE", 0.9) == "kadın"
    assert gender_tr("MALE", 0.9) == "erkek"
    assert gender_tr("FEMALE", 0.1) == "bilinmiyor"
    assert gender_tr("UNKNOWN", 1.0) == "bilinmiyor"
    assert GENDER_CONF_MIN == 0.60
    assert MATCH_THRESHOLD == 0.62
    assert MIN_MARGIN == 0.05
    assert MIN_DET_SCORE == 0.50


def test_parse_kisi_and_gender_queries():
    assert parse_face_query("Kişi 017") == {"kind": "person", "value": "person_0017"}
    assert parse_face_query("kadın")["value"] == "FEMALE"
    assert parse_face_query("erkek")["value"] == "MALE"
    assert parse_face_query("bilinmiyor")["value"] == "UNKNOWN"


def test_person_cluster_and_search_on_face_db_only(tmp_path):
    store = FaceIndexStore(
        tmp_path / "face_index.db",
        threshold=0.62,
        min_margin=0.05,
        identity_engine=BACKEND,
    )
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[0] = 0.99; b[1] = 0.01
    c = np.zeros(8, dtype=np.float32); c[1] = 1.0
    store.replace_file_faces(1, str(tmp_path / "p1.jpg"), 1, 10, [
        FaceObservation("face_001", (1, 2, 30, 40), a, "FEMALE", 0.9, "HIGH", 0.88),
        FaceObservation("face_002", (50, 2, 80, 40), c, "MALE", 0.9, "HIGH", 0.91),
    ])
    store.replace_file_faces(2, str(tmp_path / "p2.jpg"), 1, 10, [
        FaceObservation("face_001", (3, 4, 20, 22), b, "FEMALE", 0.85, "HIGH", 0.77),
    ])
    persons = {p["person_id"] for p in store.list_persons()}
    assert len(persons) == 2
    female_files = store.search(gender="FEMALE")
    male_files = store.search(gender="MALE")
    assert 1 in female_files and 2 in female_files
    assert 1 in male_files and 2 not in male_files
    pids = store.search(person_id="person_0001")
    assert 1 in pids and 2 in pids
    store.label_person("person_0001", "Ahmet")
    assert 1 in store.search_by_name("Ahmet")
    fi = FaceIntelligence(tmp_path / "face_index.db")
    assert fi.images_for_kisi(1) == pids or set(fi.query_images("Kişi 001")) >= {1, 2}
    assert "116K" not in later_archive_scan_howto() or "review" in later_archive_scan_howto().lower()


def test_face_writes_do_not_touch_pattern_index(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    patterns = data / "patterns.db"
    patterns.write_bytes(b"not-a-real-index")
    snap = snapshot_index_artifacts(
        db_path=str(patterns),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        cache_dir=str(tmp_path / "cache"),
    )
    store = FaceIndexStore(data / "face_index.db")
    vec = np.ones(4, dtype=np.float32)
    store.replace_file_faces(9, "x.jpg", 1, 2, [
        FaceObservation("face_001", (0, 0, 8, 8), vec, "UNKNOWN", 0.0, "UNKNOWN", 0.9),
    ])
    snap2 = snapshot_index_artifacts(
        db_path=str(patterns),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        cache_dir=str(tmp_path / "cache"),
    )
    assert snap == snap2
    assert patterns.read_bytes() == b"not-a-real-index"


def test_intelligence_never_uses_vision_fallback(tmp_path):
    fi = FaceIntelligence(tmp_path / "face.db")
    assert fi.engine.allow_vision_fallback is False
    assert BACKEND.startswith("insightface")


def _tiny_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (160, 160), color).save(path, "JPEG")


def _bundled_face_image() -> Path | None:
    try:
        import insightface
        root = Path(insightface.__file__).parent / "data" / "images"
        # Tom_Hanks_54745.png is 112x112 and buffalo_l det_size=640 misses it.
        t1 = root / "t1.jpg"
        return t1 if t1.is_file() else None
    except Exception:
        return None


def test_real_insightface_or_honest_skip(tmp_path):
    fi = FaceIntelligence(tmp_path / "face_index.db")
    if not fi.ready():
        pytest.skip(f"InsightFace/buffalo not usable: {fi.unavailable_reason()}")
    sample = _bundled_face_image()
    if sample is None:
        blank = tmp_path / "blank.jpg"
        _tiny_jpeg(blank, (40, 80, 120))
        det = fi.detect_embed(str(blank))
        assert det["ok"] is True
        assert det["backend"] == BACKEND
        if det["face_count"] == 0:
            pytest.skip("InsightFace loaded but fixture had no detectable face")
        return
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(sample.read_bytes())
    b.write_bytes(sample.read_bytes())
    det = fi.detect_embed(str(a))
    assert det["ok"] is True
    assert det["backend"] == BACKEND
    assert det["face_count"] >= 2  # t1.jpg is a group photo
    assert det["faces"][0]["embedding_dim"] >= 128
    assert det["faces"][0]["gender_tr"] in {"kadın", "erkek", "bilinmiyor"}
    r1 = fi.index_image(1, str(a))
    r2 = fi.index_image(2, str(b))
    assert r1["ok"] and r2["ok"]
    assert len(r1["person_ids"]) >= 2
    assert set(r1["person_ids"]) == set(r2["person_ids"])
    pid = r1["person_ids"][0]
    fi.name_person(pid, "Ahmet")
    n = int("".join(ch for ch in pid if ch.isdigit()))
    assert 1 in fi.images_for_kisi(n)
    assert 2 in fi.query_images(kisi_label(pid))
    assert 1 in fi.query_images("Ahmet")
    hits = fi.match_new_image(str(b))
    assert hits["ok"]
    assert any(m.get("person_id") == pid for m in hits["matches"])
