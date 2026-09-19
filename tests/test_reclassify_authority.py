"""Same-file reclassification: Leopard→Tiger must not leave dual positives/display."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.concept_registry import learn, positives_for_file
from core.db import Database
from core.learned_concept_search import collect_learned_hits
from core.teach_me import apply_metadata_edit_to_files, learn_from_metadata_edit
from core.user_feedback import apply_metadata_overlay_to_result
from ui.designer_labels import family_badge_for_result


def _db(tmp_path: Path) -> tuple[Database, str]:
    db_path = str(tmp_path / "patterns.db")
    db = Database(db_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db, db_path


def _file(db: Database, tmp_path: Path, name: str) -> int:
    img = tmp_path / name
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": name,
                "source_id": 1,
                "status": "indexed",
                "pattern_family": "animal_print",
            }
        )
    )
    vec = np.ones(8, dtype=np.float32).tobytes()
    db.upsert_features(fid, {"clip_embedding": vec, "texture_map": {}})
    return fid


def test_leopard_to_tiger_demotes_same_file_keeps_other(tmp_path: Path):
    db, db_path = _db(tmp_path)
    a = _file(db, tmp_path, "a.jpg")
    b = _file(db, tmp_path, "b.jpg")
    learn(db_path, "Leopard", file_id=a, parent="Animal Print", concept_type="visual_concept")
    learn(db_path, "Leopard", file_id=b, parent="Animal Print", concept_type="visual_concept")

    stats = apply_metadata_edit_to_files(
        db,
        db_path,
        [a],
        {
            "parent": "Animal Print",
            "child": "Tiger",
            "category_path": "Animal Print/Tiger",
            "pattern_family": "Leopard / Tiger / Animal",  # stale UI free text
            "animal_print_type": "tiger",
            "tags": ["tiger", "kaplan"],
        },
    )
    assert stats["taught"] >= 1
    assert stats.get("demoted", 0) >= 1

    pos_a = {str(r["canonical"]).casefold() for r in positives_for_file(db_path, a)}
    assert "tiger" in pos_a
    assert "leopard" not in pos_a

    pos_b = {str(r["canonical"]).casefold() for r in positives_for_file(db_path, b)}
    assert "leopard" in pos_b

    tiger_pack = collect_learned_hits(db_path, "tiger", db=db)
    assert a in (tiger_pack.get("exact_ids") or [])
    leopard_pack = collect_learned_hits(db_path, "leopard", db=db)
    assert a not in (leopard_pack.get("exact_ids") or [])
    assert b in (leopard_pack.get("exact_ids") or [])

    # DB classification: machine family, not free-text mash
    row = db.get_file_by_id(a) or {}
    assert str(row.get("pattern_family") or "") == "animal_print"
    tm = db.get_features(a) or {}
    # texture_map may be JSON string
    import json

    raw = tm.get("texture_map")
    if isinstance(raw, str):
        raw = json.loads(raw or "{}")
    assert isinstance(raw, dict)
    assert str(raw.get("animal_print_type") or "") == "tiger"
    assert "Leopard" not in str(raw.get("pattern_family") or "")


def test_overlay_clears_stale_learned_canonical_for_card():
    result = SimpleNamespace(
        file_id=1,
        color_family="",
        pattern_family="Leopard / Tiger / Animal",
        animal_print_type="leopard",
        debug={
            "learned_canonical": "Leopard",
            "learned_concept": True,
            "learned_concept_exact": True,
            "user_tags": ["leopard"],
            "texture_map": {"pattern_family": "Leopard / Animal"},
        },
    )
    apply_metadata_overlay_to_result(
        result,
        {
            "parent": "Animal Print",
            "child": "Tiger",
            "category_path": "Animal Print/Tiger",
            "pattern_family": "Tiger / Animal",
            "animal_print_type": "tiger",
            "tags": ["tiger", "kaplan"],
        },
    )
    assert result.pattern_family == "animal_print"
    assert result.animal_print_type == "tiger"
    assert result.debug.get("learned_canonical") == "Tiger"
    assert "leopard" not in " ".join(result.debug.get("user_tags") or []).casefold()
    badge = family_badge_for_result(result)
    assert "leopard" not in badge.casefold()
    assert "tiger" in badge.casefold()

    # Yeni kullanıcı etiketi, object-AI / stale learned olmadan admin_label kalsın
    plain = SimpleNamespace(
        file_id=2,
        color_family="",
        pattern_family="unknown",
        animal_print_type="",
        debug={"texture_map": {}},
    )
    apply_metadata_overlay_to_result(
        plain,
        {"parent": "Animal", "child": "kedi", "category_path": "Animal/kedi"},
    )
    assert not plain.debug.get("learned_canonical")
    assert plain.debug.get("user_labeled") is True
    from core.category_predictions import category_predictions_from_result
    from core.search_engine import SearchResult

    sr = SearchResult(
        file_id=2,
        path="x.jpg",
        filename="x.jpg",
        customer="",
        thumbnail_path="",
        score=0.5,
        score_percent=50,
        pattern_family=str(plain.pattern_family or ""),
        debug=dict(plain.debug),
    )
    preds = category_predictions_from_result(sr)
    assert preds and preds[0]["source"] == "admin_label"


def test_reclassify_tiger_leopard_tiger_no_duplicate_positive(tmp_path: Path):
    db, db_path = _db(tmp_path)
    fid = _file(db, tmp_path, "x.jpg")
    for child, animal in (
        ("Tiger", "tiger"),
        ("Leopard", "leopard"),
        ("Tiger", "tiger"),
    ):
        learn_from_metadata_edit(
            db,
            db_path,
            fid,
            {
                "parent": "Animal Print",
                "child": child,
                "category_path": f"Animal Print/{child}",
                "pattern_family": "animal_print",
                "animal_print_type": animal,
            },
            previous={},
        )
    pos = positives_for_file(db_path, fid)
    tiger = [r for r in pos if str(r["canonical"]).casefold() == "tiger"]
    leopard = [r for r in pos if str(r["canonical"]).casefold() == "leopard"]
    assert len(tiger) == 1
    assert len(leopard) == 0
