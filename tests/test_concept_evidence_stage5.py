"""Stage 5 — concept evidence consolidation (read-only)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.concept_evidence import (
    apply_concept_evidence_scoring,
    build_concept_evidence_profile,
    build_rival_profile,
    concept_data_quality_report,
    score_result_against_concept_evidence,
)
from core.db import Database
from core.teach_me import teach_files


def _db(tmp_path: Path) -> tuple[Database, str]:
    path = str(tmp_path / "patterns.db")
    db = Database(path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db, path


def _add_with_dna(
    db: Database,
    tmp_path: Path,
    name: str,
    *,
    scale: str = "Low",
    density: str = "High",
    colors: list | None = None,
    family: str = "animal_print",
    subtype: str = "tiger",
) -> int:
    img = tmp_path / name
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": name,
                "source_id": 1,
                "status": "indexed",
                "pattern_family": family,
                "pattern_subtype": subtype,
            }
        )
    )
    db.upsert_features(
        fid,
        {
            "phash": "abc",
            "clip_embedding": b"\x00\x01\x02\x03",
            "texture_map": {
                "pattern_family": family,
                "animal_print_type": subtype,
                "pattern_dna": {
                    "confidence": 0.75,
                    "scale": scale,
                    "density": density,
                    "dominant_colors": colors or ["black", "cream"],
                    "repeat_type": "stripe" if subtype == "tiger" else "spot",
                },
            },
        },
    )
    return fid


def test_evidence_profile_from_user_positives(tmp_path):
    db, path = _db(tmp_path)
    a = _add_with_dna(db, tmp_path, "t1.jpg", scale="Low", density="High")
    b = _add_with_dna(db, tmp_path, "t2.jpg", scale="Low", density="High")
    c = _add_with_dna(db, tmp_path, "t3.jpg", scale="High", density="Low")
    teach_files(db, path, [a, b, c], "Tiger", overlay={"parent": "Animal Print"})
    prof = build_concept_evidence_profile(path, "Tiger")
    assert prof.positive_count == 3
    assert prof.dna_available >= 2
    assert prof.user_verified is False
    assert prof.scale.value in {"low", "high"}
    assert prof.attribute_coverage > 0
    report = concept_data_quality_report(path, ["Tiger"])
    assert report[0]["DNA_available"] == prof.dna_available


def test_unknown_attr_no_penalty(tmp_path):
    db, path = _db(tmp_path)
    fid = _add_with_dna(db, tmp_path, "t.jpg", scale="High", density="High", colors=[])
    # Clear colors in DNA
    db.upsert_features(
        fid,
        {
            "texture_map": {
                "pattern_family": "animal_print",
                "animal_print_type": "tiger",
                "pattern_dna": {"confidence": 0.7, "scale": "High", "density": "High"},
            }
        },
    )
    teach_files(db, path, [fid], "Tiger")
    prof = build_concept_evidence_profile(path, "Tiger")
    row = SimpleNamespace(
        score=0.9,
        score_percent=90,
        animal_print_type="tiger",
        debug={
            "learned_concept": True,
            "learned_canonical": "Tiger",
            "texture_map": {
                "pattern_dna": {"scale": "High", "density": "High"},
            },
        },
        is_self_match=False,
    )
    meta = score_result_against_concept_evidence(
        row, prof, query_scale="small", query_density="", query_colors=["black", "cream"]
    )
    # colors unknown on file → no color penalty; scale mismatch may apply if strong
    assert meta.get("user_verified") is False
    color_part = (meta.get("parts") or {}).get("color") or {}
    assert color_part.get("delta", 0) == 0 or color_part.get("confidence") == "unknown"


def test_user_protected_not_changed(tmp_path):
    db, path = _db(tmp_path)
    fid = _add_with_dna(db, tmp_path, "t.jpg", scale="High")
    teach_files(db, path, [fid], "Tiger")
    prof = build_concept_evidence_profile(path, "Tiger")
    row = SimpleNamespace(
        score=0.95,
        score_percent=95,
        animal_print_type="tiger",
        debug={
            "learned_concept_exact": True,
            "learned_canonical": "Tiger",
            "texture_map": {"pattern_dna": {"scale": "High"}},
        },
        is_self_match=False,
    )
    before = row.score
    apply_concept_evidence_scoring(
        [row], "küçük kaplan", index_db=path, canonical="Tiger"
    )
    assert row.score == before


def test_rival_profile_not_authority(tmp_path):
    db, path = _db(tmp_path)
    t = _add_with_dna(db, tmp_path, "t.jpg", subtype="tiger")
    l = _add_with_dna(db, tmp_path, "l.jpg", subtype="leopard", scale="High")
    teach_files(db, path, [t], "Tiger", overlay={"parent": "Animal Print"})
    teach_files(db, path, [l], "Leopard", overlay={"parent": "Animal Print"})
    rival = build_rival_profile(path, "Tiger", "Leopard")
    assert rival["discriminative"]["user_verified"] is False
    assert rival["left"]["positive_count"] == 1
    assert rival["right"]["positive_count"] == 1


def test_evidence_does_not_create_or_demote(tmp_path):
    db, path = _db(tmp_path)
    t = _add_with_dna(db, tmp_path, "t.jpg")
    teach_files(db, path, [t], "Tiger")
    from core.concept_registry import concepts, positives_for_file

    n_before = len(concepts(path))
    build_concept_evidence_profile(path, "Tiger")
    build_rival_profile(path, "Tiger", "Leopard")
    assert len(concepts(path)) == n_before
    assert positives_for_file(path, t)
