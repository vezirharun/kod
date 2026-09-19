"""Canonical correction service — spelling fixes without merging distinct concepts."""
from __future__ import annotations

from pathlib import Path

from core.canonical_correction import (
    FIELD_CHILD,
    FIELD_PARENT,
    are_distinct_concepts,
    correct_learned_label,
)
from core.category_memory import dynamic_children, register_category, register_root_category
from core.concept_registry import concepts, learn


def test_are_distinct_tiger_leopard():
    assert are_distinct_concepts("Tiger", "Leopard") is True
    assert are_distinct_concepts("leopard", "Leopard") is False
    assert are_distinct_concepts("leoprad", "leopard") is False


def test_leoprad_to_leopard_updates_memory_and_concept(tmp_path: Path):
    db = str(tmp_path / "patterns.db")
    register_root_category(db, "Animal")
    register_category(db, "Animal", "leoprad")
    learn(db, "leoprad", parent="Animal", source="user")

    stats = correct_learned_label(
        db, FIELD_CHILD, "leoprad", "leopard", parent="Animal"
    )
    assert stats["ok"] is True
    kids = dynamic_children(db, "Animal")
    assert "leopard" in kids
    assert "leoprad" not in kids

    rows = {r["canonical"].lower(): r for r in concepts(db)}
    assert "leopard" in rows
    aliases = [str(a).lower() for a in __import__("json").loads(rows["leopard"]["aliases"] or "[]")]
    assert "leoprad" in aliases or "leoprad" in {rows["leopard"]["canonical"].lower()}


def test_tiger_not_merged_into_leopard(tmp_path: Path):
    db = str(tmp_path / "patterns.db")
    learn(db, "Tiger", parent="Animal", source="user")
    learn(db, "Leopard", parent="Animal", source="user")
    stats = correct_learned_label(
        db, FIELD_CHILD, "Tiger", "Leopard", parent="Animal"
    )
    assert stats["ok"] is False
    assert stats["reason"] == "distinct_concepts"
    names = {r["canonical"] for r in concepts(db)}
    assert "Tiger" in names
    assert "Leopard" in names


def test_correction_idempotent(tmp_path: Path):
    db = str(tmp_path / "patterns.db")
    register_root_category(db, "moti")
    a = correct_learned_label(db, FIELD_PARENT, "moti", "Motif")
    b = correct_learned_label(db, FIELD_PARENT, "moti", "Motif")
    c = correct_learned_label(db, FIELD_PARENT, "Motif", "Motif")
    assert a["ok"] and b["ok"] and c["ok"]


def test_edit_dialog_has_no_rename_buttons():
    from pathlib import Path as P

    src = (P(__file__).resolve().parents[1] / "ui" / "result_metadata_dialog.py").read_text(
        encoding="utf-8"
    )
    assert "Ana kategoriyi düzelt" not in src
    assert "Alt kategoriyi düzelt" not in src
    assert "Düzenle / Düzelt" in src
    assert "correct_learned_label" in src
