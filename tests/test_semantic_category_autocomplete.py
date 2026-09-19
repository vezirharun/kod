"""Semantic category autocomplete — concept vs context ranking."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.canonical_correction import are_distinct_concepts
from core.semantic_category_autocomplete import (
    flatten_children_catalog,
    split_concept_context,
    suggest_semantic,
)
from ui.result_metadata_dialog import ResultMetadataDialog, TypeaheadCombo


def _app():
    return QApplication.instance() or QApplication([])


def _leopard_catalog():
    return flatten_children_catalog(
        {
            "Animal Print": ["Leopard", "Tiger", "Zebra"],
            "Textile": ["Leopard", "Floral"],
            "Animal": ["Cat", "Dog"],
            "Texture Ground": ["Fabric Texture", "Plain Texture"],
        }
    )


def test_empty_query_returns_no_dump():
    hits = suggest_semantic("", _leopard_catalog(), allow_new=True)
    assert hits == []


def test_leo_and_leopard_list_existing_meanings():
    cat = _leopard_catalog()
    for q in ("leo", "leopard", "leopar"):
        hits = suggest_semantic(q, cat, allow_new=False)
        labels = [h.display for h in hits]
        assert any("Leopard" in lab for lab in labels), q
        # Both Animal Print and Textile leopard meanings if present
        assert any("Animal Print" in lab for lab in labels), q
        assert any("Textile" in lab for lab in labels), q


def test_leopard_doku_prefers_texture_context():
    cat = _leopard_catalog()
    for q in ("leopard d", "leopard dok", "leopard dokusu"):
        hits = suggest_semantic(q, cat, allow_new=False)
        assert hits, q
        top = hits[0]
        assert "Textile" in top.display or "texture" in top.reason or top.rank <= 3, (
            q,
            top,
        )
        # Pure animal-only must not rank first when texture context present
        assert not top.display.startswith("Animal /"), q


def test_leopard_kafasi_prefers_animal_context():
    cat = _leopard_catalog()
    hits = suggest_semantic("leopard kafası", cat, allow_new=False)
    assert hits
    # Animal Print (pattern/animal) should beat Textile texture
    top_parents = [h.parent for h in hits[:2]]
    assert "Animal Print" in top_parents or hits[0].parent == "Animal Print"


def test_kaplan_and_kaplan_dokusu():
    cat = _leopard_catalog()
    hits = suggest_semantic("kaplan", cat, allow_new=False)
    assert any(h.data == "Tiger" for h in hits)
    assert not any(h.data == "Leopard" for h in hits)

    hits2 = suggest_semantic("kaplan dokusu", cat, allow_new=False)
    # Tiger exists only under Animal Print in fixture — still concept match
    assert any(h.data == "Tiger" for h in hits2)


def test_zebra_deseni():
    cat = _leopard_catalog()
    hits = suggest_semantic("zebra deseni", cat, allow_new=False)
    assert hits
    assert hits[0].data == "Zebra"


def test_xyz_yeni_ekle():
    cat = _leopard_catalog()
    hits = suggest_semantic("xyzzyqq", cat, allow_new=True)
    assert len(hits) == 1
    assert hits[0].is_new
    assert hits[0].display == "Yeni ekle"


def test_tiger_not_merged_with_leopard():
    assert are_distinct_concepts("Tiger", "Leopard")
    parsed = split_concept_context("leopard")
    assert "tiger" not in {k.lower() for k in parsed.concept_keys}
    assert "kaplan" not in {k.lower() for k in parsed.concept_keys}


def test_typeahead_empty_and_semantic(tmp_path):
    _app()
    combo = TypeaheadCombo()
    combo.set_choices([("Leopard", "Leopard"), ("Tiger", "Tiger"), ("Zebra", "Zebra")])
    combo.set_semantic_catalog(_leopard_catalog())
    assert combo.filtered_matches("") == []
    hits = combo.filtered_matches("leopard dokusu")
    assert hits
    assert any("Leopard" in d for d, _ in hits)
    assert combo.filtered_matches("xyzzyqq") == []


def test_edit_dialog_opens_without_hang(tmp_path):
    _app()
    db = str(tmp_path / "ac.db")
    from core.db import Database

    Database(db)
    dlg = ResultMetadataDialog(db_path=db)
    # Must not block UI thread on open; sync fill is for tests only.
    dlg.ensure_options_ready()
    assert dlg._options_loaded
    assert dlg.cmb_parent.filtered_matches("") == []
    dlg.cmb_child.set_semantic_catalog(
        flatten_children_catalog(dlg._options.get("children_by_parent") or {})
    )
    hits = dlg.cmb_child.filtered_matches("leopard")
    assert any("Leopard" in d for d, _ in hits)


def test_kal_prefix_finds_kaliba():
    cat = flatten_children_catalog(
        {
            "Textile": ["Kalıba Uyarlanmış", "Kamuflaj"],
            "Animal Print": ["Leopard"],
        }
    )
    # Progressive typeahead: freeze was at ≥3 chars ("kal"); short stems may be empty.
    for q in ("kal", "kalı", "kalıba", "kaliba"):
        hits = suggest_semantic(q, cat, allow_new=False)
        assert any(h.data == "Kalıba Uyarlanmış" for h in hits), (q, [h.data for h in hits])
    # 1–2 char stems must not hang / dump unrelated rows
    for q in ("k", "ka"):
        hits = suggest_semantic(q, cat, allow_new=False)
        assert isinstance(hits, list)


def test_synonym_kaplan_still_resolves_tiger():
    cat = _leopard_catalog()
    hits = suggest_semantic("kaplan", cat, allow_new=False)
    assert any(h.data == "Tiger" for h in hits)


def test_teach_new_category_appears_after_reindex():
    from core.semantic_category_autocomplete import (
        ensure_candidates_indexed,
        invalidate_autocomplete_caches,
    )

    cat = flatten_children_catalog({"Textile": ["Leopard"]})
    assert not any(h.data == "YeniMotif" for h in suggest_semantic("yenimotif", cat, allow_new=False))
    invalidate_autocomplete_caches()
    cat2 = flatten_children_catalog({"Textile": ["Leopard", "YeniMotif"]})
    cat2 = ensure_candidates_indexed(cat2)
    hits = suggest_semantic("yenimotif", cat2, allow_new=False)
    assert any(h.data == "YeniMotif" for h in hits)


def test_invalidate_bumps_catalog_keys_gen():
    from core.semantic_category_autocomplete import (
        _catalog_gen,
        invalidate_autocomplete_caches,
    )

    g0 = _catalog_gen()
    invalidate_autocomplete_caches()
    assert _catalog_gen() == g0 + 1


def test_suggest_semantic_hot_path_skips_leaf_rebuild(monkeypatch):
    """After catalog index, suggest_semantic must not rebuild candidate leaf keys."""
    import time

    import core.semantic_category_autocomplete as sca

    calls = {"n": 0}
    real = sca._compute_candidate_keys

    def spy_compute(c):
        calls["n"] += 1
        return real(c)

    monkeypatch.setattr(sca, "_compute_candidate_keys", spy_compute)

    cat = flatten_children_catalog(
        {f"Parent{p}": [f"Leaf{p * 32 + i:03d}" for i in range(32)] for p in range(8)}
    )
    assert len(cat) >= 256
    assert all(c.match_keys is not None for c in cat)
    calls["n"] = 0

    t0 = time.perf_counter()
    for q in ("zzz", "kal", "leaf250", "leaf251", "nomatchxx"):
        hits = suggest_semantic(q, cat, allow_new=False)
        assert hits is not None
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert calls["n"] == 0, f"_compute_candidate_keys called {calls['n']} times on hot path"
    assert elapsed_ms < 100.0, f"5 keypresses took {elapsed_ms:.1f}ms (budget 100ms)"


def test_suggest_semantic_time_budget_single_keypress():
    import time

    cat = flatten_children_catalog(
        {f"Parent{p}": [f"Leaf{p * 32 + i:03d}" for i in range(32)] for p in range(8)}
    )
    # Warm
    suggest_semantic("leaf250", cat, allow_new=False)
    t0 = time.perf_counter()
    suggest_semantic("kal", cat, allow_new=False)
    ms = (time.perf_counter() - t0) * 1000
    assert ms < 20.0, f"suggest_semantic took {ms:.1f}ms (budget <20ms)"

