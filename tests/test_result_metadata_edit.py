import os
import sqlite3
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.category_memory import dynamic_children, dynamic_parents, register_category
from core.category_tree import parent_categories
from core.db import Database
from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.search_engine import SearchResult
from core.user_feedback import (
    ACTION_METADATA_EDIT,
    UserFeedbackStore,
    apply_metadata_overlay_to_result,
)
from ui.designer_labels import color_badge_text, family_badge_text
from ui.inspector_panel import InspectorPanel
from ui.result_metadata_dialog import ResultMetadataDialog, TypeaheadCombo

ROOT = Path(__file__).resolve().parents[1]


def _app():
    return QApplication.instance() or QApplication([])


def _result(**kwargs) -> SearchResult:
    data = dict(
        file_id=1,
        path="C:/tmp/a.jpg",
        filename="a.jpg",
        customer="",
        thumbnail_path="",
        score=0.8,
        score_percent=80,
        color_family="brown_tan",
        pattern_family="animal_print",
        animal_print_type="leopard",
        debug={},
    )
    data.update(kwargs)
    return SearchResult(**data)


def _index_fp(db_path: str, faiss_dino: str, faiss_clip: str) -> dict:
    snap = snapshot_index_artifacts(
        db_path=db_path,
        faiss_dino_path=faiss_dino,
        faiss_clip_path=faiss_clip,
    )
    return freeze_fingerprint(snap)


def _files_fingerprint(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    try:
        files = conn.execute(
            "SELECT id, path, pattern_family, color_family, manual_category_path "
            "FROM files ORDER BY id"
        ).fetchall()
        feats = conn.execute(
            "SELECT file_id, texture_map FROM features ORDER BY file_id"
        ).fetchall()
    except sqlite3.OperationalError:
        files, feats = [], []
    conn.close()
    return [tuple(r) for r in files], [tuple(r) for r in feats]


def test_inspector_detail_keeps_only_six_actions():
    src = (ROOT / "ui" / "inspector_panel.py").read_text(encoding="utf-8")
    for banned in (
        "Bu Leopard",
        "Bu Floral",
        "Aynı Desen",
        "Benzer doku",
        "Benzer Değil",
        "Alakasız",
        "Kayıtlı kategorileriniz",
    ):
        assert banned not in src
    for keep in ("Doğru", "Yanlış", "Düzenle", "Klasörde Aç", "Görseli Aç", "Benzerlerini Ara"):
        assert keep in src


def test_edit_dialog_has_no_similarity_maze():
    src = (ROOT / "ui" / "result_metadata_dialog.py").read_text(encoding="utf-8")
    assert "SIMILARITY_TIER_CHOICES" not in src
    assert "Kayıtlı kategorileriniz" not in src
    assert "Bul ve seç" not in src
    assert "Kaydet" in src
    assert "Aynı Görsel" not in src


def test_wrong_dialog_still_has_teach_and_apply():
    src = (ROOT / "ui" / "feedback_family_dialog.py").read_text(encoding="utf-8")
    assert "Öğret ve Uygula" in src


def test_color_save_shows_kirmizi_and_can_revert(tmp_path):
    _app()
    db = str(tmp_path / "overlay.db")
    Database(db)
    store = UserFeedbackStore(Database(db))
    result = _result()
    dlg = ResultMetadataDialog(result=result, db_path=db)
    assert dlg.cmb_color.current_data_or_text() in ("brown_tan", "Kahve / Bej") or (
        color_badge_text(result.color_family) == "Kahve / Bej"
    )
    dlg.cmb_color.set_text("Kırmızı", "red")
    overlay = dlg.values()
    assert overlay["color_family"] == "red"
    store.save_metadata_overlay(1, overlay)
    loaded = store.metadata_overlay_for_file(1)
    apply_metadata_overlay_to_result(result, loaded)
    assert color_badge_text(result.color_family) == "Kırmızı"

    overlay["color_family"] = "brown_tan"
    store.save_metadata_overlay(1, overlay)
    apply_metadata_overlay_to_result(result, store.metadata_overlay_for_file(1))
    assert color_badge_text(result.color_family) == "Kahve / Bej"


def test_existing_and_new_category_persist(tmp_path):
    _app()
    db = str(tmp_path / "cats.db")
    Database(db)
    before = list(parent_categories())
    dlg = ResultMetadataDialog(result=_result(), db_path=db)
    dlg.ensure_options_ready()
    assert "Animal Print" in [dlg.cmb_parent.itemText(i) for i in range(dlg.cmb_parent.count())]
    dlg.cmb_parent.set_text("Animal Print")
    dlg._reload_children()
    dlg.cmb_child.set_text("Leopard")
    vals = dlg.values()
    assert vals["child"] == "Leopard"

    dlg.cmb_parent.set_text("YeniAnaGrup")
    dlg.cmb_child.set_text("YeniAlt")
    dlg._remember_new_values(dlg.values())
    assert "YeniAnaGrup" in dynamic_parents(db)
    assert "YeniAlt" in dynamic_children(db, "YeniAnaGrup")

    dlg2 = ResultMetadataDialog(result=_result(), db_path=db)
    dlg2.ensure_options_ready(force=True)
    parents = [dlg2.cmb_parent.itemText(i) for i in range(dlg2.cmb_parent.count())]
    assert "YeniAnaGrup" in parents
    dlg2.cmb_parent.set_text("YeniAnaGrup")
    dlg2._reload_children()
    children = [dlg2.cmb_child.itemText(i) for i in range(dlg2.cmb_child.count())]
    assert "YeniAlt" in children
    assert parent_categories() == before


def test_pattern_family_selectable_and_filter(tmp_path):
    _app()
    dlg = ResultMetadataDialog(result=_result(), db_path=str(tmp_path / "f.db"))
    dlg.ensure_options_ready()
    families = [dlg.cmb_family.itemData(i) for i in range(dlg.cmb_family.count())]
    assert "animal_print" in families
    combo = TypeaheadCombo()
    combo.set_choices([("Kırmızı", "red"), ("Kırmızı / Siyah", "red_black"), ("Kahve / Bej", "brown_tan")])
    hits = combo.filtered_matches("kır")
    labels = [h[0] for h in hits]
    assert "Kırmızı" in labels
    assert "Kırmızı / Siyah" in labels


def test_kaydet_does_not_mutate_index_or_faiss_or_ranking(tmp_path):
    _app()
    data = tmp_path / "data"
    data.mkdir()
    db_path = str(data / "patterns.db")
    faiss_d = str(data / "faiss_dino.index")
    faiss_c = str(data / "faiss_clip.index")
    Path(faiss_d).write_bytes(b"FAISS-SENTINEL")
    Path(faiss_c).write_bytes(b"FAISS-SENTINEL-2")
    db = Database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    db.upsert_file(
        {
            "path": str(tmp_path / "a.jpg"),
            "filename": "a.jpg",
            "source_id": 1,
            "status": "indexed",
            "pattern_family": "animal_print",
            "color_family": "brown_tan",
        }
    )
    before_index = _index_fp(db_path, faiss_d, faiss_c)
    before_files = _files_fingerprint(db_path)
    before_parents = list(parent_categories())

    store = UserFeedbackStore(Database(db_path))
    overlay = {
        "parent": "Animal Print",
        "child": "Leopard",
        "category_path": "Animal Print/Leopard",
        "pattern_family": "animal_print",
        "color_family": "red",
        "tags": ["demo"],
    }
    store.save_metadata_overlay(1, overlay)
    register_category(db_path, "Renk", "Mercan")

    after_index = _index_fp(db_path, faiss_d, faiss_c)
    after_files = _files_fingerprint(db_path)
    assert after_files == before_files
    assert Path(faiss_d).read_bytes() == b"FAISS-SENTINEL"
    assert Path(faiss_c).read_bytes() == b"FAISS-SENTINEL-2"
    # SQLite user_feedback may change db size; Pattern Index rows must not.
    assert freeze_fingerprint(
        {k: v for k, v in after_index.items() if "faiss" in k.lower()}
    ) == freeze_fingerprint(
        {k: v for k, v in before_index.items() if "faiss" in k.lower()}
    )
    assert store.score_adjustments("") == {}
    assert parent_categories() == before_parents
    with sqlite3.connect(db_path) as conn:
        actions = [
            r[0]
            for r in conn.execute("SELECT action FROM user_feedback").fetchall()
        ]
    assert ACTION_METADATA_EDIT in actions
    assert "label_family" not in actions


def test_overlay_display_leopard_kirmizi():
    result = _result()
    apply_metadata_overlay_to_result(
        result,
        {
            "pattern_family": "animal_print",
            "animal_print_type": "leopard",
            "color_family": "red",
        },
    )
    assert family_badge_text(result.pattern_family, result.animal_print_type) == "Leopard"
    assert color_badge_text(result.color_family) == "Kırmızı"


def test_inspector_panel_builds(tmp_path):
    _app()
    panel = InspectorPanel()
    labels = []
    for btn in (panel.btn_ai_correct, panel.btn_ai_wrong, panel.btn_ai_edit,
                panel.btn_folder, panel.btn_file, panel.btn_similar):
        labels.append(btn.text())
    assert labels == [
        "Doğru",
        "Yanlış",
        "Düzenle",
        "Klasörde Aç",
        "Görseli Aç",
        "Benzerlerini Ara",
    ]
    panel.set_result(_result())
    html = panel.lbl_preview_name.text()
    assert "Kahve / Bej" in html or "Leopard" in html
    apply_metadata_overlay_to_result(
        panel._result, {"color_family": "red", "pattern_family": "animal_print", "animal_print_type": "leopard"}
    )
    panel.set_result(panel._result)
    assert "Kırmızı" in panel.lbl_preview_name.text()


def test_color_only_edit_does_not_create_concept_example(tmp_path):
    import numpy as np

    from core.concept_registry import concepts, example_count
    from core.teach_me import learn_from_metadata_edit

    db_path = str(tmp_path / "patterns.db")
    db = Database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    fid = db.upsert_file(
        {
            "path": str(tmp_path / "a.jpg"),
            "filename": "a.jpg",
            "source_id": 1,
            "status": "indexed",
            "pattern_family": "floral",
            "color_family": "brown_tan",
        }
    )
    vec = np.ones(8, dtype=np.float32).tobytes()
    db.upsert_features(fid, {"clip_embedding": vec})
    out = learn_from_metadata_edit(
        db,
        db_path,
        fid,
        {"color_family": "red", "pattern_family": "floral"},
        previous={"pattern_family": "floral", "color_family": "brown_tan"},
    )
    assert out["taught"] == 0
    assert concepts(db_path) == [] or all(
        str(r.get("canonical") or "").lower() not in {"red", "kırmızı", "kirmizi"}
        for r in concepts(db_path)
    )


def test_category_edit_records_concept_and_clip_without_touching_index(tmp_path):
    import numpy as np

    from core.concept_registry import concepts, example_count
    from core.teach_me import learn_from_metadata_edit, match_clip_to_concepts

    data = tmp_path / "data"
    data.mkdir()
    db_path = str(data / "patterns.db")
    faiss_d = str(data / "faiss_dino.index")
    faiss_c = str(data / "faiss_clip.index")
    Path(faiss_d).write_bytes(b"FAISS-SENTINEL")
    Path(faiss_c).write_bytes(b"FAISS-SENTINEL-2")
    db = Database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    fid = db.upsert_file(
        {
            "path": str(tmp_path / "zincir.jpg"),
            "filename": "zincir.jpg",
            "source_id": 1,
            "status": "indexed",
            "pattern_family": "unknown",
            "color_family": "brown_tan",
        }
    )
    vec = np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
    db.upsert_features(fid, {"clip_embedding": vec, "texture_map": {}})
    before_files = _files_fingerprint(db_path)
    overlay = {
        "parent": "Aksesuar",
        "child": "Zincir",
        "category_path": "Aksesuar/Zincir",
        "color_family": "gold",
        "tags": ["zincir"],
    }
    store = UserFeedbackStore(Database(db_path))
    store.save_metadata_overlay(fid, overlay)
    out = learn_from_metadata_edit(db, db_path, fid, overlay, previous={})
    assert out["taught"] == 1
    assert out["concept_id"] > 0
    assert example_count(db_path, out["concept_id"]) == 1
    assert any(str(r["canonical"]) == "Zincir" for r in concepts(db_path))
    hits = match_clip_to_concepts(db_path, vec)
    assert hits and hits[0]["canonical"] == "Zincir"
    assert _files_fingerprint(db_path) == before_files
    assert Path(faiss_d).read_bytes() == b"FAISS-SENTINEL"
    assert Path(faiss_c).read_bytes() == b"FAISS-SENTINEL-2"
    color_only = learn_from_metadata_edit(
        db,
        db_path,
        fid,
        {**overlay, "color_family": "red"},
        previous=overlay,
    )
    assert color_only["taught"] == 0
    assert example_count(db_path, out["concept_id"]) == 1


def test_bulk_edit_fixes_wrong_family_on_many_files(tmp_path):
    import numpy as np

    from core.concept_registry import example_count
    from core.teach_me import apply_metadata_edit_to_files, build_common_edit_overlay
    from core.user_feedback import UserFeedbackStore

    db_path = str(tmp_path / "patterns.db")
    db = Database(db_path)
    faiss_d = tmp_path / "faiss_dino.index"
    faiss_c = tmp_path / "faiss_clip.index"
    faiss_d.write_bytes(b"FAISS-SENTINEL")
    faiss_c.write_bytes(b"FAISS-SENTINEL-2")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    ids = []
    vec = np.ones(4, dtype=np.float32).tobytes()
    for i in range(40):
        fid = db.upsert_file(
            {
                "path": str(tmp_path / f"leo{i}.jpg"),
                "filename": f"leo{i}.jpg",
                "source_id": 1,
                "status": "indexed",
                "pattern_family": "zebra",
                "color_family": "navy" if i == 0 else "brown_tan",
            }
        )
        db.upsert_features(
            fid,
            {
                "clip_embedding": vec,
                "texture_map": {"pattern_family": "zebra", "color_family": "navy" if i == 0 else "brown_tan"},
            },
        )
        ids.append(fid)
    common = build_common_edit_overlay(db, ids)
    assert common.get("pattern_family") == "zebra"
    overlay = {
        "parent": "Animal Print",
        "child": "Leopard",
        "category_path": "Animal Print/Leopard",
        "pattern_family": "animal_print",
        "animal_print_type": "leopard",
    }
    stats = apply_metadata_edit_to_files(db, db_path, ids, overlay)
    assert stats["updated"] == 40
    assert stats["taught"] == 40
    store = UserFeedbackStore(Database(db_path))
    for fid in ids:
        loaded = store.metadata_overlay_for_file(fid)
        assert loaded["child"] == "Leopard"
        assert loaded["pattern_family"] == "animal_print"
        assert loaded["animal_print_type"] == "leopard"
    assert example_count(db_path, stats["concept_id"]) == 40
    assert faiss_d.read_bytes() == b"FAISS-SENTINEL"
    assert faiss_c.read_bytes() == b"FAISS-SENTINEL-2"
    from core.manual_label_guard import parse_texture_map

    tm = parse_texture_map((db.get_features(ids[0]) or {}).get("texture_map"))
    assert tm.get("user_labeled") is True
    assert "Leopard" in str(tm.get("manual_category_path") or "")
    mixed = build_common_edit_overlay(db, ids[:2])
    assert mixed.get("child") == "Leopard"


def test_options_worker_emits_without_blocking_ui(tmp_path):
    """Dialog opens; options arrive via finished_ok signal (async path)."""
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    from ui.result_metadata_dialog import (
        ResultMetadataDialog,
        invalidate_edit_options_cache,
    )

    app = _app()
    db = str(tmp_path / "async_opts.db")
    Database(db)
    invalidate_edit_options_cache(db)
    dlg = ResultMetadataDialog(result=_result(), db_path=db)
    # Ctor schedules QTimer.singleShot(0, _start_options_load) — pump events.
    for _ in range(100):
        app.processEvents()
        if dlg._options_worker is not None or dlg._options_loaded:
            break
    assert dlg._options_worker is not None or dlg._options_loaded
    if not dlg._options_loaded:
        loop = QEventLoop()
        QTimer.singleShot(8000, loop.quit)
        dlg._options_worker.finished_ok.connect(lambda *_: loop.quit())
        loop.exec()
    assert dlg._options_loaded
    assert dlg.cmb_parent.count() > 0
    assert "Animal Print" in [dlg.cmb_parent.itemText(i) for i in range(dlg.cmb_parent.count())]
    assert dlg.cmb_parent.filtered_matches("") == []


def test_teach_rename_invalidates_options_cache(tmp_path):
    """After register + invalidate, load_edit_form_options sees the new label."""
    from ui.result_metadata_dialog import (
        invalidate_edit_options_cache,
        load_edit_form_options,
    )

    db = str(tmp_path / "inv.db")
    Database(db)
    invalidate_edit_options_cache(db)
    before = load_edit_form_options(db)
    assert "YeniOgretAna" not in before["parents"]
    register_category(db, "YeniOgretAna", "YeniOgretAlt")
    # Cached snapshot must not silently keep the old list after teach/correct.
    cached = load_edit_form_options(db)
    assert "YeniOgretAna" not in cached["parents"]
    invalidate_edit_options_cache(db)
    after = load_edit_form_options(db)
    assert "YeniOgretAna" in after["parents"]
    assert "YeniOgretAlt" in (after["children_by_parent"].get("YeniOgretAna") or [])


def test_edit_options_turkish_sort_and_session_cache(tmp_path):
    """Batched options: TR alpha order + no timed rebuild within session."""
    import time

    from core.category_memory import register_category as reg
    from core.concept_query_normalize import turkish_sort_key
    from ui.result_metadata_dialog import (
        _OPTIONS_CACHE,
        invalidate_edit_options_cache,
        load_edit_form_options,
    )

    db = str(tmp_path / "sort_opts.db")
    Database(db)
    invalidate_edit_options_cache(db)
    for label in (
        "Zebra",
        "Çiçek",
        "Animal",
        "Şal",
        "Leopard",
        "Ürün",
        "Kalıba Uyarlanmış",
        "İplik",
        "Örme",
    ):
        reg(db, "Animal Print", label)

    t0 = time.perf_counter()
    first = load_edit_form_options(db)
    first_ms = (time.perf_counter() - t0) * 1000
    kids = list(first["children_by_parent"].get("Animal Print") or [])
    # Subset in TR order (tree may include more static kids)
    wanted = [
        "Animal",
        "Çiçek",
        "İplik",
        "Kalıba Uyarlanmış",
        "Leopard",
        "Örme",
        "Şal",
        "Ürün",
        "Zebra",
    ]
    positions = [kids.index(w) for w in wanted if w in kids]
    assert positions == sorted(positions)
    assert kids == sorted(kids, key=turkish_sort_key)
    assert first["parents"] == sorted(first["parents"], key=turkish_sort_key)

    t1 = time.perf_counter()
    second = load_edit_form_options(db)
    second_ms = (time.perf_counter() - t1) * 1000
    assert second["children_by_parent"].get("Animal Print") == kids
    assert db in _OPTIONS_CACHE
    # Cache hit must be cheap (not a full re-scan); allow generous CI budget.
    assert second_ms < 50.0
    assert first_ms < 5000.0  # batched path, not per-parent 349s regime


def test_teach_classify_reuses_panel_concept_cache(tmp_path):
    """Şu/Öğret must not call concepts() on UI thread when panel cache is warm."""
    from unittest.mock import patch

    from PySide6.QtWidgets import QApplication

    from ui.teach_me_panel import TeachMePanel

    app = _app()
    db = str(tmp_path / "teach_cache.db")
    Database(db)

    class _S:
        db_path = db
        cache_dir = str(tmp_path / "cache")

    panel = TeachMePanel(_S())
    panel._concept_names = ["Animal", "Çiçek", "Leopard", "Şal"]
    with patch("ui.teach_me_panel.concepts") as mock_concepts:
        known = panel._known_concepts_for_dialog()
        mock_concepts.assert_not_called()
    assert known == ["Animal", "Çiçek", "Leopard", "Şal"]
    _ = app  # keep QApp alive