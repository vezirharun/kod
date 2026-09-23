"""Bana Öğret — havuz, toplu seçim, kalıcı kavram; Düzenle ayrı kalır."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.concept_registry import concepts, example_count, learn
from core.db import Database
from core.settings import AppSettings
from core.teach_me import (
    assign_pool,
    assign_undecided_pool,
    competing_strong_guesses,
    list_inbox,
    list_inbox_pools,
    match_clip_to_concepts,
    teach_files,
)
from ui.result_metadata_dialog import ResultMetadataDialog
from ui.teach_me_panel import TeachMePanel


def _app():
    return QApplication.instance() or QApplication([])


def _db(tmp_path: Path) -> Database:
    return Database(tmp_path / "patterns.db")


def _add(db: Database, tmp_path: Path, name: str, **fields) -> int:
    img = tmp_path / name
    img.write_bytes(b"x")
    rec = {
        "path": str(img),
        "filename": name,
        "source_id": 1,
        "status": "pending",
        "width": 8,
        "height": 8,
        "pattern_family": fields.get("pattern_family", ""),
        "pattern_confidence": fields.get("pattern_confidence", 0),
    }
    fid = int(db.upsert_file(rec))
    fam = str(fields.get("pattern_family") or "")
    if fam:
        with db.connect() as conn:
            conn.execute(
                "UPDATE files SET pattern_family=? WHERE id=?",
                (fam, fid),
            )
    tm = fields.get("texture_map")
    clip = fields.get("clip")
    payload = {}
    if tm is not None:
        payload["texture_map"] = tm
    if clip is not None:
        payload["clip_embedding"] = clip
    if payload:
        db.upsert_features(fid, payload)
    return fid


def test_assign_pool_undefined_and_suspicious():
    assert assign_pool({"pattern_family": ""}, {}) == (
        "undefined",
        "Anlamlı bir sınıf yok",
    )
    assert assign_pool(
        {"pattern_family": "floral", "pattern_confidence": 0.4},
        {"pattern_family": "floral", "classification_confidence": 0.4},
    ) == ("suspicious", "Güven düşük")
    assert assign_pool(
        {"pattern_family": "floral"},
        {
            "pattern_family": "floral",
            "pattern_dna": {"family": "paisley"},
            "classification_confidence": 0.9,
        },
    ) == ("suspicious", "Motorlar farklı sınıf diyor")
    assert (
        assign_pool(
            {"pattern_family": "floral"},
            {
                "pattern_family": "floral",
                "classification_confidence": 0.9,
                "user_labeled": True,
            },
        )
        is None
    )


def test_bulk_teach_persists_and_appends_same_concept(tmp_path):
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    vec = np.ones(8, dtype=np.float32).tobytes()
    a = _add(db, tmp_path, "a.jpg", pattern_family="", clip=vec)
    b = _add(db, tmp_path, "b.jpg", pattern_family="", clip=vec)
    path = str(tmp_path / "patterns.db")
    out = teach_files(db, path, [a, b], "Zincir")
    assert out["taught"] == 2
    cid = out["concept_id"]
    assert cid > 0
    assert example_count(path, cid) == 2
    c = _add(db, tmp_path, "c.jpg", pattern_family="", clip=vec)
    teach_files(db, path, [c], "zincir")
    assert example_count(path, cid) == 3
    rows = concepts(path)
    names = {str(r["canonical"]).lower() for r in rows}
    assert "zincir" in names
    assert learn(path, "Zincir", file_id=c) == cid


def test_inbox_excludes_taught_and_keeps_pools(tmp_path):
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    u = _add(db, tmp_path, "u.jpg", pattern_family="", pattern_confidence=0)
    s = _add(
        db,
        tmp_path,
        "s.jpg",
        pattern_family="floral",
        pattern_confidence=0.4,
        texture_map={"pattern_family": "floral", "classification_confidence": 0.4},
    )
    path = str(tmp_path / "patterns.db")
    undef = list_inbox(db, path, pool="undefined")
    sus = list_inbox(db, path, pool="suspicious")
    assert {c.file_id for c in undef} == {u}
    assert {c.file_id for c in sus} == {s}
    teach_files(db, path, [u], "Çanta")
    undef2 = list_inbox(db, path, pool="undefined")
    assert all(c.file_id != u for c in undef2)


def test_concepts_survive_reopen(tmp_path):
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    fid = _add(db, tmp_path, "x.jpg", pattern_family="")
    path = str(tmp_path / "patterns.db")
    teach_files(db, path, [fid], "Ayakkabı")
    again = concepts(path)
    assert any(str(r["canonical"]) == "Ayakkabı" for r in again)


def test_match_learned_clip_is_ready(tmp_path):
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    vec = np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
    fid = _add(db, tmp_path, "z.jpg", pattern_family="", clip=vec)
    path = str(tmp_path / "patterns.db")
    teach_files(db, path, [fid], "Zincir")
    hits = match_clip_to_concepts(path, vec)
    assert hits and hits[0]["canonical"] == "Zincir"
    assert hits[0]["strong"] is True


def test_panel_multi_select_and_teach(tmp_path):
    _app()
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    a = _add(db, tmp_path, "a.jpg", pattern_family="")
    b = _add(db, tmp_path, "b.jpg", pattern_family="")
    settings = AppSettings()
    settings.db_path = str(tmp_path / "patterns.db")
    panel = TeachMePanel(settings)
    panel.reload()
    assert panel.list_undefined.count() >= 2
    panel.list_undefined.selectAll()
    ids = panel.selected_ids()
    assert a in ids and b in ids
    panel._apply_teach(ids, "Zincir")
    # Async teach worker — drain until idle (offscreen Qt).
    import time
    from PySide6.QtWidgets import QApplication
    from core.qthread_lifecycle import qthread_is_running

    deadline = time.time() + 30
    app = QApplication.instance()
    while time.time() < deadline:
        app.processEvents()
        w = getattr(panel, "_teach_worker", None)
        if w is None or not qthread_is_running(w):
            app.processEvents()
            if getattr(panel, "_teach_worker", None) is None:
                break
        time.sleep(0.01)
    else:
        raise AssertionError("teach worker timeout")
    path = str(tmp_path / "patterns.db")
    assert any(str(r["canonical"]) == "Zincir" for r in concepts(path))
    panel.reload()
    remaining = [
        int(panel.list_undefined.item(i).data(Qt.ItemDataRole.UserRole) or 0)
        for i in range(panel.list_undefined.count())
    ]
    assert a not in remaining and b not in remaining


def test_edit_dialog_still_single_file_and_not_bulk_teach():
    _app()
    root = Path(__file__).resolve().parents[1]
    src = (root / "ui" / "result_metadata_dialog.py").read_text(encoding="utf-8")
    assert 'setWindowTitle("Düzenle — metadata")' in src
    assert "Bana Öğret" not in src
    dlg = ResultMetadataDialog()
    assert dlg.teach_apply() is False
    assert hasattr(dlg, "btn_teach")
    insp = (root / "ui" / "inspector_panel.py").read_text(encoding="utf-8")
    assert 'QPushButton("Düzenle")' in insp


def _tiny_png(path: Path) -> None:
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
            "53de0000000c4944415408d763f8cf00000101010000e1b5d13a0000000049"
            "454e44ae426082"
        )
    )


def test_large_preview_opens_and_closes(tmp_path):
    from ui.preview_dialog import ImagePreviewDialog

    _app()
    img = tmp_path / "p.png"
    _tiny_png(img)
    dlg = ImagePreviewDialog(str(img), title="Önizleme — p.png")
    assert "Önizleme" in dlg.windowTitle()
    assert dlg.lbl_image.minimumWidth() >= 640
    dlg.close()
    assert dlg.isVisible() is False


def test_panel_preview_uses_cache_path(tmp_path):
    _app()
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    preview = tmp_path / "cache.png"
    _tiny_png(preview)
    eps = tmp_path / "desen.eps"
    eps.write_text("%!PS")
    rec = {
        "path": str(eps),
        "filename": "desen.eps",
        "source_id": 1,
        "status": "pending",
        "pattern_family": "",
        "pattern_confidence": 0,
        "thumbnail_path": str(preview),
        "feature_preview_path": str(preview),
    }
    db.upsert_file(rec)
    settings = AppSettings()
    settings.db_path = str(tmp_path / "patterns.db")
    panel = TeachMePanel(settings)
    panel.reload()
    assert panel.list_undefined.count() >= 1
    item = panel.list_undefined.item(0)
    dlg = panel._open_preview(item, exec_dialog=False)
    assert dlg is not None
    assert str(preview) in (dlg._path,)
    dlg.close()


def test_teach_writes_classification_and_leaves_pools(tmp_path):
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    vec = np.ones(8, dtype=np.float32).tobytes()
    u = _add(db, tmp_path, "u.jpg", pattern_family="", clip=vec)
    s = _add(
        db,
        tmp_path,
        "s.jpg",
        pattern_family="floral",
        pattern_confidence=0.4,
        texture_map={
            "pattern_family": "floral",
            "classification_confidence": 0.4,
            "color_family": "navy",
        },
        clip=vec,
    )
    path = str(tmp_path / "patterns.db")
    overlay = {
        "parent": "Aksesuar",
        "child": "Zincir",
        "category_path": "Aksesuar/Zincir",
        "tags": ["zincir", "aksesuar"],
        "pattern_family": "geometric",
        "color_family": "gold",
    }
    out = teach_files(db, path, [u, s], "Zincir", overlay=overlay)
    assert out["taught"] == 2
    from core.concept_registry import example_count
    from core.manual_label_guard import is_manual_labeled, parse_texture_map

    assert example_count(path, out["concept_id"]) == 2
    for fid in (u, s):
        rec = db.get_file_by_id(fid)
        assert rec["category_path"] == "Aksesuar/Zincir"
        feat = db.get_features(fid)
        tm = parse_texture_map(feat.get("texture_map"))
        assert is_manual_labeled(tm)
        assert tm["user_tags"] == ["zincir", "aksesuar"]
        assert tm["pattern_family"] == "geometric"
    undef = list_inbox(db, path, pool="undefined")
    sus = list_inbox(db, path, pool="suspicious")
    left = {c.file_id for c in undef} | {c.file_id for c in sus}
    assert u not in left and s not in left
    hits = match_clip_to_concepts(path, vec)
    assert hits and hits[0]["canonical"] == "Zincir"


def test_empty_overlay_fields_do_not_wipe_existing(tmp_path):
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    fid = _add(
        db,
        tmp_path,
        "keep.jpg",
        pattern_family="",
        texture_map={"color_family": "navy", "brand_name": "Amiri"},
    )
    path = str(tmp_path / "patterns.db")
    teach_files(
        db,
        path,
        [fid],
        "Parfüm",
        overlay={"parent": "Aksesuar", "child": "Parfüm", "category_path": "Aksesuar/Parfüm"},
    )
    from core.manual_label_guard import parse_texture_map

    tm = parse_texture_map(db.get_features(fid)["texture_map"])
    assert tm.get("color_family") == "navy"
    assert tm.get("brand_name") == "Amiri"
    assert tm.get("category_path") == "Aksesuar/Parfüm"


def test_teach_dialog_reuses_edit_fields():
    from ui.teach_me_panel import TeachClassifyDialog

    _app()
    edit = ResultMetadataDialog()
    teach = TeachClassifyDialog(file_count=3, concepts=["Zincir"])
    for name in (
        "cmb_parent",
        "cmb_child",
        "cmb_family",
        "cmb_color",
        "cmb_brand",
        "txt_tag",
        "btn_save",
    ):
        assert hasattr(edit, name) and hasattr(teach, name)
    assert teach.btn_teach.isHidden()
    teach.cmb_parent.set_text("Aksesuar")
    teach._reload_children()
    teach.cmb_child.set_text("Zincir")
    teach.txt_tag.setText("zincir")
    teach._add_tag()
    vals = teach.values()
    assert vals["parent"] == "Aksesuar"
    assert vals["child"] == "Zincir"
    assert "zincir" in vals["tags"]
    teach.cmb_known.setCurrentIndex(teach.cmb_known.findData("Zincir"))
    assert teach.concept_label() == "Zincir"


def test_edit_learn_gate_ignores_surface_fields():
    from core.teach_me import is_meaningful_concept_correction

    before = {"pattern_family": "floral", "color_family": "navy", "brand": "Amiri"}
    assert (
        is_meaningful_concept_correction(
            before, {**before, "color_family": "red", "brand": "X"}
        )
        is False
    )
    assert (
        is_meaningful_concept_correction(
            before, {**before, "child": "Zincir", "parent": "Aksesuar"}
        )
        is True
    )


def test_summarize_shows_overwrite_risk():
    from core.teach_me import summarize_classification_changes

    snaps = [
        {
            "filename": "a.jpg",
            "parent": "Giyim",
            "child": "Gömlek",
            "category_path": "Giyim/Gömlek",
            "pattern_family": "floral",
            "color_family": "navy",
            "brand": "",
            "tags": [],
        }
    ]
    text = summarize_classification_changes(
        snaps,
        {"parent": "Aksesuar", "child": "Zincir", "category_path": "Aksesuar/Zincir"},
    )
    assert "Aksesuar" in text
    assert "a.jpg" in text
    assert "Giyim" in text


def test_assign_undecided_requires_two_strong_guesses():
    assert assign_undecided_pool({"pattern_family": ""}, {}) is None
    assert (
        assign_undecided_pool(
            {"pattern_family": "floral", "pattern_confidence": 0.9},
            {"pattern_family": "floral", "classification_confidence": 0.9},
        )
        is None
    )
    goi = {
        "global_object_intelligence": {
            "objects": [
                {"label": "kedi", "confidence": 0.72},
                {"label": "elma", "confidence": 0.64},
            ]
        }
    }
    hit = assign_undecided_pool({"pattern_family": ""}, goi)
    assert hit is not None
    assert hit[0] == "undecided"
    assert "kedi" in hit[1] and "elma" in hit[1]
    labeled = assign_undecided_pool(
        {"pattern_family": ""},
        {**goi, "user_labeled": True, "manual_category_path": "Animal/kedi"},
    )
    assert labeled is None
    rivals = competing_strong_guesses({"pattern_family": ""}, goi)
    assert {n for n, _c in rivals} >= {"kedi", "elma"}


def test_common_candidate_scores_intersection_min_conf():
    from core.teach_me import TeachMeCard
    from ui.teach_me_panel import _quick_pick_candidates

    a = TeachMeCard(
        file_id=1,
        filename="a.jpg",
        path="",
        preview_path="",
        pool="undecided",
        reason="",
        guess="a",
        confidence=0.5,
        rivals=[["Leopard", 0.6], ["Çiçek", 0.25], ["Elma", 0.1]],
    )
    b = TeachMeCard(
        file_id=2,
        filename="b.jpg",
        path="",
        preview_path="",
        pool="undecided",
        reason="",
        guess="b",
        confidence=0.5,
        rivals=[["Leopard", 0.8], ["Çiçek", 0.4], ["Kaplan", 0.3]],
    )
    c = TeachMeCard(
        file_id=3,
        filename="c.jpg",
        path="",
        preview_path="",
        pool="undecided",
        reason="",
        guess="c",
        confidence=0.5,
        rivals=[["Kaplan", 0.9]],
    )
    # Tekli = BU GÖRSEL NE? listesi
    single = _quick_pick_candidates([a])
    assert [n for n, _ in single][:3] == ["Leopard", "Çiçek", "Elma"]
    # Çoklu = aynı kaynakların kesişimi (yeni aday yok)
    common = _quick_pick_candidates([a, b])
    names = {n for n, _ in common}
    assert names == {"Leopard", "Çiçek"}
    by = dict(common)
    assert abs(by["Leopard"] - 0.6) < 1e-6
    assert abs(by["Çiçek"] - 0.25) < 1e-6
    assert _quick_pick_candidates([a, b, c]) == []


def test_inbox_card_attaches_undecided_rivals(tmp_path):
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    _add(
        db,
        tmp_path,
        "mix.jpg",
        pattern_family="floral",
        pattern_confidence=0.9,
        texture_map={
            "pattern_family": "floral",
            "classification_confidence": 0.9,
            "global_object_intelligence": {
                "objects": [
                    {"label": "çiçek", "confidence": 0.7},
                    {"label": "elma", "confidence": 0.55},
                ]
            },
        },
    )
    pools = list_inbox_pools(db, str(tmp_path / "patterns.db"), limit=20)
    undec = pools.get("undecided") or []
    assert undec
    card = undec[0]
    assert card.rivals
    assert len(card.rivals) >= 2
    assert all(isinstance(r, list) and len(r) >= 2 for r in card.rivals)


def test_undecided_inbox_and_teach_leaves_pool(tmp_path):
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    empty = _add(db, tmp_path, "empty.jpg", pattern_family="", pattern_confidence=0)
    undec = _add(
        db,
        tmp_path,
        "mix.jpg",
        pattern_family="floral",
        pattern_confidence=0.9,
        texture_map={
            "pattern_family": "floral",
            "classification_confidence": 0.9,
            "pattern_dna": {"family": "paisley", "confidence": 0.86},
        },
    )
    path = str(tmp_path / "patterns.db")
    undef = list_inbox(db, path, pool="undefined")
    sus = list_inbox(db, path, pool="suspicious")
    mix = list_inbox(db, path, pool="undecided")
    assert empty in {c.file_id for c in undef}
    assert empty not in {c.file_id for c in mix}
    assert undec in {c.file_id for c in mix}
    assert all(c.preview_path is not None for c in mix)
    teach_files(db, path, [undec], "Barok")
    mix2 = list_inbox(db, path, pool="undecided")
    assert all(c.file_id != undec for c in mix2)
    assert empty in {c.file_id for c in list_inbox(db, path, pool="undefined")}


def test_panel_undecided_tab_preview_and_teach(tmp_path):
    _app()
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    preview = tmp_path / "u.png"
    _tiny_png(preview)
    img = tmp_path / "mix.jpg"
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": "mix.jpg",
                "source_id": 1,
                "status": "pending",
                "pattern_family": "floral",
                "pattern_confidence": 0.9,
                "thumbnail_path": str(preview),
            }
        )
    )
    db.upsert_features(
        fid,
        {
            "texture_map": {
                "pattern_family": "floral",
                "classification_confidence": 0.9,
                "pattern_dna": {"family": "paisley", "confidence": 0.86},
            }
        },
    )
    settings = AppSettings()
    settings.db_path = str(tmp_path / "patterns.db")
    panel = TeachMePanel(settings)
    panel.reload()
    assert panel.tabs.count() == 4
    assert "Kararsızlar" in panel.tabs.tabText(2)
    assert "Yeni Kavram" in panel.tabs.tabText(3)
    assert panel.list_undecided.count() >= 1
    panel.tabs.setCurrentIndex(2)
    item = panel.list_undecided.item(0)
    from ui.teach_me_panel import _CandidateCardWidget

    cw = panel.list_undecided.itemWidget(item)
    assert isinstance(cw, _CandidateCardWidget)
    dlg = panel._open_preview(item, exec_dialog=False)
    assert dlg is not None
    dlg.close()
    panel.list_undecided.selectAll()
    ids = panel.selected_ids()
    assert fid in ids
    panel._apply_teach(ids, "Barok")
    remaining = [
        int(panel.list_undecided.item(i).data(Qt.ItemDataRole.UserRole) or 0)
        for i in range(panel.list_undecided.count())
    ]
    assert fid not in remaining


def test_smart_undecided_one_click_teaches_without_confirm(tmp_path):
    """Kart üzeri aday checkbox + Seçilenlere öğret → dialog yok; async teach."""
    from core.autonomous_learn import upsert_review
    from core.concept_registry import positives_for_file
    from PySide6.QtWidgets import QCheckBox
    from ui.teach_me_panel import _CandidateCardWidget

    _app()
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    img = tmp_path / "amb.jpg"
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": "amb.jpg",
                "source_id": 1,
                "status": "indexed",
            }
        )
    )
    path = str(tmp_path / "patterns.db")
    upsert_review(
        path,
        fid,
        lane="undecided",
        suggested="GÜL",
        confidence=0.9,
        rivals=[["GÜL", 0.91], ["DUDAK", 0.88]],
        reason="test_smart",
    )
    settings = AppSettings()
    settings.db_path = path
    panel = TeachMePanel(settings)
    panel.show()
    panel.reload(blocking=True)
    panel.tabs.setCurrentIndex(2)
    assert panel.list_undecided.count() >= 1
    assert panel.smart_undecided.isHidden()
    item = panel.list_undecided.item(0)
    cw = panel.list_undecided.itemWidget(item)
    assert isinstance(cw, _CandidateCardWidget)
    # Normal: aday paneli gizli
    assert not cw.cand_host.isVisible()
    item.setSelected(True)
    panel.list_undecided.setCurrentItem(item)
    panel._update_review_detail()
    assert cw.cand_host.isVisible()
    checks = [c for c in cw._checks if isinstance(c, QCheckBox)]
    assert checks
    assert any("GÜL" in c.text() for c in checks)
    for c in checks:
        if "GÜL" in c.text():
            c.setChecked(True)
            break

    taught: list[tuple[str, int]] = []
    panel.taught.connect(lambda n, c: taught.append((n, c)))
    panel._on_teach()
    remaining = [
        int(panel.list_undecided.item(i).data(Qt.ItemDataRole.UserRole) or 0)
        for i in range(panel.list_undecided.count())
    ]
    assert fid not in remaining
    app = _app()
    for _ in range(200):
        app.processEvents()
        if taught:
            break
        w = getattr(panel, "_teach_worker", None)
        if w is not None:
            w.wait(20)
    assert taught and "GÜL" in taught[0][0]
    pos = positives_for_file(path, fid)
    assert any(str(p.get("canonical") or "") == "GÜL" for p in pos)


def test_candidate_card_hover_only_expands_that_card(tmp_path):
    from core.autonomous_learn import upsert_review
    from ui.teach_me_panel import _CandidateCardWidget

    _app()
    db = _db(tmp_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    path = str(tmp_path / "patterns.db")
    ids = []
    for name, rivals in (
        ("a.jpg", [["animal", 0.5], ["deer", 0.4]]),
        ("b.jpg", [["animal", 0.6], ["bird", 0.3]]),
    ):
        img = tmp_path / name
        img.write_bytes(b"x")
        fid = int(
            db.upsert_file(
                {
                    "path": str(img),
                    "filename": name,
                    "source_id": 1,
                    "status": "indexed",
                }
            )
        )
        upsert_review(
            path,
            fid,
            lane="suspicious",
            suggested=rivals[0][0],
            confidence=0.5,
            rivals=rivals,
            reason="t",
        )
        ids.append(fid)
    settings = AppSettings()
    settings.db_path = path
    panel = TeachMePanel(settings)
    panel.show()
    panel.reload(blocking=True)
    panel.tabs.setCurrentIndex(1)
    assert panel.list_suspicious.count() >= 2
    w0 = panel.list_suspicious.itemWidget(panel.list_suspicious.item(0))
    w1 = panel.list_suspicious.itemWidget(panel.list_suspicious.item(1))
    assert isinstance(w0, _CandidateCardWidget)
    assert isinstance(w1, _CandidateCardWidget)
    assert not w0.cand_host.isVisible()
    assert not w1.cand_host.isVisible()
    # Clean card meta: no Öneri / Güven / Neden / aday satırları
    meta0 = w0.lbl_meta.text()
    assert "Öneri:" not in meta0
    assert "Güven" not in meta0
    assert "deer %" not in meta0
    assert meta0.strip().startswith("a.jpg") or "a.jpg" in meta0.split("\n")[0]
    panel._on_card_widget_hover(w0, True)
    assert w0.cand_host.isVisible()
    assert not w1.cand_host.isVisible()
    panel._on_card_widget_hover(w0, False)
    panel._on_card_widget_hover(w1, True)
    assert not w0.cand_host.isVisible()
    assert w1.cand_host.isVisible()
    # Select both → both expanded
    panel.list_suspicious.selectAll()
    panel._update_review_detail()
    assert w0.cand_host.isVisible() and w1.cand_host.isVisible()
    # Bottom review must NOT dump global candidate panel fields
    review = panel.lbl_review.text()
    assert "Önerilen kavram" not in review
    assert "Alternatifler:" not in review
    assert "Neden:" not in review
    assert panel.smart_undecided.isHidden()


def test_suspicious_card_list_text_is_clean():
    from core.teach_me import TeachMeCard
    from ui.teach_me_panel import _card_list_text

    card = TeachMeCard(
        file_id=1,
        filename="x.png",
        path="/x.png",
        preview_path="",
        pool="suspicious",
        reason="Çelişen kavram: A vs B",
        guess="animal",
        suggested="animal",
        confidence=0.52,
        rivals=[["animal", 0.52], ["deer", 0.38]],
    )
    text = _card_list_text(card, show_candidates=False)
    assert text.strip() == "x.png"
    assert "Öneri" not in text
    assert "Güven" not in text
    assert "Çelişen" not in text
    assert "animal" not in text


def test_keep_host_visible_restores_hidden_dock(tmp_path):
    from PySide6.QtWidgets import QDockWidget, QMainWindow

    _app()
    settings = AppSettings()
    settings.db_path = str(tmp_path / "patterns.db")
    win = QMainWindow()
    dock = QDockWidget("Bana Öğret", win)
    panel = TeachMePanel(settings)
    dock.setWidget(panel)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    win.show()
    dock.show()
    dock.hide()
    assert dock.isHidden() is True
    panel._keep_host_visible()
    assert dock.isHidden() is False
