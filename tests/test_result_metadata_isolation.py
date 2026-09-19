import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.db import Database
from core.search_memory import overlay_wrong_ids, record_feedback_overlay
from core.user_feedback import ACTION_WRONG, UserFeedbackStore
from ui.result_metadata_dialog import ResultMetadataDialog, form_values_from_record


def _app():
    app = QApplication.instance()
    return app or QApplication([])


def _result(fid: int, *, brand: str = "", family: str = "", color: str = "", debug=None):
    return SimpleNamespace(
        file_id=fid,
        pattern_family=family,
        color_family=color,
        debug=debug or ({"texture_map": {"brand_name": brand}} if brand else {}),
    )


def test_form_values_do_not_invent_brand():
    a = form_values_from_record(_result(1, brand="POLO"), {"brand": "POLO"})
    b = form_values_from_record(_result(2, brand=""), {})
    c = form_values_from_record(_result(3), {})
    assert a["brand"] == "POLO"
    assert b["brand"] == ""
    assert c["brand"] == ""
    assert b["parent"] == ""
    assert c["pattern_family"] == ""


def test_dialog_does_not_carry_polo_to_next_file(tmp_path):
    _app()
    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    a = _result(1, brand="POLO")
    dlg_a = ResultMetadataDialog(result=a, db_path=db_path, overlay={"brand": "POLO"})
    assert dlg_a.cmb_brand.current_text() == "POLO"
    dlg_a.cmb_brand.set_text("POLO")
    dlg_a._on_save()
    dlg_a.close()

    dlg_b = ResultMetadataDialog(result=_result(2), db_path=db_path, overlay={})
    assert dlg_b.cmb_brand.current_text() == ""
    assert dlg_b.cmb_parent.current_text() == ""
    dlg_b.cmb_brand.set_text("AMIRI")
    dlg_b._on_save()
    dlg_b.close()

    dlg_a2 = ResultMetadataDialog(
        result=_result(1, brand="POLO"), db_path=db_path, overlay={"brand": "POLO"}
    )
    assert dlg_a2.cmb_brand.current_text() == "POLO"
    dlg_a2.close()

    dlg_b2 = ResultMetadataDialog(
        result=_result(2, brand="AMIRI"), db_path=db_path, overlay={"brand": "AMIRI"}
    )
    assert dlg_b2.cmb_brand.current_text() == "AMIRI"
    dlg_b2.close()

    dlg_c = ResultMetadataDialog(result=_result(3), db_path=db_path, overlay={})
    assert dlg_c.cmb_brand.current_text() == ""
    dlg_c.close()


def test_save_clears_wrong_and_does_not_duplicate(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    db = Database(db_path)
    store = UserFeedbackStore(db)
    store.record("amiri", 7, ACTION_WRONG)
    record_feedback_overlay(db_path, "amiri", 7, "wrong")
    assert 7 in overlay_wrong_ids(db_path, "amiri")

    store.save_metadata_overlay(7, {"brand": "AMIRI"}, query_path="amiri")
    assert 7 not in overlay_wrong_ids(db_path, "amiri")
    store.save_metadata_overlay(7, {"brand": "AMIRI"}, query_path="amiri")

    with db.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM user_feedback WHERE result_file_id=7 AND action='metadata_edit'"
        ).fetchone()[0]
        wrong = conn.execute(
            "SELECT COUNT(*) FROM user_feedback WHERE result_file_id=7 AND action='wrong'"
        ).fetchone()[0]
    assert int(n) == 1
    assert int(wrong) == 0
    assert store.metadata_overlay_for_file(7).get("brand") == "AMIRI"


def test_teach_overlay_does_not_mutate_other_file(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    db = Database(db_path)
    store = UserFeedbackStore(db)
    store.save_metadata_overlay(1, {"brand": "POLO"}, query_path="q")
    store.save_metadata_overlay(2, {"brand": "AMIRI"}, query_path="q")
    assert store.metadata_overlay_for_file(1).get("brand") == "POLO"
    assert store.metadata_overlay_for_file(2).get("brand") == "AMIRI"
    assert store.metadata_overlay_for_file(3) == {}
