from core.index_v3 import ui_bridge

def test_ai_gate_is_preview_only():
    gate = ui_bridge._AI_GATE
    assert "physical_preview_ready" in gate
    assert "physical_thumbnail_ready" not in gate

def test_ai_stage_sql_does_not_require_thumbnail():
    assert "physical_thumbnail_ready" not in ui_bridge._V3_DINO
    assert "physical_thumbnail_ready" not in ui_bridge._V3_CLIP
    assert "physical_thumbnail_ready" not in ui_bridge._V3_TEX
    assert "physical_thumbnail_ready" not in ui_bridge._V3_SEM
    assert "physical_thumbnail_ready" not in ui_bridge._V3_DNA
    assert "physical_thumbnail_ready" not in ui_bridge._V3_PATCH
    assert "physical_thumbnail_ready" not in ui_bridge._V3_OCR
