
from core.unified_intelligence_contracts import Evidence, QueryRequirement
from core.unified_decision_engine_v13 import decide

def test_entity_dominates_pattern():
    d=decide([Evidence("entity","kuş",.94),Evidence("pattern","kuş",.40)])
    assert d.accepted and d.label=="kuş"

def test_pattern_cannot_fake_entity():
    d=decide([Evidence("pattern","kuş",.98)],
             [QueryRequirement("kuş","entity",True,.72)])
    assert not d.accepted

def test_strong_contradiction_rejected():
    d=decide([Evidence("entity","kadın",.92),Evidence("entity","erkek",.90)])
    assert d.contradictory and not d.accepted

def test_negative_user_feedback_reduces():
    d=decide([Evidence("entity","kedi",.92),Evidence("user_negative","kedi",.95,"negative")])
    assert not d.accepted

def test_required_concept_gate():
    d=decide([Evidence("entity","togg",.91),Evidence("entity","suv",.40)],
             [QueryRequirement("togg","brand",True,.72),QueryRequirement("suv","vehicle_type",True,.72)])
    assert not d.accepted and "suv" in d.missing

def test_multiple_independent_sources_raise_confidence():
    a=decide([Evidence("entity","şahin",.84)])
    b=decide([Evidence("entity","şahin",.84),Evidence("object","şahin",.83),Evidence("face","şahin",.80)])
    assert b.confidence>a.confidence

def test_brand_requires_brand_evidence():
    d=decide([Evidence("pattern","togg",.99)],
             [QueryRequirement("togg","brand",True,.72)])
    assert not d.accepted

def test_entity_evidence_can_pass_entity_gate():
    d=decide([Evidence("entity","şahin",.90)],
             [QueryRequirement("şahin","entity",True,.72)])
    assert d.accepted

def test_low_margin_is_not_accepted():
    d=decide([Evidence("entity","kadın",.80),
              Evidence("entity","erkek",.75)])
    assert not d.accepted


def test_duplicate_same_source_does_not_fake_independence():
    one=decide([Evidence("entity","şahin",.84)])
    dup=decide([Evidence("entity","şahin",.84), Evidence("entity","şahin",.84), Evidence("entity","şahin",.84)])
    assert dup.confidence == one.confidence

def test_engine_wrapper_is_available():
    from core.unified_decision_engine_v13 import UnifiedDecisionEngine
    d=UnifiedDecisionEngine().decide([Evidence("entity","kuş",.94)])
    assert d.accepted and d.label == "kuş"

def test_unknown_source_is_conservative():
    d=decide([Evidence("unknown_source","kuş",.94)])
    assert not d.accepted

def test_negative_evidence_cannot_create_winner():
    d=decide([Evidence("user_negative","kuş",.99,"negative")])
    assert not d.accepted and not d.label
