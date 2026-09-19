from core.category_tree import infer_category_path, resolve_category_query, pattern_fields_for_path
from core.category_predictions import category_predictions_from_metadata


def test_brand_query_resolves_even_when_pattern_is_monogram():
    assert resolve_category_query('dior').category_path == 'Marka/Christian Dior'
    assert resolve_category_query('amiri').category_path == 'Marka/Amiri'


def test_brand_metadata_surfaces_brand_and_keeps_pattern_family():
    tm = {
        'pattern_family': 'monogram_logo',
        'brand_name': 'christian dior',
    }
    assert infer_category_path(tm) == 'Marka/Christian Dior'
    preds = category_predictions_from_metadata(
        'monogram_logo', '', {'brand_name': 'christian dior', 'pattern_dna': {'family': 'monogram_logo', 'confidence': .9}}
    )
    assert preds[0]['category_path'] == 'Marka/Christian Dior'
    assert any(p['category_path'] == 'Monogram Logo' for p in preds)


def test_person_subject_is_an_independent_manual_label_axis():
    fields = pattern_fields_for_path('Kişi/Ünlü/Kenan İmirzalıoğlu')
    assert fields['pattern_family'] == 'person_subject'
    assert fields['pattern_subtype'] == 'kenan imirzalioglu'
