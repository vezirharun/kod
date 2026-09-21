"""Visual family candidates — multi-signal Teach Me grouping."""
from __future__ import annotations

from pathlib import Path

from core.teach_me import TeachMeCard
from core.visual_family_candidates import (
    FamilyMember,
    build_family_candidates,
    collapse_pools_with_families,
    expand_member_ids,
    is_rejected_family,
    normalize_variant_stem,
    pair_family_signals,
    record_rejected_family,
    reject_and_split_family,
)


def _m(
    fid: int,
    filename: str,
    *,
    path: str = "",
    guess: str = "",
    pattern_family: str = "",
    category: str = "",
    pool: str = "undefined",
) -> FamilyMember:
    return FamilyMember(
        file_id=fid,
        filename=filename,
        path=path or f"D:/patterns/{filename}",
        guess=guess,
        pattern_family=pattern_family,
        category=category,
        pool=pool,
    )


def test_stem_sml_xl_and_final():
    assert normalize_variant_stem("Mickey_S.tif") == normalize_variant_stem("Mickey_XL.eps")
    assert normalize_variant_stem("Mickey_final.eps") == normalize_variant_stem("Mickey_M.png")
    assert normalize_variant_stem("Mickey_S.tif") == "mickey"


def test_same_stem_path_forms_family():
    members = [
        _m(1, "Mickey_S.tif", path="D:/a/Mickey/Mickey_S.tif", pattern_family="logo"),
        _m(2, "Mickey_M.tif", path="D:/a/Mickey/Mickey_M.tif", pattern_family="logo"),
        _m(3, "Mickey_L.tif", path="D:/a/Mickey/Mickey_L.tif", pattern_family="logo"),
        _m(4, "Mickey_final.eps", path="D:/a/Mickey/Mickey_final.eps", pattern_family="logo"),
    ]
    cands = build_family_candidates(members)
    assert len(cands) == 1
    assert set(cands[0].member_ids) == {1, 2, 3, 4}
    assert cands[0].confidence >= 0.55
    assert "stem" in cands[0].signals
    assert "path" in cands[0].signals


def test_different_folders_same_stem_still_groups_with_family():
    members = [
        _m(1, "Zebra_S.ai", path="D:/old/Zebra_S.ai", pattern_family="zebra"),
        _m(2, "Zebra_XL.ai", path="D:/new/exports/Zebra_XL.ai", pattern_family="zebra"),
    ]
    cands = build_family_candidates(members)
    assert len(cands) == 1
    assert set(cands[0].member_ids) == {1, 2}


def test_tiger_ne_leopard():
    a = _m(1, "spot_a.tif", guess="Kaplan", pattern_family="tiger")
    b = _m(2, "spot_b.tif", guess="Leopar", pattern_family="leopard")
    sigs, conf = pair_family_signals(a, b)
    assert sigs == []
    assert conf == 0.0
    assert build_family_candidates([a, b]) == []


def test_kaplan_leopar_turkish_guess_only():
    a = _m(1, "spot_a.tif", guess="Kaplan")
    b = _m(2, "spot_b.tif", guess="Leopar")
    sigs, conf = pair_family_signals(a, b)
    assert sigs == []
    assert conf == 0.0


def test_flower_ne_rose():
    a = _m(1, "bloom_1.tif", guess="Çiçek", pattern_family="floral")
    b = _m(2, "bloom_2.tif", guess="Gül", pattern_family="rose")
    cands = build_family_candidates([a, b])
    # Must not merge distinct floral leaves; stems also differ (bloom_1 vs bloom_2)
    assert cands == [] or all(set(c.member_ids) != {1, 2} for c in cands)


def test_animal_vs_textile_leopard():
    a = _m(
        1,
        "leo_photo.jpg",
        path="D:/photos/animal/leo_photo.jpg",
        guess="Leopard",
        pattern_family="leopard",
        category="animal/wildlife",
    )
    b = _m(
        2,
        "leo_print.tif",
        path="D:/fabrics/textile/leo_print.tif",
        guess="Leopard",
        pattern_family="leopard",
        category="textile/fabric",
    )
    sigs, conf = pair_family_signals(a, b)
    assert sigs == []
    assert conf == 0.0


def test_clip_alone_no_family():
    import numpy as np

    v = np.ones(8, dtype=np.float32)
    # Distinct folders — path affinity must not fire; CLIP alone is insufficient.
    a = _m(1, "alpha.tif", path="/x/alpha.tif", guess="x", pattern_family="unknown")
    b = _m(2, "beta.tif", path="/y/beta.tif", guess="y", pattern_family="unknown")
    a.clip = v
    b.clip = v.copy()
    sigs, conf = pair_family_signals(a, b)
    assert sigs == []
    assert conf == 0.0
    assert build_family_candidates([a, b]) == []


def test_low_conf_no_candidate():
    a = _m(1, "foo.tif", path="D:/mix/foo.tif")
    b = _m(2, "bar.tif", path="D:/mix/bar.tif")
    assert build_family_candidates([a, b]) == []


def test_reject_prevents_reform(tmp_path: Path):
    db = str(tmp_path / "mem.db")
    members = [
        _m(10, "Logo_S.tif", path="D:/L/Logo_S.tif", pattern_family="logo"),
        _m(11, "Logo_M.tif", path="D:/L/Logo_M.tif", pattern_family="logo"),
    ]
    cands = build_family_candidates(members, db_path=db)
    assert len(cands) == 1
    record_rejected_family(db, cands[0].member_ids, label="Logo")
    assert is_rejected_family(db, [10, 11])
    assert build_family_candidates(members, db_path=db) == []


def test_collapse_no_duplicate_and_expand():
    cards = [
        TeachMeCard(
            file_id=1,
            filename="Mickey_S.tif",
            path="D:/m/Mickey_S.tif",
            preview_path="",
            pool="undefined",
            reason="x",
            guess="?",
            confidence=0.4,
            pattern_family="logo",
        ),
        TeachMeCard(
            file_id=2,
            filename="Mickey_XL.tif",
            path="D:/m/Mickey_XL.tif",
            preview_path="",
            pool="undefined",
            reason="x",
            guess="?",
            confidence=0.4,
            pattern_family="logo",
        ),
        TeachMeCard(
            file_id=99,
            filename="Other.tif",
            path="D:/m/Other.tif",
            preview_path="",
            pool="undefined",
            reason="x",
            guess="?",
            confidence=0.4,
        ),
    ]
    pools = collapse_pools_with_families({"undefined": cards}, db_path="")
    undefined = pools["undefined"]
    ids = [int(c.file_id) for c in undefined]
    assert len(ids) == len(set(ids))
    fam = [c for c in undefined if int(getattr(c, "cluster_size", 1) or 1) > 1]
    assert len(fam) == 1
    assert set(fam[0].member_ids) == {1, 2}
    expanded = expand_member_ids(fam[0], [fam[0].file_id])
    assert set(expanded) == {1, 2}


def test_reject_split_does_not_dismiss(tmp_path: Path):
    db = str(tmp_path / "mem.db")
    card = TeachMeCard(
        file_id=5,
        filename="A_S.tif",
        path="D:/A_S.tif",
        preview_path="",
        pool="undefined",
        reason="family",
        guess="A",
        confidence=0.7,
        cluster_size=2,
        member_ids=[5, 6],
        suggested="A",
    )
    members = reject_and_split_family(db, card)
    assert set(members) == {5, 6}
    assert is_rejected_family(db, [5, 6])


def test_new_member_alone_no_family():
    alone = build_family_candidates(
        [_m(3, "X_L.tif", path="D:/X/X_L.tif", pattern_family="x")]
    )
    assert alone == []
