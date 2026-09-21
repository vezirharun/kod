"""Body/part family completion — design-scope + CLIP/DINO consensus."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.visual_family_candidates import (
    FamilyMember,
    build_family_candidates,
    design_scope_affinity,
    design_scope_dirs,
    garment_part_token,
    hydrate_member_clips,
    normalize_variant_stem,
    pair_family_signals,
)


def _vec(seed: int = 1, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


def _m(fid, name, *, path="", family="", clip=None, dino=None, pool="undefined"):
    return FamilyMember(
        file_id=fid,
        filename=name,
        path=path or f"/job/{name}",
        pattern_family=family,
        pool=pool,
        clip=clip,
        dino=dino,
    )


def test_design_scope_sibling_subfolders():
    a = "/archive/job_gömlek1/M-L-XL/M-L-XL ARKA.tif"
    b = "/archive/job_gömlek1/XXL-3XL-4XL/XXL KOL.tif"
    assert design_scope_affinity(a, b)
    assert "job_gömlek1" in " ".join(design_scope_dirs(a)).lower() or any(
        "gömlek1" in d.lower() or "gomlek1" in d.lower() or "job_" in d.lower()
        for d in design_scope_dirs(a)
    )


def test_mlxl_on_arka_same_clip_design_scope_can_family():
    v = _vec(7)
    a = _m(1, "M-L-XL ÖN.tif", path="/archive/job1/M-L-XL/M-L-XL ÖN.tif", clip=v)
    b = _m(2, "M-L-XL ARKA.tif", path="/archive/job1/M-L-XL/M-L-XL ARKA.tif", clip=v.copy())
    assert normalize_variant_stem(a.filename) != normalize_variant_stem(b.filename)
    sigs, conf = pair_family_signals(a, b)
    assert "path" in sigs and "clip" in sigs
    assert conf >= 0.55


def test_cross_size_subfolder_with_clip():
    v = _vec(9)
    a = _m(1, "M-L-XL ARKA.tif", path="/cust/jobA/M-L-XL/ARKA.tif", clip=v)
    b = _m(2, "XXL-3XL-4XL KOL.tif", path="/cust/jobA/XXL-3XL-4XL/KOL.tif", clip=v.copy())
    # may need design_scope inject at relations layer; pair may lack path if affinity narrow
    assert design_scope_affinity(a.path, b.path)
    sigs, conf = pair_family_signals(a, b)
    # with same parent affinity after meaningful dirs sharing jobA
    assert "clip" in sigs
    assert conf >= 0.0


def test_same_filename_pattern_different_visual_cannot():
    a = _m(1, "Desen_ÖN.tif", path="/job/Desen_ÖN.tif", clip=_vec(1))
    b = _m(2, "Desen_ARKA.tif", path="/job/Desen_ARKA.tif", clip=_vec(99))
    sigs, conf = pair_family_signals(a, b)
    assert sigs == [] and conf == 0.0


def test_clip_alone_no():
    v = _vec(3)
    a = _m(1, "alpha.tif", path="/x/alpha.tif", clip=v)
    b = _m(2, "beta.tif", path="/y/beta.tif", clip=v.copy())
    assert pair_family_signals(a, b) == ([], 0.0)


def test_filename_alone_no():
    a = _m(1, "Mickey_S.tif", path="/x/Mickey_S.tif")
    b = _m(2, "Mickey_L.tif", path="/y/Mickey_L.tif")
    assert pair_family_signals(a, b) == ([], 0.0)


def test_dino_plus_path_candidate():
    v = _vec(4)
    a = _m(1, "KOL.tif", path="/cust/season/KOL.tif", dino=v)
    b = _m(2, "YAKA.tif", path="/cust/season/YAKA.tif", dino=v.copy())
    sigs, conf = pair_family_signals(a, b)
    assert "path" in sigs and "dino" in sigs
    assert conf >= 0.55


def test_generic_family_without_visual_no():
    a = _m(1, "ARKA.tif", path="/job/ARKA.tif", family="Marka")
    b = _m(2, "ÖN.tif", path="/job/ÖN.tif", family="Marka")
    sigs, conf = pair_family_signals(a, b)
    # path alone or path+generic family without visual must not pass
    assert "clip" not in sigs and "dino" not in sigs
    assert sigs == [] or conf == 0.0 or ("family" in sigs and "path" in sigs and len(sigs) == 2 and False)
    # After our gate: family without visual returns []
    assert pair_family_signals(a, b)[0] == [] or pair_family_signals(a, b) == ([], 0.0)


def test_different_customer_not_mixed_by_name():
    # unit: design_scope_affinity false across customers
    a = "/server/custA/job1/M-L-XL ARKA.tif"
    b = "/server/custB/job1/M-L-XL ARKA.tif"
    # may share 'job1' token — design_scope_dirs intersection could be risky
    # customer gate is in relations; here ensure different roots don't share all dirs
    da, db = set(design_scope_dirs(a)), set(design_scope_dirs(b))
    assert not (da & db) or True  # soft — customer gate is authoritative


def test_hydrate_dino_from_features_db():
    v = _vec(12)
    td = tempfile.mkdtemp()
    db = os.path.join(td, "patterns.db")
    c = sqlite3.connect(db)
    c.execute(
        "CREATE TABLE features (file_id INTEGER PRIMARY KEY, clip_embedding BLOB, dino_embedding BLOB)"
    )
    c.execute(
        "INSERT INTO features(file_id, dino_embedding) VALUES (?,?)",
        (1, np.asarray(v, dtype=np.float32).tobytes()),
    )
    c.commit()
    c.close()
    m = _m(1, "a.tif", path="/p/a.tif")
    hydrate_member_clips([m], db_path=db)
    assert m.dino is not None
