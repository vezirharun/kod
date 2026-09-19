"""BU DESENİN DİĞERLERİ — pattern_relations unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.duplicate_detection import DUP_COLOR, DUP_EXACT, DUP_NEAR, DUP_SCALE
from core.pattern_relations import (
    SECTION_FAMILY,
    SECTION_SAME,
    SECTION_SIMILAR,
    SECTION_VARIANT,
    get_pattern_relations,
    invalidate_pattern_relations_cache,
)
from core.visual_family_candidates import (
    FamilyMember,
    record_rejected_family,
)


class _FakeDB:
    """Minimal DB stand-in for relation layer (no Index/Search side effects)."""

    def __init__(self) -> None:
        self.files: dict[int, dict] = {}
        self.groups: dict[int, dict] = {}
        self.members: dict[int, list[dict]] = {}
        self._file_group: dict[int, int] = {}

    def add_file(self, fid: int, **kwargs) -> None:
        path = kwargs.get("path") or f"D:/c1/p/{kwargs.get('filename', f'{fid}.tif')}"
        tm = kwargs.get("texture_map") or {}
        self.files[fid] = {
            "id": fid,
            "path": path,
            "filename": kwargs.get("filename") or Path(path).name,
            "customer": kwargs.get("customer", ""),
            "partial_hash": kwargs.get("partial_hash", ""),
            "phash": kwargs.get("phash", ""),
            "dhash": kwargs.get("dhash", ""),
            "whash": kwargs.get("whash", ""),
            "texture_map": tm,
            "pattern_family": kwargs.get("pattern_family", tm.get("pattern_family", "")),
            "color_family": kwargs.get("color_family", tm.get("color_family", "")),
            "status": "indexed",
        }

    def link_group(self, gid: int, members: list[tuple[int, str, float]], label: str = "F") -> None:
        self.groups[gid] = {"id": gid, "label": label}
        self.members[gid] = []
        for fid, rel, score in members:
            self.members[gid].append(
                {"file_id": fid, "relation": rel, "score": score, "filename": "", "path": ""}
            )
            self._file_group[fid] = gid

    def get_indexed_files_by_ids(self, file_ids, **kwargs):
        return [dict(self.files[i]) for i in file_ids if i in self.files]

    def get_file_by_id(self, file_id):
        return dict(self.files[file_id]) if file_id in self.files else None

    def get_features(self, file_id):
        f = self.files.get(file_id)
        if not f:
            return None
        return {
            "phash": f.get("phash", ""),
            "dhash": f.get("dhash", ""),
            "whash": f.get("whash", ""),
            "texture_map": f.get("texture_map") or {},
        }

    def get_pattern_group_for_file(self, file_id):
        gid = self._file_group.get(int(file_id))
        return dict(self.groups[gid]) if gid in self.groups else None

    def list_pattern_group_members(self, group_id):
        return list(self.members.get(int(group_id), []))

    def connect(self):
        raise RuntimeError("fake db has no sqlite connect")


@pytest.fixture(autouse=True)
def _clear_rel_cache():
    invalidate_pattern_relations_cache()
    yield
    invalidate_pattern_relations_cache()


def test_exact_dups_exclude_source():
    db = _FakeDB()
    db.add_file(1, filename="a.tif", path="D:/c1/a.tif", phash="aaaaaaaaaaaaaaaa", partial_hash="P1")
    db.add_file(2, filename="a_copy.tif", path="D:/c1/a_copy.tif", phash="aaaaaaaaaaaaaaaa", partial_hash="P1")
    db.link_group(10, [(1, "representative", 1.0), (2, DUP_EXACT, 0.99)])
    bundle = get_pattern_relations(db, 1, use_cache=False, stage="meta")
    ids = [i.file_id for s in bundle.sections for i in s.items]
    assert 1 not in ids
    assert 2 in ids
    same = next(s for s in bundle.sections if s.key == SECTION_SAME)
    assert same.count == 1


def test_same_pattern_different_folders_customers():
    db = _FakeDB()
    db.add_file(
        1,
        filename="logo.tif",
        path="D:/MusteriA/logo.tif",
        customer="MusteriA",
        phash="bbbbbbbbbbbbbbbb",
        partial_hash="PX",
    )
    db.add_file(
        2,
        filename="logo.tif",
        path="D:/MusteriB/exports/logo.tif",
        customer="MusteriB",
        phash="bbbbbbbbbbbbbbbb",
        partial_hash="PX",
    )
    db.link_group(11, [(1, "representative", 1.0), (2, DUP_EXACT, 0.99)])
    bundle = get_pattern_relations(db, 1, use_cache=False, stage="meta")
    item = next(i for s in bundle.sections for i in s.items if i.file_id == 2)
    assert item.section == SECTION_SAME
    assert item.customer == "MusteriB"


def test_variants_in_right_bucket():
    db = _FakeDB()
    db.add_file(1, filename="Mickey_S.tif", path="D:/a/Mickey_S.tif", pattern_family="logo")
    db.add_file(
        2,
        filename="Mickey_M.tif",
        path="D:/a/Mickey_M.tif",
        pattern_family="logo",
        color_family="red",
    )
    db.add_file(
        3,
        filename="Mickey_L.tif",
        path="D:/a/Mickey_L.tif",
        pattern_family="logo",
        color_family="blue",
    )
    db.link_group(
        12,
        [
            (1, "representative", 1.0),
            (2, DUP_SCALE, 0.88),
            (3, DUP_COLOR, 0.86),
        ],
    )
    bundle = get_pattern_relations(db, 1, use_cache=False, stage="meta")
    by_sec = {s.key: {i.file_id for i in s.items} for s in bundle.sections}
    assert 2 in by_sec.get(SECTION_VARIANT, set())
    assert 3 in by_sec.get(SECTION_VARIANT, set())
    assert 2 not in by_sec.get(SECTION_SAME, set())


def test_tiger_ne_leopard_not_same():
    db = _FakeDB()
    db.add_file(
        1,
        filename="t1.tif",
        path="D:/x/t1.tif",
        pattern_family="tiger",
        texture_map={"pattern_family": "tiger", "guess": "Kaplan"},
    )
    db.add_file(
        2,
        filename="l1.tif",
        path="D:/x/l1.tif",
        pattern_family="leopard",
        texture_map={"pattern_family": "leopard", "guess": "Leopar"},
    )
    db.link_group(13, [(1, "representative", 1.0), (2, DUP_NEAR, 0.97)])
    bundle = get_pattern_relations(db, 1, use_cache=False, stage="meta")
    ids = [i.file_id for s in bundle.sections for i in s.items]
    assert 2 not in ids


def test_animal_ne_textile_domain():
    from core.visual_family_candidates import _domain_bucket

    animal = FamilyMember(
        file_id=1,
        filename="leo_photo.jpg",
        path="D:/photos/animal/leo.jpg",
        category="animal",
        pattern_family="leopard",
        guess="Leopar",
    )
    textile = FamilyMember(
        file_id=2,
        filename="leo_print.tif",
        path="D:/fabric/textile/leo_print.tif",
        category="textile",
        pattern_family="leopard",
        guess="Leopar",
    )
    assert _domain_bucket(animal) != _domain_bucket(textile) or (
        _domain_bucket(animal) and _domain_bucket(textile)
    )
    db = _FakeDB()
    db.add_file(
        1,
        filename="leo_photo.jpg",
        path="D:/photos/animal/leo.jpg",
        texture_map={"category": "animal", "pattern_family": "leopard", "guess": "Leopar"},
    )
    db.add_file(
        2,
        filename="leo_print.tif",
        path="D:/fabric/textile/leo_print.tif",
        texture_map={"category": "textile", "pattern_family": "leopard", "guess": "Leopar"},
    )
    db.link_group(14, [(1, "representative", 1.0), (2, DUP_EXACT, 0.99)])
    bundle = get_pattern_relations(db, 1, use_cache=False, stage="meta")
    ids = [i.file_id for s in bundle.sections for i in s.items]
    assert 2 not in ids


def test_dedupe_across_sections():
    db = _FakeDB()
    db.add_file(1, filename="x.tif", path="D:/a/x.tif", partial_hash="Z1", phash="cccccccccccccccc")
    db.add_file(2, filename="x2.tif", path="D:/a/x2.tif", partial_hash="Z1", phash="cccccccccccccccc")
    db.link_group(15, [(1, "representative", 1.0), (2, DUP_EXACT, 0.99)])
    bundle = get_pattern_relations(db, 1, use_cache=False, stage="meta")
    ids = [i.file_id for s in bundle.sections for i in s.items]
    assert ids.count(2) == 1


def test_near_not_in_ayni():
    db = _FakeDB()
    db.add_file(1, filename="a.tif", path="D:/a/a.tif", pattern_family="logo")
    db.add_file(2, filename="b.tif", path="D:/a/b.tif", pattern_family="logo")
    db.link_group(16, [(1, "representative", 1.0), (2, DUP_NEAR, 0.85)])
    bundle = get_pattern_relations(db, 1, use_cache=False, stage="meta")
    by_sec = {s.key: {i.file_id for i in s.items} for s in bundle.sections}
    assert 2 not in by_sec.get(SECTION_SAME, set())
    # May land in benzer or variant via stem — not AYNI
    if SECTION_SIMILAR in by_sec:
        assert 2 in by_sec[SECTION_SIMILAR] or 2 in by_sec.get(SECTION_VARIANT, set())


def test_taught_family_and_banned(tmp_path: Path):
    db_path = str(tmp_path / "t.db")
    # Minimal concept_examples schema via concept_registry helpers
    from core.concept_registry import learn

    learn(db_path, "Mickey Logo", file_id=1, file_path="D:/a/m1.tif", source="user")
    learn(db_path, "Mickey Logo", file_id=2, file_path="D:/a/m2.tif", source="user")
    learn(db_path, "Mickey Logo", file_id=3, file_path="D:/a/m3.tif", source="user")

    db = _FakeDB()
    db.add_file(1, filename="m1.tif", path="D:/a/m1.tif")
    db.add_file(2, filename="m2.tif", path="D:/a/m2.tif")
    db.add_file(3, filename="m3.tif", path="D:/a/m3.tif")

    bundle = get_pattern_relations(db, 1, db_path=db_path, use_cache=False, stage="meta")
    fam = next((s for s in bundle.sections if s.key == SECTION_FAMILY), None)
    assert fam is not None
    assert {i.file_id for i in fam.items} == {2, 3}

    record_rejected_family(db_path, [1, 2, 3], label="Mickey Logo")
    invalidate_pattern_relations_cache()
    bundle2 = get_pattern_relations(db, 1, db_path=db_path, use_cache=False, stage="meta")
    fam2 = next((s for s in bundle2.sections if s.key == SECTION_FAMILY), None)
    assert fam2 is None or fam2.count == 0


def test_cache_invalidation():
    db = _FakeDB()
    db.add_file(1, filename="a.tif", path="D:/a/a.tif")
    db.add_file(2, filename="b.tif", path="D:/a/b.tif")
    db.link_group(17, [(1, "representative", 1.0), (2, DUP_EXACT, 0.99)])
    b1 = get_pattern_relations(db, 1, use_cache=True, stage="meta")
    assert b1.total_count() == 1
    # Mutate underlying group — cache should still serve old until invalidate
    db.link_group(17, [(1, "representative", 1.0)])
    b2 = get_pattern_relations(db, 1, use_cache=True, stage="meta")
    assert b2.total_count() == 1
    invalidate_pattern_relations_cache(1)
    b3 = get_pattern_relations(db, 1, use_cache=True, stage="meta")
    assert b3.total_count() == 0


def test_no_invented_customer_names():
    db = _FakeDB()
    db.add_file(1, filename="a.tif", path="D:/SomeRandomRoot/sub/a.tif", customer="")
    db.add_file(2, filename="b.tif", path="D:/OtherRoot/sub/b.tif", customer="")
    db.link_group(18, [(1, "representative", 1.0), (2, DUP_EXACT, 0.99)])
    bundle = get_pattern_relations(db, 1, use_cache=False, stage="meta")
    item = next(i for s in bundle.sections for i in s.items if i.file_id == 2)
    assert item.customer == ""


def test_self_excluded_and_empty_ok():
    db = _FakeDB()
    db.add_file(1, filename="solo.tif", path="D:/a/solo.tif")
    bundle = get_pattern_relations(db, 1, use_cache=False, stage="full")
    assert bundle.source_file_id == 1
    assert bundle.total_count() == 0
    assert all(1 not in [i.file_id for i in s.items] for s in bundle.sections)
