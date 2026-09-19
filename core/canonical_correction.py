"""Canonical spelling correction for learned category / concept records.

Single entry point used by Edit panel and teach/apply flows.
Updates real learned records (category_memory, concept_registry, brand_memory).
Preserves distinct concept identity (e.g. Tiger ≠ Leopard). Idempotent.
"""

from __future__ import annotations

import json
from typing import Any

from core.logger import setup_logger

logger = setup_logger(__name__)

# UI / service field keys
FIELD_PARENT = "parent"
FIELD_CHILD = "child"
FIELD_TAG = "tag"
FIELD_FAMILY = "pattern_family"
FIELD_COLOR = "color"
FIELD_BRAND = "brand"

_MEMORY_PARENT = {
    FIELD_TAG: "Etiket",
    FIELD_FAMILY: "Desen ailesi",
    FIELD_COLOR: "Renk",
    FIELD_BRAND: "Marka",
}


def _clean(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _key(text: str) -> str:
    try:
        from core.textile_terms import normalize_turkish

        return normalize_turkish(text or "")
    except Exception:
        return " ".join(str(text or "").strip().lower().split())


def _concept_key(text: str) -> str:
    try:
        from core.concept_query_normalize import concept_match_key

        return concept_match_key(text)
    except Exception:
        return _key(text)


def _taxonomy_leaf_keys() -> set[str]:
    """Known distinct child labels across the static category tree."""
    keys: set[str] = set()
    try:
        from core.category_tree import CATEGORY_TREE

        for children in CATEGORY_TREE.values():
            for child in children.keys():
                k = _concept_key(child)
                if k:
                    keys.add(k)
    except Exception:
        pass
    return keys


def are_distinct_concepts(old_label: str, new_label: str) -> bool:
    """True when both labels are known distinct taxonomy leaves (Tiger ≠ Leopard)."""
    a = _concept_key(old_label)
    b = _concept_key(new_label)
    if not a or not b or a == b:
        return False
    leaves = _taxonomy_leaf_keys()
    return a in leaves and b in leaves


def _rename_concept_canonical(
    db_path: str,
    old_label: str,
    new_label: str,
    *,
    parent: str = "",
) -> dict[str, Any]:
    """Rename one concept row; refuse merging distinct taxonomy siblings."""
    from core.concept_registry import _conn, _now, _norm, _row_by_canonical, ensure

    stats: dict[str, Any] = {
        "ok": False,
        "renamed": 0,
        "merged": 0,
        "reason": "",
        "final": new_label,
    }
    old_label = _clean(old_label)
    new_label = _clean(new_label)
    if not db_path or not old_label or not new_label:
        stats["reason"] = "empty"
        return stats
    if _concept_key(old_label) == _concept_key(new_label) and old_label == new_label:
        stats["ok"] = True
        stats["reason"] = "unchanged"
        return stats
    if are_distinct_concepts(old_label, new_label):
        stats["reason"] = "distinct_concepts"
        logger.info(
            "canonical_correction refused distinct merge: %r → %r",
            old_label,
            new_label,
        )
        return stats

    ensure(db_path)
    c = _conn(db_path)
    try:
        def _full_row(cid: int):
            return c.execute(
                "SELECT id, aliases, confidence, canonical, parent FROM concept_registry WHERE id=?",
                (int(cid),),
            ).fetchone()

        found = _row_by_canonical(c, old_label)
        old_row = _full_row(int(found["id"])) if found is not None else None
        if old_row is None:
            # Match via aliases
            want = _norm(old_label)
            for r in c.execute(
                "SELECT id, aliases, confidence, canonical, parent FROM concept_registry"
            ):
                vals = [r["canonical"]]
                try:
                    vals.extend(json.loads(r["aliases"] or "[]"))
                except Exception:
                    pass
                if any(_norm(v) == want for v in vals if v):
                    old_row = r
                    break
        if old_row is None:
            # Create corrected concept with old spelling as alias (teach path)
            from core.concept_registry import upsert

            c.close()
            upsert(
                db_path,
                new_label,
                parent=parent,
                aliases=[old_label],
                source="user",
            )
            stats["ok"] = True
            stats["renamed"] = 1
            stats["reason"] = "created"
            stats["final"] = new_label
            logger.info("canonical_correction concept created: %r → %r", old_label, new_label)
            return stats

        found_new = _row_by_canonical(c, new_label)
        new_row = _full_row(int(found_new["id"])) if found_new is not None else None
        old_aliases: list[str] = []
        try:
            old_aliases = json.loads(old_row["aliases"] or "[]")
        except Exception:
            old_aliases = []

        if new_row is not None and int(new_row["id"]) != int(old_row["id"]):
            # Spelling variant merge only (already guarded by are_distinct_concepts)
            new_aliases: list[str] = []
            try:
                new_aliases = json.loads(new_row["aliases"] or "[]")
            except Exception:
                new_aliases = []
            merged = list(
                dict.fromkeys(
                    [
                        *new_aliases,
                        *old_aliases,
                        old_row["canonical"],
                        old_label,
                        new_row["canonical"],
                        new_label,
                    ]
                )
            )
            display = _clean(new_row["canonical"]) or new_label
            c.execute(
                "UPDATE concept_registry SET aliases=?, updated_at=? WHERE id=?",
                (json.dumps(merged, ensure_ascii=False), _now(), int(new_row["id"])),
            )
            # Retarget examples to surviving concept; drop empty duplicate shell
            c.execute(
                "UPDATE concept_examples SET concept_id=? WHERE concept_id=?",
                (int(new_row["id"]), int(old_row["id"])),
            )
            c.execute("DELETE FROM concept_registry WHERE id=?", (int(old_row["id"]),))
            stats["merged"] = 1
            stats["renamed"] = 1
            stats["final"] = display
        else:
            merged = list(
                dict.fromkeys([*old_aliases, old_row["canonical"], old_label, new_label])
            )
            parent_sql = parent or str(old_row["parent"] or "")
            c.execute(
                "UPDATE concept_registry SET canonical=?, aliases=?, parent=COALESCE(NULLIF(?,''),parent), updated_at=? WHERE id=?",
                (
                    new_label,
                    json.dumps(merged, ensure_ascii=False),
                    parent_sql,
                    _now(),
                    int(old_row["id"]),
                ),
            )
            stats["renamed"] = 1
            stats["final"] = new_label

        # Parent field retarget on other concepts when renaming a parent shell
        if not parent and _key(str(old_row["canonical"])) != _key(stats["final"]):
            c.execute(
                "UPDATE concept_registry SET parent=? WHERE parent=?",
                (stats["final"], old_row["canonical"]),
            )

        c.commit()
        stats["ok"] = True
        logger.info(
            "canonical_correction concept: %r → %r (merged=%s)",
            old_label,
            stats["final"],
            stats["merged"],
        )
        return stats
    except Exception as exc:
        stats["reason"] = str(exc)
        logger.exception("canonical_correction concept failed")
        return stats
    finally:
        try:
            c.close()
        except Exception:
            pass


def correct_learned_label(
    db_path: str,
    field: str,
    old_label: str,
    new_label: str,
    *,
    parent: str = "",
) -> dict[str, Any]:
    """Correct a taught/spelled label in canonical stores.

    field: parent | child | tag | pattern_family | color | brand
    """
    field = str(field or "").strip()
    old_label = _clean(old_label)
    new_label = _clean(new_label)
    parent = _clean(parent)
    stats: dict[str, Any] = {
        "ok": False,
        "field": field,
        "old": old_label,
        "new": new_label,
        "final": new_label,
        "reason": "",
        "memory": {},
        "concept": {},
        "brand": {},
    }
    if not db_path or not field or not old_label or not new_label:
        stats["reason"] = "empty"
        return stats
    if _key(old_label) == _key(new_label) and old_label == new_label:
        stats["ok"] = True
        stats["reason"] = "unchanged"
        return stats
    if field in (FIELD_CHILD, FIELD_TAG, FIELD_FAMILY, FIELD_COLOR) and are_distinct_concepts(
        old_label, new_label
    ):
        stats["reason"] = "distinct_concepts"
        logger.info(
            "canonical_correction refused: %s %r → %r (distinct)",
            field,
            old_label,
            new_label,
        )
        return stats

    from core.category_memory import rename_child_category, rename_root_category

    if field == FIELD_PARENT:
        mem = rename_root_category(db_path, old_label, new_label)
        stats["memory"] = mem
        concept = _rename_concept_canonical(db_path, old_label, new_label)
        stats["concept"] = concept
        stats["ok"] = bool(mem.get("ok") or concept.get("ok"))
        stats["final"] = str(mem.get("final") or concept.get("final") or new_label)
        stats["reason"] = "" if stats["ok"] else (mem.get("reason") or concept.get("reason") or "fail")
    elif field == FIELD_CHILD:
        if not parent:
            stats["reason"] = "parent_required"
            return stats
        mem = rename_child_category(db_path, parent, old_label, new_label)
        stats["memory"] = mem
        concept = _rename_concept_canonical(
            db_path, old_label, new_label, parent=parent
        )
        stats["concept"] = concept
        stats["ok"] = bool(mem.get("ok") or concept.get("ok"))
        stats["final"] = str(concept.get("final") or new_label)
        stats["reason"] = "" if stats["ok"] else (mem.get("reason") or concept.get("reason") or "fail")
    elif field in _MEMORY_PARENT:
        mem_parent = _MEMORY_PARENT[field]
        mem = rename_child_category(db_path, mem_parent, old_label, new_label)
        stats["memory"] = mem
        concept = _rename_concept_canonical(
            db_path, old_label, new_label, parent=mem_parent
        )
        stats["concept"] = concept
        if field == FIELD_BRAND:
            try:
                from core.brand_aliases import register_brand_alias

                ok_brand = register_brand_alias(db_path, old_label, new_label)
                stats["brand"] = {"ok": bool(ok_brand)}
            except Exception as exc:
                stats["brand"] = {"ok": False, "reason": str(exc)}
        stats["ok"] = bool(mem.get("ok") or concept.get("ok") or stats["brand"].get("ok"))
        stats["final"] = str(concept.get("final") or new_label)
        stats["reason"] = "" if stats["ok"] else (mem.get("reason") or concept.get("reason") or "fail")
    else:
        stats["reason"] = "unknown_field"
        return stats

    if stats["ok"]:
        logger.info(
            "canonical_correction %s: %r → %r",
            field,
            old_label,
            stats["final"],
        )
    return stats
