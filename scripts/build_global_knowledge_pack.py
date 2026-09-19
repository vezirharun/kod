"""Compile Global Knowledge Pack from product ontologies.

Read-only. Does not touch production DB, FAISS, or customer Visual Memory.

python scripts/build_global_knowledge_pack.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "knowledge" / "vezir_core.json"


def _uniq(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        n = str(x or "").strip().lower()
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def compile_pack() -> dict:
    from core.brand_aliases import BRAND_ALIASES
    from core.natural_language_query import COLOR_PHRASES, MOTIF_PHRASES
    from core.semantic_pattern_intel import _ONTOLOGY
    from core.universal_visual_intel import CAPABILITIES, ONTOLOGY
    from core.visual_memory.knowledge import KNOWLEDGE_PACK_ID, KNOWLEDGE_PACK_VERSION

    objects = []
    types = []
    for nid, meta in ONTOLOGY.items():
        row = {
            "id": nid,
            "parent": meta.get("parent") or "",
            "level": meta.get("level") or "",
            "status": meta.get("status") or "experimental",
            "aliases": list(meta.get("aliases") or ()),
        }
        if meta.get("level") in ("category", "family"):
            objects.append(row)
        else:
            types.append(row)

    families = []
    brands_sem = []
    for cid, meta in _ONTOLOGY.items():
        spec = str(meta.get("specificity") or "")
        row = {
            "id": cid,
            "family": meta.get("family") or "",
            "motif": meta.get("motif") or cid,
            "specificity": spec,
        }
        if spec == "brand":
            brands_sem.append(row)
        else:
            families.append(row)

    colors = _uniq(list(COLOR_PHRASES.values()))
    motifs = _uniq(list(MOTIF_PHRASES.values()))
    brands = _uniq(list(BRAND_ALIASES.values()) + ["togg"])

    return {
        "id": KNOWLEDGE_PACK_ID,
        "version": KNOWLEDGE_PACK_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "compiled_ontologies",
        "customer_memory": False,
        "production_db": False,
        "faiss": False,
        "sources": [
            "core.universal_visual_intel.ONTOLOGY",
            "core.semantic_pattern_intel._ONTOLOGY",
            "core.natural_language_query.COLOR_PHRASES",
            "core.natural_language_query.MOTIF_PHRASES",
            "core.brand_aliases.BRAND_ALIASES",
            "core.universal_visual_intel.CAPABILITIES",
        ],
        "objects": objects,
        "types": types,
        "colors": colors,
        "motifs": motifs,
        "textures": ["scale", "spot", "stripe", "denim", "skin", "knit"],
        "pattern_families": _uniq([str(m.get("family") or "") for m in families]),
        "semantic_concepts": families,
        "brands": brands,
        "brand_concepts": brands_sem,
        "textile_concepts": ["repeat", "placement", "allover", "skin", "desen", "kumas"],
        "composition": [
            {"id": "count", "status": "unsupported"},
            {"id": "orientation", "status": "unsupported"},
            {"id": "spatial", "status": "unsupported"},
        ],
        "channels": [
            "visual",
            "ocr",
            "visual_dna",
            "filename",
            "metadata",
            "family",
            "texture",
            "semantic",
        ],
        "capabilities": dict(CAPABILITIES),
        "unsupported": [
            "count",
            "orientation",
            "spatial",
            "claimed_class_auto",
            "dedicated_togg_classifier",
            "dedicated_suv_body_classifier",
            "face_recognition",
            "spatial_object_boxes",
        ],
    }


def main() -> None:
    pack = compile_pack()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "wrote",
        OUT,
        "objects",
        len(pack["objects"]),
        "types",
        len(pack["types"]),
        "brands",
        len(pack["brands"]),
        "customer_memory",
        pack["customer_memory"],
    )


if __name__ == "__main__":
    main()
