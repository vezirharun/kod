"""Mevcut indeks için varlık kanıtını AI çalıştırmadan geriye dönük doldurur.

Bu işlem yalnızca mevcut texture_map/category/semantic metadata'dan kanıt çıkarır;
CLIP/COCO yeniden çalıştırmaz. Böylece V12.1 entity katmanı mevcut indeksle de
hemen kullanılabilir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import Database
from core.entity_evidence import extract_entity_evidence
from core.settings import AppSettings
from core.text_index import build_text_search_blob, extract_index_fields
from core.texture_profile import TextureProfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--detect",
        action="store_true",
        help="COCO/Global Object AI ile mevcut görselleri de tarar; uygun ortamda nesne kanıtını kalıcılaştırır.",
    )
    args = ap.parse_args()

    settings = AppSettings.load()
    db = Database(settings.db_path)
    changed = 0
    detected = 0
    scanned = 0
    object_ai = None
    if args.detect:
        from core.global_object_intelligence import GlobalObjectIntelligence
        if GlobalObjectIntelligence.capability().get("available"):
            object_ai = GlobalObjectIntelligence(
                enabled=True,
                min_confidence=float(getattr(settings, "global_object_detection_min_confidence", 0.45) or 0.45),
                max_detections=int(getattr(settings, "global_object_max_detections", 50) or 50),
            )

    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT f.id, f.path, f.filename, f.customer, f.ocr_text,
                   f.category_path, f.manual_category_path, f.text_search_blob,
                   fe.texture_map
            FROM files f
            LEFT JOIN features fe ON fe.file_id=f.id
            WHERE f.status='indexed'
            ORDER BY f.id
            """
        ).fetchall()

    for row in rows:
        if args.limit and scanned >= args.limit:
            break
        scanned += 1
        rec = dict(row)
        try:
            tm = json.loads(rec.get("texture_map") or "{}")
        except Exception:
            tm = {}
        if not isinstance(tm, dict):
            tm = {}
        if object_ai is not None and not tm.get("global_object_intelligence"):
            image_path = str(rec.get("path") or "")
            if image_path and Path(image_path).is_file():
                try:
                    object_meta = object_ai.analyze(image_path)
                    if object_meta.get("objects") or object_meta.get("person_count"):
                        tm["global_object_intelligence"] = object_meta
                        detected += 1
                except Exception:
                    pass

        ev = sorted(extract_entity_evidence({**rec, "texture_map": tm}))
        if not ev and not tm.get("global_object_intelligence"):
            continue
        if tm.get("entity_evidence") == ev:
            continue
        tm["entity_evidence"] = ev
        changed += 1
        if args.dry_run:
            continue
        pf, pt, tf = extract_index_fields(tm)
        blob = build_text_search_blob(
            filename=rec.get("filename") or "",
            path=rec.get("path") or "",
            customer=rec.get("customer") or "",
            ocr_text=rec.get("ocr_text") or "",
            texture_map=tm,
            pattern_family=pf,
            pattern_type=pt,
            texture_family=tf,
            pattern_subtype=TextureProfile.from_dict(tm).pattern_subtype or pt,
            category_path=str(rec.get("manual_category_path") or rec.get("category_path") or ""),
            category_aliases=list(tm.get("category_aliases") or []),
            semantic_enabled=bool(getattr(settings, "semantic_text_search_enabled", True)),
        )
        with db.connect() as conn:
            conn.execute(
                "UPDATE features SET texture_map=? WHERE file_id=?",
                (json.dumps(tm, ensure_ascii=False), int(rec["id"])),
            )
            conn.execute(
                "UPDATE files SET text_search_blob=? WHERE id=?",
                (blob, int(rec["id"])),
            )

    print(json.dumps({
        "scanned": scanned,
        "changed": changed,
        "detected": detected,
        "detect_enabled": bool(args.detect),
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
