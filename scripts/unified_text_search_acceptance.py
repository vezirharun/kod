"""Kabul testi: ortak metin/görsel alaka sıralaması.

Gerçek üretim DB/FAISS olmadan da sıralama sözleşmesini deterministik olarak
kanıtlar. Gerçek veri benchmark'ı için uygulamanın kendi DB/FAISS ortamında
ayrıca çalıştırılmalıdır.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.search_engine import SearchResult
from core.unified_relevance_ranker import apply_unified_text_ranking


def r(fid, score, bd=None, dbg=None):
    return SearchResult(
        file_id=fid, path=f"/accept/{fid}.jpg", filename=f"{fid}.jpg",
        customer="", thumbnail_path="", score=score, score_percent=score * 100,
        breakdown=dict(bd or {}), debug=dict(dbg or {}),
    )


def run():
    checks = []

    # Leopard: strong visual leopard must outrank filename-only coincidence.
    rows = apply_unified_text_ranking([
        r(1, .94, {"filename_score": .94}, {"clip_score": .14}),
        r(2, .71, {"family_score": .71}, {"clip_score": .95, "visual_grade": "visual_strong"}),
        r(3, .83, {"semantic_score": .88}, {"clip_score": .89, "visual_grade": "visual_exact"}),
    ])
    checks.append(("leopard_visual_priority", [x.file_id for x in rows] == [2, 1, 3]))

    # Woman: non-human filename coincidence cannot outrank gender evidence.
    rows = apply_unified_text_ranking([
        r(10, .99, {"filename_score": .99}, {"clip_score": .10}),
        r(11, .76, {"semantic_score": .76}, {"gender_visual_score": .99, "human_semantic_score": .99}),
        r(12, .72, {"semantic_score": .70}, {"gender_visual_score": .86, "human_semantic_score": .86}),
    ], human_query=True)
    checks.append(("woman_visual_priority", [x.file_id for x in rows] == [11, 12, 10]))

    # AI unavailable: exact metadata must remain searchable and ranked first.
    rows = apply_unified_text_ranking([
        r(20, .70, {"family_score": .70}),
        r(21, .95, {"filename_score": .95}),
        r(22, .84, {"ocr_score": .84}),
    ])
    checks.append(("metadata_fallback", [x.file_id for x in rows] == [21, 22, 20]))

    # Visual mode must remain visually dominant; text ranker is not allowed to
    # rewrite the visual engine itself.
    from core.unified_relevance_ranker import compute_unified_relevance
    img = r(30, .91, {"texture_score": .80}, {"clip_score": .96})
    img_score, _ = compute_unified_relevance(img, mode="image")
    checks.append(("image_visual_base_preserved", img_score >= .80))

    # Score monotonicity: higher same-query visual evidence must never score lower.
    rows = apply_unified_text_ranking([
        r(40, .60, {"family_score": .60}, {"clip_score": .55}),
        r(41, .60, {"family_score": .60}, {"clip_score": .80}),
        r(42, .60, {"family_score": .60}, {"clip_score": .92}),
    ])
    checks.append(("visual_monotonicity", [x.file_id for x in rows] == [42, 41, 40]))

    passed = sum(ok for _, ok in checks)
    report = {
        "suite": "unified_text_search_acceptance_v1",
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "checks": [{"name": n, "passed": ok} for n, ok in checks],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(run())
