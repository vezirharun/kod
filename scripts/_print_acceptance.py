import json
from pathlib import Path

d = json.loads(Path("data/reports/visual_intel_acceptance.json").read_text(encoding="utf-8"))
for q, v in d["queries"].items():
    print("=" * 60)
    print(
        q,
        "n=",
        v["n"],
        "comp",
        v["composite_pass"],
        "reject",
        v["rejected_competing_subtype"],
        "ch",
        v["evidence_sources_used"][:12],
    )
    for i, r in enumerate(v["top10"], 1):
        qev = r.get("query_evidence") or {}
        states = ",".join(
            f"{k}:{str(c.get('state') or '')[:4]}"
            for k, c in (qev.get("concepts") or {}).items()
        )
        print(f"  {i:2} {r['score']:.3f} {r['filename'][:72]}")
        print(
            f"     fn={r['filename_score']:.2f} ocr={r['ocr_score']:.2f} "
            f"fam={r['family_score']:.2f} sem={r['semantic_score']:.2f} "
            f"{r.get('v2_reason','')} {qev.get('composite','')} {states}"
        )
