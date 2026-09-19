import json
from pathlib import Path

d = json.loads(
    Path("data/reports/visual_intel_capability_benchmark.json").read_text(encoding="utf-8")
)
print("coverage", d["target_coverage"])
print("\n-- object --")
for q, v in d["object"].items():
    print(
        f"{q:10} n={v['n']:3} t={v['ttfr_s']:.2f}s fnGT={v['gt_filename_n']:3} "
        f"rec={v['filename_gt_recalled_in_top20']:3} ocrGT={v['gt_ocr_n']} "
        f"clip={v['clip_in_top']} ch={v['evidence_channels'][:6]}"
    )
    if v["filename_gt_missed_sample"][:3]:
        print("   missed", [x["filename"][:50] for x in v["filename_gt_missed_sample"][:4]])

print("\n-- snake audit --")
a = d["visual_vs_filename"]["snake_filename_audit"]
print("fn", a["filename_snake_n"], "retrieved", a["retrieved_n"], "verdict", a["verdict"])
print("recalled", len(a["recalled"]), [x["filename"][:40] for x in a["recalled"][:8]])
print("missed", [(x["filename"][:50], x["type"], x["family"]) for x in a["missed"][:10]])

print("\n-- attr AND --")
for q, v in d["attribute"].items():
    print(q, "n", v["n"], v.get("color_object_and"), "comp", v["composite_pass"])
    if v["top10"]:
        p = v["top10"][0]
        print("  #1", round(p["score"], 3), p["filename"][:50], p.get("composite"), p.get("concepts"))

print("\n-- composite --")
for q, v in d["composite"].items():
    print(q, "n", v["n"], "pass", v["composite_pass"], v["verdict"], "empty", v["empty_and_fail"])
    if v["top10"]:
        print("  #1", v["top10"][0]["filename"][:55], v["top10"][0].get("composite"))

print("\n-- gucci --")
g = d["ocr_brand"]["gucci"]
print("n", g["n"], "fn", g["filename_gucci_n"], "ocr_only_in", g["ocr_only_in_results"], g["ocr_on_image_without_filename"])

print("\n-- fine --")
print(d["fine_grained"])

print("\n-- count/orient --")
for q, v in d["count_orient"].items():
    print(q, v["n"], v["count_status"], v["orient_status"], v["relation_status"], v["top3"][:2])

print("\n-- sim --")
print(d["visual_similarity"]["same_query_top2"])
print(d["visual_similarity"]["snake_vs_leopard_top1"])
