import json
import sqlite3

c = sqlite3.connect("data/patterns.db")
c.row_factory = sqlite3.Row
r = c.execute(
    "SELECT dominant_colors, texture_map FROM features "
    "WHERE texture_map LIKE '%detected_colors%' LIMIT 1"
).fetchone()
dc = json.loads(r["dominant_colors"])
tm = json.loads(r["texture_map"])
ev = tm["color_evidence"]
print("rgb", dc[:4])
print("detected", ev["detected_colors"])
print("family", ev.get("color_family"))
print("mood", ev.get("color_mood"))
print("ratios", ev.get("color_ratios"))
print("dna", (tm.get("pattern_dna") or {}).get("dominant_colors"))
