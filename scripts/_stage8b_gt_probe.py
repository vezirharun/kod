import json
import sqlite3
from pathlib import Path

MEMORY = Path("data/search_memory.db")
c = sqlite3.connect(str(MEMORY))
c.row_factory = sqlite3.Row
rows = c.execute(
    "SELECT id, canonical, aliases FROM concept_registry"
).fetchall()
needles = ["dudak", "lip", "pip", "puan", "polka", "dot"]
for r in rows:
    can = str(r["canonical"] or "")
    al = str(r["aliases"] or "")
    blob = (can + " " + al).lower()
    if any(n in blob for n in needles):
        n = c.execute(
            "SELECT COUNT(*) n FROM concept_examples WHERE concept_id=? AND role='positive'",
            (r["id"],),
        ).fetchone()["n"]
        print(r["id"], can, "pos=", n, "aliases=", al[:80])
print("--- tiger ---")
for r in rows:
    if str(r["canonical"]).lower() in {"tiger", "kaplan"}:
        n = c.execute(
            "SELECT COUNT(*) n FROM concept_examples WHERE concept_id=? AND role='positive'",
            (r["id"],),
        ).fetchone()["n"]
        print(r["id"], r["canonical"], "pos=", n)
