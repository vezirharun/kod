"""Positive/negative teaching memory with auditable records."""

from __future__ import annotations
import sqlite3
import time
from pathlib import Path

class LearningMemory:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS learning_feedback(
                id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                item_id TEXT NOT NULL,
                label TEXT NOT NULL,
                polarity TEXT NOT NULL,
                reason TEXT,
                created_at REAL NOT NULL
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_learning_query ON learning_feedback(query)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_learning_item ON learning_feedback(item_id)")

    def teach(self, query, item_id, label, polarity="positive", reason=""):
        if polarity not in ("positive", "negative"):
            raise ValueError("polarity must be positive or negative")
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO learning_feedback(query,item_id,label,polarity,reason,created_at) VALUES(?,?,?,?,?,?)",
                (query, str(item_id), label, polarity, reason, time.time()),
            )

    def score_adjustment(self, query, item_id, label):
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute(
                "SELECT polarity FROM learning_feedback WHERE query=? AND item_id=? AND label=?",
                (query, str(item_id), label),
            ).fetchall()
        if not rows:
            return 0.0
        pos = sum(r[0] == "positive" for r in rows)
        neg = sum(r[0] == "negative" for r in rows)
        return max(-0.30, min(0.30, 0.10 * (pos - neg)))
