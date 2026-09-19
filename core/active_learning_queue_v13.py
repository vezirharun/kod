"""
V12.4.13 Active Learning Queue
Stores only auditable teaching candidates. It never auto-teaches uncertain results.
"""
from __future__ import annotations
import sqlite3,time
from pathlib import Path

class ActiveLearningQueue:
    def __init__(self, db_path):
        self.db_path=str(db_path); Path(self.db_path).parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.db_path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS active_learning_queue(
                id INTEGER PRIMARY KEY, query TEXT, item_id TEXT, label TEXT,
                confidence REAL, contradiction INTEGER, priority REAL,
                status TEXT DEFAULT 'pending', created_at REAL)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_alq_status ON active_learning_queue(status)")

    def add(self, query,item_id,label,confidence,contradiction=False):
        priority=(1.0-float(confidence)) + (0.35 if contradiction else 0.0)
        with sqlite3.connect(self.db_path) as c:
            c.execute("""INSERT INTO active_learning_queue
                (query,item_id,label,confidence,contradiction,priority,created_at)
                VALUES(?,?,?,?,?,?,?)""",
                (query,str(item_id),label,float(confidence),int(contradiction),priority,time.time()))

    def next_batch(self,limit=20):
        with sqlite3.connect(self.db_path) as c:
            return c.execute("""SELECT id,query,item_id,label,confidence,contradiction
                FROM active_learning_queue WHERE status='pending'
                ORDER BY priority DESC, created_at ASC LIMIT ?""",(int(limit),)).fetchall()
