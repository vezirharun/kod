import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.db import Database
from core.settings import AppSettings

db = Database(AppSettings.load().db_path)

with db.connect() as conn:
    print("STATUS", [dict(r) for r in conn.execute(
        "SELECT status, COUNT(*) n FROM files WHERE source_id IN (8,9) GROUP BY status"
    )])
    blob_leo = conn.execute(
        "SELECT COUNT(*) FROM files WHERE source_id IN (8,9) AND lower(coalesce(text_search_blob,'')) LIKE '%leopard%'"
    ).fetchone()[0]
    fname_leo = conn.execute(
        "SELECT COUNT(*) FROM files WHERE source_id IN (8,9) AND lower(filename) LIKE '%leopard%'"
    ).fetchone()[0]
    fname_leopar = conn.execute(
        "SELECT COUNT(*) FROM files WHERE source_id IN (8,9) AND (lower(filename) LIKE '%leopar%' OR lower(filename) LIKE '%leo%')"
    ).fetchone()[0]
    fts = conn.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
    fts_leo = conn.execute(
        "SELECT COUNT(*) FROM files_fts WHERE text_search_blob MATCH 'leopard'"
    ).fetchone()[0] if True else 0
    print("blob_contains_leopard", blob_leo)
    print("filename_leopard", fname_leo)
    print("filename_leopar_or_leo", fname_leopar)
    print("fts_rows", fts)
    print("fts_match_leopard", fts_leo)
    pf = conn.execute(
        "SELECT pattern_family, COUNT(*) n FROM files WHERE source_id IN (8,9) GROUP BY pattern_family ORDER BY n DESC LIMIT 10"
    ).fetchall()
    print("files.pattern_family", [dict(r) for r in pf])
    indexed_leo = conn.execute(
        """SELECT COUNT(*) FROM files
           WHERE source_id IN (8,9) AND status='indexed'
             AND (lower(coalesce(text_search_blob,'')) LIKE '%leopard%'
                  OR lower(filename) LIKE '%leopard%')"""
    ).fetchone()[0]
    print("indexed_status_and_leopard_text", indexed_leo)
    any_indexed = conn.execute(
        "SELECT COUNT(*) FROM files WHERE status='indexed'"
    ).fetchone()[0]
    print("status_indexed_all", any_indexed)
