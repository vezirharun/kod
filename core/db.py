"""SQLite veritabanı katmanı."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable

_MUTATING_SQL = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|VACUUM|REINDEX|ATTACH|DETACH)\b",
    re.IGNORECASE,
)

from core.utils import normalize_path, normalize_source_root, source_root_key
from core.textile_terms import normalize_turkish

# Aktif DbWriteBatch (index sırasında disk I/O azaltma)
_active_write_batch = None
_active_write_batch_lock = __import__("threading").Lock()


def sql_visual_ready_clause(*, file_alias: str = "f", feature_alias: str = "fe") -> str:
    """Görsel arama için hazır kayıt — search pool ile aynı iç koşul.

    ``get_indexed_files(include_processing_ready=True)`` ile birebir.
    """
    f, fe = file_alias, feature_alias
    return f"""(
            {f}.status NOT IN ('missing','excluded_internal')
            AND coalesce({fe}.phash,'')<>''
            AND coalesce({fe}.texture_map,'') NOT IN ('','{{}}')
            AND (
                ({fe}.dino_embedding IS NOT NULL AND length({fe}.dino_embedding)>0)
                OR ({fe}.clip_embedding IS NOT NULL AND length({fe}.clip_embedding)>0)
            )
        )"""


def sql_processing_ready_clause(
    *,
    file_alias: str = "f",
    feature_alias: str = "fe",
    indexed_status: str = "indexed",
) -> str:
    """``status=indexed`` VEYA görsel-hazır (pending dahil)."""
    f = file_alias
    return (
        f"({f}.status = '{indexed_status}' OR "
        f"{sql_visual_ready_clause(file_alias=file_alias, feature_alias=feature_alias)})"
    )


def set_active_write_batch(batch) -> None:
    global _active_write_batch
    with _active_write_batch_lock:
        _active_write_batch = batch


def clear_active_write_batch() -> None:
    global _active_write_batch
    with _active_write_batch_lock:
        _active_write_batch = None


def get_active_write_batch():
    with _active_write_batch_lock:
        return _active_write_batch


def expand_fts_tokens(term: str) -> list[str]:
    from core.textile_terms import expand_query_terms, normalize_turkish

    return expand_query_terms(term) or [normalize_turkish(term)]


SCHEMA_VERSION = 19

# Progressive Index / OCR UI (count_v3_ssot) require these on files.
_OCR_PATCH_FILE_COLUMNS = (
    ("ocr_processed", "INTEGER DEFAULT 0"),
    ("ocr_error", "TEXT DEFAULT ''"),
    ("patch_error", "TEXT DEFAULT ''"),
)


def _ensure_files_columns(
    conn: sqlite3.Connection, specs: Iterable[tuple[str, str]]
) -> None:
    """ALTER TABLE files ADD COLUMN only when missing. No row data changes."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
    for name, typedef in specs:
        if name not in existing:
            conn.execute(f"ALTER TABLE files ADD COLUMN {name} {typedef}")
            existing.add(name)


def _json_object_ready_sql(column: str, key: str) -> str:
    """Top-level, non-empty JSON object check used by artifact SSOT queries."""
    safe_json = (
        f"(CASE WHEN json_valid(COALESCE({column},'')) "
        f"THEN {column} ELSE '{{}}' END)"
    )
    path = f"$.{key}"
    return (
        f"(json_type({safe_json}, '{path}')='object' "
        f"AND COALESCE(json_extract({safe_json}, '{path}'),'{{}}')!='{{}}')"
    )


def _merge_feature_payload(old_row: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Keep existing non-empty embeddings/tags when a parallel writer omits them."""

    def _parse(val: Any, default: Any) -> Any:
        if val is None:
            return default
        if isinstance(val, (dict, list)):
            return val
        if isinstance(val, (bytes, bytearray)):
            return default if default != val else val
        try:
            return json.loads(val)
        except Exception:
            return default

    def _blob_ok(val: Any) -> bool:
        if val is None:
            return False
        if isinstance(val, (bytes, bytearray, memoryview)):
            return len(val) > 0
        return bool(val)

    old_tm = _parse(old_row.get("texture_map"), {})
    new_tm = _parse(incoming.get("texture_map"), {})
    merged_tm = {**old_tm, **new_tm}
    if not new_tm.get("semantic_tags") and old_tm.get("semantic_tags"):
        merged_tm["semantic_tags"] = old_tm["semantic_tags"]
    if not new_tm.get("pattern_dna") and old_tm.get("pattern_dna"):
        merged_tm["pattern_dna"] = old_tm["pattern_dna"]
    new_dc = _parse(incoming.get("dominant_colors", []), [])
    new_tf = _parse(incoming.get("texture_features", []), [])
    new_patch = _parse(
        incoming.get("patch_embeddings_meta", incoming.get("patch_embeddings", [])),
        [],
    )
    old_dc = _parse(old_row.get("dominant_colors"), [])
    old_tf = _parse(old_row.get("texture_features"), [])
    old_patch = _parse(
        old_row.get("patch_embeddings_meta", old_row.get("patch_embeddings")),
        [],
    )
    inc_dino = incoming.get("dino_embedding")
    inc_clip = incoming.get("clip_embedding")
    return {
        "phash": incoming.get("phash") or old_row.get("phash") or "",
        "dhash": incoming.get("dhash") or old_row.get("dhash") or "",
        "whash": incoming.get("whash") or old_row.get("whash") or "",
        "color_hist": incoming.get("color_hist")
        if incoming.get("color_hist") is not None
        else old_row.get("color_hist"),
        "dominant_colors": new_dc if new_dc else old_dc,
        "texture_features": new_tf if new_tf else old_tf,
        "dino_embedding": inc_dino if _blob_ok(inc_dino) else old_row.get("dino_embedding"),
        "clip_embedding": inc_clip if _blob_ok(inc_clip) else old_row.get("clip_embedding"),
        "patch_embeddings_meta": new_patch if new_patch else old_patch,
        "texture_map": merged_tm,
    }


class _SearchReadConnection:
    """SELECT-only view of patterns.db while a search session is active.

    `Database.connect()` used to call guard_index_write for every connection,
    including reads, so a writable Database opened during search crashed with
    INDEX_FROZEN_WRITE_BLOCKED. Mutations still raise that error.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def execute(self, sql, parameters=()):
        text = str(sql or "")
        if _MUTATING_SQL.match(text):
            from core.index_freeze import guard_index_write

            guard_index_write("sqlite.write", "core.db", sql=text)
        return self._conn.execute(sql, parameters)

    def executemany(self, sql, seq_of_parameters):
        text = str(sql or "")
        if _MUTATING_SQL.match(text):
            from core.index_freeze import guard_index_write

            guard_index_write("sqlite.write", "core.db", sql=text)
        return self._conn.executemany(sql, seq_of_parameters)

    def executescript(self, sql):
        from core.index_freeze import guard_index_write

        guard_index_write("sqlite.write", "core.db", sql=str(sql or "")[:240])
        return self._conn.executescript(sql)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class Database:
    def __init__(self, db_path: str | Path, *, read_only: bool = False):
        self.db_path = str(db_path)
        self.read_only = bool(read_only)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        if not self.read_only:
            from core.index_freeze import INDEX_FROZEN
            from core.index_freeze import search_write_protection_active

            # Never CREATE/ALTER during search — even under allow_index_writes.
            if INDEX_FROZEN and search_write_protection_active():
                pass
            else:
                self._init_schema()

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        from core.index_freeze import (
            INDEX_FROZEN,
            guard_index_write,
            search_write_protection_active,
        )

        search_readonly = INDEX_FROZEN and search_write_protection_active()
        if self.read_only or search_readonly:
            uri = f"file:{Path(self.db_path).as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                yield _SearchReadConnection(conn) if search_readonly else conn
            finally:
                conn.close()
            return

        guard_index_write("sqlite.write", "core.db")

        batch = get_active_write_batch()
        if (
            batch is not None
            and str(getattr(batch, "db_path", "")) == self.db_path
            and hasattr(batch, "exclusive")
        ):
            # Nested transaction yok — SAVEPOINT + exclusive writer lock
            with batch.exclusive() as conn:
                yield conn
            return

        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    customer TEXT DEFAULT '',
                    file_size INTEGER DEFAULT 0,
                    mtime REAL DEFAULT 0,
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    thumbnail_path TEXT DEFAULT '',
                    partial_hash TEXT DEFAULT '',
                    full_hash TEXT DEFAULT '',
                    error_msg TEXT DEFAULT '',
                    ocr_text TEXT DEFAULT '',
                    indexed_at TEXT DEFAULT '',
                    created_at TEXT DEFAULT '',
                    updated_at TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
                CREATE INDEX IF NOT EXISTS idx_files_customer ON files(customer);
                CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);

                CREATE TABLE IF NOT EXISTS features (
                    file_id INTEGER PRIMARY KEY,
                    phash TEXT DEFAULT '',
                    dhash TEXT DEFAULT '',
                    whash TEXT DEFAULT '',
                    color_hist BLOB,
                    dominant_colors TEXT DEFAULT '[]',
                    texture_features TEXT DEFAULT '[]',
                    dino_embedding BLOB,
                    clip_embedding BLOB,
                    patch_embeddings TEXT DEFAULT '[]',
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS index_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    last_error TEXT DEFAULT '',
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_queue_status ON index_queue(status);
                """)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', '1')",
            )
            self._migrate_schema(conn)
            self._canonicalize_source_roots(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        version = int(row["value"]) if row else 1

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'local_pc',
                root_path TEXT NOT NULL UNIQUE,
                is_active INTEGER DEFAULT 1,
                last_scan_at TEXT DEFAULT '',
                last_quick_scan_at TEXT DEFAULT '',
                last_deep_scan_at TEXT DEFAULT '',
                scan_interval_hours INTEGER DEFAULT 6,
                deep_scan_interval_days INTEGER DEFAULT 7,
                file_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                cache_status TEXT DEFAULT 'empty',
                created_at TEXT DEFAULT '',
                updated_at TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);
            CREATE INDEX IF NOT EXISTS idx_sources_active ON sources(is_active);
            """)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
        if "source_id" not in cols:
            conn.execute("ALTER TABLE files ADD COLUMN source_id INTEGER DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_files_source ON files(source_id)")

        if version < SCHEMA_VERSION:
            feat_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(features)").fetchall()
            }
            if "texture_map" not in feat_cols:
                conn.execute(
                    "ALTER TABLE features ADD COLUMN texture_map TEXT DEFAULT '{}'"
                )
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            for col, typedef in (
                ("feature_version", "INTEGER DEFAULT 0"),
                ("texture_version", "INTEGER DEFAULT 0"),
                ("thumbnail_version", "INTEGER DEFAULT 0"),
                ("last_feature_at", "TEXT DEFAULT ''"),
                ("last_texture_at", "TEXT DEFAULT ''"),
                ("feature_preview_path", "TEXT DEFAULT ''"),
                ("feature_preview_version", "INTEGER DEFAULT 0"),
                ("feature_preview_mtime", "REAL DEFAULT 0"),
                ("unsupported_preview", "INTEGER DEFAULT 0"),
            ):
                if col not in file_cols:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} {typedef}")

            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pattern_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    representative_file_id INTEGER NOT NULL,
                    group_type TEXT DEFAULT 'auto',
                    label TEXT DEFAULT '',
                    pattern_family TEXT DEFAULT '',
                    animal_print_type TEXT DEFAULT '',
                    color_family TEXT DEFAULT '',
                    created_at TEXT DEFAULT '',
                    updated_at TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS pattern_group_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    file_id INTEGER NOT NULL,
                    relation TEXT DEFAULT 'similar_texture',
                    score REAL DEFAULT 0,
                    created_at TEXT DEFAULT '',
                    FOREIGN KEY (group_id) REFERENCES pattern_groups(id) ON DELETE CASCADE,
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                    UNIQUE(group_id, file_id)
                );
                CREATE INDEX IF NOT EXISTS idx_pgm_group ON pattern_group_members(group_id);
                CREATE INDEX IF NOT EXISTS idx_pgm_file ON pattern_group_members(file_id);

                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_file_id INTEGER DEFAULT 0,
                    query_path TEXT DEFAULT '',
                    result_file_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    label TEXT DEFAULT '',
                    note TEXT DEFAULT '',
                    created_at TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_query ON user_feedback(query_path);
                CREATE INDEX IF NOT EXISTS idx_feedback_result ON user_feedback(result_file_id);

                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    customer TEXT DEFAULT '',
                    created_at TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS collection_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER NOT NULL,
                    file_id INTEGER NOT NULL,
                    note TEXT DEFAULT '',
                    created_at TEXT DEFAULT '',
                    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                    UNIQUE(collection_id, file_id)
                );
                CREATE INDEX IF NOT EXISTS idx_collection_items_collection
                    ON collection_items(collection_id);
                """)
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )

        if version < 6:
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            for col, typedef in (
                ("pattern_family", "TEXT DEFAULT ''"),
                ("pattern_type", "TEXT DEFAULT ''"),
                ("texture_family", "TEXT DEFAULT ''"),
                ("text_search_blob", "TEXT DEFAULT ''"),
            ):
                if col not in file_cols:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} {typedef}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_pattern_family ON files(pattern_family)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_text_blob ON files(text_search_blob)"
            )
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                        file_id UNINDEXED,
                        text_search_blob,
                        tokenize='unicode61 remove_diacritics 1'
                    )
                    """)
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )

        if version < 7:
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            for col, typedef in (
                ("pattern_subtype", "TEXT DEFAULT ''"),
                ("pattern_confidence", "REAL DEFAULT 0"),
                ("preview_status", "TEXT DEFAULT ''"),
            ):
                if col not in file_cols:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} {typedef}")
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("7",),
            )

        if version < 8:
            q_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(index_queue)").fetchall()
            }
            for col, typedef in (
                ("source_id", "INTEGER DEFAULT 0"),
                ("file_path", "TEXT DEFAULT ''"),
                ("task_type", "TEXT DEFAULT ''"),
                ("priority", "INTEGER DEFAULT 50"),
                ("retry_count", "INTEGER DEFAULT 0"),
                ("error_message", "TEXT DEFAULT ''"),
                ("created_at", "TEXT DEFAULT ''"),
                ("updated_at", "TEXT DEFAULT ''"),
            ):
                if col not in q_cols:
                    conn.execute(f"ALTER TABLE index_queue ADD COLUMN {col} {typedef}")
            # Eski stage → task_type
            conn.execute(
                "UPDATE index_queue SET task_type=stage WHERE (task_type IS NULL OR task_type='') AND stage IS NOT NULL"
            )
            conn.execute(
                "UPDATE index_queue SET error_message=last_error WHERE (error_message IS NULL OR error_message='') AND last_error IS NOT NULL"
            )
            conn.execute(
                "UPDATE index_queue SET retry_count=attempts WHERE retry_count=0 AND attempts>0"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_queue_source ON index_queue(source_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_queue_file ON index_queue(file_id)"
            )
            # processing dosyaları için kuyruk oluştur
            now = datetime.now(timezone.utc).isoformat()
            rows = conn.execute(
                "SELECT id, source_id, path FROM files WHERE status='processing'"
            ).fetchall()
            for row in rows:
                fid, sid, fpath = (
                    int(row["id"]),
                    int(row["source_id"] or 0),
                    row["path"],
                )
                exists = conn.execute(
                    "SELECT 1 FROM index_queue WHERE file_id=? AND status IN ('pending','running') LIMIT 1",
                    (fid,),
                ).fetchone()
                if not exists:
                    for task in ("thumbnail", "hash", "texture_map", "taxonomy"):
                        conn.execute(
                            """
                            INSERT INTO index_queue(
                                file_id, stage, status, source_id, file_path,
                                task_type, priority, created_at, updated_at
                            ) VALUES (?, ?, 'pending', ?, ?, ?, 50, ?, ?)
                            """,
                            (fid, task, sid, fpath, task, now, now),
                        )
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("8",),
            )

        if version < 9:
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            for col, typedef in (
                ("category_path", "TEXT DEFAULT ''"),
                ("manual_category_path", "TEXT DEFAULT ''"),
                ("category_confidence", "REAL DEFAULT 0"),
                ("category_source", "TEXT DEFAULT ''"),
                ("category_aliases", "TEXT DEFAULT ''"),
                ("is_searchable_visual", "INTEGER DEFAULT 1"),
            ):
                if col not in file_cols:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} {typedef}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_category_path ON files(category_path)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_manual_category ON files(manual_category_path)"
            )
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("9",),
            )

        if version < 10:
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            for col, typedef in (
                ("index_stage", "TEXT DEFAULT 'pending_light'"),
                ("needs_review", "INTEGER DEFAULT 0"),
            ):
                if col not in file_cols:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} {typedef}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_index_stage ON files(index_stage)"
            )
            conn.execute("""
                UPDATE files SET index_stage='full_done'
                WHERE status='indexed' AND (index_stage IS NULL OR index_stage='' OR index_stage='pending_light')
                """)
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("10",),
            )

        if version < 11:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS category_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    old_predictions TEXT DEFAULT '[]',
                    old_category TEXT DEFAULT '',
                    correct_category TEXT NOT NULL,
                    correct_subcategory TEXT DEFAULT '',
                    actor TEXT DEFAULT '',
                    is_admin INTEGER DEFAULT 0,
                    propagation_scope TEXT DEFAULT 'single',
                    status TEXT DEFAULT 'pending_review',
                    created_at TEXT DEFAULT '',
                    undone_at TEXT DEFAULT '',
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_category_correction_file
                    ON category_corrections(file_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_category_correction_status
                    ON category_corrections(status);
                """)
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("11",),
            )

        if version < 12:
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            for col, typedef in (
                ("format_family", "TEXT DEFAULT ''"),
                ("format_extension", "TEXT DEFAULT ''"),
                ("format_capabilities", "TEXT DEFAULT '[]'"),
                ("format_metadata", "TEXT DEFAULT '{}'"),
                ("thumbnail_status", "TEXT DEFAULT ''"),
                ("metadata_status", "TEXT DEFAULT ''"),
                ("semantic_status", "TEXT DEFAULT ''"),
                ("parser_name", "TEXT DEFAULT ''"),
                ("parser_version", "TEXT DEFAULT ''"),
                ("parser_error", "TEXT DEFAULT ''"),
                ("project_group_id", "TEXT DEFAULT ''"),
                ("project_group_confidence", "REAL DEFAULT 0"),
                ("project_group_reason", "TEXT DEFAULT ''"),
                ("project_group_status", "TEXT DEFAULT ''"),
            ):
                if col not in file_cols:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} {typedef}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_project_group ON files(project_group_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_format_family ON files(format_family)"
            )
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("12",),
            )

        if version < 13:
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            if "quarantine_reason" not in file_cols:
                conn.execute(
                    "ALTER TABLE files ADD COLUMN quarantine_reason TEXT DEFAULT ''"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_quarantine ON files(quarantine_reason)"
            )
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("13",),
            )

        if version < 14:
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            for col, typedef in (
                ("light_status", "TEXT DEFAULT 'pending'"),
                ("heavy_status", "TEXT DEFAULT 'pending'"),
                ("processing_lock", "TEXT DEFAULT ''"),
                ("light_done_at", "TEXT DEFAULT ''"),
                ("heavy_done_at", "TEXT DEFAULT ''"),
                ("needs_medium_preview", "INTEGER DEFAULT 0"),
                ("needs_embedding", "INTEGER DEFAULT 0"),
                ("needs_ai", "INTEGER DEFAULT 0"),
                ("needs_semantic", "INTEGER DEFAULT 0"),
                ("needs_dna", "INTEGER DEFAULT 0"),
                ("needs_ocr", "INTEGER DEFAULT 0"),
                ("needs_texture", "INTEGER DEFAULT 0"),
            ):
                if col not in file_cols:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} {typedef}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_light_status ON files(light_status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_heavy_status ON files(heavy_status)"
            )
            conn.execute("""
                UPDATE files SET light_status='done'
                WHERE index_stage IN ('light_done','full_done','skipped_heavy_format')
            """)
            conn.execute("""
                UPDATE files SET heavy_status='done'
                WHERE index_stage='full_done'
            """)
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("14",),
            )

        if version < 15:
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            for col, typedef in (
                ("physical_preview_ready", "INTEGER DEFAULT NULL"),
                ("physical_thumbnail_ready", "INTEGER DEFAULT NULL"),
                ("physical_verified_at", "TEXT DEFAULT ''"),
            ):
                if col not in file_cols:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} {typedef}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_physical_preview "
                "ON files(physical_preview_ready)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_physical_thumbnail "
                "ON files(physical_thumbnail_ready)"
            )
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("15",),
            )

        if version < 16:
            file_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()
            }
            for col, typedef in (
                ("repair_retry_count", "INTEGER DEFAULT 0"),
                ("repair_failure_reason", "TEXT DEFAULT ''"),
                ("repair_last_error", "TEXT DEFAULT ''"),
                ("repair_retryable", "INTEGER DEFAULT 1"),
            ):
                if col not in file_cols:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} {typedef}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_repair_retryable "
                "ON files(repair_retryable)"
            )
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("16",),
            )

        if version < 17:
            _ensure_files_columns(conn, (("ocr_processed", "INTEGER DEFAULT 0"),))
            # Eski dolu ocr_text → processed say (yalnız v17 yükseltmesi)
            conn.execute(
                """
                UPDATE files SET ocr_processed=1
                WHERE ifnull(ocr_processed,0)=0 AND ifnull(ocr_text,'')!=''
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_ocr_processed "
                "ON files(ocr_processed)"
            )
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                ("17",),
            )

        # v17 DB'lerde 18/19 atlanmış olabilir — version'dan bağımsız, idempotent.
        _ensure_files_columns(conn, _OCR_PATCH_FILE_COLUMNS)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_ocr_processed ON files(ocr_processed)"
        )
        if version < 19:
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )

    def _canonicalize_source_roots(self, conn: sqlite3.Connection) -> None:
        """Merge legacy slash/case variants while preserving file ownership."""
        rows = conn.execute("SELECT * FROM sources ORDER BY id").fetchall()
        kept: dict[str, int] = {}
        for row in rows:
            source = dict(row)
            canonical = normalize_source_root(source.get("root_path", ""))
            key = source_root_key(canonical)
            if key in kept:
                target_id = kept[key]
                conn.execute(
                    "UPDATE files SET source_id=? WHERE source_id=?",
                    (target_id, source["id"]),
                )
                conn.execute("DELETE FROM sources WHERE id=?", (source["id"],))
                continue
            kept[key] = int(source["id"])
            if canonical != source.get("root_path"):
                conn.execute(
                    "UPDATE sources SET root_path=? WHERE id=?",
                    (canonical, source["id"]),
            )

    def get_meta(self, key: str, default: str = "") -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def list_physical_verification_batch(
        self,
        *,
        after_id: int = 0,
        limit: int = 250,
        verified_before: str = "",
    ) -> list[dict[str, Any]]:
        """Cache doğrulaması için küçük, kaldığı yerden devam eden batch."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, path, filename, status, light_status, heavy_status,
                       thumbnail_path, feature_preview_path,
                       physical_thumbnail_ready, physical_preview_ready,
                       physical_verified_at, repair_retry_count,
                       repair_failure_reason, repair_last_error, repair_retryable
                FROM files
                WHERE id > ?
                  AND status NOT IN ('excluded_internal', 'missing')
                  AND (
                        physical_verified_at=''
                        OR physical_verified_at IS NULL
                        OR ?=''
                        OR physical_verified_at < ?
                      )
                ORDER BY id
                LIMIT ?
                """,
                (
                    max(0, int(after_id)),
                    str(verified_before or ""),
                    str(verified_before or ""),
                    max(1, int(limit)),
                ),
            ).fetchall()
            return [dict(row) for row in rows]

    def record_light_repair_failure(
        self,
        file_id: int,
        *,
        reason: str,
        error_msg: str,
        retryable: bool,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Repair hatasını kalıcılaştır; geçici hataları sınırlı retry ile tut."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(repair_retry_count,0) AS retry_count "
                "FROM files WHERE id=?",
                (int(file_id),),
            ).fetchone()
            retry_count = int((row or {"retry_count": 0})["retry_count"] or 0) + 1
            can_retry = bool(retryable and retry_count < max(1, int(max_retries)))
            conn.execute(
                """
                UPDATE files SET
                    repair_retry_count=?,
                    repair_failure_reason=?,
                    repair_last_error=?,
                    repair_retryable=?,
                    quarantine_reason=CASE WHEN ?=0 THEN ? ELSE quarantine_reason END,
                    error_msg=?,
                    processing_lock='',
                    updated_at=?
                WHERE id=?
                """,
                (
                    retry_count,
                    str(reason or "unknown"),
                    str(error_msg or reason or "repair_failed"),
                    1 if can_retry else 0,
                    1 if can_retry else 0,
                    str(reason or "unknown"),
                    str(error_msg or reason or "repair_failed"),
                    now,
                    int(file_id),
                ),
            )
        return {
            "retry_count": retry_count,
            "retryable": can_retry,
            "reason": str(reason or "unknown"),
        }

    def clear_light_repair_failure(self, file_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET
                    repair_retry_count=0,
                    repair_failure_reason='',
                    repair_last_error='',
                    repair_retryable=1
                WHERE id=?
                """,
                (int(file_id),),
            )

    def list_ai_integrity_candidates(
        self,
        *,
        after_id: int = 0,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        """DB-only scan: artifact kaybetmiş heavy_done satırları."""
        dna_ready = _json_object_ready_sql("fe.texture_map", "pattern_dna")
        semantic_ready = _json_object_ready_sql("fe.texture_map", "semantic_tags")
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT f.id, f.path, f.filename, f.status,
                       f.light_status, f.heavy_status,
                       f.thumbnail_path, f.feature_preview_path,
                       CASE WHEN fe.dino_embedding IS NOT NULL
                                  AND length(fe.dino_embedding)>0
                            THEN 0 ELSE 1 END AS missing_dino,
                       CASE WHEN fe.clip_embedding IS NOT NULL
                                  AND length(fe.clip_embedding)>0
                            THEN 0 ELSE 1 END AS missing_clip,
                       CASE WHEN COALESCE(fe.phash,'')!=''
                                  AND COALESCE(fe.texture_features,'') NOT IN ('','[]')
                            THEN 0 ELSE 1 END AS missing_texture,
                       CASE WHEN {semantic_ready} THEN 0 ELSE 1 END
                            AS missing_semantic,
                       CASE WHEN {dna_ready} THEN 0 ELSE 1 END
                            AS missing_dna,
                       CASE WHEN COALESCE(fe.patch_embeddings,'') NOT IN ('','[]')
                            THEN 0 ELSE 1 END AS missing_patch
                FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                WHERE f.id>?
                  AND f.status NOT IN ('excluded_internal','missing')
                  AND f.light_status='done'
                  AND f.heavy_status='done'
                  AND (
                        fe.dino_embedding IS NULL
                        OR length(fe.dino_embedding)=0
                        OR fe.clip_embedding IS NULL
                        OR length(fe.clip_embedding)=0
                        OR COALESCE(fe.phash,'')=''
                        OR COALESCE(fe.texture_features,'') IN ('','[]')
                        OR NOT {semantic_ready}
                        OR NOT {dna_ready}
                        OR COALESCE(fe.patch_embeddings,'') IN ('','[]')
                      )
                ORDER BY f.id
                LIMIT ?
                """,
                (max(0, int(after_id)), max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_physical_readiness(
        self,
        file_id: int,
        *,
        thumbnail_ready: bool,
        preview_ready: bool,
        verified_at: str | None = None,
        requeue_missing: bool = True,
    ) -> None:
        """Fiziksel cache SSOT; stale kayıtları preview-repair lane'ine al."""
        now = verified_at or datetime.now(timezone.utc).isoformat()
        # V3 heavy/AI kapısı yalnız gerçek Preview'a bağlıdır.
        # Thumbnail ayrı bir UI artifact'idir ve Preview hazırken AI'yi bloke etmez.
        ready = bool(preview_ready)
        with self.connect() as conn:
            if ready or not requeue_missing:
                conn.execute(
                    """
                    UPDATE files SET
                        physical_thumbnail_ready=?,
                        physical_preview_ready=?,
                        physical_verified_at=?,
                        needs_medium_preview=CASE WHEN ? THEN 0
                                                  ELSE needs_medium_preview END,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        1 if thumbnail_ready else 0,
                        1 if preview_ready else 0,
                        now,
                        1 if ready else 0,
                        now,
                        int(file_id),
                    ),
                )
                return
            conn.execute(
                """
                UPDATE files SET
                    physical_thumbnail_ready=?,
                    physical_preview_ready=?,
                    physical_verified_at=?,
                    heavy_status='pending',
                    index_stage='needs_medium_preview',
                    needs_medium_preview=1,
                    processing_lock='',
                    updated_at=?
                WHERE id=?
                  AND status NOT IN ('excluded_internal','missing')
                """,
                (
                    1 if thumbnail_ready else 0,
                    1 if preview_ready else 0,
                    now,
                    now,
                    int(file_id),
                ),
            )

    def count_physical_readiness(self) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN physical_thumbnail_ready=1
                                  AND physical_preview_ready=1
                             THEN 1 ELSE 0 END) AS physical_ready,
                    SUM(CASE WHEN physical_thumbnail_ready=0
                                  OR physical_preview_ready=0
                             THEN 1 ELSE 0 END) AS physical_missing,
                    SUM(CASE WHEN physical_thumbnail_ready IS NULL
                                  OR physical_preview_ready IS NULL
                             THEN 1 ELSE 0 END) AS physical_unverified,
                    SUM(CASE WHEN coalesce(thumbnail_path,'')!=''
                                  AND coalesce(feature_preview_path,'')!=''
                             THEN 1 ELSE 0 END) AS db_path_ready,
                    MAX(coalesce(physical_verified_at,'')) AS last_verified_at
                FROM files
                WHERE status NOT IN ('excluded_internal','missing')
                """
            ).fetchone()
            return {
                key: (
                    str(row[key] or "")
                    if key == "last_verified_at"
                    else int(row[key] or 0)
                )
                for key in row.keys()
            }

    def upsert_file(self, record: dict[str, Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        path = normalize_path(record["path"])
        with self.connect() as conn:
            existing_row = conn.execute(
                "SELECT * FROM files WHERE path = ?", (path,)
            ).fetchone()
            if existing_row:
                existing = dict(existing_row)
                merged = {
                    **existing,
                    **{
                        k: v
                        for k, v in record.items()
                        if v not in (None, "", 0)
                        or k in ("status", "error_msg", "parser_error")
                    },
                }
                # status ve error_msg her zaman güncellenebilir
                merged["status"] = record.get(
                    "status", existing.get("status", "pending")
                )
                if "error_msg" in record:
                    merged["error_msg"] = record["error_msg"]
                conn.execute(
                    """
                    UPDATE files SET
                        filename=?, customer=?, file_size=?, mtime=?,
                        width=?, height=?, status=?, thumbnail_path=?,
                        partial_hash=?, full_hash=?, error_msg=?, ocr_text=?,
                        indexed_at=?, source_id=?, feature_version=?,
                        texture_version=?, thumbnail_version=?,
                        last_feature_at=?, last_texture_at=?,
                        feature_preview_path=?, feature_preview_version=?,
                        feature_preview_mtime=?, unsupported_preview=?,
                        pattern_subtype=?, pattern_confidence=?, preview_status=?,
                        format_family=?, format_extension=?, format_capabilities=?,
                        format_metadata=?, thumbnail_status=?, metadata_status=?,
                        semantic_status=?, parser_name=?, parser_version=?, parser_error=?,
                        project_group_id=?, project_group_confidence=?,
                        project_group_reason=?, project_group_status=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        merged.get("filename", ""),
                        merged.get("customer", ""),
                        merged.get("file_size", 0),
                        merged.get("mtime", 0),
                        merged.get("width", 0),
                        merged.get("height", 0),
                        merged.get("status", "pending"),
                        merged.get("thumbnail_path", ""),
                        merged.get("partial_hash", ""),
                        merged.get("full_hash", ""),
                        merged.get("error_msg", ""),
                        merged.get("ocr_text", ""),
                        merged.get("indexed_at", ""),
                        merged.get("source_id", 0),
                        merged.get(
                            "feature_version", existing.get("feature_version", 0)
                        ),
                        merged.get(
                            "texture_version", existing.get("texture_version", 0)
                        ),
                        merged.get(
                            "thumbnail_version", existing.get("thumbnail_version", 0)
                        ),
                        merged.get(
                            "last_feature_at", existing.get("last_feature_at", "")
                        ),
                        merged.get(
                            "last_texture_at", existing.get("last_texture_at", "")
                        ),
                        merged.get(
                            "feature_preview_path",
                            existing.get("feature_preview_path", ""),
                        ),
                        int(
                            merged.get(
                                "feature_preview_version",
                                existing.get("feature_preview_version", 0),
                            )
                        ),
                        float(
                            merged.get(
                                "feature_preview_mtime",
                                existing.get("feature_preview_mtime", 0),
                            )
                        ),
                        int(
                            merged.get(
                                "unsupported_preview",
                                existing.get("unsupported_preview", 0),
                            )
                        ),
                        merged.get(
                            "pattern_subtype", existing.get("pattern_subtype", "")
                        ),
                        float(
                            merged.get(
                                "pattern_confidence",
                                existing.get("pattern_confidence", 0),
                            )
                            or 0
                        ),
                        merged.get(
                            "preview_status", existing.get("preview_status", "")
                        ),
                        merged.get("format_family", existing.get("format_family", "")),
                        merged.get("format_extension", existing.get("format_extension", "")),
                        merged.get("format_capabilities", existing.get("format_capabilities", "[]")),
                        merged.get("format_metadata", existing.get("format_metadata", "{}")),
                        merged.get("thumbnail_status", existing.get("thumbnail_status", "")),
                        merged.get("metadata_status", existing.get("metadata_status", "")),
                        merged.get("semantic_status", existing.get("semantic_status", "")),
                        merged.get("parser_name", existing.get("parser_name", "")),
                        merged.get("parser_version", existing.get("parser_version", "")),
                        merged.get("parser_error", existing.get("parser_error", "")),
                        merged.get("project_group_id", existing.get("project_group_id", "")),
                        float(merged.get("project_group_confidence", existing.get("project_group_confidence", 0)) or 0),
                        merged.get("project_group_reason", existing.get("project_group_reason", "")),
                        merged.get("project_group_status", existing.get("project_group_status", "")),
                        now,
                        existing["id"],
                    ),
                )
                return existing["id"]
            cur = conn.execute(
                """
                INSERT INTO files(
                    path, filename, customer, file_size, mtime,
                    width, height, status, thumbnail_path,
                    partial_hash, full_hash, error_msg, ocr_text,
                    indexed_at, source_id, feature_version, texture_version,
                    thumbnail_version, last_feature_at, last_texture_at,
                    feature_preview_path, feature_preview_version, feature_preview_mtime,
                    unsupported_preview, pattern_subtype, pattern_confidence, preview_status,
                    format_family, format_extension, format_capabilities, format_metadata,
                    thumbnail_status, metadata_status, semantic_status,
                    parser_name, parser_version, parser_error,
                    project_group_id, project_group_confidence,
                    project_group_reason, project_group_status,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    path,
                    record.get("filename", ""),
                    record.get("customer", ""),
                    record.get("file_size", 0),
                    record.get("mtime", 0),
                    record.get("width", 0),
                    record.get("height", 0),
                    record.get("status", "pending"),
                    record.get("thumbnail_path", ""),
                    record.get("partial_hash", ""),
                    record.get("full_hash", ""),
                    record.get("error_msg", ""),
                    record.get("ocr_text", ""),
                    record.get("indexed_at", ""),
                    record.get("source_id", 0),
                    record.get("feature_version", 0),
                    record.get("texture_version", 0),
                    record.get("thumbnail_version", 0),
                    record.get("last_feature_at", ""),
                    record.get("last_texture_at", ""),
                    record.get("feature_preview_path", ""),
                    record.get("feature_preview_version", 0),
                    float(record.get("feature_preview_mtime", 0)),
                    int(record.get("unsupported_preview", 0)),
                    record.get("pattern_subtype", ""),
                    float(record.get("pattern_confidence", 0) or 0),
                    record.get("preview_status", ""),
                    record.get("format_family", ""),
                    record.get("format_extension", ""),
                    record.get("format_capabilities", "[]"),
                    record.get("format_metadata", "{}"),
                    record.get("thumbnail_status", ""),
                    record.get("metadata_status", ""),
                    record.get("semantic_status", ""),
                    record.get("parser_name", ""),
                    record.get("parser_version", ""),
                    record.get("parser_error", ""),
                    record.get("project_group_id", ""),
                    float(record.get("project_group_confidence", 0) or 0),
                    record.get("project_group_reason", ""),
                    record.get("project_group_status", ""),
                    now,
                    now,
                ),
            )
            return cur.lastrowid

    def get_source_discovery_rows(self, source_id: int) -> list[dict[str, Any]]:
        """Read only discovery columns for a source in one query."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, path, file_size, mtime, status, source_id
                FROM files
                WHERE source_id=?
                """,
                (int(source_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    def bulk_insert_source_files(self, records: list[dict[str, Any]]) -> dict[str, int]:
        """Fast discovery insert: one SQLite transaction for many new source files.

        This intentionally writes only discovery fields. Existing artifacts and
        feature columns are never touched, so an incremental rescan cannot
        destroy previously completed indexing.
        """
        if not records:
            return {}
        now = datetime.now(timezone.utc).isoformat()
        result: dict[str, int] = {}
        with self.connect() as conn:
            conn.execute("BEGIN")
            for rec in records:
                path = normalize_path(str(rec["path"]))
                conn.execute(
                    """
                    INSERT OR IGNORE INTO files(
                        path, filename, source_id, file_size, mtime,
                        status, light_status, heavy_status, created_at, updated_at
                    ) VALUES(?,?,?,?,?,'pending','pending','pending',?,?)
                    """,
                    (
                        path,
                        str(rec.get("filename") or Path(path).name),
                        int(rec.get("source_id") or 0),
                        int(rec.get("file_size") or 0),
                        float(rec.get("mtime") or 0),
                        now,
                        now,
                    ),
                )
            paths = [normalize_path(str(r["path"])) for r in records]
            result = {}
            chunk = 400
            for i in range(0, len(paths), chunk):
                part = paths[i : i + chunk]
                placeholders = ",".join("?" for _ in part)
                rows = conn.execute(
                    f"SELECT id, path FROM files WHERE path IN ({placeholders})",
                    part,
                ).fetchall()
                result.update(
                    {normalize_path(str(r["path"])): int(r["id"]) for r in rows}
                )
            conn.commit()
        return result

    def get_source_artifact_rows(self, source_id: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Fetch file + feature state for a source in one DB round trip."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT f.*, fe.phash, fe.dhash, fe.whash, fe.color_hist,
                       fe.dominant_colors, fe.texture_features,
                       fe.dino_embedding, fe.clip_embedding,
                       fe.patch_embeddings, fe.texture_map
                FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                WHERE f.source_id=? AND f.status!='missing'
                ORDER BY f.id
                """,
                (int(source_id),),
            ).fetchall()
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in rows:
            d = dict(row)
            feat = {
                "phash": d.pop("phash", ""),
                "dhash": d.pop("dhash", ""),
                "whash": d.pop("whash", ""),
                "color_hist": d.pop("color_hist", None),
                "dominant_colors": d.pop("dominant_colors", "[]"),
                "texture_features": d.pop("texture_features", "[]"),
                "dino_embedding": d.pop("dino_embedding", None),
                "clip_embedding": d.pop("clip_embedding", None),
                "patch_embeddings": d.pop("patch_embeddings", "[]"),
                "patch_embeddings_meta": d.get("patch_embeddings", "[]"),
                "texture_map": d.pop("texture_map", "{}"),
            }
            # Preserve JSON decoding contract of Database.get_features().
            for key in ("dominant_colors", "texture_features", "patch_embeddings_meta", "texture_map"):
                try:
                    feat[key] = json.loads(feat[key] or ("{}" if key == "texture_map" else "[]"))
                except Exception:
                    feat[key] = {} if key == "texture_map" else []
            out.append((d, feat))
        return out

    def get_artifact_rows_for_ids(
        self, file_ids: list[int]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Streaming discovery: yalnız yeni/değişen file_id satırları."""
        ids = [int(x) for x in (file_ids or []) if int(x or 0) > 0]
        if not ids:
            return []
        sql = """
                SELECT f.*, fe.phash, fe.dhash, fe.whash, fe.color_hist,
                       fe.dominant_colors, fe.texture_features,
                       fe.dino_embedding, fe.clip_embedding,
                       fe.patch_embeddings, fe.texture_map
                FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                WHERE f.id IN ({ph})
                """
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        chunk = 400
        with self.connect() as conn:
            for i in range(0, len(ids), chunk):
                part = ids[i : i + chunk]
                ph = ",".join("?" for _ in part)
                rows = conn.execute(sql.format(ph=ph), part).fetchall()
                for row in rows:
                    d = dict(row)
                    feat = {
                        "phash": d.pop("phash", ""),
                        "dhash": d.pop("dhash", ""),
                        "whash": d.pop("whash", ""),
                        "color_hist": d.pop("color_hist", None),
                        "dominant_colors": d.pop("dominant_colors", "[]"),
                        "texture_features": d.pop("texture_features", "[]"),
                        "dino_embedding": d.pop("dino_embedding", None),
                        "clip_embedding": d.pop("clip_embedding", None),
                        "patch_embeddings": d.pop("patch_embeddings", "[]"),
                        "patch_embeddings_meta": d.get("patch_embeddings", "[]"),
                        "texture_map": d.pop("texture_map", "{}"),
                    }
                    for key in (
                        "dominant_colors",
                        "texture_features",
                        "patch_embeddings_meta",
                        "texture_map",
                    ):
                        try:
                            feat[key] = json.loads(
                                feat[key] or ("{}" if key == "texture_map" else "[]")
                            )
                        except Exception:
                            feat[key] = {} if key == "texture_map" else []
                    out.append((d, feat))
        return out

    def get_file_by_path(self, path: str) -> dict[str, Any] | None:
        path = normalize_path(path)
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
            return dict(row) if row else None

    def find_reusable_thumbnail(
        self,
        *,
        filename: str,
        file_size: int,
        mtime: float,
        exclude_file_id: int = 0,
    ) -> dict[str, Any] | None:
        """Find an already-rendered copy without reopening a large network image."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM files
                WHERE id!=?
                  AND filename=? COLLATE NOCASE
                  AND file_size=?
                  AND abs(mtime-?)<=0.5
                  AND status='indexed'
                  AND coalesce(thumbnail_path,'')!=''
                ORDER BY id DESC
                LIMIT 1
                """,
                (int(exclude_file_id), filename, int(file_size), float(mtime)),
            ).fetchone()
            return dict(row) if row else None

    def get_file_by_id(self, file_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,)
            ).fetchone()
            return dict(row) if row else None

    def mark_missing_files(self, known_paths: Iterable[str]) -> int:
        known = {normalize_path(p) for p in known_paths}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, path FROM files WHERE status != 'missing'"
            ).fetchall()
            count = 0
            for row in rows:
                if row["path"] not in known:
                    conn.execute(
                        "UPDATE files SET status='missing', updated_at=? WHERE id=?",
                        (datetime.now(timezone.utc).isoformat(), row["id"]),
                    )
                    count += 1
            return count

    def update_feature_hashes(
        self,
        file_id: int,
        *,
        phash: str,
        dhash: str,
        whash: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE features SET phash=?, dhash=?, whash=? WHERE file_id=?",
                (phash, dhash, whash, file_id),
            )

    def upsert_features(self, file_id: int, features: dict[str, Any]) -> None:
        """Insert or merge features so parallel artifact workers cannot clobber
        each other's blobs (DINO vs CLIP vs patch vs texture_map tags).
        """
        with self.connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                pass
            row = conn.execute(
                "SELECT * FROM features WHERE file_id=?", (int(file_id),)
            ).fetchone()
            if row:
                old = dict(row)
                merged = _merge_feature_payload(old, features)
            else:
                merged = {
                    "phash": features.get("phash", "") or "",
                    "dhash": features.get("dhash", "") or "",
                    "whash": features.get("whash", "") or "",
                    "color_hist": features.get("color_hist"),
                    "dominant_colors": features.get("dominant_colors", []),
                    "texture_features": features.get("texture_features", []),
                    "dino_embedding": features.get("dino_embedding"),
                    "clip_embedding": features.get("clip_embedding"),
                    "patch_embeddings_meta": features.get("patch_embeddings_meta", []),
                    "texture_map": features.get("texture_map", {}),
                }
            conn.execute(
                """
                INSERT INTO features(
                    file_id, phash, dhash, whash, color_hist, dominant_colors,
                    texture_features, dino_embedding, clip_embedding, patch_embeddings,
                    texture_map
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(file_id) DO UPDATE SET
                    phash=excluded.phash, dhash=excluded.dhash, whash=excluded.whash,
                    color_hist=excluded.color_hist, dominant_colors=excluded.dominant_colors,
                    texture_features=excluded.texture_features,
                    dino_embedding=excluded.dino_embedding,
                    clip_embedding=excluded.clip_embedding,
                    patch_embeddings=excluded.patch_embeddings,
                    texture_map=excluded.texture_map
                """,
                (
                    file_id,
                    merged.get("phash", ""),
                    merged.get("dhash", ""),
                    merged.get("whash", ""),
                    merged.get("color_hist"),
                    json.dumps(merged.get("dominant_colors", []), ensure_ascii=False),
                    json.dumps(merged.get("texture_features", []), ensure_ascii=False),
                    merged.get("dino_embedding"),
                    merged.get("clip_embedding"),
                    json.dumps(merged.get("patch_embeddings_meta", []), ensure_ascii=False),
                    json.dumps(merged.get("texture_map", {}), ensure_ascii=False),
                ),
            )

    def upsert_texture_map(self, file_id: int, texture_map: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO features(file_id, texture_map)
                VALUES (?, ?)
                ON CONFLICT(file_id) DO UPDATE SET texture_map=excluded.texture_map
                """,
                (file_id, json.dumps(texture_map, ensure_ascii=False)),
            )

    def count_missing_texture_maps(self) -> int:
        with self.connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) AS c FROM files f
                LEFT JOIN features fe ON f.id = fe.file_id
                WHERE f.status='indexed'
                  AND (fe.texture_map IS NULL OR fe.texture_map='' OR fe.texture_map='{}')
                """).fetchone()
            return int(row["c"]) if row else 0

    def get_features(self, file_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM features WHERE file_id = ?", (file_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["dominant_colors"] = json.loads(d.get("dominant_colors") or "[]")
            d["texture_features"] = json.loads(d.get("texture_features") or "[]")
            d["patch_embeddings_meta"] = json.loads(d.get("patch_embeddings") or "[]")
            d["texture_map"] = json.loads(d.get("texture_map") or "{}")
            return d

    # --- Kaynak yönetimi ---

    def upsert_source(self, record: dict[str, Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        root = normalize_source_root(record["root_path"])
        with self.connect() as conn:
            if record.get("id"):
                conn.execute(
                    """
                    UPDATE sources SET
                        name=?, source_type=?, root_path=?, is_active=?,
                        scan_interval_hours=?, deep_scan_interval_days=?,
                        file_count=?, error_count=?, cache_status=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        record.get("name", ""),
                        record.get("source_type", "local_pc"),
                        root,
                        int(record.get("is_active", 1)),
                        int(record.get("scan_interval_hours", 6)),
                        int(record.get("deep_scan_interval_days", 7)),
                        int(record.get("file_count", 0)),
                        int(record.get("error_count", 0)),
                        record.get("cache_status", "empty"),
                        now,
                        record["id"],
                    ),
                )
                return int(record["id"])
            existing = conn.execute(
                "SELECT id FROM sources WHERE root_path=?", (root,)
            ).fetchone()
            if existing:
                record["id"] = existing["id"]
                conn.execute(
                    """
                    UPDATE sources SET
                        name=?, source_type=?, is_active=?,
                        scan_interval_hours=?, deep_scan_interval_days=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        record.get("name", ""),
                        record.get("source_type", "local_pc"),
                        int(record.get("is_active", 1)),
                        int(record.get("scan_interval_hours", 6)),
                        int(record.get("deep_scan_interval_days", 7)),
                        now,
                        existing["id"],
                    ),
                )
                return int(existing["id"])
            cur = conn.execute(
                """
                INSERT INTO sources(
                    name, source_type, root_path, is_active,
                    scan_interval_hours, deep_scan_interval_days,
                    file_count, error_count, cache_status,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.get("name", ""),
                    record.get("source_type", "local_pc"),
                    root,
                    int(record.get("is_active", 1)),
                    int(record.get("scan_interval_hours", 6)),
                    int(record.get("deep_scan_interval_days", 7)),
                    0,
                    0,
                    record.get("cache_status", "empty"),
                    now,
                    now,
                ),
            )
            return cur.lastrowid

    def get_source(self, source_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE id=?", (source_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_source_by_path(self, root_path: str) -> dict[str, Any] | None:
        root = normalize_source_root(root_path)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE root_path=? COLLATE NOCASE", (root,)
            ).fetchone()
            return dict(row) if row else None

    def list_sources(self, active_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM sources"
        if active_only:
            query += " WHERE is_active=1"
        query += " ORDER BY name"
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(query).fetchall()]

    def delete_source(self, source_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM sources WHERE id=?", (source_id,))

    def deactivate_source(self, source_id: int) -> None:
        """Kaynağı arama havuzundan çıkar — dosyaları silmez."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                "UPDATE sources SET is_active=0, updated_at=? WHERE id=?",
                (now, source_id),
            )

    def list_files_for_source(self, source_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, path, thumbnail_path, feature_preview_path FROM files WHERE source_id=?",
                (source_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_files_for_source(self, source_id: int) -> list[int]:
        """source_id'ye bağlı tüm index kayıtlarını sil."""
        removed = self.purge_source_records(source_id)
        return list(range(removed))

    def list_file_artifact_paths(
        self,
        *,
        file_ids: list[int] | None = None,
        source_id: int | None = None,
        missing_only: bool = False,
    ) -> list[str]:
        clauses: list[str] = []
        params: list[Any] = []
        if file_ids is not None:
            ids = [int(value) for value in file_ids if int(value) > 0]
            if not ids:
                return []
            clauses.append(f"id IN ({','.join('?' * len(ids))})")
            params.extend(ids)
        if source_id is not None:
            clauses.append("source_id=?")
            params.append(int(source_id))
        if missing_only:
            clauses.append("status='missing'")
        where = " AND ".join(clauses) if clauses else "1=1"
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT thumbnail_path, feature_preview_path
                FROM files WHERE {where}
                """,
                params,
            ).fetchall()
        return [
            str(path)
            for row in rows
            for path in (row["thumbnail_path"], row["feature_preview_path"])
            if str(path or "").strip()
        ]

    @staticmethod
    def remove_cache_artifacts(paths: list[str], cache_dir: str | Path) -> int:
        """Yalnız cache root altındaki bilinen artifact dosyalarını sil."""
        root = Path(cache_dir).resolve()
        removed = 0
        seen: set[str] = set()
        for raw in paths:
            value = str(raw or "").strip()
            if not value:
                continue
            candidate = Path(value)
            candidates = (
                [candidate]
                if candidate.is_absolute()
                else [Path.cwd() / candidate, root.parent / candidate]
            )
            for item in candidates:
                try:
                    resolved = item.resolve()
                    key = str(resolved).lower()
                    if key in seen or not resolved.is_relative_to(root):
                        continue
                    seen.add(key)
                    if resolved.is_file():
                        resolved.unlink()
                        removed += 1
                    break
                except OSError:
                    continue
        return removed

    def purge_source_records(
        self, source_id: int, *, cache_dir: str | Path | None = None
    ) -> int:
        """Kaynak index kayıtlarını toplu sil — büyük kaynaklarda UI donmasın."""
        artifact_paths = (
            self.list_file_artifact_paths(source_id=source_id) if cache_dir else []
        )
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM files WHERE source_id=?",
                (source_id,),
            ).fetchone()
            count = int(row["c"]) if row else 0
            if count == 0:
                conn.execute("DELETE FROM index_queue WHERE source_id=?", (source_id,))
                return 0

            sub = "SELECT id FROM files WHERE source_id=?"
            conn.execute(
                f"DELETE FROM pattern_group_members WHERE file_id IN ({sub})",
                (source_id,),
            )
            conn.execute(
                f"DELETE FROM user_feedback WHERE result_file_id IN ({sub})",
                (source_id,),
            )
            conn.execute(
                f"DELETE FROM collection_items WHERE file_id IN ({sub})",
                (source_id,),
            )
            try:
                conn.execute(
                    f"DELETE FROM files_fts WHERE file_id IN ({sub})",
                    (source_id,),
                )
            except sqlite3.OperationalError:
                pass
            conn.execute("DELETE FROM index_queue WHERE source_id=?", (source_id,))
            conn.execute("DELETE FROM files WHERE source_id=?", (source_id,))
            if cache_dir:
                self.remove_cache_artifacts(artifact_paths, cache_dir)
            return count

    def count_missing_files(self, source_id: int | None = None) -> int:
        with self.connect() as conn:
            if source_id:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM files WHERE status='missing' AND source_id=?",
                    (source_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM files WHERE status='missing'",
                ).fetchone()
            return int(row["c"] or 0) if row else 0

    def purge_file_ids(
        self,
        file_ids: list[int],
        *,
        cache_dir: str | Path | None = None,
    ) -> int:
        """Verilen file_id kayıtlarını ve bağlı index verilerini sil."""
        ids = [int(i) for i in file_ids if int(i) > 0]
        if not ids:
            return 0
        artifact_paths = (
            self.list_file_artifact_paths(file_ids=ids) if cache_dir else []
        )
        removed = 0
        with self.connect() as conn:
            for offset in range(0, len(ids), 500):
                batch = ids[offset : offset + 500]
                placeholders = ",".join("?" * len(batch))
                conn.execute(
                    f"DELETE FROM pattern_group_members WHERE file_id IN ({placeholders})",
                    batch,
                )
                conn.execute(
                    f"DELETE FROM user_feedback WHERE result_file_id IN ({placeholders})",
                    batch,
                )
                conn.execute(
                    f"DELETE FROM collection_items WHERE file_id IN ({placeholders})",
                    batch,
                )
                try:
                    conn.execute(
                        f"DELETE FROM files_fts WHERE file_id IN ({placeholders})",
                        batch,
                    )
                except sqlite3.OperationalError:
                    pass
                conn.execute(
                    f"DELETE FROM features WHERE file_id IN ({placeholders})",
                    batch,
                )
                conn.execute(
                    f"DELETE FROM index_queue WHERE file_id IN ({placeholders})",
                    batch,
                )
                cur = conn.execute(
                    f"DELETE FROM files WHERE id IN ({placeholders})",
                    batch,
                )
                removed += max(0, cur.rowcount)
        if cache_dir:
            self.remove_cache_artifacts(artifact_paths, cache_dir)
        return removed

    def purge_missing_files(
        self,
        source_id: int | None = None,
        *,
        cache_dir: str | Path | None = None,
    ) -> int:
        """status=missing kayıtlarını veritabanından tamamen kaldır."""
        with self.connect() as conn:
            if source_id:
                rows = conn.execute(
                    "SELECT id FROM files WHERE status='missing' AND source_id=?",
                    (source_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM files WHERE status='missing'",
                ).fetchall()
        ids = [int(r["id"]) for r in rows]
        if not ids:
            return 0
        removed = self.purge_file_ids(ids, cache_dir=cache_dir)
        if removed:
            self.set_meta("faiss_needs_rebuild", "1")
        return removed

    def update_file_text_index(
        self,
        file_id: int,
        *,
        pattern_family: str = "",
        pattern_type: str = "",
        texture_family: str = "",
        pattern_subtype: str = "",
        pattern_confidence: float = 0.0,
        text_search_blob: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET
                    pattern_family=?, pattern_type=?, texture_family=?,
                    pattern_subtype=?, pattern_confidence=?,
                    text_search_blob=?, updated_at=?
                WHERE id=?
                """,
                (
                    pattern_family,
                    pattern_type,
                    texture_family,
                    pattern_subtype,
                    pattern_confidence,
                    text_search_blob,
                    datetime.now(timezone.utc).isoformat(),
                    file_id,
                ),
            )
            self._upsert_fts(conn, file_id, text_search_blob)

    def update_file_category(
        self,
        file_id: int,
        *,
        category_path: str = "",
        manual_category_path: str = "",
        category_confidence: float = 0.0,
        category_source: str = "",
        category_aliases: list[str] | None = None,
        pattern_family: str = "",
        pattern_subtype: str = "",
        pattern_type: str = "",
        text_search_blob: str = "",
        clear_wrong_legacy: bool = False,
    ) -> None:
        """Kategori ağacı alanlarını ve ilişkili pattern sütunlarını güncelle."""
        aliases_json = json.dumps(category_aliases or [], ensure_ascii=False)
        with self.connect() as conn:
            fields = {
                "category_path": category_path,
                "manual_category_path": manual_category_path,
                "category_confidence": float(category_confidence),
                "category_source": category_source,
                "category_aliases": aliases_json,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if pattern_family:
                fields["pattern_family"] = pattern_family
            if pattern_subtype:
                fields["pattern_subtype"] = pattern_subtype
            if pattern_type:
                fields["pattern_type"] = pattern_type
            if text_search_blob:
                fields["text_search_blob"] = text_search_blob
            if clear_wrong_legacy and pattern_family:
                # manual animal illustration should not keep animal_print traces
                if pattern_family not in ("animal_print",):
                    pass  # animal_print_type cleared via texture_map in learning module
            set_clause = ", ".join(f"{k}=?" for k in fields)
            conn.execute(
                f"UPDATE files SET {set_clause} WHERE id=?",
                (*fields.values(), file_id),
            )
            if text_search_blob:
                self._upsert_fts(conn, file_id, text_search_blob)

    def list_file_ids_by_category_path(
        self,
        category_path: str,
        *,
        limit: int = 5000,
    ) -> list[int]:
        """Kategoriye göre dosya ID'lerini döndür.

        Marka kategorisi için yalnızca ``files.category_path`` alanına güvenme.
        Eski indekslerde marka kanıtı OCR, category_aliases, text_search_blob veya
        features.texture_map içinde kalmış olabilir. Arama motoru ile kategori
        panelinin aynı kanıt kümesini kullanması özellikle Marka için zorunludur:
        ``Amiri`` altında görünen her dosya ``Marka`` altında da görünmelidir.
        """
        path = (category_path or "").strip().strip("/")
        if not path:
            return []
        like = path + "/%"

        # Kişi/Ünlü legacy kayıtları da aynı şekilde dosya adı, OCR ve
        # text_search_blob içinde kalmış olabilir. Burada yalnızca metadata
        # aranır; görüntüden kişi adı çıkarılmaz.
        person_prefixes = ("Kişi/", "Kişi /", "Kisi/", "Kisi /")
        if path == "Kişi" or path == "Kisi" or any(path.startswith(x) for x in person_prefixes):
            wanted = ""
            if "/" in path:
                wanted = path.split("/", 1)[1].strip()
                if wanted.lower().startswith("ünlü/") or wanted.lower().startswith("unlu/"):
                    wanted = wanted.split("/", 1)[1].strip()
                wanted_norm = normalize_turkish(wanted)
            else:
                wanted_norm = ""
            with self.connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT f.id, f.filename, f.ocr_text, f.category_path,
                           f.manual_category_path, f.text_search_blob
                    FROM files f
                    LEFT JOIN features fe ON fe.file_id=f.id
                    WHERE {sql_processing_ready_clause()}
                    ORDER BY f.id DESC
                    """
                ).fetchall()
            out = []
            for row in rows:
                rec = dict(row)
                hay = " ".join(str(rec.get(k) or "") for k in (
                    "filename", "ocr_text", "category_path",
                    "manual_category_path", "text_search_blob"
                ))
                nhay = normalize_turkish(hay)
                generic_celebrity = wanted_norm in {"unlu", "celebrity", ""}
                if wanted_norm and not generic_celebrity:
                    if wanted_norm not in nhay:
                        continue
                else:
                    # Generic Kişi/Ünlü category: only explicit person/celebrity
                    # metadata is accepted, not every file containing "person".
                    if not any(x in nhay for x in ("kisi", "unlu", "celebrity", "portrait", "portre")):
                        continue
                out.append(int(rec["id"]))
                if limit > 0 and len(out) >= int(limit):
                    break
            return out

        # Marka ağacında legacy kayıtların category_path'ı boş/yanlış olabilir.
        # Aynı ortak evidence extractor'ı kullanarak kesin küme üret. Bu yol
        # yalnızca kategori filtresinde çalışır; indeksleme hot-path'ine girmez.
        if path == "Marka" or path.startswith("Marka/"):
            from core.brand_evidence import extract_brand_evidence
            from core.brand_aliases import normalize_brand_key, resolve_brand_alias

            wanted = ""
            if path.startswith("Marka/"):
                wanted_raw = path.split("/", 1)[1].strip()
                wanted = normalize_brand_key(
                    resolve_brand_alias(wanted_raw) or wanted_raw
                )

            with self.connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT f.id, f.filename, f.ocr_text, f.category_path,
                           f.manual_category_path, f.category_aliases,
                           f.text_search_blob, fe.texture_map
                    FROM files f
                    LEFT JOIN features fe ON fe.file_id=f.id
                    WHERE {sql_processing_ready_clause()}
                      AND (f.source_id=0 OR NOT EXISTS (
                          SELECT 1 FROM sources s
                          WHERE s.id=f.source_id AND s.is_active=0
                      ))
                    ORDER BY f.id DESC
                    """
                ).fetchall()

            out: list[int] = []
            for row in rows:
                rec = dict(row)
                try:
                    rec["texture_map"] = json.loads(rec.get("texture_map") or "{}")
                except Exception:
                    rec["texture_map"] = {}
                evidence = extract_brand_evidence(rec, self.db_path)
                if not evidence:
                    continue
                if wanted and wanted not in evidence:
                    continue
                out.append(int(rec["id"]))
                if limit > 0 and len(out) >= int(limit):
                    break
            return out

        sql_limit = int(limit) if int(limit) > 0 else -1
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT f.id FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                WHERE {sql_processing_ready_clause()}
                  AND (
                    f.category_path=? OR f.category_path LIKE ?
                    OR f.manual_category_path=? OR f.manual_category_path LIKE ?
                  )
                ORDER BY f.id DESC
                LIMIT ?
                """,
                (path, like, path, like, sql_limit),
            ).fetchall()
        return [int(r["id"]) for r in rows]

    def _upsert_fts(self, conn: sqlite3.Connection, file_id: int, blob: str) -> None:
        try:
            conn.execute("DELETE FROM files_fts WHERE file_id=?", (file_id,))
            if blob:
                conn.execute(
                    "INSERT INTO files_fts(file_id, text_search_blob) VALUES (?,?)",
                    (file_id, blob),
                )
        except sqlite3.OperationalError:
            pass

    def _delete_fts_ids(self, conn: sqlite3.Connection, file_ids: list[int]) -> None:
        if not file_ids:
            return
        try:
            batch_size = 500
            for i in range(0, len(file_ids), batch_size):
                chunk = file_ids[i : i + batch_size]
                placeholders = ",".join("?" * len(chunk))
                conn.execute(
                    f"DELETE FROM files_fts WHERE file_id IN ({placeholders})",
                    chunk,
                )
        except sqlite3.OperationalError:
            pass

    def search_text_candidates(
        self,
        terms: list[str],
        limit: int = 5000,
        customer: str = "",
        source_types: list[str] | None = None,
        source_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Metin arama adayları — FTS5 veya LIKE fallback."""
        if not terms:
            return []
        # Kısa / ham terimler önce — "allover leopard spot" dosya adını kaçırmasın
        ordered: list[str] = []
        for t in sorted(
            (str(x).strip() for x in terms if str(x).strip()),
            key=lambda s: (len(s.split()), len(s)),
        ):
            if t not in ordered:
                ordered.append(t)
        terms = ordered
        hits: dict[int, dict[str, Any]] = {}
        fts_cap = min(max(limit * 2, 200), 800)
        fts_ids = self._fts_search(terms, fts_cap)
        if fts_ids:
            for fid in fts_ids:
                hits[int(fid)] = {"id": int(fid)}
        # Dosya adı LIKE her zaman — FTS generic sonuçlarda kaçmasın
        primary = [t for t in terms if len(str(t).split()) <= 3][:3]
        filename_ids: set[int] = set()
        if primary:
            fname_hits = self._blob_like_search(
                primary, min(limit, 200), filename_only=True
            )
            filename_ids = set(fname_hits.keys())
            hits.update(fname_hits)
        # FTS + dosya adı yetmezse pahalı path/blob LIKE
        min_enough = max(40, min(limit, 120))
        if len(hits) < min_enough:
            hits.update(
                self._blob_like_search(primary[:2], min(limit, 400), filename_only=False)
            )
        ocr_ids: set[int] = set()
        ocr_terms = [t for t in (primary or terms) if str(t).strip()][:3]
        for term in ocr_terms:
            for rec in self.search_ocr_text(term, limit=min(80, max(limit, 50))):
                fid = int(rec["id"])
                hits.setdefault(fid, {"id": fid})
                ocr_ids.add(fid)
                if len(ocr_ids) >= min(80, max(limit, 50)):
                    break
            if len(ocr_ids) >= min(80, max(limit, 50)):
                break
        label_pool_limit = 0
        if int(limit or 0) > 0:
            label_pool_limit = min(int(limit), 400)
        hits.update(self._label_text_search(primary[:4], label_pool_limit or 800))
        if not hits:
            return []
        query = """
            SELECT f.*, fe.texture_map,
                   s.name AS source_name, s.source_type,
                   g.label AS pattern_group_label
            FROM files f
            LEFT JOIN features fe ON fe.file_id=f.id
            LEFT JOIN sources s ON s.id=f.source_id
            LEFT JOIN pattern_group_members pgm ON pgm.file_id=f.id
            LEFT JOIN pattern_groups g ON g.id=pgm.group_id
            WHERE (f.status='indexed' OR COALESCE(f.physical_preview_ready,0)=1)
              AND f.status NOT IN ('missing','excluded_internal')
              AND (f.source_id=0 OR s.id IS NULL OR s.is_active=1)
              AND f.id IN ({placeholders})
        """
        # Dosya adı önce; OCR-only limit kesiminde düşmesin
        preferred = [i for i in filename_ids if i in hits]
        ocr_only = [i for i in ocr_ids if i in hits and i not in filename_ids]
        rest = [
            i for i in hits.keys() if i not in filename_ids and i not in ocr_ids
        ]
        cap = max(limit * 2, len(preferred) + len(ocr_only))
        if int(limit or 0) <= 0:
            cap = len(preferred) + len(ocr_only) + len(rest)
        ids = (preferred + ocr_only + rest)[:cap]
        placeholders = ",".join("?" * len(ids))
        params: list[Any] = list(ids)
        sql = query.format(placeholders=placeholders)
        if customer:
            sql += " AND f.customer=?"
            params.append(customer)
        if source_ids:
            ph = ",".join("?" * len(source_ids))
            sql += f" AND f.source_id IN ({ph})"
            params.extend(source_ids)
        elif source_types:
            ph = ",".join("?" * len(source_types))
            sql += f" AND s.source_type IN ({ph})"
            params.extend(source_types)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        by_id = {int(dict(row)["id"]): dict(row) for row in rows}
        result = []
        seen: set[int] = set()
        for fid in ids:
            d = by_id.get(int(fid))
            if not d or fid in seen:
                continue
            seen.add(fid)
            d["texture_map"] = json.loads(d.get("texture_map") or "{}")
            d["feedback_labels"] = []
            result.append(d)
            if int(limit or 0) > 0 and len(result) >= limit:
                break
        return result

    def search_filename_path_candidates(
        self,
        terms: list[str],
        limit: int = 400,
        customer: str = "",
        source_types: list[str] | None = None,
        source_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Filename/path LIKE without preview/indexed gate.

        Brand recall must include pending files that have the brand in the
        name but no embeddings yet. Does not change FTS/folder candidate SQL.
        """
        needles: list[str] = []
        for raw in terms or []:
            t = str(raw or "").strip()
            if len(t) < 2:
                continue
            esc = t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            if esc not in needles:
                needles.append(esc)
            if "&" in t:
                compact = t.replace("&", "")
                if len(compact) >= 2 and compact not in needles:
                    needles.append(compact.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"))
        if not needles:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        for n in needles[:12]:
            like = f"%{n}%"
            clauses.append(
                "(f.filename LIKE ? ESCAPE '\\' COLLATE NOCASE"
                " OR f.path LIKE ? ESCAPE '\\' COLLATE NOCASE)"
            )
            params.extend([like, like])
        sql = (
            "SELECT f.*, fe.texture_map, s.name AS source_name, s.source_type "
            "FROM files f "
            "LEFT JOIN features fe ON fe.file_id=f.id "
            "LEFT JOIN sources s ON s.id=f.source_id "
            "WHERE f.status NOT IN ('missing','excluded_internal') "
            "AND (f.source_id=0 OR s.id IS NULL OR s.is_active=1) "
            "AND (" + " OR ".join(clauses) + ")"
        )
        if customer:
            sql += " AND f.customer=?"
            params.append(customer)
        if source_ids:
            ph = ",".join("?" * len(source_ids))
            sql += f" AND f.source_id IN ({ph})"
            params.extend(int(x) for x in source_ids)
        elif source_types:
            ph = ",".join("?" * len(source_types))
            sql += f" AND s.source_type IN ({ph})"
            params.extend(source_types)
        sql += " LIMIT ?"
        params.append(max(1, int(limit or 400)))
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for row in rows:
            d = dict(row)
            fid = int(d.get("id") or 0)
            if fid <= 0 or fid in seen:
                continue
            seen.add(fid)
            d["texture_map"] = json.loads(d.get("texture_map") or "{}")
            out.append(d)
        return out

    def _category_path_label_search(
        self, terms: list[str], limit: int
    ) -> dict[int, dict[str, Any]]:
        """Taught category_path / manual_category_path term match (generic)."""
        from core.teach_search_wire import run_category_path_label_search

        if not terms:
            return {}
        cap = int(limit) if int(limit or 0) > 0 else 800
        try:
            with self.connect() as conn:
                return run_category_path_label_search(conn, terms, cap)
        except Exception:
            return {}

    def _label_text_search(
        self, terms: list[str], limit: int
    ) -> dict[int, dict[str, Any]]:
        """Görsel sınıflandırma etiketinden aday — blob/FTS boş olsa da."""
        from core.textile_terms import query_family_hints

        hits: dict[int, dict[str, Any]] = {}
        if not terms:
            return hits
        cap = int(limit) if int(limit or 0) > 0 else 800
        families: set[str] = set()
        animals: set[str] = set()
        subtypes: set[str] = set()
        if int(limit or 0) > 0:
            for term in terms:
                hints = query_family_hints(str(term), include_semantic=False)
                if hints.get("pattern_family"):
                    families.add(str(hints["pattern_family"]))
                if hints.get("animal_print_type"):
                    animals.add(str(hints["animal_print_type"]))
                subtype = str(
                    hints.get("pattern_subtype") or hints.get("pattern_type") or ""
                ).strip()
                if subtype:
                    subtypes.add(subtype)
        # Always merge taught category-path hits (in addition to family/custom_tag).
        hits.update(self._category_path_label_search(terms, cap))
        if not families and not animals and not subtypes:
            hits.update(self._custom_tag_label_search(terms, cap))
            return hits
        clauses: list[str] = []
        params: list[Any] = []
        for fam in families:
            clauses.append("f.pattern_family=?")
            params.append(fam)
            clauses.append("fe.texture_map LIKE ?")
            params.append(f'%"pattern_family": "{fam}"%')
        for animal in animals:
            clauses.append("f.pattern_type=?")
            params.append(animal)
            clauses.append("fe.texture_map LIKE ?")
            params.append(f'%"animal_print_type": "{animal}"%')
            clauses.append("fe.texture_map LIKE ?")
            params.append(f'%"animal_print_type":"{animal}"%')
        for subtype in subtypes:
            clauses.append("f.pattern_subtype=?")
            params.append(subtype)
            clauses.append("fe.texture_map LIKE ?")
            params.append(f'%"pattern_subtype": "{subtype}"%')
            clauses.append("fe.texture_map LIKE ?")
            params.append(f'%"pattern_subtype":"{subtype}"%')
        sql = (
            "SELECT f.id FROM files f "
            "LEFT JOIN features fe ON fe.file_id=f.id "
            "WHERE f.status NOT IN ('missing','excluded_internal') "
            "AND (f.status='indexed' OR COALESCE(f.physical_preview_ready,0)=1) "
            "AND (" + " OR ".join(clauses) + ") LIMIT ?"
        )
        params.append(int(cap))
        with self.connect() as conn:
            for row in conn.execute(sql, params):
                hits[int(row["id"])] = {"id": int(row["id"])}
        hits.update(self._custom_tag_label_search(terms, cap))
        return hits

    def _custom_tag_label_search(
        self, terms: list[str], limit: int
    ) -> dict[int, dict[str, Any]]:
        """Learned custom_tag hits — no reindex / FAISS."""
        hits: dict[int, dict[str, Any]] = {}
        needles = [
            str(t or "").strip().lower()
            for t in terms
            if str(t or "").strip()
        ]
        if not needles:
            return hits
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    """
                    SELECT result_file_id, label FROM user_feedback
                    WHERE action='custom_tag' AND TRIM(label) != ''
                    ORDER BY id DESC
                    """
                ).fetchall()
        except Exception:
            rows = []
        for row in rows:
            fid = int(row["result_file_id"] or 0)
            lab = str(row["label"] or "").strip().lower()
            if fid <= 0 or lab not in needles:
                continue
            hits[fid] = {"id": fid, "_learned_custom_tag": True}
            if len(hits) >= max(1, int(limit or 800)):
                break
        if hits:
            return hits
        try:
            from core.search_memory import get_search_memory

            mem = get_search_memory(self.db_path)
            with mem._connect() as con:
                ov = con.execute(
                    """SELECT file_id, label FROM ranking_overlay
                       WHERE lower(action) IN ('custom_tag','öğret','ogret','teach')
                         AND TRIM(label) != ''"""
                ).fetchall()
            for file_id, label in ov:
                if str(label or "").strip().lower() in needles:
                    fid = int(file_id)
                    hits[fid] = {"id": fid, "_learned_custom_tag": True}
                    if len(hits) >= max(1, int(limit or 800)):
                        break
        except Exception:
            pass
        return hits

    def _blob_like_search(
        self, terms: list[str], limit: int, *, filename_only: bool = False
    ) -> dict[int, dict[str, Any]]:
        from core.textile_terms import normalize_turkish

        hits: dict[int, dict[str, Any]] = {}
        with self.connect() as conn:
            for term in terms[:3]:
                raw = (term or "").strip()
                if not raw or len(raw.split()) > 3:
                    continue
                # 1) Hızlı: sadece dosya adı
                rows = conn.execute(
                    """
                    SELECT id FROM files
                    WHERE status NOT IN ('missing','excluded_internal')
                      AND (status='indexed' OR COALESCE(physical_preview_ready,0)=1)
                      AND filename LIKE ?
                    LIMIT ?
                    """,
                    (f"%{raw}%", min(limit, 400)),
                ).fetchall()
                for row in rows:
                    hits[int(row["id"])] = {"id": int(row["id"])}
                if len(hits) >= limit or filename_only:
                    if filename_only:
                        continue
                    break
                # 2) Path/blob LIKE indeksiz — sadece dosya adı yetmezse
                norm = normalize_turkish(raw)
                rows = conn.execute(
                    """
                    SELECT id FROM files
                    WHERE status NOT IN ('missing','excluded_internal')
                      AND (status='indexed' OR COALESCE(physical_preview_ready,0)=1)
                      AND (path LIKE ? OR coalesce(text_search_blob,'') LIKE ?)
                    LIMIT ?
                    """,
                    (f"%{raw}%", f"%{norm}%", min(limit, 100)),
                ).fetchall()
                for row in rows:
                    hits.setdefault(int(row["id"]), {"id": int(row["id"])})
                if len(hits) >= limit:
                    break
        return hits

    def _fts_search(self, terms: list[str], limit: int) -> list[int]:
        try:
            with self.connect() as conn:
                ids: list[int] = []
                for term in terms[:6]:
                    if not term.strip():
                        continue
                    q = " OR ".join(f'"{t}"' for t in expand_fts_tokens(term)[:4])
                    if not q:
                        continue
                    rows = conn.execute(
                        f"SELECT file_id FROM files_fts WHERE files_fts MATCH ? LIMIT ?",
                        (q, limit),
                    ).fetchall()
                    ids.extend(int(r["file_id"]) for r in rows)
                return list(dict.fromkeys(ids))[:limit]
        except sqlite3.OperationalError:
            return []

    def count_sources_summary(self) -> dict[str, int]:
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM sources").fetchone()["c"]
            active = conn.execute(
                "SELECT COUNT(*) AS c FROM sources WHERE is_active=1"
            ).fetchone()["c"]
            return {
                "total": int(total),
                "active": int(active),
                "inactive": int(total) - int(active),
            }

    def update_source_scan_times(
        self, source_id: int, scan_mode: str = "quick"
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            if scan_mode == "deep":
                conn.execute(
                    """
                    UPDATE sources SET
                        last_scan_at=?, last_quick_scan_at=?, last_deep_scan_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (now, now, now, now, source_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE sources SET last_scan_at=?, last_quick_scan_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (now, now, now, source_id),
                )

    def update_source_stats(
        self,
        source_id: int,
        file_count: int = 0,
        error_count: int = 0,
        cache_status: str = "ok",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE sources SET file_count=?, error_count=?, cache_status=?,
                updated_at=? WHERE id=?
                """,
                (file_count, error_count, cache_status, now, source_id),
            )

    def count_files_for_source(self, source_id: int) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM files WHERE source_id=? GROUP BY status",
                (source_id,),
            ).fetchall()
            return {r["status"]: r["cnt"] for r in rows}

    def mark_missing_files_for_source(
        self,
        source_id: int,
        known_paths: Iterable[str],
    ) -> int:
        known = {normalize_path(p) for p in known_paths}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, path FROM files WHERE source_id=? AND status != 'missing'",
                (source_id,),
            ).fetchall()
            count = 0
            now = datetime.now(timezone.utc).isoformat()
            for row in rows:
                if row["path"] not in known:
                    conn.execute(
                        "UPDATE files SET status='missing', updated_at=? WHERE id=?",
                        (now, row["id"]),
                    )
                    count += 1
            return count

    def exclude_internal_artifacts(self, roots: list[str]) -> int:
        """Keep generated app artifacts in DB history but out of search/index."""
        normalized = [normalize_path(root).rstrip("\\/") for root in roots if root]
        if not normalized:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        total = 0
        with self.connect() as conn:
            for root in normalized:
                ids = [
                    int(row["id"])
                    for row in conn.execute(
                        """
                        SELECT id FROM files
                        WHERE path=? COLLATE NOCASE OR path LIKE ? COLLATE NOCASE
                        """,
                        (root, root + os.sep + "%"),
                    ).fetchall()
                ]
                cur = conn.execute(
                    """
                    UPDATE files SET status='excluded_internal', updated_at=?
                    WHERE status!='excluded_internal'
                      AND (path=? COLLATE NOCASE OR path LIKE ? COLLATE NOCASE)
                    """,
                    (now, root, root + os.sep + "%"),
                )
                total += max(0, cur.rowcount)
                if ids:
                    for offset in range(0, len(ids), 500):
                        batch = ids[offset : offset + 500]
                        placeholders = ",".join("?" * len(batch))
                        conn.execute(
                            f"""
                            UPDATE index_queue SET status='skipped', updated_at=?
                            WHERE file_id IN ({placeholders})
                              AND status IN ('pending','running')
                            """,
                            [now, *batch],
                        )
        return total

    def total_searchable_files(self) -> int:
        with self.connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) AS c FROM files
                WHERE status NOT IN ('excluded_internal', 'missing')
                """).fetchone()
            return int(row["c"] or 0) if row else 0

    def exclude_os_metadata_artifacts(self) -> int:
        """Keep OS metadata sidecars out of search and resumable index work."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            rows = conn.execute("""
                SELECT f.id FROM files f
                WHERE (
                    substr(f.filename, 1, 2)='._'
                    OR lower(replace(f.path, '/', '\\')) LIKE '%\\.appledouble\\%'
                    OR lower(replace(f.path, '/', '\\')) LIKE '%\\__macosx\\%'
                    OR lower(replace(f.path, '/', '\\')) LIKE '%\\#recycle\\%'
                    OR lower(replace(f.path, '/', '\\')) LIKE '%\\@recycle\\%'
                    OR lower(replace(f.path, '/', '\\')) LIKE '%\\$recycle.bin\\%'
                    OR lower(replace(f.path, '/', '\\')) LIKE '%\\recycler\\%'
                    OR lower(replace(f.path, '/', '\\')) LIKE '%\\system volume information\\%'
                    OR lower(replace(f.path, '/', '\\')) LIKE '%\\.snapshot\\%'
                    OR lower(replace(f.path, '/', '\\')) LIKE '%\\@eadir\\%'
                    OR lower(f.filename) IN ('.ds_store', 'thumbs.db', 'desktop.ini')
                )
                AND (
                    f.status!='excluded_internal'
                    OR EXISTS (
                        SELECT 1 FROM index_queue q
                        WHERE q.file_id=f.id
                          AND q.status IN ('pending','running','error')
                    )
                )
                """).fetchall()
            ids = [int(row["id"]) for row in rows]
            for offset in range(0, len(ids), 500):
                batch = ids[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                conn.execute(
                    f"UPDATE files SET status='excluded_internal', updated_at=? "
                    f"WHERE id IN ({placeholders})",
                    [now, *batch],
                )
                conn.execute(
                    f"UPDATE index_queue SET status='skipped', updated_at=? "
                    f"WHERE file_id IN ({placeholders}) "
                    "AND status IN ('pending','running','error')",
                    [now, *batch],
                )
            return len(ids)

    def get_indexed_files(
        self,
        customer: str = "",
        status: str = "indexed",
        source_types: list[str] | None = None,
        source_id: int = 0,
        source_ids: list[int] | None = None,
        source_roots: list[str] | None = None,
        folder_prefix: str = "",
        lightweight: bool = False,
        include_processing_ready: bool = False,
    ) -> list[dict[str, Any]]:
        file_select = "f.*"
        if lightweight:
            file_select = """
                f.id, f.path, f.filename, f.customer, f.file_size, f.mtime,
                f.status, f.thumbnail_path, f.partial_hash, f.full_hash,
                f.source_id, f.pattern_family, f.pattern_type,
                f.pattern_subtype, f.pattern_confidence, f.ocr_text,
                f.category_path, f.manual_category_path,
                f.category_confidence, f.category_source, f.category_aliases,
                f.text_search_blob
            """
        feature_select = (
            """
                   fe.phash, fe.dhash, fe.whash, fe.texture_map
        """
            if lightweight
            else """
                   fe.phash, fe.dhash, fe.whash, fe.color_hist,
                   fe.dominant_colors, fe.texture_features,
                   fe.dino_embedding, fe.clip_embedding, fe.patch_embeddings,
                   fe.texture_map
        """
        )
        visual_ready_clause = sql_visual_ready_clause()
        status_clause = "f.status = ?"
        if include_processing_ready:
            status_clause = f"(f.status = ? OR {visual_ready_clause})"
        query = f"""
            SELECT {file_select}, {feature_select},
                   s.name as source_name, s.source_type
            FROM files f
            LEFT JOIN features fe ON f.id = fe.file_id
            LEFT JOIN sources s ON f.source_id = s.id
            WHERE {status_clause}
              AND (f.source_id = 0 OR s.id IS NULL OR s.is_active = 1)
        """
        params: list[Any] = [status]
        if customer:
            query += " AND f.customer = ?"
            params.append(customer)

        scope_parts: list[str] = []
        if source_ids:
            placeholders = ",".join("?" * len(source_ids))
            scope_parts.append(f"f.source_id IN ({placeholders})")
            params.extend(source_ids)
        elif source_id:
            scope_parts.append("f.source_id = ?")
            params.append(source_id)

        if source_roots:
            for root in source_roots:
                prefix = normalize_path(root)
                if not prefix.endswith(("\\", "/")):
                    prefix += os.sep
                scope_parts.append("f.path LIKE ?")
                params.append(prefix + "%")

        if scope_parts:
            query += " AND (" + " OR ".join(scope_parts) + ")"

        if source_types:
            placeholders = ",".join("?" * len(source_types))
            query += f" AND s.source_type IN ({placeholders})"
            params.extend(source_types)
        if folder_prefix:
            prefix = normalize_path(folder_prefix)
            p = Path(prefix)
            if p.suffix and p.is_file():
                prefix = str(p.parent)
            if not prefix.endswith(("\\", "/")):
                prefix += os.sep
            query += " AND f.path LIKE ?"
            params.append(prefix + "%")
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            result = []
            seen: set[int] = set()
            for row in rows:
                d = dict(row)
                fid = d.get("id")
                if fid in seen:
                    continue
                seen.add(fid)
                if not lightweight:
                    d["dominant_colors"] = json.loads(d.get("dominant_colors") or "[]")
                    d["texture_features"] = json.loads(
                        d.get("texture_features") or "[]"
                    )
                    d["patch_embeddings_meta"] = json.loads(
                        d.get("patch_embeddings") or "[]"
                    )
                d["texture_map"] = json.loads(d.get("texture_map") or "{}")
                result.append(d)
            return result

    def get_indexed_files_by_ids(
        self,
        file_ids: list[int],
        *,
        include_processing_ready: bool = False,
        include_inactive_sources: bool = False,
        lightweight: bool = False,
        include_pending: bool = False,
    ) -> list[dict[str, Any]]:
        if not file_ids:
            return []
        placeholders = ",".join("?" * len(file_ids))
        visual_ready_clause = sql_visual_ready_clause()
        status_clause = "f.status='indexed'"
        if include_pending:
            status_clause = "f.status NOT IN ('missing','excluded_internal')"
        elif include_processing_ready:
            status_clause = f"(f.status='indexed' OR {visual_ready_clause})"
        source_clause = (
            "1=1"
            if include_inactive_sources
            else ("(f.source_id=0 OR s.id IS NULL OR s.is_active=1)")
        )
        feature_select = (
            "fe.phash, fe.dhash, fe.whash, fe.texture_map"
            if lightweight
            else """fe.phash, fe.dhash, fe.whash, fe.color_hist,
                   fe.dominant_colors, fe.texture_features,
                   fe.dino_embedding, fe.clip_embedding, fe.patch_embeddings,
                   fe.texture_map"""
        )
        query = f"""
            SELECT f.*, {feature_select},
                   s.name AS source_name, s.source_type
            FROM files f
            LEFT JOIN features fe ON f.id=fe.file_id
            LEFT JOIN sources s ON f.source_id=s.id
            WHERE {status_clause}
              AND {source_clause}
              AND f.id IN ({placeholders})
        """
        with self.connect() as conn:
            rows = conn.execute(query, file_ids).fetchall()
        by_id: dict[int, dict[str, Any]] = {}
        for row in rows:
            d = dict(row)
            if not lightweight:
                d["dominant_colors"] = json.loads(d.get("dominant_colors") or "[]")
                d["texture_features"] = json.loads(d.get("texture_features") or "[]")
                d["patch_embeddings_meta"] = json.loads(d.get("patch_embeddings") or "[]")
            else:
                d["dominant_colors"] = []
                d["texture_features"] = []
                d["patch_embeddings_meta"] = []
                d["color_hist"] = b""
                d["dino_embedding"] = b""
                d["clip_embedding"] = b""
            d["texture_map"] = json.loads(d.get("texture_map") or "{}")
            by_id[int(d["id"])] = d
        return [by_id[fid] for fid in file_ids if fid in by_id]

    def get_identity_candidates(
        self,
        *,
        path: str = "",
        partial_hash: str = "",
        full_hash: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return exact-identity records even while their deep index is pending."""
        clauses: list[str] = []
        params: list[Any] = []
        if path:
            clauses.append("f.path=? COLLATE NOCASE")
            params.append(normalize_path(path))
        if full_hash:
            clauses.append("f.full_hash=?")
            params.append(full_hash)
        if partial_hash:
            clauses.append("f.partial_hash=?")
            params.append(partial_hash)
        if not clauses:
            return []

        query = f"""
            SELECT f.*, fe.phash, fe.dhash, fe.whash, fe.color_hist,
                   fe.dominant_colors, fe.texture_features,
                   fe.dino_embedding, fe.clip_embedding, fe.patch_embeddings,
                   fe.texture_map, s.name AS source_name, s.source_type
            FROM files f
            LEFT JOIN features fe ON f.id=fe.file_id
            LEFT JOIN sources s ON f.source_id=s.id
            WHERE f.status IN ('indexed','processing')
              AND (f.source_id=0 OR s.id IS NULL OR s.is_active=1)
              AND ({" OR ".join(clauses)})
            ORDER BY CASE f.status WHEN 'indexed' THEN 0 ELSE 1 END, f.id
            LIMIT ?
        """
        params.append(max(1, int(limit)))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            d["dominant_colors"] = json.loads(d.get("dominant_colors") or "[]")
            d["texture_features"] = json.loads(d.get("texture_features") or "[]")
            d["patch_embeddings_meta"] = json.loads(d.get("patch_embeddings") or "[]")
            d["texture_map"] = json.loads(d.get("texture_map") or "{}")
            result.append(d)
        return result

    def count_embeddings(self) -> dict[str, int]:
        """Aktif dosyalarda benzersiz DINO/CLIP (features.file_id = 1:1)."""
        with self.connect() as conn:
            row = conn.execute("""
                SELECT
                    SUM(CASE WHEN fe.dino_embedding IS NOT NULL
                              AND length(fe.dino_embedding)>0 THEN 1 ELSE 0 END) AS dino,
                    SUM(CASE WHEN fe.clip_embedding IS NOT NULL
                              AND length(fe.clip_embedding)>0 THEN 1 ELSE 0 END) AS clip
                FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                WHERE f.status NOT IN ('excluded_internal','missing')
                """).fetchone()
            return {"dino": int(row["dino"] or 0), "clip": int(row["clip"] or 0)}

    def feature_gap_counts(self, ai_enabled: bool = False) -> dict[str, int]:
        with self.connect() as conn:
            row = conn.execute("""
                SELECT
                    SUM(CASE WHEN f.thumbnail_path IS NULL OR f.thumbnail_path='' THEN 1 ELSE 0 END) AS thumbnail,
                    SUM(CASE WHEN f.feature_preview_path IS NULL OR f.feature_preview_path='' THEN 1 ELSE 0 END) AS preview,
                    SUM(CASE WHEN fe.patch_embeddings IS NULL OR fe.patch_embeddings='' OR fe.patch_embeddings='[]' THEN 1 ELSE 0 END) AS patch,
                    SUM(CASE WHEN fe.texture_map IS NULL OR fe.texture_map='' OR fe.texture_map='{}' THEN 1 ELSE 0 END) AS texture_map,
                    SUM(CASE WHEN fe.phash IS NULL OR fe.phash='' OR fe.texture_features IS NULL OR fe.texture_features='[]' THEN 1 ELSE 0 END) AS structure,
                    SUM(CASE WHEN fe.dino_embedding IS NULL OR length(fe.dino_embedding)=0 THEN 1 ELSE 0 END) AS ai_dino,
                    SUM(CASE WHEN fe.clip_embedding IS NULL OR length(fe.clip_embedding)=0 THEN 1 ELSE 0 END) AS ai_clip
                FROM files f LEFT JOIN features fe ON fe.file_id=f.id
                WHERE f.status='indexed'
                """).fetchone()
        result = {key: int(row[key] or 0) for key in row.keys()}
        if not ai_enabled:
            result["ai_dino"] = 0
            result["ai_clip"] = 0
        return result

    def count_pattern_intelligence(self) -> dict[str, int]:
        """Pattern DNA / semantic/OCR kapsamını hafif SQLite sayımlarıyla döndür."""
        dna_ready = _json_object_ready_sql("fe.texture_map", "pattern_dna")
        semantic_ready = _json_object_ready_sql("fe.texture_map", "semantic_tags")
        with self.connect() as conn:
            row = conn.execute(f"""
                SELECT
                    SUM(CASE WHEN {dna_ready} THEN 1 ELSE 0 END) AS pattern_dna,
                    SUM(CASE WHEN {semantic_ready} THEN 1 ELSE 0 END) AS semantic_tags,
                    SUM(CASE WHEN f.ocr_text IS NOT NULL AND f.ocr_text!='' THEN 1 ELSE 0 END) AS ocr_done,
                    SUM(CASE WHEN f.index_stage IN ('light_done','full_done') THEN 1 ELSE 0 END) AS indexed_stage_done,
                    SUM(CASE WHEN {dna_ready} OR {semantic_ready} THEN 1 ELSE 0 END) AS ai_analyzed,
                    SUM(CASE WHEN f.status='error' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN f.index_stage IN ('skipped_heavy_format','skipped') OR f.status='skipped' THEN 1 ELSE 0 END) AS skipped,
                    SUM(CASE WHEN f.status IN ('pending','processing','changed') OR f.index_stage IN ('pending_light','pending_full','failed') THEN 1 ELSE 0 END) AS total_remaining
                FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                """).fetchone()
        return {key: int(row[key] or 0) for key in row.keys()}

    def count_index_pipeline_dashboard(
        self, source_ids: list[int] | None = None
    ) -> dict[str, int]:
        """Preview-first pipeline: hazır / bekleyen / karantina sayımları.

        source_ids verilirse yalnız o kaynak(lar)ın dosyaları sayılır (UI yüzde paydası).
        """
        dna_ready = _json_object_ready_sql("fe.texture_map", "pattern_dna")
        semantic_ready = _json_object_ready_sql("fe.texture_map", "semantic_tags")
        ids = [int(x) for x in (source_ids or []) if int(x) > 0]
        where_sql = ""
        params: tuple[int, ...] = ()
        if ids:
            ph = ",".join("?" * len(ids))
            where_sql = f" WHERE f.source_id IN ({ph})"
            params = tuple(ids)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing') THEN 1 ELSE 0 END) AS total_searchable,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        THEN 1 ELSE 0 END) AS preview_ready,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND f.physical_thumbnail_ready=1
                        AND f.physical_preview_ready=1
                        THEN 1 ELSE 0 END) AS verified_preview_ready,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        THEN 1 ELSE 0 END) AS preview_db_path_ready,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND (f.physical_thumbnail_ready=0
                             OR f.physical_preview_ready=0)
                        THEN 1 ELSE 0 END) AS physical_preview_missing,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND (f.physical_thumbnail_ready IS NULL
                             OR f.physical_preview_ready IS NULL)
                        THEN 1 ELSE 0 END) AS physical_preview_unverified,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        AND fe.dino_embedding IS NOT NULL
                        AND length(fe.dino_embedding)>0
                        AND fe.clip_embedding IS NOT NULL
                        AND length(fe.clip_embedding)>0
                        THEN 1 ELSE 0 END) AS embedding_ready,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        AND fe.dino_embedding IS NOT NULL
                        AND length(fe.dino_embedding)>0
                        AND fe.clip_embedding IS NOT NULL
                        AND length(fe.clip_embedding)>0
                        AND {dna_ready}
                        AND {semantic_ready}
                        AND COALESCE(fe.phash,'')!=''
                        AND COALESCE(fe.texture_features,'') NOT IN ('','[]')
                        AND COALESCE(fe.patch_embeddings,'') NOT IN ('','[]')
                        THEN 1 ELSE 0 END) AS patch_embedding_ready,
                    SUM(CASE WHEN f.status='indexed' AND {dna_ready}
                        THEN 1 ELSE 0 END) AS deep_content_ready,
                    SUM(CASE WHEN f.status='indexed' AND {dna_ready}
                        THEN 1 ELSE 0 END) AS ai_analyzed_ready,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        AND fe.dino_embedding IS NOT NULL
                        AND length(fe.dino_embedding)>0
                        AND fe.clip_embedding IS NOT NULL
                        AND length(fe.clip_embedding)>0
                        AND {dna_ready}
                        THEN 1 ELSE 0 END) AS pattern_dna_ready,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        AND fe.dino_embedding IS NOT NULL
                        AND length(fe.dino_embedding)>0
                        AND fe.clip_embedding IS NOT NULL
                        AND length(fe.clip_embedding)>0
                        AND {dna_ready}
                        AND {semantic_ready}
                        THEN 1 ELSE 0 END) AS semantic_tag_ready,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.ocr_text,'')!=''
                        THEN 1 ELSE 0 END) AS ocr_ready,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        AND fe.dino_embedding IS NOT NULL
                        AND length(fe.dino_embedding)>0
                        AND fe.clip_embedding IS NOT NULL
                        AND length(fe.clip_embedding)>0
                        AND {dna_ready}
                        AND {semantic_ready}
                        AND COALESCE(fe.phash,'')!=''
                        AND COALESCE(fe.texture_features,'') NOT IN ('','[]')
                        THEN 1 ELSE 0 END) AS texture_ready,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        AND fe.dino_embedding IS NOT NULL
                        AND length(fe.dino_embedding)>0
                        AND fe.clip_embedding IS NOT NULL
                        AND length(fe.clip_embedding)>0
                        AND COALESCE(fe.patch_embeddings,'') NOT IN ('','[]')
                        AND {dna_ready}
                        AND {semantic_ready}
                        AND COALESCE(fe.phash,'')!=''
                        AND COALESCE(fe.texture_features,'') NOT IN ('','[]')
                        THEN 1 ELSE 0 END) AS ai_final_ready,
                    SUM(CASE WHEN f.status IN ('indexed','pending','processing','changed')
                        AND (
                            COALESCE(f.thumbnail_path,'')=''
                            OR COALESCE(f.feature_preview_path,'')=''
                        ) THEN 1 ELSE 0 END) AS pending_preview,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        AND (fe.dino_embedding IS NULL OR length(fe.dino_embedding)=0
                             OR fe.clip_embedding IS NULL OR length(fe.clip_embedding)=0)
                        THEN 1 ELSE 0 END) AS pending_embedding,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND f.index_stage IN ('light_done','pending_full')
                        AND NOT {dna_ready}
                        THEN 1 ELSE 0 END) AS pending_ai_analysis,
                    SUM(CASE WHEN f.status='indexed'
                        AND NOT {dna_ready}
                        THEN 1 ELSE 0 END) AS pending_pattern_dna,
                    SUM(CASE WHEN f.status='indexed'
                        AND NOT {semantic_ready}
                        THEN 1 ELSE 0 END) AS pending_semantic_tag,
                    SUM(CASE WHEN f.status='indexed'
                        AND COALESCE(f.ocr_text,'')=''
                        THEN 1 ELSE 0 END) AS pending_ocr,
                    SUM(CASE WHEN COALESCE(f.quarantine_reason,'')!='' THEN 1 ELSE 0 END) AS quarantine_total,
                    SUM(CASE WHEN f.preview_status='unsupported_preview'
                        OR f.unsupported_preview=1 THEN 1 ELSE 0 END) AS unsupported_format,
                    SUM(CASE WHEN f.preview_status='preview_missing_dependency' THEN 1 ELSE 0 END) AS dependency_missing,
                    SUM(CASE WHEN f.status='indexed' AND f.index_stage='full_done'
                        THEN 1 ELSE 0 END) AS full_done_stage,
                    SUM(CASE WHEN f.status='indexed' AND f.heavy_status='done'
                        THEN 1 ELSE 0 END) AS heavy_done,
                    -- Bağımsız aşama SSOT (önkoşul yığılmaz; aktif file_id)
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND COALESCE(f.thumbnail_path,'')!=''
                        THEN 1 ELSE 0 END) AS stage_thumbnail,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND COALESCE(f.thumbnail_path,'')!=''
                        AND COALESCE(f.feature_preview_path,'')!=''
                        THEN 1 ELSE 0 END) AS stage_preview,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND fe.dino_embedding IS NOT NULL
                        AND length(fe.dino_embedding)>0
                        THEN 1 ELSE 0 END) AS stage_dino,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND fe.clip_embedding IS NOT NULL
                        AND length(fe.clip_embedding)>0
                        THEN 1 ELSE 0 END) AS stage_clip,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND COALESCE(fe.patch_embeddings,'') NOT IN ('','[]')
                        THEN 1 ELSE 0 END) AS stage_patch,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND {dna_ready}
                        THEN 1 ELSE 0 END) AS stage_dna,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND {semantic_ready}
                        THEN 1 ELSE 0 END) AS stage_semantic,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND COALESCE(fe.phash,'')!=''
                        AND COALESCE(fe.texture_features,'') NOT IN ('','[]')
                        THEN 1 ELSE 0 END) AS stage_texture,
                    SUM(CASE WHEN f.status NOT IN ('excluded_internal','missing')
                        AND COALESCE(f.ocr_text,'')!=''
                        THEN 1 ELSE 0 END) AS stage_ocr
                FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                {where_sql}
                """,
                params,
            ).fetchone()
        return {key: int(row[key] or 0) for key in row.keys()}

    def count_quarantine_by_reason(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(quarantine_reason,'') AS reason, COUNT(*) AS c
                FROM files
                WHERE COALESCE(quarantine_reason,'')!=''
                GROUP BY quarantine_reason
                ORDER BY c DESC
                """
            ).fetchall()
        return {str(r["reason"]): int(r["c"]) for r in rows}

    def get_pending_files(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM files
                WHERE status IN ('pending', 'error', 'changed')
                ORDER BY id LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM files GROUP BY status"
            ).fetchall()
            return {r["status"]: r["cnt"] for r in rows}

    def total_files(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM files").fetchone()
            return row["cnt"] if row else 0

    def list_customers(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT customer FROM files WHERE customer != '' ORDER BY customer"
            ).fetchall()
            return [r["customer"] for r in rows]

    def search_ocr_text(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        """Aday retrieval: yalnız dolu OCR metni. Dosya adı eşleşmesi burada yok.

        pending + preview_ready dosyalar (OCR'ı bitmiş, light henüz pending)
        filename yoluna düşmesin diye indexed şartı yok. Boş OCR = pending OCR,
        eşleşme sayılmaz.
        """
        q = (query or "").strip()
        if not q or limit <= 0:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, path, ocr_text, status, source_id
                FROM files
                WHERE TRIM(COALESCE(ocr_text,'')) != ''
                  AND ocr_text LIKE ?
                  AND (status='indexed' OR COALESCE(physical_preview_ready,0)=1)
                  AND status NOT IN ('missing','excluded_internal')
                LIMIT ?
                """,
                (f"%{q}%", int(limit)),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_files_missing_feature_preview(self) -> int:
        with self.connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) AS c FROM files
                WHERE status='indexed'
                  AND (feature_preview_path IS NULL OR feature_preview_path='')
                  AND unsupported_preview=0
                """).fetchone()
            return int(row["c"]) if row else 0

    # --- Pattern groups ---

    def create_pattern_group(self, record: dict[str, Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO pattern_groups(
                    representative_file_id, group_type, label,
                    pattern_family, animal_print_type, color_family,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    record.get("representative_file_id", 0),
                    record.get("group_type", "auto"),
                    record.get("label", ""),
                    record.get("pattern_family", ""),
                    record.get("animal_print_type", ""),
                    record.get("color_family", ""),
                    now,
                    now,
                ),
            )
            return cur.lastrowid

    def add_pattern_group_member(
        self,
        group_id: int,
        file_id: int,
        relation: str,
        score: float,
        created_at: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO pattern_group_members(group_id, file_id, relation, score, created_at)
                VALUES (?,?,?,?,?)
                ON CONFLICT(group_id, file_id) DO UPDATE SET
                    relation=excluded.relation,
                    score=MAX(pattern_group_members.score, excluded.score)
                """,
                (group_id, file_id, relation, score, created_at),
            )

    def get_pattern_group_for_file(self, file_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT g.* FROM pattern_groups g
                JOIN pattern_group_members m ON m.group_id=g.id
                WHERE m.file_id=?
                LIMIT 1
                """,
                (file_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_pattern_group_members(self, group_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*, f.filename, f.path FROM pattern_group_members m
                JOIN files f ON f.id=m.file_id
                WHERE m.group_id=?
                ORDER BY m.score DESC
                """,
                (group_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def merge_pattern_groups(self, source_id: int, target_id: int) -> None:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT file_id, relation, score, created_at FROM pattern_group_members WHERE group_id=?",
                (source_id,),
            ).fetchall()
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO pattern_group_members(group_id, file_id, relation, score, created_at)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(group_id, file_id) DO NOTHING
                    """,
                    (
                        target_id,
                        row["file_id"],
                        row["relation"],
                        row["score"],
                        row["created_at"],
                    ),
                )
            conn.execute("DELETE FROM pattern_groups WHERE id=?", (source_id,))

    def get_pattern_groups_for_files(
        self, file_ids: list[int]
    ) -> dict[int, dict[str, Any]]:
        if not file_ids:
            return {}
        placeholders = ",".join("?" * len(file_ids))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT m.file_id, g.* FROM pattern_group_members m
                JOIN pattern_groups g ON g.id=m.group_id
                WHERE m.file_id IN ({placeholders})
                """,
                file_ids,
            ).fetchall()
            return {int(r["file_id"]): dict(r) for r in rows}

    # --- User feedback ---

    def insert_category_correction(self, record: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO category_corrections(
                    file_id, old_predictions, old_category,
                    correct_category, correct_subcategory,
                    actor, is_admin, propagation_scope,
                    status, created_at, undone_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    int(record.get("file_id", 0)),
                    json.dumps(record.get("old_predictions", []), ensure_ascii=False),
                    record.get("old_category", ""),
                    record.get("correct_category", ""),
                    record.get("correct_subcategory", ""),
                    record.get("actor", ""),
                    int(bool(record.get("is_admin", False))),
                    record.get("propagation_scope", "single"),
                    record.get("status", "pending_review"),
                    record.get("created_at", ""),
                    record.get("undone_at", ""),
                ),
            )
            return int(cur.lastrowid)

    def list_category_corrections(
        self,
        *,
        file_id: int = 0,
        status: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if file_id:
            clauses.append("file_id=?")
            params.append(int(file_id))
        if status:
            clauses.append("status=?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, int(limit)))
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM category_corrections{where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["old_predictions"] = json.loads(item.get("old_predictions") or "[]")
            result.append(item)
        return result

    def undo_category_correction(self, correction_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE category_corrections
                SET status='undone', undone_at=?
                WHERE id=? AND status!='undone'
                """,
                (datetime.now(timezone.utc).isoformat(), int(correction_id)),
            )
            return cur.rowcount > 0

    def insert_user_feedback(self, record: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO user_feedback(
                    query_file_id, query_path, result_file_id,
                    action, label, note, created_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    record.get("query_file_id", 0),
                    record.get("query_path", ""),
                    record.get("result_file_id", 0),
                    record.get("action", ""),
                    record.get("label", ""),
                    record.get("note", ""),
                    record.get("created_at", ""),
                ),
            )
            return cur.lastrowid

    def get_feedback_for_query(
        self,
        query_path: str,
        candidate_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if candidate_ids:
                placeholders = ",".join("?" * len(candidate_ids))
                rows = conn.execute(
                    f"""
                    SELECT * FROM user_feedback
                    WHERE query_path=? AND result_file_id IN ({placeholders})
                    ORDER BY created_at DESC
                    """,
                    [query_path, *candidate_ids],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM user_feedback WHERE query_path=? ORDER BY created_at DESC",
                    (query_path,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_feedback_labels_for_file(self, file_id: int) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT label FROM user_feedback
                WHERE result_file_id=? AND label!='' AND action='label_family'
                """,
                (file_id,),
            ).fetchall()
            return [r["label"] for r in rows]

    def get_feedback_custom_tags_for_file(self, file_id: int) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT label FROM user_feedback
                WHERE result_file_id=? AND label!='' AND action='custom_tag'
                ORDER BY created_at DESC
                """,
                (file_id,),
            ).fetchall()
        seen: set[str] = set()
        tags: list[str] = []
        for r in rows:
            tag = (r["label"] or "").strip()
            key = tag.lower()
            if tag and key not in seen:
                seen.add(key)
                tags.append(tag)
        return tags

    def get_feedback_family_overrides_for_file(self, file_id: int) -> dict[str, object]:
        """
        For a file, compute a simple family override:
        - positive: latest ACTION_LABEL_FAMILY label (if any)
        - negative: set of ACTION_LABEL_NOT_FAMILY labels
        If there is a clear action, it resets both.
        """
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT action, label, created_at FROM user_feedback
                WHERE result_file_id=?
                  AND action IN ('label_family','label_not_family','clear_family_labels')
                ORDER BY created_at DESC
                """,
                (file_id,),
            ).fetchall()
            positive = ""
            negative: set[str] = set()
            for r in rows:
                action = (r["action"] or "").strip()
                label = (r["label"] or "").strip()
                if action == "clear_family_labels":
                    break
                if action == "label_family" and label and not positive:
                    positive = label
                if action == "label_not_family" and label:
                    negative.add(label)
            return {"positive": positive, "negative": negative}

    def get_feedback_family_overrides_for_files(
        self,
        file_ids: list[int],
    ) -> dict[int, dict[str, object]]:
        if not file_ids:
            return {}
        placeholders = ",".join("?" * len(file_ids))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT result_file_id, action, label, created_at
                FROM user_feedback
                WHERE result_file_id IN ({placeholders})
                  AND action IN ('label_family','label_not_family','clear_family_labels')
                ORDER BY result_file_id, created_at DESC, id DESC
                """,
                file_ids,
            ).fetchall()

        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["result_file_id"]), []).append(dict(row))

        result: dict[int, dict[str, object]] = {}
        for file_id, items in grouped.items():
            positive = ""
            negative: set[str] = set()
            for item in items:
                action = (item.get("action") or "").strip()
                label = (item.get("label") or "").strip()
                if action == "clear_family_labels":
                    break
                if action == "label_family" and label and not positive:
                    positive = label
                elif action == "label_not_family" and label:
                    negative.add(label)
            result[file_id] = {"positive": positive, "negative": negative}
        return result

    def get_learned_tier_for_file(self, file_id: int) -> dict[str, str]:
        """Dosya için öğrenilmiş benzerlik sınıfı (texture_map + feedback)."""
        detailed = self.get_indexed_files_by_ids(
            [file_id],
            include_processing_ready=True,
            include_inactive_sources=True,
        )
        tm = (detailed[0].get("texture_map") if detailed else {}) or {}
        tier_id = str(tm.get("user_similarity_tier") or "").strip()
        tier_label = str(tm.get("user_tier_label") or "").strip()
        if tier_id or tier_label:
            return {
                "tier_id": tier_id,
                "tier_label": tier_label,
                "pattern_family": str(tm.get("pattern_family") or ""),
                "pattern_subtype": str(tm.get("pattern_subtype") or ""),
            }
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT label, note FROM user_feedback
                WHERE result_file_id=? AND action='similarity_tier' AND label!=''
                ORDER BY created_at DESC LIMIT 1
                """,
                (file_id,),
            ).fetchone()
        if not row:
            return {}
        return {
            "tier_id": str(row["note"] or "").strip(),
            "tier_label": str(row["label"] or "").strip(),
            "pattern_family": "",
            "pattern_subtype": "",
        }

    def get_learned_tiers_for_files(
        self, file_ids: list[int]
    ) -> dict[int, dict[str, str]]:
        if not file_ids:
            return {}
        out: dict[int, dict[str, str]] = {}
        detailed = self.get_indexed_files_by_ids(
            file_ids,
            include_processing_ready=True,
            include_inactive_sources=True,
        )
        by_id = {int(r["id"]): r for r in detailed}
        missing: list[int] = []
        for fid in file_ids:
            rec = by_id.get(int(fid))
            tm = (rec.get("texture_map") if rec else {}) or {}
            tier_id = str(tm.get("user_similarity_tier") or "").strip()
            tier_label = str(tm.get("user_tier_label") or "").strip()
            if tier_id or tier_label:
                out[int(fid)] = {
                    "tier_id": tier_id,
                    "tier_label": tier_label,
                    "pattern_family": str(tm.get("pattern_family") or ""),
                    "pattern_subtype": str(tm.get("pattern_subtype") or ""),
                }
            else:
                missing.append(int(fid))
        if missing:
            placeholders = ",".join("?" * len(missing))
            with self.connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT result_file_id, label, note, created_at
                    FROM user_feedback
                    WHERE result_file_id IN ({placeholders})
                      AND action='similarity_tier' AND label!=''
                    ORDER BY result_file_id, created_at DESC
                    """,
                    missing,
                ).fetchall()
            seen: set[int] = set()
            for row in rows:
                fid = int(row["result_file_id"])
                if fid in seen:
                    continue
                seen.add(fid)
                out[fid] = {
                    "tier_id": str(row["note"] or "").strip(),
                    "tier_label": str(row["label"] or "").strip(),
                    "pattern_family": "",
                    "pattern_subtype": "",
                }
        return out

    def list_files_by_pattern_signature(
        self,
        pattern_family: str,
        pattern_subtype: str = "",
        *,
        limit: int = 80,
        exclude_file_id: int = 0,
    ) -> list[dict[str, Any]]:
        """Aynı öğrenilmiş desen imzasına sahip dosyalar."""
        if not pattern_family:
            return []
        subtype = (pattern_subtype or "").strip()
        with self.connect() as conn:
            if subtype:
                rows = conn.execute(
                    """
                    SELECT id, path, filename, pattern_family, pattern_subtype
                    FROM files
                    WHERE status='indexed'
                      AND pattern_family=?
                      AND pattern_subtype=?
                      AND id!=?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (pattern_family, subtype, int(exclude_file_id), int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, path, filename, pattern_family, pattern_subtype
                    FROM files
                    WHERE status='indexed'
                      AND pattern_family=?
                      AND id!=?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (pattern_family, int(exclude_file_id), int(limit)),
                ).fetchall()
        return [dict(r) for r in rows]

    # --- Project Search candidates (non-destructive until admin approval) ---

    def list_files_in_directory(self, directory: str, limit: int = 2000) -> list[dict[str, Any]]:
        normalized = normalize_path(directory).rstrip("\\/")
        escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT f.*, fe.phash
                   FROM files f LEFT JOIN features fe ON fe.file_id=f.id
                   WHERE f.path LIKE ? ESCAPE '\\' ORDER BY f.id LIMIT ?""",
                (escaped + "\\\\%", int(limit)),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_project_siblings(
        self, directory: str, base_filename: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        normalized = normalize_path(directory).rstrip("\\/")
        path_prefix = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stem = str(base_filename or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT f.*, fe.phash
                   FROM files f LEFT JOIN features fe ON fe.file_id=f.id
                   WHERE f.path LIKE ? ESCAPE '\\'
                     AND lower(f.filename) LIKE lower(?) ESCAPE '\\'
                   ORDER BY f.id LIMIT ?""",
                (path_prefix + "\\\\%", stem + ".%", int(limit)),
            ).fetchall()
            return [dict(row) for row in rows]

    def set_project_candidate(
        self,
        file_ids: list[int],
        group_id: str,
        confidence: float,
        reason: str,
        status: str = "candidate",
    ) -> None:
        if not file_ids:
            return
        placeholders = ",".join("?" for _ in file_ids)
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                f"""
                UPDATE files SET project_group_id=?, project_group_confidence=?,
                    project_group_reason=?, project_group_status=?, updated_at=?
                WHERE id IN ({placeholders})
                """,
                (group_id, float(confidence), reason, status, now, *file_ids),
            )

    def get_project_candidate_members(self, group_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE project_group_id=? ORDER BY id",
                (group_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # --- Collections ---

    def create_collection(self, name: str, customer: str = "") -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM collections WHERE name=? AND customer=?",
                (name.strip(), customer.strip()),
            ).fetchone()
            if existing:
                return int(existing["id"])
            cur = conn.execute(
                "INSERT INTO collections(name, customer, created_at) VALUES(?,?,?)",
                (name.strip(), customer.strip(), now),
            )
            return int(cur.lastrowid)

    def list_collections(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("""
                SELECT c.*, COUNT(ci.id) AS item_count
                FROM collections c
                LEFT JOIN collection_items ci ON ci.collection_id=c.id
                GROUP BY c.id ORDER BY c.created_at DESC
                """).fetchall()
            return [dict(r) for r in rows]

    def add_collection_item(
        self, collection_id: int, file_id: int, note: str = ""
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO collection_items(
                    collection_id, file_id, note, created_at
                ) VALUES(?,?,?,?)
                """,
                (collection_id, file_id, note, now),
            )
            return cur.rowcount > 0

    def list_collection_items(self, collection_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ci.*, f.filename, f.path, f.thumbnail_path
                FROM collection_items ci
                JOIN files f ON f.id=ci.file_id
                WHERE ci.collection_id=? ORDER BY ci.created_at DESC
                """,
                (collection_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Index queue (resumable) ---

    def reset_index_queue_running(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE index_queue SET status='pending', updated_at=?
                WHERE status='running'
                """,
                (now,),
            )
            return cur.rowcount

    def enqueue_index_tasks(
        self,
        file_id: int,
        source_id: int,
        file_path: str,
        task_types: list[str],
        priority: int = 50,
    ) -> None:
        """No-op — tek kuyruk SSOT (files.light_status / heavy_status)."""
        _ = (file_id, source_id, file_path, task_types, priority)
        return None

    def get_pending_index_queue(
        self,
        source_id: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if source_id:
                rows = conn.execute(
                    """
                    SELECT q.*, f.filename, f.status AS file_status
                    FROM index_queue q
                    JOIN files f ON f.id=q.file_id
                    WHERE q.status='pending' AND q.source_id=?
                    ORDER BY q.priority ASC, q.id ASC
                    LIMIT ?
                    """,
                    (source_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT q.*, f.filename, f.status AS file_status
                    FROM index_queue q
                    JOIN files f ON f.id=q.file_id
                    WHERE q.status='pending'
                    ORDER BY q.priority ASC, q.id ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def update_index_queue_item(
        self,
        queue_id: int,
        status: str,
        error_message: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE index_queue SET
                    status=?, error_message=?, last_error=?, updated_at=?,
                    attempts=attempts+1, retry_count=retry_count+1
                WHERE id=?
                """,
                (status, error_message, error_message, now, queue_id),
            )

    def mark_queue_tasks_done_for_file(self, file_id: int) -> None:
        """Sadece artifact'ı doğrulanan görevleri DONE yap; eksikleri pending bırak.

        False-done (done yazılmış ama features/thumb/hash yok) üretimini engeller.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            from core.index_integrity import task_artifact_ok
        except Exception:
            task_artifact_ok = None  # type: ignore

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, COALESCE(task_type, stage) AS task
                FROM index_queue
                WHERE file_id=? AND status IN ('pending','running','done')
                """,
                (int(file_id),),
            ).fetchall()

        for row in rows:
            qid = int(row["id"])
            task = str(row["task"] or "")
            ok = True
            if task_artifact_ok is not None:
                try:
                    ok = bool(task_artifact_ok(self, int(file_id), task))
                except Exception:
                    ok = False
            if ok:
                with self.connect() as conn:
                    conn.execute(
                        """
                        UPDATE index_queue SET status='done', updated_at=?,
                               error_message='', last_error=''
                        WHERE id=?
                        """,
                        (now, qid),
                    )
            else:
                with self.connect() as conn:
                    conn.execute(
                        """
                        UPDATE index_queue SET status='pending', updated_at=?,
                               error_message=?, last_error=?
                        WHERE id=?
                        """,
                        (
                            now,
                            f"artifact_missing_for_task:{task}",
                            f"artifact_missing_for_task:{task}",
                            qid,
                        ),
                    )

    def file_queue_all_done(self, file_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c FROM index_queue
                WHERE file_id=? AND status NOT IN ('done','skipped')
                """,
                (file_id,),
            ).fetchone()
            pending = int(row["c"]) if row else 0
            if pending > 0:
                return False
            has_done = conn.execute(
                "SELECT 1 FROM index_queue WHERE file_id=? AND status='done' LIMIT 1",
                (file_id,),
            ).fetchone()
            return bool(has_done)

    def count_queue_for_source(self, source_id: int) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS c FROM index_queue
                WHERE source_id=? GROUP BY status
                """,
                (source_id,),
            ).fetchall()
            return {r["status"]: r["c"] for r in rows}

    def get_processing_files(
        self, source_id: int | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if source_id:
                rows = conn.execute(
                    """
                    SELECT * FROM files
                    WHERE status='processing' AND source_id=?
                    ORDER BY
                        CASE WHEN lower(filename) LIKE '%.tif'
                                   OR lower(filename) LIKE '%.tiff'
                             THEN 1 ELSE 0 END,
                        file_size ASC,
                        id ASC
                    LIMIT ?
                    """,
                    (source_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM files WHERE status='processing'
                    ORDER BY
                        CASE WHEN lower(filename) LIKE '%.tif'
                                   OR lower(filename) LIKE '%.tiff'
                             THEN 1 ELSE 0 END,
                        file_size ASC,
                        id ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def count_pending_for_sources(self, source_ids: list[int] | None = None) -> int:
        with self.connect() as conn:
            if source_ids:
                ph = ",".join("?" * len(source_ids))
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS c FROM files
                    WHERE status IN ('processing','pending') AND source_id IN ({ph})
                    """,
                    source_ids,
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM files WHERE status IN ('processing','pending')"
                ).fetchone()
            return int(row["c"]) if row else 0

    def set_file_index_stage(
        self, file_id: int, stage: str, *, needs_review: int | None = None
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            if needs_review is None:
                conn.execute(
                    "UPDATE files SET index_stage=?, updated_at=? WHERE id=?",
                    (stage, now, int(file_id)),
                )
            else:
                conn.execute(
                    "UPDATE files SET index_stage=?, needs_review=?, updated_at=? WHERE id=?",
                    (stage, int(needs_review), now, int(file_id)),
                )

    def count_by_index_stage(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT index_stage, COUNT(*) AS c FROM files GROUP BY index_stage"
            ).fetchall()
            return {str(r["index_stage"] or "pending_light"): int(r["c"]) for r in rows}

    def count_index_queue_active(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c FROM index_queue
                WHERE status IN ('pending', 'running', 'error')
                """
            ).fetchone()
            return int(row["c"]) if row else 0

    def count_pending_full(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM files WHERE index_stage='pending_full'"
            ).fetchone()
            return int(row["c"]) if row else 0

    def list_backfill_candidates(
        self,
        source_id: int = 0,
        limit: int = 50000,
    ) -> list[dict[str, Any]]:
        stages = (
            "pending_light",
            "pending_full",
            "failed",
            "processing",
            "skipped_heavy_format",
            "light_done",
            "needs_medium_preview",
        )
        ph = ",".join("?" * len(stages))
        with self.connect() as conn:
            if source_id:
                rows = conn.execute(
                    f"""
                    SELECT id, path, filename, index_stage, status, mtime
                    FROM files
                    WHERE source_id=? AND index_stage IN ({ph})
                    ORDER BY id LIMIT ?
                    """,
                    (source_id, *stages, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT id, path, filename, index_stage, status, mtime
                    FROM files
                    WHERE index_stage IN ({ph})
                    ORDER BY id LIMIT ?
                    """,
                    (*stages, limit),
                ).fetchall()
            return [dict(r) for r in rows]

    def list_light_work_candidates(
        self, source_id: int = 0, limit: int = 50000
    ) -> list[dict[str, Any]]:
        """Hızlı index — yalnızca light_status != done.

        Not: light_done ama thumb/preview eksik dosyalar buraya GİRMEZ.
        Aksi halde bitmiş dosyalar yeniden işlenir; UI kuyruğu sahte düşer,
        restart'ta DB'ye dönünce kuyruk 'geriye gider' gibi görünür.
        """
        with self.connect() as conn:
            sql = """
                SELECT id, path, filename, index_stage, status, mtime
                FROM files
                WHERE status IN ('indexed','processing','pending','error')
                  AND coalesce(light_status,'') != 'done'
                ORDER BY id LIMIT ?
            """
            if source_id:
                sql = sql.replace(
                    "WHERE status",
                    "WHERE source_id=? AND status",
                    1,
                )
                rows = conn.execute(sql, (source_id, limit)).fetchall()
            else:
                rows = conn.execute(sql, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def list_night_full_candidates(
        self, source_id: int = 0, limit: int = 50000
    ) -> list[dict[str, Any]]:
        """Genel AI: light hazır / heavy bekleyen + eksik preview."""
        with self.connect() as conn:
            sql = """
                SELECT id, path, filename, index_stage, status
                FROM files
                WHERE (
                    (light_status='done' AND coalesce(heavy_status,'')!='done')
                    OR index_stage IN (
                        'pending_full','light_done','skipped_heavy_format',
                        'needs_medium_preview'
                    )
                )
                AND status IN ('indexed','processing','pending','error')
                ORDER BY
                  CASE WHEN light_status='done' AND coalesce(heavy_status,'')!='done'
                       THEN 0 ELSE 1 END,
                  id
                LIMIT ?
            """
            if source_id:
                sql = sql.replace("WHERE (", "WHERE source_id=? AND (", 1)
                rows = conn.execute(sql, (source_id, limit)).fetchall()
            else:
                rows = conn.execute(sql, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def purge_broken_index_records(self) -> dict[str, int]:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            r1 = conn.execute("""
                UPDATE files SET status='pending', index_stage='pending_light', error_msg=''
                WHERE status='processing'
                """).rowcount
            r2 = conn.execute("""
                UPDATE files SET index_stage='pending_light'
                WHERE status='error' AND index_stage='failed'
                """).rowcount
            r3 = conn.execute(
                """
                UPDATE index_queue SET status='pending', updated_at=?
                WHERE status='running'
                """,
                (now,),
            ).rowcount
            return {"processing_reset": r1, "failed_reset": r2, "queue_reset": r3}

    def requeue_files_missing_thumbnails(self, limit: int = 50000) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id FROM files
                WHERE status='indexed' AND (
                    thumbnail_path IS NULL OR thumbnail_path=''
                    OR index_stage IN ('failed','pending_light')
                )
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            n = 0
            for row in rows:
                fid = int(row["id"])
                conn.execute(
                    "UPDATE files SET index_stage='pending_light', status='processing' WHERE id=?",
                    (fid,),
                )
                n += 1
            return {"queued": n}

    def queue_files_missing_ai_embeddings(self, limit: int = 50000) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id FROM files f
                LEFT JOIN features fe ON fe.file_id=f.id
                WHERE f.status='indexed'
                  AND (fe.dino_embedding IS NULL OR fe.clip_embedding IS NULL
                       OR length(fe.dino_embedding)=0 OR length(fe.clip_embedding)=0)
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            n = 0
            for row in rows:
                fid = int(row["id"])
                conn.execute(
                    "UPDATE files SET index_stage='pending_full' WHERE id=?",
                    (fid,),
                )
                n += 1
            return {"queued": n}

    def queue_files_missing_ocr(self, limit: int = 50000) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id, f.source_id, f.path FROM files f
                WHERE f.status='indexed'
                  AND COALESCE(f.ocr_text,'')=''
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            n = 0
            for row in rows:
                fid = int(row["id"])
                conn.execute(
                    "UPDATE files SET needs_ocr=1, index_stage='pending_full' WHERE id=?",
                    (fid,),
                )
                sid = int(row["source_id"] or 0)
                fpath = str(row["path"] or "")
                exists = conn.execute(
                    """
                    SELECT id FROM index_queue
                    WHERE file_id=? AND task_type='ocr' AND status IN ('pending','running')
                    """,
                    (fid,),
                ).fetchone()
                if not exists and sid and fpath:
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """
                        INSERT INTO index_queue(
                            file_id, stage, status, attempts, last_error,
                            source_id, file_path, task_type, priority,
                            retry_count, error_message, created_at, updated_at
                        ) VALUES (?, 'ocr', 'pending', 0, '', ?, ?, 'ocr', 60, 0, '', ?, ?)
                        """,
                        (fid, sid, fpath, now, now),
                    )
                n += 1
            return {"queued": n}

    def count_files_by_status(self, status: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM files WHERE status=?",
                (status,),
            ).fetchone()
        return int(row["c"] or 0) if row else 0

    def try_acquire_processing_lock(self, file_id: int, worker: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT processing_lock FROM files WHERE id=?", (int(file_id),)
            ).fetchone()
            if not row:
                return False
            lock = str(row["processing_lock"] or "")
            if lock and lock != worker:
                return False
            conn.execute(
                "UPDATE files SET processing_lock=?, updated_at=? WHERE id=?",
                (worker, now, int(file_id)),
            )
            return True

    def release_processing_lock(self, file_id: int, worker: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET processing_lock='', updated_at=?
                WHERE id=? AND processing_lock=?
                """,
                (now, int(file_id), worker),
            )

    def mark_light_processing(self, file_id: int) -> None:
        """Done/failed üzerine yazılmaz — tek yön SSOT."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET light_status='processing', updated_at=?
                WHERE id=? AND light_status IN ('pending', 'processing')
                """,
                (now, int(file_id)),
            )

    def mark_heavy_processing(self, file_id: int) -> None:
        """Done/failed üzerine yazılmaz — tek yön SSOT."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET heavy_status='processing', updated_at=?
                WHERE id=? AND coalesce(nullif(heavy_status,''),'pending')
                      IN ('pending', 'processing')
                """,
                (now, int(file_id)),
            )

    def mark_light_done(self, file_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET
                    light_status='done',
                    light_done_at=?,
                    needs_medium_preview=CASE
                        WHEN physical_thumbnail_ready=1
                         AND physical_preview_ready=1 THEN 0
                        ELSE 1
                    END,
                    processing_lock='',
                    updated_at=?
                WHERE id=?
                """,
                (now, now, int(file_id)),
            )

    def is_ai_final_ready(self, file_id: int) -> bool:
        dna_ready = _json_object_ready_sql("fe.texture_map", "pattern_dna")
        semantic_ready = _json_object_ready_sql("fe.texture_map", "semantic_tags")
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT 1
                FROM files f
                JOIN features fe ON fe.file_id=f.id
                WHERE f.id=?
                  AND f.status='indexed'
                  AND COALESCE(f.feature_preview_path,'')!=''
                  AND COALESCE(f.physical_preview_ready,0)=1
                  AND fe.dino_embedding IS NOT NULL
                  AND length(fe.dino_embedding)>0
                  AND fe.clip_embedding IS NOT NULL
                  AND length(fe.clip_embedding)>0
                  AND {dna_ready}
                  AND {semantic_ready}
                  AND COALESCE(fe.phash,'')!=''
                  AND COALESCE(fe.texture_features,'') NOT IN ('','[]')
                  AND COALESCE(fe.patch_embeddings,'') NOT IN ('','[]')
                LIMIT 1
                """,
                (int(file_id),),
            ).fetchone()
            return row is not None

    def mark_heavy_incomplete(self, file_id: int) -> None:
        """Eksik zorunlu aşama heavy done olamaz; yeniden kuyruğa döner."""
        now = datetime.now(timezone.utc).isoformat()
        dna_ready = _json_object_ready_sql("fe.texture_map", "pattern_dna")
        semantic_ready = _json_object_ready_sql("fe.texture_map", "semantic_tags")
        with self.connect() as conn:
            conn.execute(
                f"""
                UPDATE files SET
                    heavy_status='pending',
                    heavy_done_at='',
                    index_stage='light_done',
                    needs_embedding=CASE WHEN EXISTS(
                        SELECT 1 FROM features fe WHERE fe.file_id=files.id
                          AND fe.dino_embedding IS NOT NULL
                          AND length(fe.dino_embedding)>0
                          AND fe.clip_embedding IS NOT NULL
                          AND length(fe.clip_embedding)>0
                    ) THEN 0 ELSE 1 END,
                    needs_dna=CASE WHEN EXISTS(
                        SELECT 1 FROM features fe WHERE fe.file_id=files.id
                          AND {dna_ready}
                    ) THEN 0 ELSE 1 END,
                    needs_semantic=CASE WHEN EXISTS(
                        SELECT 1 FROM features fe WHERE fe.file_id=files.id
                          AND {semantic_ready}
                    ) THEN 0 ELSE 1 END,
                    needs_texture=CASE WHEN EXISTS(
                        SELECT 1 FROM features fe WHERE fe.file_id=files.id
                          AND COALESCE(fe.phash,'')!=''
                          AND COALESCE(fe.texture_features,'') NOT IN ('','[]')
                    ) THEN 0 ELSE 1 END,
                    needs_ai=1,
                    updated_at=?
                WHERE id=?
                """,
                (now, int(file_id)),
            )

    def mark_heavy_done(self, file_id: int) -> bool:
        if not self.is_ai_final_ready(file_id):
            self.mark_heavy_incomplete(file_id)
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE files SET
                    heavy_status='done',
                    heavy_done_at=?,
                    needs_embedding=0,
                    needs_ai=0,
                    needs_semantic=0,
                    needs_dna=0,
                    needs_ocr=0,
                    needs_texture=0,
                    processing_lock='',
                    updated_at=?
                WHERE id=?
                """,
                (now, now, int(file_id)),
            )
            return cur.rowcount > 0

    def mark_light_failed(
        self,
        file_id: int,
        *,
        reason: str = "",
        error_msg: str = "",
    ) -> None:
        if not reason and error_msg:
            from core.quarantine import classify_quarantine_reason

            reason = classify_quarantine_reason(error_msg)
        reason = reason or "other"
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET
                    light_status='failed',
                    status='error',
                    processing_lock='',
                    quarantine_reason=CASE
                        WHEN ? != '' THEN ?
                        ELSE coalesce(nullif(quarantine_reason,''), ?)
                    END,
                    error_msg=CASE
                        WHEN ? != '' THEN ?
                        ELSE error_msg
                    END,
                    updated_at=?
                WHERE id=? AND light_status != 'done'
                """,
                (
                    reason,
                    reason,
                    reason or "other",
                    error_msg,
                    error_msg,
                    now,
                    int(file_id),
                ),
            )

    def mark_heavy_failed(
        self,
        file_id: int,
        *,
        reason: str = "",
        error_msg: str = "",
    ) -> None:
        if not reason and error_msg:
            from core.quarantine import classify_quarantine_reason

            reason = classify_quarantine_reason(error_msg)
        reason = reason or "other"
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET
                    heavy_status='failed',
                    processing_lock='',
                    quarantine_reason=CASE
                        WHEN ? != '' THEN ?
                        ELSE coalesce(nullif(quarantine_reason,''), ?)
                    END,
                    error_msg=CASE
                        WHEN ? != '' THEN ?
                        ELSE error_msg
                    END,
                    updated_at=?
                WHERE id=? AND heavy_status != 'done'
                """,
                (
                    reason,
                    reason,
                    reason or "other",
                    error_msg,
                    error_msg,
                    now,
                    int(file_id),
                ),
            )

    def mark_heavy_waiting_preview(self, file_id: int) -> None:
        """Retry edilebilir preview önkoşulu; gerçek heavy failure değildir."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET
                    heavy_status='pending',
                    index_stage='needs_medium_preview',
                    needs_medium_preview=1,
                    physical_preview_ready=0,
                    physical_verified_at=?,
                    processing_lock='',
                    quarantine_reason='',
                    error_msg='',
                    updated_at=?
                WHERE id=? AND heavy_status != 'done'
                """,
                (now, now, int(file_id)),
            )

    def set_needs_medium_preview_flag(self, file_id: int, value: bool = True) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE files SET
                    needs_medium_preview=?,
                    heavy_status='pending',
                    physical_preview_ready=CASE WHEN ? THEN 0
                                                ELSE physical_preview_ready END,
                    physical_verified_at=CASE WHEN ? THEN ?
                                               ELSE physical_verified_at END,
                    updated_at=?
                WHERE id=?
                """,
                (
                    1 if value else 0,
                    1 if value else 0,
                    1 if value else 0,
                    now,
                    now,
                    int(file_id),
                ),
            )

    def mark_file_pipeline_reset_on_change(
        self,
        file_id: int,
        *,
        content_changed: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Yalnızca gerçek içerik değişiminde done→pending.

        content_changed=False ise no-op (SSOT: done asla rastgele bozulmaz).
        """
        if not content_changed:
            return
        now = datetime.now(timezone.utc).isoformat()
        artifact_paths = (
            self.list_file_artifact_paths(file_ids=[int(file_id)])
            if cache_dir
            else []
        )
        with self.connect() as conn:
            conn.execute("DELETE FROM features WHERE file_id=?", (int(file_id),))
            try:
                conn.execute("DELETE FROM files_fts WHERE file_id=?", (int(file_id),))
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                UPDATE files SET
                    light_status='pending',
                    heavy_status='pending',
                    status='pending',
                    processing_lock='',
                    light_done_at='',
                    heavy_done_at='',
                    needs_medium_preview=1,
                    needs_embedding=1,
                    needs_ai=1,
                    needs_semantic=1,
                    needs_dna=1,
                    needs_ocr=1,
                    needs_texture=1,
                    physical_preview_ready=NULL,
                    physical_thumbnail_ready=NULL,
                    physical_verified_at='',
                    thumbnail_path='',
                    feature_preview_path='',
                    thumbnail_status='pending',
                    preview_status='pending',
                    ocr_text='',
                    ocr_processed=0,
                    ocr_error='',
                    patch_error='',
                    text_search_blob='',
                    partial_hash='',
                    full_hash='',
                    pattern_family='',
                    pattern_type='',
                    texture_family='',
                    pattern_subtype='',
                    pattern_confidence=0,
                    quarantine_reason='',
                    error_msg='',
                    updated_at=?
                WHERE id=?
                """,
                (now, int(file_id)),
            )
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('faiss_needs_rebuild','1') "
                "ON CONFLICT(key) DO UPDATE SET value='1'"
            )
        if cache_dir:
            self.remove_cache_artifacts(artifact_paths, cache_dir)

    def count_light_heavy_pipeline(self) -> dict[str, int]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN light_status='pending' THEN 1 ELSE 0 END) AS light_pending,
                    SUM(CASE WHEN light_status='processing' THEN 1 ELSE 0 END) AS light_processing,
                    SUM(CASE WHEN light_status='done' THEN 1 ELSE 0 END) AS light_done,
                    SUM(CASE WHEN light_status='failed' THEN 1 ELSE 0 END) AS light_failed,
                    SUM(CASE WHEN heavy_status='pending' THEN 1 ELSE 0 END) AS heavy_pending,
                    SUM(CASE WHEN heavy_status='processing' THEN 1 ELSE 0 END) AS heavy_processing,
                    SUM(CASE WHEN heavy_status='done' THEN 1 ELSE 0 END) AS heavy_done,
                    SUM(CASE WHEN heavy_status='failed' THEN 1 ELSE 0 END) AS heavy_failed,
                    SUM(CASE WHEN needs_medium_preview=1 THEN 1 ELSE 0 END) AS needs_medium_preview_count
                FROM files
                WHERE status NOT IN ('excluded_internal','missing')
                """
            ).fetchone()
        return {key: int(row[key] or 0) for key in row.keys()}
