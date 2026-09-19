"""Isolated object index. Never writes patterns.db, FAISS, or preview cache."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS object_files(
  file_id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  mtime REAL DEFAULT 0,
  file_size INTEGER DEFAULT 0,
  object_count INTEGER DEFAULT 0,
  scanned_at TEXT DEFAULT '',
  detector TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS object_instances(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id INTEGER NOT NULL,
  instance_index INTEGER NOT NULL,
  label TEXT NOT NULL,
  label_tr TEXT DEFAULT '',
  confidence REAL NOT NULL,
  bbox TEXT NOT NULL,
  area_ratio REAL DEFAULT 0,
  embedding BLOB,
  embedding_dim INTEGER DEFAULT 0,
  embedding_backend TEXT DEFAULT '',
  instance_id TEXT DEFAULT '',
  UNIQUE(file_id, instance_index),
  FOREIGN KEY(file_id) REFERENCES object_files(file_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_obj_label ON object_instances(label);
CREATE INDEX IF NOT EXISTS idx_obj_file ON object_instances(file_id);
CREATE TABLE IF NOT EXISTS object_concepts(
  file_id INTEGER NOT NULL,
  lemma TEXT NOT NULL,
  label_tr TEXT DEFAULT '',
  category TEXT DEFAULT '',
  confidence REAL NOT NULL,
  evidence TEXT NOT NULL,
  detected INTEGER DEFAULT 0,
  PRIMARY KEY (file_id, lemma, evidence)
);
CREATE INDEX IF NOT EXISTS idx_obj_concept_lemma ON object_concepts(lemma);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blob(vec: np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).reshape(-1).tobytes()


def _vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


class ObjectIndexStore:
    def __init__(self, db_path: str | Path, *, readonly: bool = False):
        self.path = str(db_path)
        self.readonly = bool(readonly)
        self._missing = False
        if self.readonly:
            self._missing = not Path(self.path).is_file()
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        if self.readonly:
            if self._missing:
                raise sqlite3.OperationalError("object_index missing")
            uri = Path(self.path).resolve().as_posix()
            con = sqlite3.connect(f"file:{uri}?mode=ro", uri=True, timeout=60)
            con.row_factory = sqlite3.Row
            return con
        con = sqlite3.connect(self.path, timeout=60)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init(self) -> None:
        with self._connect() as con:
            con.executescript(SCHEMA)
            cols = {str(r[1]) for r in con.execute("PRAGMA table_info(object_instances)")}
            if "instance_id" not in cols:
                con.execute("ALTER TABLE object_instances ADD COLUMN instance_id TEXT DEFAULT ''")
            for name, spec in (
                ("detector", "TEXT DEFAULT ''"),
                ("evidence", "TEXT DEFAULT ''"),
                ("model_version", "TEXT DEFAULT ''"),
                ("vocab_version", "TEXT DEFAULT ''"),
                ("ovd_threshold", "REAL DEFAULT 0"),
            ):
                if name not in cols:
                    con.execute(f"ALTER TABLE object_instances ADD COLUMN {name} {spec}")
            con.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_ovd_instance_key
                   ON object_instances(file_id, detector, label, bbox)
                   WHERE detector IN ('grounding_dino','owlv2')"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS ovd_file_scans(
                    file_id INTEGER PRIMARY KEY,
                    mtime REAL DEFAULT 0,
                    file_size INTEGER DEFAULT 0,
                    vocab_version TEXT DEFAULT '',
                    model TEXT DEFAULT '',
                    threshold REAL DEFAULT 0,
                    scanned_at TEXT DEFAULT '',
                    box_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'done',
                    error TEXT DEFAULT '',
                    last_path TEXT DEFAULT ''
                )"""
            )
            ocols = {str(r[1]) for r in con.execute("PRAGMA table_info(ovd_file_scans)")}
            for name, spec in (
                ("status", "TEXT DEFAULT 'done'"),
                ("error", "TEXT DEFAULT ''"),
                ("last_path", "TEXT DEFAULT ''"),
            ):
                if name not in ocols:
                    con.execute(f"ALTER TABLE ovd_file_scans ADD COLUMN {name} {spec}")
            con.executescript(
                """CREATE TABLE IF NOT EXISTS object_concepts(
  file_id INTEGER NOT NULL,
  lemma TEXT NOT NULL,
  label_tr TEXT DEFAULT '',
  category TEXT DEFAULT '',
  confidence REAL NOT NULL,
  evidence TEXT NOT NULL,
  detected INTEGER DEFAULT 0,
  PRIMARY KEY (file_id, lemma, evidence)
);
CREATE INDEX IF NOT EXISTS idx_obj_concept_lemma ON object_concepts(lemma);"""
            )
            con.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('kind','object_intelligence')"
            )

    def has_scan(self, file_id: int) -> bool:
        """True if this file already has an object/concept pass (resume skip)."""
        if self.readonly and self._missing:
            return False
        with self._connect() as con:
            try:
                row = con.execute(
                    "SELECT scanned_at FROM object_files WHERE file_id=?",
                    (int(file_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        return bool(row and str(row["scanned_at"] or "").strip())

    def replace_file_objects(
        self,
        file_id: int,
        path: str,
        objects: Iterable[dict[str, Any]],
        *,
        mtime: float = 0,
        file_size: int = 0,
        detector: str = "",
    ) -> int:
        if self.readonly:
            raise sqlite3.OperationalError("object_index is read-only")
        rows = list(objects)
        with self._connect() as con:
            try:
                con.execute(
                    """DELETE FROM object_instances
                       WHERE file_id=? AND IFNULL(detector,'') NOT IN ('grounding_dino','owlv2')""",
                    (int(file_id),),
                )
            except sqlite3.OperationalError:
                con.execute("DELETE FROM object_instances WHERE file_id=?", (int(file_id),))
            con.execute(
                """INSERT INTO object_files(file_id,path,mtime,file_size,object_count,scanned_at,detector)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(file_id) DO UPDATE SET
                     path=excluded.path, mtime=excluded.mtime, file_size=excluded.file_size,
                     object_count=excluded.object_count, scanned_at=excluded.scanned_at,
                     detector=excluded.detector""",
                (int(file_id), str(path), float(mtime or 0), int(file_size or 0),
                 len(rows), _now(), str(detector or "")),
            )
            for i, obj in enumerate(rows):
                bbox = obj.get("bbox") or (0, 0, 0, 0)
                emb = obj.get("embedding")
                con.execute(
                    """INSERT INTO object_instances(
                         file_id,instance_index,label,label_tr,confidence,bbox,area_ratio,
                         embedding,embedding_dim,embedding_backend,instance_id,detector)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        int(file_id),
                        i,
                        str(obj.get("label") or ""),
                        str(obj.get("label_tr") or ""),
                        float(obj.get("confidence") or 0),
                        json.dumps([int(x) for x in bbox]),
                        float(obj.get("area_ratio") or 0),
                        _blob(emb),
                        int(0 if emb is None else np.asarray(emb).size),
                        str(obj.get("embedding_backend") or ""),
                        str(obj.get("instance_id") or ""),
                        str(detector or obj.get("detector") or ""),
                    ),
                )
        return len(rows)

    def replace_file_concepts(
        self,
        file_id: int,
        concepts: Iterable[dict[str, Any]],
    ) -> int:
        """Persist Nesne/Kavram DNA rows. Isolated from patterns.db."""
        if self.readonly:
            raise sqlite3.OperationalError("object_index is read-only")
        rows = [c for c in concepts if isinstance(c, dict) and (c.get("lemma") or c.get("label"))]
        with self._connect() as con:
            con.execute("DELETE FROM object_concepts WHERE file_id=?", (int(file_id),))
            for c in rows:
                lemma = str(c.get("lemma") or c.get("label") or "").strip().lower()
                if not lemma:
                    continue
                con.execute(
                    """INSERT OR REPLACE INTO object_concepts(
                         file_id, lemma, label_tr, category, confidence, evidence, detected)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        int(file_id),
                        lemma,
                        str(c.get("label_tr") or ""),
                        str(c.get("category") or c.get("category_tr") or ""),
                        float(c.get("confidence") or 0.0),
                        str(c.get("evidence") or "visual_concept"),
                        1 if c.get("detected") else 0,
                    ),
                )
        return len(rows)

    def concepts_for_file(self, file_id: int) -> list[dict[str, Any]]:
        if self.readonly and self._missing:
            return []
        with self._connect() as con:
            try:
                rows = con.execute(
                    "SELECT * FROM object_concepts WHERE file_id=?",
                    (int(file_id),),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(r) for r in rows]

    def search_concepts(self, lemmas: Iterable[str], *, limit: int = 400) -> list[int]:
        wanted = {str(x).strip().lower() for x in lemmas if str(x).strip()}
        if not wanted:
            return []
        if self.readonly and self._missing:
            return []
        placeholders = ",".join("?" * len(wanted))
        with self._connect() as con:
            try:
                rows = con.execute(
                    f"""SELECT file_id FROM object_concepts
                        WHERE lemma IN ({placeholders})
                        GROUP BY file_id
                        ORDER BY MAX(confidence) DESC LIMIT ?""",
                    (*wanted, int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [int(r["file_id"]) for r in rows]

    def instances_for_file(self, file_id: int) -> list[dict[str, Any]]:
        if self.readonly and self._missing:
            return []
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM object_instances WHERE file_id=? ORDER BY instance_index",
                (int(file_id),),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["bbox"] = tuple(json.loads(d["bbox"]))
            d["embedding"] = _vec(d.get("embedding"))
            out.append(d)
        return out

    def instances_for_files(self, file_ids: Iterable[int]) -> dict[int, list[dict[str, Any]]]:
        ids = [int(x) for x in file_ids]
        out: dict[int, list[dict[str, Any]]] = {i: [] for i in ids}
        if not ids:
            return out
        if self.readonly and self._missing:
            return out
        placeholders = ",".join("?" * len(ids))
        with self._connect() as con:
            rows = con.execute(
                f"""SELECT * FROM object_instances
                    WHERE file_id IN ({placeholders})
                    ORDER BY file_id, instance_index""",
                ids,
            ).fetchall()
        for r in rows:
            d = dict(r)
            d["bbox"] = tuple(json.loads(d["bbox"]))
            d["embedding"] = _vec(d.get("embedding"))
            out.setdefault(int(d["file_id"]), []).append(d)
        return out

    def file_paths(self, file_ids: Iterable[int]) -> dict[int, str]:
        ids = [int(x) for x in file_ids]
        if not ids or (self.readonly and self._missing):
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._connect() as con:
            rows = con.execute(
                f"SELECT file_id, path FROM object_files WHERE file_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {int(r["file_id"]): str(r["path"] or "") for r in rows}

    def counts_for_file(self, file_id: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for inst in self.instances_for_file(file_id):
            lab = str(inst["label"])
            counts[lab] = counts.get(lab, 0) + 1
        return counts

    def search_label(self, labels: Iterable[str], *, min_count: int = 1, limit: int = 400) -> list[int]:
        wanted = {str(x) for x in labels if str(x)}
        if not wanted:
            return []
        if self.readonly and self._missing:
            return []
        placeholders = ",".join("?" * len(wanted))
        extra = "AND IFNULL(detector,'') NOT IN ('grounding_dino','owlv2')"
        with self._connect() as con:
            try:
                rows = con.execute(
                    f"""SELECT file_id, COUNT(*) AS n FROM object_instances
                        WHERE label IN ({placeholders}) {extra}
                        GROUP BY file_id HAVING n >= ?
                        ORDER BY n DESC LIMIT ?""",
                    (*wanted, int(min_count), int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = con.execute(
                    f"""SELECT file_id, COUNT(*) AS n FROM object_instances
                        WHERE label IN ({placeholders})
                        GROUP BY file_id HAVING n >= ?
                        ORDER BY n DESC LIMIT ?""",
                    (*wanted, int(min_count), int(limit)),
                ).fetchall()
        return [int(r["file_id"]) for r in rows]

    def rtdetr_count(self, file_id: int) -> int:
        if self.readonly and self._missing:
            return 0
        with self._connect() as con:
            try:
                row = con.execute(
                    """SELECT COUNT(*) AS n FROM object_instances
                       WHERE file_id=? AND IFNULL(detector,'') NOT IN ('grounding_dino','owlv2')""",
                    (int(file_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                row = con.execute(
                    "SELECT COUNT(*) AS n FROM object_instances WHERE file_id=?",
                    (int(file_id),),
                ).fetchone()
        return int(row["n"] if row else 0)

    def search_ovd_labels(self, labels: Iterable[str], *, limit: int = 400) -> list[int]:
        wanted = {str(x).strip().lower() for x in labels if str(x).strip()}
        if not wanted or (self.readonly and self._missing):
            return []
        placeholders = ",".join("?" * len(wanted))
        with self._connect() as con:
            try:
                rows = con.execute(
                    f"""SELECT file_id FROM object_instances
                        WHERE detector IN ('owlv2','grounding_dino') AND lower(label) IN ({placeholders})
                        GROUP BY file_id
                        ORDER BY MAX(confidence) DESC LIMIT ?""",
                    (*wanted, int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [int(r["file_id"]) for r in rows]

    def ovd_scan_key_matches(
        self,
        file_id: int,
        *,
        mtime: float,
        file_size: int,
        vocab_version: str,
        model: str,
        threshold: float,
    ) -> bool:
        if self.readonly and self._missing:
            return False
        with self._connect() as con:
            try:
                row = con.execute(
                    "SELECT * FROM ovd_file_scans WHERE file_id=?",
                    (int(file_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        if not row:
            return False
        try:
            st = str(row["status"] or "done")
        except (KeyError, IndexError):
            st = "done"
        if st in {"failed", "retry"}:
            return False
        return (
            abs(float(row["mtime"] or 0) - float(mtime or 0)) < 0.01
            and int(row["file_size"] or 0) == int(file_size or 0)
            and str(row["vocab_version"] or "") == str(vocab_version)
            and str(row["model"] or "") == str(model)
            and abs(float(row["threshold"] or 0) - float(threshold)) < 1e-6
        )

    def ovd_has_done_scan(self, file_id: int) -> bool:
        """OCR/Object Index resume: file already had an OWL pass. Preview pool is not rewritten."""
        if self.readonly and self._missing:
            return False
        with self._connect() as con:
            try:
                row = con.execute(
                    "SELECT status FROM ovd_file_scans WHERE file_id=?",
                    (int(file_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        if not row:
            return False
        try:
            st = str(row["status"] or "done")
        except (KeyError, IndexError):
            st = "done"
        return st not in {"failed", "retry"}

    def exact_object_labels(self, file_id: int) -> list[str]:
        """RT-DETR / EXACT_OBJECT labels only. Does not include owlv2."""
        if self.readonly and self._missing:
            return []
        with self._connect() as con:
            try:
                rows = con.execute(
                    """SELECT DISTINCT lower(label) FROM object_instances
                       WHERE file_id=? AND IFNULL(detector,'') NOT IN ('owlv2','grounding_dino')""",
                    (int(file_id),),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        out: list[str] = []
        for r in rows:
            lab = str(r[0] or "").strip().lower()
            if lab and lab not in out:
                out.append(lab)
        return out

    def upsert_open_vocab_objects(
        self,
        file_id: int,
        path: str,
        objects: Iterable[dict[str, Any]],
        *,
        mtime: float = 0,
        file_size: int = 0,
        vocab_version: str = "",
        model: str = "grounding_dino",
        model_version: str = "tiny-v1",
        threshold: float = 0.35,
        scan_status: str = "done",
        scan_error: str = "",
        last_path: str = "",
    ) -> int:
        """Write OWL/OVD rows only. Never deletes RT-DETR instances."""
        if self.readonly:
            raise sqlite3.OperationalError("object_index is read-only")
        from core.index_freeze import guard_index_write

        guard_index_write("ovd_upsert", "object_index")
        rows = list(objects)
        det = "owlv2" if "owlv2" in str(model or "") else "grounding_dino"
        with self._connect() as con:
            con.execute(
                "DELETE FROM object_instances WHERE file_id=? AND detector=?",
                (int(file_id), det),
            )
            con.execute(
                """INSERT INTO object_files(file_id,path,mtime,file_size,object_count,scanned_at,detector)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(file_id) DO UPDATE SET path=excluded.path""",
                (
                    int(file_id),
                    str(path),
                    float(mtime or 0),
                    int(file_size or 0),
                    0,
                    _now(),
                    "",
                ),
            )
            base = 1_000_000
            for i, obj in enumerate(rows):
                bbox = obj.get("bbox") or [0, 0, 0, 0]
                bbox_j = json.dumps([round(float(x), 6) for x in bbox])
                con.execute(
                    """INSERT INTO object_instances(
                         file_id,instance_index,label,label_tr,confidence,bbox,area_ratio,
                         embedding,embedding_dim,embedding_backend,instance_id,
                         detector,evidence,model_version,vocab_version,ovd_threshold)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        int(file_id),
                        base + i,
                        str(obj.get("label") or "").strip().lower(),
                        str(obj.get("label_tr") or ""),
                        float(obj.get("confidence") or 0),
                        bbox_j,
                        float(obj.get("area_ratio") or 0),
                        None,
                        0,
                        det,
                        str(obj.get("instance_id") or f"{file_id}:{det}:{i}"),
                        det,
                        "OPEN_VOCAB_OBJECT",
                        str(model_version or "tiny-v1"),
                        str(vocab_version or ""),
                        float(threshold),
                    ),
                )
            con.execute(
                """INSERT INTO ovd_file_scans(
                     file_id,mtime,file_size,vocab_version,model,threshold,scanned_at,box_count,
                     status,error,last_path)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(file_id) DO UPDATE SET
                     mtime=excluded.mtime, file_size=excluded.file_size,
                     vocab_version=excluded.vocab_version, model=excluded.model,
                     threshold=excluded.threshold, scanned_at=excluded.scanned_at,
                     box_count=excluded.box_count, status=excluded.status,
                     error=excluded.error, last_path=excluded.last_path""",
                (
                    int(file_id),
                    float(mtime or 0),
                    int(file_size or 0),
                    str(vocab_version or ""),
                    str(model or det),
                    float(threshold),
                    _now(),
                    len(rows),
                    str(scan_status or "done"),
                    str(scan_error or "")[:500],
                    str(last_path or path or ""),
                ),
            )
            con.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('owlv2_last_path',?)",
                (str(last_path or path or ""),),
            )
        return len(rows)

    def owlv2_scan_progress(self, *, total: int) -> dict[str, Any]:
        """Count OWLv2 scans from live ovd_file_scans columns (no status required)."""
        total = max(0, int(total or 0))
        out = {
            "total": total, "completed": 0, "pending": total, "failed": 0, "retry": 0,
            "percent": 0.0, "last_file": "", "unsupported": 0,
        }
        if self.readonly and self._missing:
            return out
        with self._connect() as con:
            tables = {
                str(r[0])
                for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "ovd_file_scans" not in tables:
                logger.error("owlv2_scan_progress: ovd_file_scans table missing in %s", self.path)
                return out
            cols = {str(r[1]) for r in con.execute("PRAGMA table_info(ovd_file_scans)")}
            required = {"file_id", "model"}
            missing = required - cols
            if missing:
                logger.error(
                    "owlv2_scan_progress: ovd_file_scans missing columns %s in %s",
                    sorted(missing),
                    self.path,
                )
                raise sqlite3.OperationalError(
                    f"ovd_file_scans missing columns: {sorted(missing)}"
                )
            completed = int(
                con.execute(
                    """SELECT COUNT(*) FROM ovd_file_scans
                       WHERE instr(lower(IFNULL(model,'')), 'owlv2') > 0"""
                ).fetchone()[0]
            )
            failed = retry = unsupported = 0
            if "status" in cols:
                rows = con.execute(
                    """SELECT IFNULL(status,'done') AS status, COUNT(*) AS n
                       FROM ovd_file_scans
                       WHERE instr(lower(IFNULL(model,'')), 'owlv2') > 0
                       GROUP BY IFNULL(status,'done')"""
                ).fetchall()
                by = {str(r["status"] or "done"): int(r["n"] or 0) for r in rows}
                failed = int(by.get("failed", 0) or 0)
                unsupported = int(by.get("unsupported", 0) or 0)
                retry = int(by.get("retry", 0) or 0)
            last_file = ""
            if "last_path" in cols:
                row = con.execute(
                    """SELECT last_path FROM ovd_file_scans
                       WHERE instr(lower(IFNULL(model,'')), 'owlv2') > 0
                         AND IFNULL(last_path,'') != ''
                       ORDER BY scanned_at DESC, file_id DESC LIMIT 1"""
                ).fetchone()
                if row:
                    last_file = str(row["last_path"] or "")
            if not last_file:
                try:
                    meta = con.execute(
                        "SELECT value FROM meta WHERE key='owlv2_last_path'"
                    ).fetchone()
                except sqlite3.OperationalError:
                    meta = None
                last_file = str(meta["value"] if meta else "") or ""
        pending = max(0, total - completed)
        pct = (100.0 * completed / total) if total > 0 else 0.0
        out.update({
            "completed": completed,
            "pending": pending,
            "failed": failed + unsupported,
            "retry": retry,
            "unsupported": unsupported,
            "percent": round(pct, 1),
            "last_file": last_file,
        })
        return out

    def search_labels_and(self, label_groups: list, *, limit: int = 400) -> list[int]:
        if not label_groups:
            return []
        if self.readonly and self._missing:
            return []
        sets: list[set[int]] = []
        for group in label_groups:
            if isinstance(group, dict):
                labels = group.get("labels") or {group.get("label")}
                min_count = int(group.get("min_count") or 1)
            else:
                labels, min_count = group, 1
            ids = set(self.search_label(labels, min_count=min_count, limit=max(limit, 4000)))
            sets.append(ids)
        if not sets:
            return []
        common = set.intersection(*sets)
        return list(common)[: int(limit)]

    def all_embeddings(self) -> list[tuple[int, int, str, np.ndarray]]:
        if self.readonly and self._missing:
            return []
        out: list[tuple[int, int, str, np.ndarray]] = []
        with self._connect() as con:
            rows = con.execute(
                "SELECT file_id,instance_index,label,embedding FROM object_instances WHERE embedding IS NOT NULL"
            ).fetchall()
        for r in rows:
            vec = _vec(r["embedding"])
            if vec is None or vec.size == 0:
                continue
            out.append((int(r["file_id"]), int(r["instance_index"]), str(r["label"]), vec))
        return out

    def stats(self) -> dict[str, int]:
        if self.readonly and self._missing:
            return {"files": 0, "instances": 0}
        with self._connect() as con:
            files = int(con.execute("SELECT COUNT(*) FROM object_files").fetchone()[0])
            inst = int(con.execute("SELECT COUNT(*) FROM object_instances").fetchone()[0])
        return {"files": files, "instances": inst}
