"""Persistent, isolated Face Index for Vezir Pattern Search.

This uses a separate SQLite file. The existing pattern DB/FAISS indexes are
never modified by face indexing, so face work cannot corrupt the pattern index.
"""
from __future__ import annotations

import sqlite3
import json
import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.face_identity import FaceIdentityEngine, FaceObservation, _cosine

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS face_files(
  file_id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  mtime REAL DEFAULT 0,
  file_size INTEGER DEFAULT 0,
  source_hash TEXT DEFAULT '',
  status TEXT DEFAULT 'pending',
  face_count INTEGER DEFAULT 0,
  last_error TEXT DEFAULT '',
  scanned_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS persons(
  person_id TEXT PRIMARY KEY,
  display_name TEXT DEFAULT '',
  gender TEXT DEFAULT 'UNKNOWN',
  gender_confidence REAL DEFAULT 0,
  centroid BLOB,
  sample_count INTEGER DEFAULT 0,
  created_at TEXT DEFAULT '',
  updated_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS face_observations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id INTEGER NOT NULL,
  face_index INTEGER NOT NULL,
  person_id TEXT NOT NULL,
  bbox TEXT NOT NULL,
  gender TEXT DEFAULT 'UNKNOWN',
  gender_confidence REAL DEFAULT 0,
  embedding BLOB NOT NULL,
  embedding_dim INTEGER NOT NULL,
  created_at TEXT DEFAULT '',
  UNIQUE(file_id, face_index),
  FOREIGN KEY(file_id) REFERENCES face_files(file_id) ON DELETE CASCADE,
  FOREIGN KEY(person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS person_aliases(
  person_id TEXT NOT NULL,
  alias TEXT NOT NULL,
  alias_norm TEXT NOT NULL,
  source TEXT DEFAULT 'filename',
  created_at TEXT DEFAULT '',
  PRIMARY KEY(person_id, alias_norm),
  FOREIGN KEY(person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_person_alias_norm ON person_aliases(alias_norm);
CREATE INDEX IF NOT EXISTS idx_face_obs_person ON face_observations(person_id);
CREATE INDEX IF NOT EXISTS idx_face_obs_file ON face_observations(file_id);
CREATE INDEX IF NOT EXISTS idx_face_obs_gender ON face_observations(gender);
CREATE INDEX IF NOT EXISTS idx_face_files_status ON face_files(status);
CREATE TABLE IF NOT EXISTS face_gallery_members(
  person_id TEXT NOT NULL,
  file_id INTEGER NOT NULL,
  PRIMARY KEY(person_id,file_id),
  FOREIGN KEY(person_id) REFERENCES persons(person_id) ON DELETE CASCADE,
  FOREIGN KEY(file_id) REFERENCES face_files(file_id) ON DELETE CASCADE
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).reshape(-1).tobytes()


def _vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9ğüşöçıİĞÜŞÖÇ_ -]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


_GENERIC_NAME_TOKENS = {
    "image", "img", "photo", "picture", "foto", "whatsapp", "screenshot",
    "shutterstock", "png", "jpg", "jpeg", "webp", "woman", "man", "female",
    "male", "kadin", "erkek", "unknown", "copy", "final", "new", "old",
}


def _filename_alias(path: str) -> str:
    """Conservative filename metadata alias; never a pixel-based identity claim."""
    name = Path(str(path or "")).name
    for _ in range(3):
        name = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", name)
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\b(?:copy|kopya)\s*\d*\b", " ", name, flags=re.I)
    alias = re.sub(r"\s+", " ", name).strip()
    norm = _norm_text(alias)
    tokens = [t for t in norm.split() if t and t not in _GENERIC_NAME_TOKENS]
    if len(tokens) < 2:
        return ""
    alpha_tokens = [t for t in tokens if sum(c.isalpha() for c in t) >= 3]
    if len(alpha_tokens) < 2:
        return ""
    if any(len(t) > 24 and sum(c.isdigit() for c in t) > 5 for t in tokens):
        return ""
    return " ".join(alpha_tokens[:6]).strip()


class FaceIndexStore:
    def __init__(self, db_path: str | Path, *, threshold: float = 0.62, min_margin: float = 0.05, identity_engine: str = ""):

        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)
        self.identity_engine = str(identity_engine or "exemplar_v11")
        from core.index_freeze import in_search_session, process_search_active

        self._search_ro = bool(process_search_active() or in_search_session())
        if not self._search_ro:
            self._init()

    def _connect(self):
        if getattr(self, "_search_ro", False):
            if not Path(self.path).is_file():
                raise sqlite3.OperationalError("face index missing")
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

    def _init(self):
        with self._connect() as con:
            con.executescript(SCHEMA)
            con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version','4')")
            row = con.execute("SELECT value FROM meta WHERE key='identity_engine'").fetchone()
            old_engine = str(row[0] if row else "")
            if old_engine and old_engine != self.identity_engine:
                # Embedding dimensions/models are not safely mixable. Keep the
                # authoritative pattern DB untouched, but force a clean face
                # re-index when the backend changes (e.g. V9 ArcFace -> V10
                # local DINO/CLIP fallback).
                con.execute("DELETE FROM face_observations")
                con.execute("DELETE FROM face_gallery_members")
                con.execute("DELETE FROM persons")
                con.execute("UPDATE face_files SET status='pending', face_count=0, last_error='' ")
            con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('identity_engine',?)", (self.identity_engine,))

    def is_current(self, file_id: int, path: str, mtime: float, file_size: int) -> bool:
        with self._connect() as con:
            row = con.execute("SELECT path,mtime,file_size,status FROM face_files WHERE file_id=?", (int(file_id),)).fetchone()
        return bool(row and row["status"] == "done" and row["path"] == path and abs(float(row["mtime"]) - float(mtime)) < 0.001 and int(row["file_size"]) == int(file_size))

    def mark_scan(self, file_id: int, path: str, mtime: float, file_size: int, status: str, face_count: int = 0, error: str = ""):
        with self._connect() as con:
            con.execute("""INSERT INTO face_files(file_id,path,mtime,file_size,status,face_count,last_error,scanned_at)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(file_id) DO UPDATE SET path=excluded.path,mtime=excluded.mtime,file_size=excluded.file_size,status=excluded.status,face_count=excluded.face_count,last_error=excluded.last_error,scanned_at=excluded.scanned_at""",
                        (int(file_id), path, float(mtime), int(file_size), status, int(face_count), error[:1000], _now()))

    def _next_person_id(self, con) -> str:
        row = con.execute("SELECT person_id FROM persons ORDER BY CAST(SUBSTR(person_id,8) AS INTEGER) DESC LIMIT 1").fetchone()
        n = int(row[0][7:]) + 1 if row else 1
        return f"person_{n:04d}"

    def _gallery(self, con) -> dict[str, np.ndarray]:
        rows = con.execute("SELECT person_id,centroid FROM persons WHERE centroid IS NOT NULL AND sample_count>0").fetchall()
        return {str(r["person_id"]): _vec(r["centroid"]) for r in rows}

    def _match_person(self, con, emb: np.ndarray) -> tuple[str | None, float, float]:
        """Match against *face exemplars*, not only the moving centroid.

        Centroids are used only as a cheap candidate generator. The final
        identity decision is made from the best real face observation belonging
        to each candidate person. This is critical for old/young, side/front,
        black-and-white/colour and hairstyle changes.
        """
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.size == 0:
            return None, 0.0, 0.0
        # Broad candidate gate; the final decision remains stricter.
        candidate_gate = min(0.48, self.threshold)
        persons = con.execute(
            "SELECT person_id,centroid,gender,gender_confidence FROM persons WHERE centroid IS NOT NULL AND sample_count>0"
        ).fetchall()
        candidates: list[str] = []
        for row in persons:
            # A known gender conflict is a hard identity incompatibility.
            # Without this guard a high appearance similarity could merge two
            # different people into one gallery and later turn the person's
            # gender into UNKNOWN. UNKNOWN never blocks matching.
            row_gender = str(row["gender"] or "UNKNOWN").upper()
            obs_gender = str(getattr(self, "_current_observation_gender", "UNKNOWN") or "UNKNOWN").upper()
            if row_gender in {"FEMALE", "MALE"} and obs_gender in {"FEMALE", "MALE"} and row_gender != obs_gender:
                continue
            score = _cosine(emb, _vec(row["centroid"]))
            if score >= candidate_gate:
                candidates.append(str(row["person_id"]))
        if not candidates:
            return None, 0.0, 0.0

        marks = ",".join("?" for _ in candidates)
        rows = con.execute(
            f"SELECT person_id,embedding FROM face_observations WHERE person_id IN ({marks})",
            candidates,
        ).fetchall()
        best_by_person: dict[str, float] = {pid: -1.0 for pid in candidates}
        for row in rows:
            pid = str(row["person_id"])
            score = _cosine(emb, _vec(row["embedding"]))
            if score > best_by_person.get(pid, -1.0):
                best_by_person[pid] = score
        scores = sorted(((pid, score) for pid, score in best_by_person.items() if score >= 0),
                        key=lambda x: x[1], reverse=True)
        if not scores:
            return None, 0.0, 0.0
        pid, score = scores[0]
        second = scores[1][1] if len(scores) > 1 else 0.0
        margin = score - second
        # Strong match does not need a large margin; medium matches do.
        strong = max(self.threshold + 0.08, 0.62)
        accepted = score >= self.threshold and (score >= strong or len(scores) == 1 or margin >= self.min_margin)
        if not accepted:
            return None, score, margin
        return pid, score, margin

    def _assign_person(self, con, emb: np.ndarray, gender: str, gender_conf: float) -> str:
        # Pass the observation gender into the matcher without changing its
        # public API. This prevents an explicitly known FEMALE/MALE observation
        # from being merged into the opposite known gallery.
        self._current_observation_gender = str(gender or "UNKNOWN").upper()
        try:
            pid, _, _ = self._match_person(con, emb)
        finally:
            self._current_observation_gender = "UNKNOWN"
        now = _now()
        if pid is None:
            pid = self._next_person_id(con)
            con.execute(
                "INSERT INTO persons(person_id,gender,gender_confidence,centroid,sample_count,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (pid, gender if gender in {"FEMALE", "MALE"} else "UNKNOWN", float(gender_conf), _blob(emb), 1, now, now),
            )
            return pid
        row = con.execute(
            "SELECT centroid,sample_count,gender,gender_confidence FROM persons WHERE person_id=?",
            (pid,),
        ).fetchone()
        old = _vec(row["centroid"])
        count = int(row["sample_count"] or 0)
        centroid = (old * count + np.asarray(emb, dtype=np.float32)) / max(1, count + 1)
        oldg = str(row["gender"] or "UNKNOWN")
        newg = gender if gender in {"FEMALE", "MALE"} else oldg
        if oldg != "UNKNOWN" and gender in {"FEMALE", "MALE"} and oldg != gender:
            newg = "UNKNOWN"
        gconf = max(float(row["gender_confidence"] or 0), float(gender_conf or 0))
        con.execute(
            "UPDATE persons SET centroid=?,sample_count=?,gender=?,gender_confidence=?,updated_at=? WHERE person_id=?",
            (_blob(centroid), count + 1, newg, gconf, now, pid),
        )
        return pid

    def _recompute_person(self, con, person_id: str):
        rows = con.execute("SELECT embedding,gender,gender_confidence FROM face_observations WHERE person_id=?", (person_id,)).fetchall()
        if not rows:
            row = con.execute("SELECT display_name FROM persons WHERE person_id=?", (person_id,)).fetchone()
            if row and not str(row[0] or '').strip():
                con.execute("DELETE FROM persons WHERE person_id=?", (person_id,))
            else:
                con.execute("UPDATE persons SET centroid=NULL,sample_count=0,gender='UNKNOWN',gender_confidence=0,updated_at=? WHERE person_id=?", (_now(),person_id))
            return
        vecs=[_vec(r['embedding']) for r in rows]
        centroid=np.mean(np.stack(vecs),axis=0).astype(np.float32)
        genders=[str(r['gender'] or 'UNKNOWN') for r in rows if str(r['gender'] or 'UNKNOWN') in {'FEMALE','MALE'}]
        gender='UNKNOWN'
        if genders:
            f=genders.count('FEMALE'); m=genders.count('MALE')
            if f>m: gender='FEMALE'
            elif m>f: gender='MALE'
        gconf=max(float(r['gender_confidence'] or 0) for r in rows)
        con.execute("UPDATE persons SET centroid=?,sample_count=?,gender=?,gender_confidence=?,updated_at=? WHERE person_id=?", (_blob(centroid),len(rows),gender,gconf,_now(),person_id))

    def replace_file_faces(self, file_id: int, path: str, mtime: float, file_size: int, observations: list[FaceObservation]):
        with self._connect() as con:
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("INSERT INTO face_files(file_id,path,mtime,file_size,status,face_count,last_error,scanned_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(file_id) DO UPDATE SET path=excluded.path,mtime=excluded.mtime,file_size=excluded.file_size,status=excluded.status,last_error=excluded.last_error,scanned_at=excluded.scanned_at",
                        (int(file_id),path,float(mtime),int(file_size),'processing',0,'',_now()))
            old_persons=[r[0] for r in con.execute("SELECT DISTINCT person_id FROM face_observations WHERE file_id=?", (int(file_id),)).fetchall()]
            con.execute("DELETE FROM face_observations WHERE file_id=?", (int(file_id),))
            con.execute("DELETE FROM face_gallery_members WHERE file_id=?", (int(file_id),))
            for old_pid in old_persons:
                self._recompute_person(con, old_pid)
            assigned=[]
            # Filename aliases are metadata hints, not face identity. In a
            # multi-person image a single filename must never label every face
            # with the same person name. Only infer a filename alias when the
            # image contains exactly one face; user-created aliases remain
            # authoritative.
            alias = _filename_alias(path) if len(observations) == 1 else ""
            for idx, obs in enumerate(observations, start=1):
                if obs.embedding is None:
                    continue
                pid = self._assign_person(con, obs.embedding, obs.gender, obs.gender_confidence)
                con.execute("INSERT INTO face_observations(file_id,face_index,person_id,bbox,gender,gender_confidence,embedding,embedding_dim,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                            (int(file_id), idx, pid, json.dumps(list(obs.bbox)), obs.gender, float(obs.gender_confidence), _blob(obs.embedding), int(np.asarray(obs.embedding).size), _now()))
                con.execute("INSERT OR IGNORE INTO face_gallery_members(person_id,file_id) VALUES(?,?)", (pid,int(file_id)))
                if alias:
                    self._add_alias(con, pid, alias, source="filename")
                assigned.append(pid)
            con.execute("INSERT INTO face_files(file_id,path,mtime,file_size,status,face_count,last_error,scanned_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(file_id) DO UPDATE SET path=excluded.path,mtime=excluded.mtime,file_size=excluded.file_size,status='done',face_count=excluded.face_count,last_error='',scanned_at=excluded.scanned_at",
                        (int(file_id),path,float(mtime),int(file_size),'done',len(assigned),'',_now()))
            return assigned

    def mark_error(self, file_id:int,path:str,mtime:float,file_size:int,error:str):
        self.mark_scan(file_id,path,mtime,file_size,"error",0,error)

    def delete_file(self, file_id:int):
        with self._connect() as con:
            con.execute("DELETE FROM face_files WHERE file_id=?", (int(file_id),))
            # Empty persons are removed; named persons with no members are retained only if explicitly named.
            con.execute("DELETE FROM persons WHERE person_id NOT IN (SELECT DISTINCT person_id FROM face_observations) AND COALESCE(display_name,'')=''")

    def reconcile_missing(self, valid_file_ids: set[int]) -> int:
        with self._connect() as con:
            rows = con.execute("SELECT file_id FROM face_files").fetchall()
            removed = 0
            for row in rows:
                fid = int(row[0])
                if fid not in valid_file_ids:
                    con.execute("DELETE FROM face_files WHERE file_id=?", (fid,))
                    removed += 1
            con.execute("DELETE FROM persons WHERE person_id NOT IN (SELECT DISTINCT person_id FROM face_observations) AND COALESCE(display_name,'')=''")
            return removed

    def list_persons(self, limit:int=10000) -> list[dict[str,Any]]:
        with self._connect() as con:
            rows=con.execute("SELECT person_id,display_name,gender,gender_confidence,sample_count,created_at,updated_at FROM persons ORDER BY CAST(SUBSTR(person_id,8) AS INTEGER) LIMIT ?",(int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def _add_alias(self, con, person_id: str, alias: str, source: str = "filename") -> None:
        alias = str(alias or "").strip()
        alias_norm = _norm_text(alias)
        if not alias_norm:
            return
        con.execute(
            "INSERT OR IGNORE INTO person_aliases(person_id,alias,alias_norm,source,created_at) VALUES(?,?,?,?,?)",
            (str(person_id), alias[:200], alias_norm[:200], str(source or "filename"), _now()),
        )

    def label_person(self, person_id: str, display_name: str) -> None:
        """Explicit user label for an appearance cluster."""
        name = str(display_name or "").strip()[:200]
        with self._connect() as con:
            con.execute(
                "UPDATE persons SET display_name=?,updated_at=? WHERE person_id=?",
                (name, _now(), str(person_id)),
            )
            if name:
                self._add_alias(con, str(person_id), name, source="user")

    def rename_person(self, person_id:str, display_name:str):
        name = str(display_name).strip()[:200]
        with self._connect() as con:
            con.execute("UPDATE persons SET display_name=?,updated_at=? WHERE person_id=?", (name,_now(),person_id))
            if name:
                self._add_alias(con, str(person_id), name, source="user")

    def person(self, person_id:str) -> dict[str,Any] | None:
        with self._connect() as con:
            r=con.execute("SELECT * FROM persons WHERE person_id=?",(person_id,)).fetchone()
            return dict(r) if r else None

    def consolidate_identities(self, *, merge_threshold: float = 0.70, max_pairs: int = 2000) -> int:
        """Merge clearly duplicated person clusters conservatively."""
        merged = 0
        checked = 0
        while checked < int(max_pairs):
            with self._connect() as con:
                rows = con.execute(
                    "SELECT person_id,display_name,centroid,sample_count FROM persons WHERE centroid IS NOT NULL AND sample_count>0 ORDER BY sample_count DESC"
                ).fetchall()
                people = [
                    (str(r["person_id"]), str(r["display_name"] or "").strip(), _vec(r["centroid"]), int(r["sample_count"]))
                    for r in rows
                ]
                found = None
                for i in range(len(people)):
                    for j in range(i + 1, len(people)):
                        checked += 1
                        if checked > int(max_pairs):
                            break
                        pid_a, name_a, ca, count_a = people[i]
                        pid_b, name_b, cb, count_b = people[j]
                        if name_a and name_b:
                            continue
                        if _cosine(ca, cb) < float(merge_threshold):
                            continue
                        survivor, victim = (pid_a, pid_b) if count_a >= count_b else (pid_b, pid_a)
                        found = (survivor, victim)
                        break
                    if found or checked >= int(max_pairs):
                        break
                if not found:
                    return merged
                survivor, victim = found
                con.execute("UPDATE face_observations SET person_id=? WHERE person_id=?", (survivor, victim))
                # Ignore duplicate membership rows created by the merge.
                con.execute("INSERT OR IGNORE INTO face_gallery_members(person_id,file_id) SELECT ?,file_id FROM face_gallery_members WHERE person_id=?", (survivor, victim))
                con.execute("DELETE FROM face_gallery_members WHERE person_id=?", (victim,))
                con.execute("DELETE FROM persons WHERE person_id=?", (victim,))
                self._recompute_person(con, survivor)
                merged += 1
        return merged

    def search_by_embeddings(self, embeddings: Iterable[np.ndarray], *, threshold: float | None = None, limit: int = 2000, top_persons: int | None = None, min_candidate_threshold: float = 0.44) -> dict[int, dict[str, Any]]:
        """Retrieve persistent people with exemplar-aware face matching.

        The old implementation compared only one centroid and then truncated to
        50 people. That caused real same-person photos (different angle, age,
        lighting or hairstyle) to disappear. V8 uses the centroid as a cheap
        candidate generator, then rescoring with the best stored face exemplar
        for each candidate person. No arbitrary 50-person ceiling is applied.
        """
        qs = [np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings if e is not None]
        qs = [q for q in qs if q.size > 0 and float(np.linalg.norm(q)) > 1e-8]
        if not qs:
            return {}
        th = self.threshold if threshold is None else float(threshold)
        candidate_th = min(0.40, float(min_candidate_threshold), th)
        with self._connect() as con:
            persons = con.execute(
                "SELECT person_id,centroid,sample_count FROM persons WHERE centroid IS NOT NULL AND sample_count>0"
            ).fetchall()
            centroid_scores: list[tuple[str, float]] = []
            for row in persons:
                centroid = _vec(row["centroid"])
                best = max((_cosine(q, centroid) for q in qs), default=0.0)
                if best >= candidate_th:
                    centroid_scores.append((str(row["person_id"]), float(best)))
            centroid_scores.sort(key=lambda x: x[1], reverse=True)
            # Candidate cap is only a performance guard for enormous galleries;
            # it is intentionally much larger than the old hard 50-person limit.
            if top_persons is not None and int(top_persons) > 0:
                centroid_scores = centroid_scores[:int(top_persons)]
            else:
                centroid_scores = centroid_scores[:5000]
            if not centroid_scores:
                return {}

            pids = [p for p, _ in centroid_scores]
            marks = ",".join("?" for _ in pids)
            rows = con.execute(
                f"SELECT person_id,file_id,embedding FROM face_observations WHERE person_id IN ({marks})",
                pids,
            ).fetchall()

        # Rescore against real face exemplars. This fixes centroid drift and
        # allows one person to have many appearances without collapsing them.
        best_person: dict[str, float] = {p: score for p, score in centroid_scores}
        best_file: dict[tuple[str, int], float] = {}
        for row in rows:
            pid = str(row["person_id"])
            emb = _vec(row["embedding"])
            score = max((_cosine(q, emb) for q in qs), default=0.0)
            if score > best_person.get(pid, -1.0):
                best_person[pid] = float(score)
            key = (pid, int(row["file_id"]))
            best_file[key] = max(best_file.get(key, -1.0), float(score))

        ranked_people = [(pid, score) for pid, score in best_person.items() if score >= th]
        ranked_people.sort(key=lambda x: x[1], reverse=True)
        if top_persons is not None and int(top_persons) > 0:
            ranked_people = ranked_people[:int(top_persons)]
        if not ranked_people:
            return {}
        allowed = {pid: score for pid, score in ranked_people}

        # Expand each matched person to every file in the persistent gallery.
        with self._connect() as con:
            marks = ",".join("?" for _ in allowed)
            members = con.execute(
                f"SELECT DISTINCT person_id,file_id FROM face_gallery_members WHERE person_id IN ({marks})",
                list(allowed),
            ).fetchall()
        out: dict[int, dict[str, Any]] = {}
        for row in members:
            fid = int(row["file_id"])
            pid = str(row["person_id"])
            score = float(allowed.get(pid, 0.0))
            # Per-file exemplar score is retained as extra evidence but the
            # person score controls ranking so every photo of that person stays
            # together even when that particular photo is low quality.
            exemplar = float(best_file.get((pid, fid), 0.0))
            current = out.get(fid)
            payload = {
                "person_id": pid,
                "similarity": round(score, 5),
                "exemplar_similarity": round(exemplar, 5),
                "face_match_type": "same_person" if score >= max(th, 0.60) else "probable_person",
            }
            if current is None or score > float(current["similarity"]):
                out[fid] = payload
        ordered = sorted(out.items(), key=lambda kv: (-float(kv[1]["similarity"]), int(kv[0])))
        return dict(ordered[:max(1, int(limit))])

    def search_by_name(self, display_name:str, limit:int=400) -> list[int]:
        name = str(display_name or '').strip()
        norm = _norm_text(name)
        if not norm:
            return []
        like = f"%{norm}%"
        with self._connect() as con:
            rows = con.execute(
                """SELECT DISTINCT o.file_id
                   FROM face_observations o
                   JOIN persons p ON p.person_id=o.person_id
                   LEFT JOIN person_aliases a ON a.person_id=p.person_id
                   WHERE (a.alias_norm=? OR a.alias_norm LIKE ? OR lower(p.display_name)=lower(?))
                   ORDER BY o.file_id LIMIT ?""",
                (norm, like, name, int(limit)),
            ).fetchall()
        return [int(r[0]) for r in rows]

    def search(self, *, person_id:str="", gender:str="", limit:int=400) -> list[int]:
        clauses=[]; params=[]
        if person_id:
            clauses.append("o.person_id=?"); params.append(person_id)
        if gender:
            clauses.append("o.gender=?"); params.append(gender.upper())
        where = " AND ".join(clauses) or "1=1"
        with self._connect() as con:
            rows=con.execute(f"SELECT DISTINCT o.file_id FROM face_observations o WHERE {where} ORDER BY o.file_id LIMIT ?", (*params,int(limit))).fetchall()
        return [int(r[0]) for r in rows]

    def stats(self) -> dict[str,int]:
        with self._connect() as con:
            return {
                "files": con.execute("SELECT COUNT(*) FROM face_files WHERE status='done'").fetchone()[0],
                "faces": con.execute("SELECT COUNT(*) FROM face_observations").fetchone()[0],
                "persons": con.execute("SELECT COUNT(*) FROM persons").fetchone()[0],
                "errors": con.execute("SELECT COUNT(*) FROM face_files WHERE status='error'").fetchone()[0],
            }
