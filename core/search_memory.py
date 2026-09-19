"""Kalıcı metin arama hafızası.

Aynı sorgu daha önce çalıştırıldıysa ilk ekranı uygulama yeniden açıldıktan sonra
bile anında göstermek için kullanılır. Bu katman 'öğrenilmiş sonuç' ile canlı
arama arasında bir köprü kurar: kayıt anında gösterilir, ardından canlı arama
arka planda çalışarak hafızayı günceller.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
import zlib
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from core.search_models import SearchResponse, SearchStats

try:
    from core.search_engine import SearchResult
except Exception:  # pragma: no cover - import sıralaması için
    SearchResult = None  # type: ignore


class PersistentSearchMemory:
    def __init__(self, db_path: str, *, max_entries: int = 512, max_results: int = 300):
        self.path = str(Path(db_path).with_name("search_memory.db"))
        self.max_entries = max(32, int(max_entries))
        self.max_results = max(1, int(max_results))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure()

    def _connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _ensure(self):
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS search_memory (
                    cache_key TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    payload BLOB NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_search_memory_updated ON search_memory(updated_at)")
            con.execute("""
                CREATE TABLE IF NOT EXISTS query_rewrites (
                    src TEXT PRIMARY KEY,
                    dst TEXT NOT NULL,
                    kind TEXT DEFAULT 'typo',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS brand_memory (
                    alias TEXT PRIMARY KEY,
                    canonical TEXT NOT NULL,
                    created_at TEXT DEFAULT '',
                    updated_at TEXT DEFAULT ''
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS taught_file_brands (
                    file_id INTEGER NOT NULL,
                    brand_key TEXT NOT NULL,
                    display TEXT NOT NULL,
                    created_at TEXT DEFAULT '',
                    updated_at TEXT DEFAULT '',
                    PRIMARY KEY (file_id, brand_key)
                )
            """)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_taught_file_brands_key "
                "ON taught_file_brands(brand_key)"
            )
            con.execute("""
                CREATE TABLE IF NOT EXISTS ranking_overlay (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_key TEXT NOT NULL,
                    file_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    label TEXT DEFAULT '',
                    delta REAL DEFAULT 0,
                    extra TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_ranking_overlay_query ON ranking_overlay(query_key)"
            )
            con.execute("""
                CREATE TABLE IF NOT EXISTS memory_undo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS ocr_cache (
                    cache_key TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    confidence REAL DEFAULT 0,
                    languages TEXT DEFAULT 'tr,en',
                    brands TEXT DEFAULT '[]',
                    updated_at REAL NOT NULL
                )
            """)

    @staticmethod
    def key_hash(key: tuple[Any, ...]) -> str:
        raw = json.dumps(key, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): PersistentSearchMemory._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [PersistentSearchMemory._jsonable(v) for v in value]
        return str(value)

    def _pack_response(self, response: SearchResponse) -> bytes:
        results = list(response.all_results or [])[: self.max_results]
        result_dicts = [self._jsonable(asdict(r)) for r in results]
        visible_ids = {int(getattr(r, "file_id", 0)) for r in (response.results or [])}
        payload = {
            "results": result_dicts,
            "visible_ids": list(visible_ids),
            "stats": self._jsonable(asdict(response.stats)),
            "below": self._jsonable(list(response.below_threshold_preview or [])[:20]),
            "meta": self._jsonable(response.meta or {}),
        }
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return zlib.compress(raw, level=6)

    @staticmethod
    def _restore_result(data: dict[str, Any]):
        if SearchResult is None:
            from core.search_engine import SearchResult as _SearchResult
            cls = _SearchResult
        else:
            cls = SearchResult
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def _unpack_response(self, blob: bytes) -> SearchResponse | None:
        try:
            payload = json.loads(zlib.decompress(blob).decode("utf-8"))
            results = [self._restore_result(x) for x in payload.get("results", [])]
            visible_ids = {int(x) for x in payload.get("visible_ids", [])}
            visible = [r for r in results if int(r.file_id) in visible_ids]
            stats_data = payload.get("stats") or {}
            stats_allowed = {f.name for f in fields(SearchStats)}
            stats = SearchStats(**{k: v for k, v in stats_data.items() if k in stats_allowed})
            response = SearchResponse(
                all_results=results,
                results=visible,
                stats=stats,
                below_threshold_preview=payload.get("below", []),
                meta=dict(payload.get("meta") or {}),
            )
            return response
        except Exception:
            return None

    def get(self, key: tuple[Any, ...]) -> SearchResponse | None:
        h = self.key_hash(key)
        with self._connect() as con:
            row = con.execute(
                "SELECT payload FROM search_memory WHERE cache_key=?", (h,)
            ).fetchone()
            if not row:
                return None
            con.execute("UPDATE search_memory SET updated_at=? WHERE cache_key=?", (time.time(), h))
        response = self._unpack_response(row[0])
        if response is not None:
            response = copy.deepcopy(response)
            response.meta = {
                **(response.meta or {}),
                "search_memory_hit": True,
                "search_memory_stale_while_revalidate": True,
            }
        return response

    def put(self, key: tuple[Any, ...], response: SearchResponse) -> None:
        h = self.key_hash(key)
        query_text = str(response.meta.get("text") or response.meta.get("text_query") or "")
        payload = self._pack_response(response)
        now = time.time()
        with self._connect() as con:
            con.execute(
                """INSERT INTO search_memory(cache_key,query_text,created_at,updated_at,payload)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                     query_text=excluded.query_text,
                     updated_at=excluded.updated_at,
                     payload=excluded.payload""",
                (h, query_text, now, now, payload),
            )
            # LRU: keep the most recently used queries, not only the last search.
            con.execute(
                """DELETE FROM search_memory
                   WHERE cache_key IN (
                       SELECT cache_key FROM search_memory
                       ORDER BY updated_at DESC
                       LIMIT -1 OFFSET ?
                   )""",
                (self.max_entries,),
            )

    def clear(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM search_memory")


_memory_by_db: dict[str, PersistentSearchMemory] = {}


def get_search_memory(db_path: str) -> PersistentSearchMemory:
    key = str(Path(db_path).resolve())
    mem = _memory_by_db.get(key)
    if mem is None:
        mem = PersistentSearchMemory(key)
        _memory_by_db[key] = mem
    return mem


def memory_db_path(db_path: str) -> str:
    return str(Path(db_path).with_name("search_memory.db"))


def _norm_query(text: str) -> str:
    """Safe TR fold: çilek==cilek, çilek!=çiçek. Not a semantic synonym map."""
    try:
        from core.textile_terms import normalize_turkish

        folded = normalize_turkish(text)
    except Exception:
        folded = str(text or "").strip().lower()
    return " ".join(folded.split())


def _boost_for_action(action: str) -> float:
    a = str(action or "").strip().lower()
    if a in {"same_pattern", "promote", "correct", "ai_category_correct", "dogru", "doğru"}:
        return 0.15
    if a in {"similar_texture"}:
        return 0.08
    if a in {"wrong", "yanlis", "yanlış"}:
        return -0.55
    if a in {"demote", "not_similar"}:
        return -0.35
    return 0.0


def apply_query_memory(db_path: str, text: str) -> tuple[str, dict[str, Any]]:
    """Search-time rewrite from search_memory.db (typo/synonym/brand)."""
    raw = str(text or "").strip()
    meta: dict[str, Any] = {"memory_rewrite": "", "memory_brand": ""}
    if not raw or not db_path:
        return raw, meta
    src = _norm_query(raw)
    mem = get_search_memory(db_path)
    dst = ""
    with mem._connect() as con:
        row = con.execute(
            "SELECT dst FROM query_rewrites WHERE src=?", (src,)
        ).fetchone()
        if row:
            dst = str(row[0] or "").strip()
    if dst and dst.lower() != src:
        meta["memory_rewrite"] = dst
        raw = dst
    try:
        from core.brand_aliases import resolve_brand_alias

        brand = resolve_brand_alias(raw, db_path) or ""
        if brand:
            meta["memory_brand"] = brand
    except Exception:
        pass
    return raw, meta


def remember_query_rewrite(
    db_path: str, src: str, dst: str, *, kind: str = "typo"
) -> bool:
    a = _norm_query(src)
    b = " ".join(str(dst or "").strip().split())
    if not db_path or not a or not b or a == _norm_query(b):
        return False
    mem = get_search_memory(db_path)
    now = time.time()
    with mem._connect() as con:
        prev = con.execute(
            "SELECT dst FROM query_rewrites WHERE src=?", (a,)
        ).fetchone()
        _push_undo(
            con,
            "query_rewrite",
            {"src": a, "prev_dst": (prev[0] if prev else None), "dst": b, "kind": kind},
        )
        con.execute(
            """INSERT INTO query_rewrites(src,dst,kind,created_at,updated_at)
               VALUES(?,?,?,?,?)
               ON CONFLICT(src) DO UPDATE SET
                 dst=excluded.dst, kind=excluded.kind, updated_at=excluded.updated_at""",
            (a, b, kind, now, now),
        )
    return True


def learn_concept_token(
    db_path: str,
    token: str,
    concept_type: str,
    *,
    parent: str = "",
    aliases: tuple | list = (),
) -> int:
    """Persist a user-taught concept token in search_memory.db (INDEX_FROZEN)."""
    from core.concept_registry import learn

    return int(
        learn(
            db_path,
            token,
            concept_type=concept_type,
            parent=parent,
            aliases=aliases,
        )
        or 0
    )


def load_brand_memory(db_path: str) -> dict[str, str]:
    if not db_path:
        return {}
    out: dict[str, str] = {}
    mem_path = memory_db_path(db_path)
    for path in (mem_path, str(db_path)):
        p = Path(path)
        if not p.is_file():
            continue
        try:
            uri = f"file:{p.as_posix()}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=5)
            try:
                rows = con.execute(
                    "SELECT alias, canonical FROM brand_memory"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            finally:
                con.close()
            for alias, canonical in rows:
                if alias and canonical and str(alias) not in out:
                    out[str(alias)] = str(canonical)
        except Exception:
            continue
    return out


def register_brand_memory(db_path: str, alias: str, canonical: str) -> bool:
    from core.brand_aliases import canonical_brand_display, normalize_brand_key

    a = normalize_brand_key(alias)
    c = canonical_brand_display(canonical) or " ".join(str(canonical or "").strip().split())
    if not db_path or not a or not c:
        return False
    mem = get_search_memory(db_path)
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    with mem._connect() as con:
        prev = con.execute(
            "SELECT canonical FROM brand_memory WHERE alias=?", (a,)
        ).fetchone()
        _push_undo(
            con,
            "brand_memory",
            {"alias": a, "prev": (prev[0] if prev else None), "canonical": c},
        )
        con.execute(
            """INSERT INTO brand_memory(alias, canonical, created_at, updated_at)
               VALUES(?,?,?,?)
               ON CONFLICT(alias) DO UPDATE SET
                 canonical=excluded.canonical, updated_at=excluded.updated_at""",
            (a, c, now, now),
        )
    return True


def _brand_from_overlay_payload(label: str, extra_raw: Any) -> str:
    extra: dict[str, Any] = {}
    if isinstance(extra_raw, dict):
        extra = extra_raw
    else:
        try:
            parsed = json.loads(extra_raw or "{}")
            if isinstance(parsed, dict):
                extra = parsed
        except Exception:
            extra = {}
    path = str(extra.get("category_path") or label or "").strip()
    parent = str(extra.get("parent") or "").strip()
    child = str(extra.get("child") or extra.get("brand") or "").strip()
    if path.lower().startswith("marka/"):
        return path.split("/", 1)[1].strip()
    if parent.casefold() == "marka" and child:
        return child
    brand = str(extra.get("brand") or "").strip()
    if brand:
        return brand
    return ""


def teach_file_brand(db_path: str, file_id: int, brand: str) -> bool:
    """Öğret → search_memory.db. file_id + canonical brand unique. Index yok."""
    from core.brand_aliases import canonical_brand_display, normalize_brand_key

    fid = int(file_id or 0)
    display = canonical_brand_display(brand, db_path)
    key = normalize_brand_key(display or brand)
    if not db_path or fid <= 0 or not key:
        return False
    from core.brand_aliases import register_brand_alias

    register_brand_alias(db_path, key, display)
    try:
        from core.category_memory import register_category

        register_category(db_path, "Marka", display)
    except Exception:
        pass
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    mem = get_search_memory(db_path)
    with mem._connect() as con:
        con.execute(
            """INSERT INTO taught_file_brands(file_id, brand_key, display, created_at, updated_at)
               VALUES(?,?,?,?,?)
               ON CONFLICT(file_id, brand_key) DO UPDATE SET
                 display=excluded.display, updated_at=excluded.updated_at""",
            (fid, key, display, now, now),
        )
    return True


def taught_brand_keys_for_file(db_path: str, file_id: int) -> set[str]:
    fid = int(file_id or 0)
    if not db_path or fid <= 0:
        return set()
    out: set[str] = set()
    from core.brand_aliases import normalize_brand_key

    try:
        mem = get_search_memory(db_path)
        with mem._connect() as con:
            try:
                rows = con.execute(
                    "SELECT brand_key FROM taught_file_brands WHERE file_id=?",
                    (fid,),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for (bk,) in rows:
                k = normalize_brand_key(str(bk or ""))
                if k:
                    out.add(k)
            try:
                ov = con.execute(
                    """SELECT label, extra FROM ranking_overlay
                       WHERE file_id=? AND lower(action) IN
                       ('teach','metadata_edit','duzenle','düzenle','edit')""",
                    (fid,),
                ).fetchall()
            except sqlite3.OperationalError:
                ov = []
            for label, extra in ov:
                raw = _brand_from_overlay_payload(str(label or ""), extra)
                k = normalize_brand_key(raw)
                if k:
                    out.add(k)
    except Exception:
        return out
    return out


def file_ids_for_taught_brand(db_path: str, brand: str) -> list[int]:
    """Search reads the same canonical key teach writes."""
    from core.brand_aliases import canonical_brand_display, normalize_brand_key

    key = normalize_brand_key(canonical_brand_display(brand, db_path) or brand)
    if not db_path or not key:
        return []
    seen: set[int] = set()
    try:
        mem = get_search_memory(db_path)
        with mem._connect() as con:
            try:
                rows = con.execute(
                    "SELECT file_id FROM taught_file_brands WHERE brand_key=?",
                    (key,),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for (fid,) in rows:
                i = int(fid or 0)
                if i > 0:
                    seen.add(i)
            try:
                ov = con.execute(
                    """SELECT file_id, label, extra FROM ranking_overlay
                       WHERE lower(action) IN
                       ('teach','metadata_edit','duzenle','düzenle','edit')"""
                ).fetchall()
            except sqlite3.OperationalError:
                ov = []
            for fid, label, extra in ov:
                raw = _brand_from_overlay_payload(str(label or ""), extra)
                if normalize_brand_key(raw) == key:
                    i = int(fid or 0)
                    if i > 0:
                        seen.add(i)
    except Exception:
        return sorted(seen)
    return sorted(seen)


def record_feedback_overlay(
    db_path: str,
    query_key: str,
    file_id: int,
    action: str,
    *,
    label: str = "",
    extra: dict[str, Any] | None = None,
) -> int:
    if not db_path or int(file_id or 0) <= 0:
        return 0
    q = _norm_query(query_key)
    if not q:
        return 0
    delta = _boost_for_action(action)
    mem = get_search_memory(db_path)
    payload = json.dumps(extra or {}, ensure_ascii=False)
    with mem._connect() as con:
        _push_undo(
            con,
            "ranking_overlay",
            {
                "query_key": q,
                "file_id": int(file_id),
                "action": action,
                "label": label,
                "delta": delta,
            },
        )
        existing = con.execute(
            """SELECT id FROM ranking_overlay
               WHERE query_key=? AND file_id=?
               ORDER BY id DESC LIMIT 1""",
            (q, int(file_id)),
        ).fetchone()
        if existing:
            con.execute(
                """UPDATE ranking_overlay
                   SET action=?, label=?, delta=?, extra=?, created_at=?
                   WHERE id=?""",
                (
                    str(action or ""),
                    str(label or ""),
                    float(delta),
                    payload,
                    time.time(),
                    int(existing[0]),
                ),
            )
            con.execute(
                "DELETE FROM ranking_overlay WHERE query_key=? AND file_id=? AND id!=?",
                (q, int(file_id), int(existing[0])),
            )
            return int(existing[0])
        cur = con.execute(
            """INSERT INTO ranking_overlay(query_key,file_id,action,label,delta,extra,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (q, int(file_id), str(action or ""), str(label or ""), float(delta), payload, time.time()),
        )
        return int(cur.lastrowid or 0)


def overlay_adjustments(db_path: str, query_key: str) -> dict[int, float]:
    q = _norm_query(query_key)
    if not db_path or not q:
        return {}
    mem = get_search_memory(db_path)
    deltas: dict[int, float] = {}
    with mem._connect() as con:
        rows = con.execute(
            "SELECT file_id, delta FROM ranking_overlay WHERE query_key=?",
            (q,),
        ).fetchall()
    for fid, delta in rows:
        fid_i = int(fid)
        deltas[fid_i] = deltas.get(fid_i, 0.0) + float(delta or 0.0)
    return {k: max(-0.70, min(0.20, v)) for k, v in deltas.items()}


def clear_overlay_actions(db_path: str, file_id: int, actions: tuple[str, ...]) -> int:
    """Remove ranking_overlay rows for one file. Search ranking is not rebuilt."""
    if not db_path or int(file_id or 0) <= 0 or not actions:
        return 0
    mem = get_search_memory(db_path)
    lowers = tuple(str(a).strip().lower() for a in actions if str(a).strip())
    if not lowers:
        return 0
    ph = ",".join("?" * len(lowers))
    with mem._connect() as con:
        cur = con.execute(
            f"DELETE FROM ranking_overlay WHERE file_id=? AND lower(action) IN ({ph})",
            (int(file_id), *lowers),
        )
        return int(cur.rowcount or 0)


def overlay_wrong_ids(db_path: str, query_key: str) -> set[int]:
    q = _norm_query(query_key)
    if not db_path or not q:
        return set()
    mem = get_search_memory(db_path)
    with mem._connect() as con:
        rows = con.execute(
            """SELECT file_id FROM ranking_overlay
               WHERE query_key=? AND lower(action) IN ('wrong','yanlis','yanlış')""",
            (q,),
        ).fetchall()
    return {int(r[0]) for r in rows}


def overlay_positive_ids(db_path: str, query_key: str) -> set[int]:
    q = _norm_query(query_key)
    if not db_path or not q:
        return set()
    mem = get_search_memory(db_path)
    with mem._connect() as con:
        rows = con.execute(
            """SELECT file_id FROM ranking_overlay
               WHERE query_key=? AND lower(action) IN (
                   'correct','promote','same_pattern','ai_category_correct',
                   'dogru','doğru'
               )""",
            (q,),
        ).fetchall()
    return {int(r[0]) for r in rows}


def overlay_custom_tags(db_path: str, file_id: int) -> list[str]:
    if not db_path or int(file_id or 0) <= 0:
        return []
    mem = get_search_memory(db_path)
    with mem._connect() as con:
        rows = con.execute(
            """SELECT label FROM ranking_overlay
               WHERE file_id=? AND lower(action) IN ('custom_tag','öğret','ogret','teach')
               ORDER BY id DESC""",
            (int(file_id),),
        ).fetchall()
    tags: list[str] = []
    seen: set[str] = set()
    for (label,) in rows:
        t = str(label or "").strip()
        if t and t not in seen:
            seen.add(t)
            tags.append(t)
    return tags


def overlay_metadata(db_path: str, file_id: int) -> dict[str, Any]:
    if not db_path or int(file_id or 0) <= 0:
        return {}
    mem = get_search_memory(db_path)
    with mem._connect() as con:
        row = con.execute(
            """SELECT extra, label FROM ranking_overlay
               WHERE file_id=? AND (
                   lower(action) IN ('metadata_edit','duzenle','düzenle','edit')
                   OR extra LIKE '%"source": "metadata_edit"%'
               )
               ORDER BY id DESC LIMIT 1""",
            (int(file_id),),
        ).fetchone()
    if not row:
        return {}
    raw = row[0] or row[1] or "{}"
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def record_category_overlay(
    db_path: str,
    file_id: int,
    category_path: str,
    *,
    extra: dict[str, Any] | None = None,
) -> int:
    payload = {"category_path": category_path, **(extra or {})}
    return record_feedback_overlay(
        db_path,
        category_path or "",
        file_id,
        "teach",
        label=category_path,
        extra=payload,
    )


def _push_undo(con: sqlite3.Connection, kind: str, payload: dict[str, Any]) -> None:
    con.execute(
        "INSERT INTO memory_undo(kind, payload, created_at) VALUES(?,?,?)",
        (kind, json.dumps(payload, ensure_ascii=False), time.time()),
    )


def undo_last(db_path: str) -> dict[str, Any]:
    """Restore previous search-memory state. Does not touch Pattern Index."""
    if not db_path:
        return {"undone": False, "kind": ""}
    mem = get_search_memory(db_path)
    with mem._connect() as con:
        row = con.execute(
            "SELECT id, kind, payload FROM memory_undo ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return {"undone": False, "kind": ""}
        uid, kind, raw = int(row[0]), str(row[1]), row[2]
        try:
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        if kind == "query_rewrite":
            src = payload.get("src")
            prev = payload.get("prev_dst")
            if src and prev is None:
                con.execute("DELETE FROM query_rewrites WHERE src=?", (src,))
            elif src and prev is not None:
                con.execute(
                    "UPDATE query_rewrites SET dst=?, updated_at=? WHERE src=?",
                    (prev, time.time(), src),
                )
        elif kind == "brand_memory":
            alias = payload.get("alias")
            prev = payload.get("prev")
            if alias and prev is None:
                con.execute("DELETE FROM brand_memory WHERE alias=?", (alias,))
            elif alias and prev is not None:
                con.execute(
                    "UPDATE brand_memory SET canonical=?, updated_at=? WHERE alias=?",
                    (prev, time.time(), alias),
                )
        elif kind == "ranking_overlay":
            con.execute(
                """DELETE FROM ranking_overlay WHERE id=(
                    SELECT id FROM ranking_overlay ORDER BY id DESC LIMIT 1
                )"""
            )
        con.execute("DELETE FROM memory_undo WHERE id=?", (uid,))
    return {"undone": True, "kind": kind, "payload": payload}


def ocr_cache_get(db_path: str, cache_key: str) -> dict[str, Any] | None:
    if not db_path or not cache_key:
        return None
    mem = get_search_memory(db_path)
    with mem._connect() as con:
        row = con.execute(
            "SELECT text, confidence, languages, brands FROM ocr_cache WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
    if not row:
        return None
    try:
        brands = json.loads(row[3] or "[]")
    except Exception:
        brands = []
    return {
        "text": row[0],
        "confidence": float(row[1] or 0),
        "languages": row[2],
        "brands": brands,
    }


def ocr_cache_put(
    db_path: str,
    cache_key: str,
    text: str,
    *,
    confidence: float = 0.0,
    languages: str = "tr,en",
    brands: list[str] | None = None,
) -> None:
    if not db_path or not cache_key:
        return
    mem = get_search_memory(db_path)
    with mem._connect() as con:
        con.execute(
            """INSERT INTO ocr_cache(cache_key,text,confidence,languages,brands,updated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(cache_key) DO UPDATE SET
                 text=excluded.text, confidence=excluded.confidence,
                 languages=excluded.languages, brands=excluded.brands,
                 updated_at=excluded.updated_at""",
            (
                cache_key,
                text or "",
                float(confidence or 0),
                languages,
                json.dumps(brands or [], ensure_ascii=False),
                time.time(),
            ),
        )
