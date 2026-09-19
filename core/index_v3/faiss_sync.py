"""V3 → mevcut FaissStore senkronu (ikinci FAISS sistemi yok).

DB embedding READY olduktan sonra çağrılır. AI Final'i etkilemez:
FAISS hatası artifact job'u FAIL etmez; meta/gap ile izlenir.
"""

from __future__ import annotations

import threading
from typing import Any

from core.faiss_store import FaissStore
from core.logger import setup_logger

logger = setup_logger(__name__)

_STORE_LOCK = threading.Lock()
_CACHED_STORE: FaissStore | None = None
_CACHED_KEY = ""
_DIRTY = 0
_SAVE_EVERY = 32


def _store_from_settings(settings: Any) -> FaissStore | None:
    global _CACHED_STORE, _CACHED_KEY
    if settings is None:
        return None
    dino = str(getattr(settings, "faiss_dino_path", "") or "")
    clip = str(getattr(settings, "faiss_clip_path", "") or "")
    if not dino or not clip:
        return None
    key = f"{dino}|{clip}"
    with _STORE_LOCK:
        if _CACHED_STORE is not None and _CACHED_KEY == key:
            return _CACHED_STORE
        try:
            store = FaissStore(dino, clip)
        except Exception as exc:
            logger.warning("faiss_store_open_failed: %s", exc)
            return None
        _CACHED_STORE = store
        _CACHED_KEY = key
        return store


def flush_cached_faiss() -> None:
    global _DIRTY
    with _STORE_LOCK:
        if _CACHED_STORE is None or _DIRTY <= 0:
            return
        try:
            _CACHED_STORE.save()
            _DIRTY = 0
        except Exception as exc:
            logger.warning("faiss_flush_failed: %s", exc)


def _note_dirty() -> None:
    global _DIRTY
    with _STORE_LOCK:
        _DIRTY += 1
        if _DIRTY >= _SAVE_EVERY and _CACHED_STORE is not None:
            try:
                _CACHED_STORE.save()
                _DIRTY = 0
            except Exception as exc:
                logger.warning("faiss_batch_save_failed: %s", exc)


def _set_meta(db: Any, key: str, value: str) -> None:
    try:
        db.set_meta(key, value)
    except Exception:
        pass


def sync_file_embeddings(
    db: Any,
    settings: Any,
    file_id: int,
    *,
    kinds: tuple[str, ...] = ("dino", "clip"),
) -> dict[str, bool]:
    """DB'deki embedding(ler)i FAISS'e upsert et. Duplicate id üretmez."""
    out = {"dino": False, "clip": False}
    if not bool(getattr(settings, "ai_embedding_enabled", True)):
        return out
    store = _store_from_settings(settings)
    if store is None or not store.available:
        _set_meta(db, "faiss_sync_error", "store_unavailable")
        return out
    feat = db.get_features(int(file_id)) or {}
    try:
        if "dino" in kinds:
            blob = feat.get("dino_embedding")
            if blob:
                out["dino"] = bool(store.upsert_dino(int(file_id), blob, persist=False))
        if "clip" in kinds:
            blob = feat.get("clip_embedding")
            if blob:
                out["clip"] = bool(store.upsert_clip(int(file_id), blob, persist=False))
        if any(out.values()):
            _note_dirty()
            _set_meta(db, "faiss_sync_error", "")
    except Exception as exc:
        logger.warning("faiss_sync_file_%s_failed: %s", file_id, exc)
        _set_meta(db, "faiss_sync_error", str(exc)[:200])
        _set_meta(db, "faiss_needs_rebuild", "1")
    return out


def exclude_file_ids(settings: Any, file_ids: list[int] | set[int]) -> bool:
    store = _store_from_settings(settings)
    if store is None or not store.available:
        return False
    try:
        return bool(store.exclude_ids(file_ids))
    except Exception as exc:
        logger.warning("faiss_exclude_failed: %s", exc)
        return False


def rebuild_active_from_db(db: Any, settings: Any) -> dict[str, int]:
    """Aktif (non-missing) dosyalardaki embeddinglerden FAISS yeniden kur."""
    store = _store_from_settings(settings)
    if store is None or not store.available:
        return {"dino": 0, "clip": 0}
    sql = """
        SELECT f.id AS id, fe.dino_embedding, fe.clip_embedding
        FROM files f
        JOIN features fe ON fe.file_id = f.id
        WHERE f.status NOT IN ('excluded_internal','missing')
          AND (
            (fe.dino_embedding IS NOT NULL AND length(fe.dino_embedding)>0)
            OR (fe.clip_embedding IS NOT NULL AND length(fe.clip_embedding)>0)
          )
    """
    try:
        with db.connect() as conn:
            rows = [dict(r) for r in conn.execute(sql).fetchall()]
        store.rebuild_from_db(rows)
        _set_meta(db, "faiss_needs_rebuild", "0")
        _set_meta(db, "faiss_sync_error", "")
        return {"dino": store.dino_count, "clip": store.clip_count}
    except Exception as exc:
        logger.warning("faiss_rebuild_active_failed: %s", exc)
        _set_meta(db, "faiss_needs_rebuild", "1")
        _set_meta(db, "faiss_sync_error", str(exc)[:200])
        return {"dino": 0, "clip": 0}


def ensure_consistent(db: Any, settings: Any) -> dict[str, Any]:
    """Restart / oturum sonu: DB ↔ FAISS sayıları uyuşmuyorsa rebuild."""
    flush_cached_faiss()
    store = _store_from_settings(settings)
    if store is None or not store.available:
        return {"ok": False, "reason": "unavailable"}
    try:
        with db.connect() as conn:
            dino_db = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM features fe
                    JOIN files f ON f.id=fe.file_id
                    WHERE f.status NOT IN ('excluded_internal','missing')
                      AND fe.dino_embedding IS NOT NULL
                      AND length(fe.dino_embedding)>0
                    """
                ).fetchone()[0]
            )
            clip_db = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM features fe
                    JOIN files f ON f.id=fe.file_id
                    WHERE f.status NOT IN ('excluded_internal','missing')
                      AND fe.clip_embedding IS NOT NULL
                      AND length(fe.clip_embedding)>0
                    """
                ).fetchone()[0]
            )
        flag = ""
        try:
            flag = str(db.get_meta("faiss_needs_rebuild") or "")
        except Exception:
            flag = ""
        cons = store.consistency({"dino": dino_db, "clip": clip_db})
        if cons.get("clip_partial_map"):
            logger.warning(
                "faiss_ensure_consistent skipped rebuild: recovered CLIP map is partial"
            )
            return {
                "ok": True,
                "reason": "clip_partial_map",
                "rebuilt": False,
                **cons,
            }
        if cons.get("rebuild_blocked") or cons.get("map_unreliable"):
            logger.warning(
                "faiss_ensure_consistent skipped rebuild: id map unavailable"
            )
            return {
                "ok": False,
                "reason": "id_map_unavailable",
                "rebuilt": False,
                **cons,
            }
        if flag in ("1", "true", "True") or not cons.get("consistent"):
            counts = rebuild_active_from_db(db, settings)
            return {"ok": True, "rebuilt": True, **counts, **cons}
        return {"ok": True, "rebuilt": False, **cons}
    except Exception as exc:
        logger.warning("faiss_ensure_consistent_failed: %s", exc)
        return {"ok": False, "reason": str(exc)}
