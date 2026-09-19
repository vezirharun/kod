"""FAISS vektör indeks yönetimi."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.logger import setup_logger

logger = setup_logger(__name__)

NUMPY_MAGIC = b"\x93NUMPY"
CLIP_MAP_UNAVAILABLE = "CLIP map unavailable"
DINO_MAP_UNAVAILABLE = "DINO map unavailable"
VISUAL_SIM_FALLBACK = "Visual similarity fallback active"

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss bulunamadı — vektör araması sınırlı olacak")


class FaissMapError(Exception):
    """id-map dosyası okunamadı veya schema geçersiz."""

    def __init__(self, path: Path, reason: str):
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"{reason}: {self.path}")


def faiss_id_map_path(index_path: str | Path) -> Path:
    """Mevcut sözleşme: faiss_clip.index → faiss_clip.map.npy"""
    return Path(index_path).with_suffix(".map.npy")


def legacy_faiss_id_map_path(index_path: str | Path) -> Path:
    """Eski ad: faiss_clip.index → faiss_clip_map.npy"""
    p = Path(index_path)
    return p.with_name(f"{p.stem}_map.npy")


def resolve_faiss_id_map_path(index_path: str | Path) -> Path:
    """Var olan map yolunu seç; yoksa güncel adı döndür."""
    primary = faiss_id_map_path(index_path)
    if primary.exists():
        return primary
    legacy = legacy_faiss_id_map_path(index_path)
    if legacy.exists():
        return legacy
    return primary


def inspect_faiss_id_map(path: str | Path) -> dict[str, Any]:
    """Read-only dosya teşhisi (içerik değiştirmez)."""
    p = Path(path)
    info: dict[str, Any] = {
        "path": str(p),
        "exists": p.exists(),
        "size": 0,
        "kind": "missing",
        "is_npy": False,
        "dtype": None,
        "shape": None,
    }
    if not p.exists():
        return info
    info["size"] = int(p.stat().st_size)
    if info["size"] == 0:
        info["kind"] = "empty"
        return info
    with p.open("rb") as fh:
        head = fh.read(16)
        if info["size"] > 16:
            fh.seek(max(0, info["size"] - 16))
            tail = fh.read(16)
        else:
            tail = head
    if head.startswith(NUMPY_MAGIC):
        info["is_npy"] = True
        info["kind"] = "npy"
        try:
            arr = np.load(p, allow_pickle=False, mmap_mode="r")
            info["dtype"] = str(arr.dtype)
            info["shape"] = tuple(arr.shape)
        except ValueError as exc:
            msg = str(exc).lower()
            if "pickle" in msg or "object" in msg:
                info["kind"] = "npy_object"
                try:
                    arr = np.load(p, allow_pickle=True)
                    info["dtype"] = str(getattr(arr, "dtype", "object"))
                    info["shape"] = tuple(getattr(arr, "shape", ()))
                except Exception as inner:
                    info["kind"] = "npy_header_invalid"
                    info["header_error"] = str(inner)
            else:
                info["kind"] = "npy_header_invalid"
                info["header_error"] = str(exc)
        except Exception as exc:
            info["kind"] = "npy_header_invalid"
            info["header_error"] = str(exc)
        return info
    if head[:1] == b"\x80":
        info["kind"] = "raw_pickle"
        return info
    if head == b"\x00" * 16 and tail == b"\x00" * min(16, len(tail)):
        info["kind"] = "zeroed"
        return info
    info["kind"] = "unknown"
    return info


def _coerce_id_list(arr: Any, path: Path) -> list[int]:
    if isinstance(arr, np.ndarray):
        if arr.ndim == 0:
            arr = np.atleast_1d(arr)
        if arr.ndim != 1:
            raise FaissMapError(path, f"unsupported_shape_{arr.shape}")
        if arr.size == 0:
            return []
        if arr.dtype.kind in ("i", "u"):
            return [int(x) for x in arr.tolist()]
        if arr.dtype.kind == "O":
            out: list[int] = []
            for item in arr.tolist():
                if isinstance(item, (list, tuple, np.ndarray, dict)):
                    raise FaissMapError(path, "unsupported_object_schema")
                try:
                    out.append(int(item))
                except (TypeError, ValueError) as exc:
                    raise FaissMapError(path, "object_values_not_int") from exc
            return out
        raise FaissMapError(path, f"unsupported_dtype_{arr.dtype}")
    if isinstance(arr, (list, tuple)):
        try:
            return [int(x) for x in arr]
        except (TypeError, ValueError) as exc:
            raise FaissMapError(path, "list_values_not_int") from exc
    raise FaissMapError(path, f"unsupported_type_{type(arr).__name__}")


def load_faiss_id_map(path: str | Path) -> list[int]:
    """Tek giriş noktası: FAISS position→file_id map.

    Beklenen schema: 1-D integer ndarray (int64). Eski object-array .npy
    yalnızca dosya gerçekten NUMPY magic + object dtype ise pickle ile açılır.
    Ham pickle / sıfırlanmış / sahte .npy yüklenmez.
    """
    p = Path(path)
    if not p.exists():
        raise FaissMapError(p, "missing")
    size = int(p.stat().st_size)
    if size == 0:
        raise FaissMapError(p, "empty")
    with p.open("rb") as fh:
        magic = fh.read(6)
    if magic != NUMPY_MAGIC:
        info = inspect_faiss_id_map(p)
        if info["kind"] == "zeroed":
            raise FaissMapError(p, "corrupt_zeroed_not_npy")
        if info["kind"] == "raw_pickle":
            raise FaissMapError(p, "raw_pickle_not_npy")
        raise FaissMapError(p, f"invalid_format_{info['kind']}")
    try:
        arr = np.load(p, allow_pickle=False)
    except ValueError as exc:
        msg = str(exc).lower()
        if "pickle" not in msg and "object" not in msg:
            raise FaissMapError(p, f"npy_load_failed_{type(exc).__name__}") from exc
        try:
            arr = np.load(p, allow_pickle=True)
        except Exception as inner:
            raise FaissMapError(p, "object_npy_unreadable") from inner
    except Exception as exc:
        raise FaissMapError(p, f"npy_load_failed_{type(exc).__name__}") from exc
    return _coerce_id_list(arr, p)


def save_faiss_id_map(path: str | Path, ids: Sequence[int]) -> None:
    """Atomik int64 .npy yazımı — pickle kullanılmaz."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray([int(x) for x in ids], dtype=np.int64)
    tmp = dest.parent / f".{dest.name}.tmp.npy"
    try:
        np.save(tmp, arr, allow_pickle=False)
        os.replace(tmp, dest)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def recover_clip_id_map_from_stored_embeddings(
    clip_index_path: str | Path,
    db_path: str | Path,
    *,
    min_cosine: float = 0.999,
    min_margin: float = 0.001,
) -> tuple[list[int], dict[str, Any]]:
    """Rebuild position→file_id map from FAISS vectors vs stored DB embeddings.

    Does not recompute CLIP from images and does not rewrite the FAISS index.
    Unmatched / zero slots are  -1.
    """
    import sqlite3

    if not HAS_FAISS:
        raise FaissMapError(Path(clip_index_path), "faiss_unavailable")
    index_path = Path(clip_index_path)
    index = faiss.read_index(str(index_path))
    n = int(getattr(index, "ntotal", 0) or 0)
    d = int(getattr(index, "d", 0) or 0)
    if n <= 0 or d <= 0:
        raise FaissMapError(index_path, "empty_clip_index")
    xb = np.zeros((n, d), dtype=np.float32)
    index.reconstruct_n(0, n, xb)
    norms = np.linalg.norm(xb, axis=1)
    live = norms >= 1e-6
    xn = xb / np.maximum(norms.reshape(-1, 1), 1e-8)

    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    rows = con.execute(
        "SELECT file_id, clip_embedding FROM features "
        "WHERE clip_embedding IS NOT NULL AND length(clip_embedding)=?",
        (d * 4,),
    ).fetchall()
    con.close()
    if not rows:
        raise FaissMapError(index_path, "no_db_clip_embeddings")
    fids = np.array([int(r[0]) for r in rows], dtype=np.int64)
    dbm = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    dbn = np.linalg.norm(dbm, axis=1, keepdims=True)
    dbm = dbm / np.maximum(dbn, 1e-8)

    recovered = np.full(n, -1, dtype=np.int64)
    claimed: dict[int, tuple[int, float]] = {}
    idx_live = np.where(live)[0]
    chunk = 500
    for start in range(0, len(idx_live), chunk):
        sl = idx_live[start : start + chunk]
        sims = xn[sl] @ dbm.T
        part = np.argpartition(sims, -2, axis=1)[:, -2:]
        for row, pos in enumerate(sl):
            a, b = int(part[row, 0]), int(part[row, 1])
            sa, sb = float(sims[row, a]), float(sims[row, b])
            if sa >= sb:
                best_i, best_s, second_s = a, sa, sb
            else:
                best_i, best_s, second_s = b, sb, sa
            if best_s < min_cosine or (best_s - second_s) < min_margin:
                continue
            fid = int(fids[best_i])
            if fid <= 0:
                continue
            prev = claimed.get(fid)
            if prev is not None:
                old_pos, old_s = prev
                if best_s <= old_s:
                    continue
                recovered[old_pos] = -1
            recovered[pos] = fid
            claimed[fid] = (int(pos), best_s)
    mapped = recovered[recovered > 0]
    stats = {
        "clip_ntotal": n,
        "zero_slots": int((~live).sum()),
        "live_slots": int(live.sum()),
        "unique_ok": int(len(claimed)),
        "unique_ids": int(len(set(mapped.tolist()))),
        "duplicate_assigned": 0,
        "unmapped_slots": int((recovered <= 0).sum()),
    }
    if stats["unique_ok"] != stats["unique_ids"]:
        raise FaissMapError(index_path, "recovery_duplicate_ids")
    return [int(x) for x in recovered.tolist()], stats


def restore_clip_id_map_atomic(
    clip_index_path: str | Path,
    ids: Sequence[int],
    *,
    backup: bool = True,
) -> Path:
    """Write only the CLIP id map. Never rewrites faiss_clip.index."""
    import shutil

    index_path = Path(clip_index_path)
    dest = faiss_id_map_path(index_path)
    if backup and dest.exists():
        bak = dest.with_name(dest.name + ".zeroed.bak")
        if not bak.exists():
            shutil.copy2(dest, bak)
    save_faiss_id_map(dest, ids)
    return dest


def audit_clip_id_map(
    ids: Sequence[int],
    *,
    clip_index_path: str | Path,
    db_path: str | Path,
    probe_n: int = 5,
) -> dict[str, Any]:
    """Read-only restore checks. Does not write the FAISS index or DB."""
    import sqlite3

    index_path = Path(clip_index_path)
    id_list = [int(x) for x in ids]
    ntotal = 0
    zero_slots = 0
    probes: list[dict[str, Any]] = []
    if HAS_FAISS and index_path.exists():
        index = faiss.read_index(str(index_path))
        ntotal = int(getattr(index, "ntotal", 0) or 0)
        if ntotal > 0:
            xb = np.zeros((ntotal, int(index.d)), dtype=np.float32)
            index.reconstruct_n(0, ntotal, xb)
            norms = np.linalg.norm(xb, axis=1)
            zero_slots = int((norms < 1e-6).sum())
            mapped_pos = [i for i, fid in enumerate(id_list) if fid > 0]
            for pos in mapped_pos[: max(0, probe_n)]:
                fid = id_list[pos]
                q = xb[pos : pos + 1].copy()
                nrm = float(np.linalg.norm(q))
                if nrm < 1e-6:
                    probes.append({"pos": pos, "file_id": fid, "ok": False, "reason": "zero_vector"})
                    continue
                q = q / nrm
                scores, idxs = index.search(q.astype(np.float32), 1)
                hit_idx = int(idxs[0][0]) if idxs.size else -1
                hit_fid = id_list[hit_idx] if 0 <= hit_idx < len(id_list) else -1
                probes.append(
                    {
                        "pos": pos,
                        "file_id": fid,
                        "search_idx": hit_idx,
                        "search_fid": hit_fid,
                        "score": float(scores[0][0]) if scores.size else 0.0,
                        "ok": hit_fid == fid,
                    }
                )

    mapped = [x for x in id_list if x > 0]
    unmapped = [i for i, x in enumerate(id_list) if x <= 0]
    dupes = len(mapped) - len(set(mapped))
    db_ids: set[int] = set()
    paths: dict[int, str] = {}
    if db_path:
        uri = f"file:{Path(db_path).as_posix()}?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        db_ids = {
            int(r[0])
            for r in con.execute("SELECT id FROM files").fetchall()
        }
        want = [p["file_id"] for p in probes if p.get("file_id")]
        if want:
            qmarks = ",".join("?" * len(want))
            for fid, path, name in con.execute(
                f"SELECT id, path, filename FROM files WHERE id IN ({qmarks})",
                want,
            ):
                paths[int(fid)] = f"{path}|{name}"
        con.close()
    for p in probes:
        p["path"] = paths.get(int(p.get("file_id") or 0), "")
    in_db = sum(1 for x in mapped if x in db_ids) if db_ids else 0
    oob = [x for x in mapped if db_ids and x not in db_ids]
    return {
        "faiss_ntotal": ntotal,
        "map_length": len(id_list),
        "valid_id_count": len(mapped),
        "unique_valid_ids": len(set(mapped)),
        "db_id_count": len(db_ids),
        "mapped_in_db": in_db,
        "duplicate_id_count": dupes,
        "out_of_range_ids": oob[:20],
        "out_of_range_count": len(oob),
        "unmapped_slot_count": len(unmapped),
        "zero_vector_slots": zero_slots,
        "length_matches_ntotal": bool(ntotal and len(id_list) == ntotal),
        "no_positive_on_zero_rule": True,
        "probes": probes,
        "ok": bool(
            ntotal
            and len(id_list) == ntotal
            and dupes == 0
            and not oob
            and len(mapped) > 0
            and (not db_ids or in_db == len(mapped))
            and all(p.get("ok") for p in probes)
        ),
    }


def _validate_map_against_index(ids: list[int], index: Any, path: Path) -> None:
    ntotal = int(getattr(index, "ntotal", 0) or 0)
    if ntotal > 0 and len(ids) == 0:
        raise FaissMapError(path, "empty_map_nonempty_index")
    if ntotal > 0 and len(ids) != ntotal:
        raise FaissMapError(path, f"length_mismatch_map_{len(ids)}_index_{ntotal}")


class FaissStore:
    """DINO ve CLIP embeddingleri için ayrı FAISS indeksleri."""

    def __init__(
        self, dino_path: str, clip_path: str, dim_dino: int = 384, dim_clip: int = 512
    ):
        self.dino_path = Path(dino_path)
        self.clip_path = Path(clip_path)
        self.dim_dino = dim_dino
        self.dim_clip = dim_clip
        self.dino_index: Any = None
        self.clip_index: Any = None
        self.dino_id_map: list[int] = []
        self.clip_id_map: list[int] = []
        self.needs_rebuild = False
        self.dino_map_unavailable = False
        self.clip_map_unavailable = False
        self.dino_map_error = ""
        self.clip_map_error = ""
        self._init_indexes()

    def _load_kind(self, *, kind: str, index_path: Path, dim: int) -> tuple[Any, list[int]]:
        if not index_path.exists():
            return faiss.IndexFlatIP(dim), []
        # FAISS dosyası arka planda yazılırken arama okuyucusu asla yarım
        # dosyayı okumamalı. Yazma tarafı atomik replace kullanır; eski
        # sürümden kalmış bozuk dosyada da aramayı çökertmek yerine boş
        # güvenli indeksle devam edip yeniden oluşturma bayrağını kaldırma.
        index = None
        last_exc = None
        for attempt in range(3):
            try:
                index = faiss.read_index(str(index_path))
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))
        if index is None:
            self.needs_rebuild = True
            msg = f"FAISS {kind} dosyası okunamadı; güvenli boş indeks kullanılacak: {index_path}"
            logger.error("%s (%s)", msg, last_exc)
            if kind == "clip":
                self.clip_map_error = f"read_error: {last_exc}"
            else:
                self.dino_map_error = f"read_error: {last_exc}"
            return faiss.IndexFlatIP(dim), []
        map_path = resolve_faiss_id_map_path(index_path)
        try:
            if not map_path.exists():
                if int(getattr(index, "ntotal", 0) or 0) > 0:
                    raise FaissMapError(map_path, "missing")
                return index, []
            ids = load_faiss_id_map(map_path)
            _validate_map_against_index(ids, index, map_path)
            return index, ids
        except FaissMapError as exc:
            if kind == "clip":
                self.clip_map_unavailable = True
                self.clip_map_error = str(exc)
                logger.error("%s: %s", CLIP_MAP_UNAVAILABLE, exc)
            else:
                self.dino_map_unavailable = True
                self.dino_map_error = str(exc)
                logger.error("%s: %s", DINO_MAP_UNAVAILABLE, exc)
            return None, []

    def _init_indexes(self) -> None:
        if not HAS_FAISS:
            return
        self.dino_index, self.dino_id_map = self._load_kind(
            kind="dino", index_path=self.dino_path, dim=self.dim_dino
        )
        self.clip_index, self.clip_id_map = self._load_kind(
            kind="clip", index_path=self.clip_path, dim=self.dim_clip
        )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vectors / norms

    def add_dino(self, file_id: int, embedding: bytes) -> None:
        from core.index_freeze import guard_index_write

        guard_index_write("faiss.add_dino", "core.faiss_store")
        if self.dino_map_unavailable:
            return
        if not HAS_FAISS or self.dino_index is None or not embedding:
            return
        if file_id in self.dino_id_map:
            self.needs_rebuild = True
            return
        if len(embedding) % 4 != 0 or len(embedding) != self.dim_dino * 4:
            return
        if len(embedding) % 4 != 0 or len(embedding) != self.dim_dino * 4:
            return False
        vec = np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dim_dino:
            return
        vec = self._normalize(vec)
        self.dino_index.add(vec)
        self.dino_id_map.append(file_id)

    def add_clip(self, file_id: int, embedding: bytes) -> None:
        from core.index_freeze import guard_index_write

        guard_index_write("faiss.add_clip", "core.faiss_store")
        if self.clip_map_unavailable:
            return
        if not HAS_FAISS or self.clip_index is None or not embedding:
            return
        if file_id in self.clip_id_map:
            self.needs_rebuild = True
            return
        if len(embedding) % 4 != 0 or len(embedding) != self.dim_clip * 4:
            return
        if len(embedding) % 4 != 0 or len(embedding) != self.dim_clip * 4:
            return False
        vec = np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dim_clip:
            return
        vec = self._normalize(vec)
        self.clip_index.add(vec)
        self.clip_id_map.append(file_id)

    def upsert_dino(self, file_id: int, embedding: bytes, *, persist: bool = True) -> bool:
        """Add or replace DINO vector (no duplicate ids)."""
        from core.index_freeze import guard_index_write

        guard_index_write("faiss.upsert_dino", "core.faiss_store")
        if self.dino_map_unavailable:
            return False
        if not HAS_FAISS or self.dino_index is None or not embedding:
            return False
        vec = np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dim_dino:
            return False
        fid = int(file_id)
        if fid in self.dino_id_map:
            self._replace_vector(
                kind="dino", file_id=fid, embedding=embedding
            )
        else:
            self.add_dino(fid, embedding)
            if fid not in self.dino_id_map:
                return False
        if persist:
            self.save()
        self.needs_rebuild = False
        return True

    def upsert_clip(self, file_id: int, embedding: bytes, *, persist: bool = True) -> bool:
        """Add or replace CLIP vector (no duplicate ids)."""
        from core.index_freeze import guard_index_write

        guard_index_write("faiss.upsert_clip", "core.faiss_store")
        if self.clip_map_unavailable:
            return False
        if not HAS_FAISS or self.clip_index is None or not embedding:
            return False
        vec = np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dim_clip:
            return False
        fid = int(file_id)
        if fid in self.clip_id_map:
            self._replace_vector(
                kind="clip", file_id=fid, embedding=embedding
            )
        else:
            self.add_clip(fid, embedding)
            if fid not in self.clip_id_map:
                return False
        if persist:
            self.save()
        self.needs_rebuild = False
        return True

    def _replace_vector(
        self, *, kind: str, file_id: int, embedding: bytes
    ) -> None:
        """Rebuild one index replacing a single file_id vector."""
        if kind == "dino":
            index, id_map, dim = self.dino_index, self.dino_id_map, self.dim_dino
        else:
            index, id_map, dim = self.clip_index, self.clip_id_map, self.dim_clip
        if index is None:
            return
        new_index = faiss.IndexFlatIP(dim)
        new_map: list[int] = []
        if len(embedding) % 4 != 0 or len(embedding) != dim * 4:
            return
        new_vec = self._normalize(
            np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
        )
        for i, fid in enumerate(id_map):
            if int(fid) == int(file_id):
                new_index.add(new_vec)
                new_map.append(int(file_id))
                continue
            try:
                old = index.reconstruct(i).reshape(1, -1).astype(np.float32)
            except Exception:
                continue
            new_index.add(self._normalize(old))
            new_map.append(int(fid))
        if int(file_id) not in new_map:
            new_index.add(new_vec)
            new_map.append(int(file_id))
        if kind == "dino":
            self.dino_index, self.dino_id_map = new_index, new_map
        else:
            self.clip_index, self.clip_id_map = new_index, new_map

    def exclude_ids(self, file_ids: set[int] | list[int]) -> bool:
        """Drop vectors for missing/deleted files (rebuild without those ids)."""
        from core.index_freeze import guard_index_write

        guard_index_write("faiss.exclude_ids", "core.faiss_store")
        if not HAS_FAISS:
            return False
        if self.dino_map_unavailable and self.clip_map_unavailable:
            return False
        drop = {int(x) for x in file_ids}
        if not drop:
            return False
        changed = False
        for kind in ("dino", "clip"):
            if kind == "dino" and self.dino_map_unavailable:
                continue
            if kind == "clip" and self.clip_map_unavailable:
                continue
            index = self.dino_index if kind == "dino" else self.clip_index
            id_map = self.dino_id_map if kind == "dino" else self.clip_id_map
            dim = self.dim_dino if kind == "dino" else self.dim_clip
            if index is None or not id_map:
                continue
            if not any(int(fid) in drop for fid in id_map):
                continue
            new_index = faiss.IndexFlatIP(dim)
            new_map: list[int] = []
            for i, fid in enumerate(id_map):
                if int(fid) in drop:
                    changed = True
                    continue
                try:
                    old = index.reconstruct(i).reshape(1, -1).astype(np.float32)
                except Exception:
                    continue
                new_index.add(self._normalize(old))
                new_map.append(int(fid))
            if kind == "dino":
                self.dino_index, self.dino_id_map = new_index, new_map
            else:
                self.clip_index, self.clip_id_map = new_index, new_map
        if changed:
            self.save()
            self.needs_rebuild = False
        return changed

    def search_dino(self, query: np.ndarray, k: int = 50) -> list[tuple[int, float]]:
        return self._search(self.dino_index, self.dino_id_map, query, k)

    def search_clip(self, query: np.ndarray, k: int = 50) -> list[tuple[int, float]]:
        return self._search(self.clip_index, self.clip_id_map, query, k)

    def _search(
        self,
        index: Any,
        id_map: list[int],
        query: np.ndarray,
        k: int,
    ) -> list[tuple[int, float]]:
        if not HAS_FAISS or index is None or index.ntotal == 0:
            return []
        q = query.astype(np.float32).reshape(1, -1)
        if q.shape[1] != index.d:
            return []
        q = self._normalize(q)
        k = min(k, index.ntotal)
        scores, indices = index.search(q, k)
        results: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(id_map):
                continue
            fid = int(id_map[idx])
            if fid <= 0:
                continue
            results.append((fid, float(score)))
        return results

    @staticmethod
    def _write_index_atomic(index: Any, destination: Path) -> None:
        """FAISS indeksini geçici dosyaya yazıp tek atomik rename ile yayınla."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.parent / f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            faiss.write_index(index, str(tmp))
            # Aynı disk üzerinde replace: okuyucu ya eski tam dosyayı ya yeni
            # tam dosyayı görür; hiçbir zaman yarım FAISS görmez.
            os.replace(tmp, destination)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def save(self) -> None:
        from core.index_freeze import guard_index_write

        guard_index_write("faiss.save", "core.faiss_store")
        if not HAS_FAISS:
            return
        self.dino_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            self.dino_index is not None
            and self.dino_index.ntotal > 0
            and not self.dino_map_unavailable
        ):
            self._write_index_atomic(self.dino_index, self.dino_path)
            save_faiss_id_map(faiss_id_map_path(self.dino_path), self.dino_id_map)
        if (
            self.clip_index is not None
            and self.clip_index.ntotal > 0
            and not self.clip_map_unavailable
        ):
            self._write_index_atomic(self.clip_index, self.clip_path)
            save_faiss_id_map(faiss_id_map_path(self.clip_path), self.clip_id_map)

    def rebuild_from_db(self, records: list[dict[str, Any]]) -> None:
        """Veritabanından FAISS indeksini yeniden oluştur."""
        from core.index_freeze import guard_index_write

        guard_index_write("faiss.rebuild_from_db", "core.faiss_store")
        if not HAS_FAISS:
            return
        self.dino_index = faiss.IndexFlatIP(self.dim_dino)
        self.clip_index = faiss.IndexFlatIP(self.dim_clip)
        self.dino_id_map = []
        self.clip_id_map = []
        self.dino_map_unavailable = False
        self.clip_map_unavailable = False
        self.dino_map_error = ""
        self.clip_map_error = ""
        for rec in records:
            fid = rec["id"]
            if rec.get("dino_embedding"):
                self.add_dino(fid, rec["dino_embedding"])
            if rec.get("clip_embedding"):
                self.add_clip(fid, rec["clip_embedding"])
        self.save()
        self.needs_rebuild = False
        logger.info(
            "FAISS yeniden oluşturuldu: DINO=%d, CLIP=%d",
            len(self.dino_id_map),
            len(self.clip_id_map),
        )

    @property
    def available(self) -> bool:
        return HAS_FAISS

    @property
    def dino_count(self) -> int:
        return len(self.dino_id_map)

    @property
    def clip_count(self) -> int:
        return sum(1 for x in self.clip_id_map if int(x) > 0)

    def consistency(self, db_counts: dict[str, int]) -> dict[str, Any]:
        dino_db = int(db_counts.get("dino", 0))
        clip_db = int(db_counts.get("clip", 0))
        map_unreliable = bool(self.dino_map_unavailable or self.clip_map_unavailable)
        clip_pos = [int(x) for x in self.clip_id_map if int(x) > 0]
        clip_unmapped = sum(1 for x in self.clip_id_map if int(x) <= 0)
        clip_partial = bool(clip_pos and clip_unmapped and not self.clip_map_unavailable)
        clip_unique = len(set(clip_pos)) == len(clip_pos)
        dino_unique = len(set(self.dino_id_map)) == len(self.dino_id_map)
        # Partial recovered CLIP maps are valid: unmapped slots stay -1.
        # Do not require clip_db == clip_count (that would force a rebuild).
        clip_ok = (
            not self.clip_map_unavailable
            and clip_unique
            and self.clip_count > 0
            and (clip_db <= 0 or self.clip_count <= clip_db)
        )
        return {
            "dino_db": dino_db,
            "clip_db": clip_db,
            "dino_faiss": self.dino_count,
            "clip_faiss": self.clip_count,
            "clip_unmapped": clip_unmapped,
            "clip_partial_map": clip_partial,
            "clip_map_unavailable": self.clip_map_unavailable,
            "dino_map_unavailable": self.dino_map_unavailable,
            "map_unreliable": map_unreliable,
            "rebuild_blocked": map_unreliable or clip_partial,
            "consistent": (
                not map_unreliable
                and not self.needs_rebuild
                and dino_db == self.dino_count
                and clip_ok
                and dino_unique
            ),
        }
