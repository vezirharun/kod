"""Yardımcı fonksiyonlar — yol normalizasyonu, hash, müşteri adı."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Callable

from core.logger import setup_logger

logger = setup_logger(__name__)


def strip_extended_path_prefix(path: str | Path) -> str:
    """Windows \\\\?\\ / \\\\?\\UNC\\ önekini kaldır (DB kimliği için)."""
    raw = str(path or "").strip().replace("/", "\\")
    if raw.startswith("\\\\?\\UNC\\"):
        return "\\\\" + raw[8:]
    if raw.startswith("\\\\?\\"):
        return raw[4:]
    return raw


def fs_access_path(path: str | Path) -> str:
    """
    OS açılışı için Windows extended-length path.
    UNC: \\\\server\\share\\a → \\\\?\\UNC\\server\\share\\a
    Yerel: C:\\a → \\\\?\\C:\\a
    DB'de saklanan kanonik yolu değiştirmez; yalnızca open/stat için kullanın.
    """
    canonical = normalize_path(path)
    if not canonical:
        return canonical
    if os.name != "nt":
        return canonical
    raw = strip_extended_path_prefix(canonical).replace("/", "\\")
    if raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        # \\\\server\\share\\... → \\\\?\\UNC\\server\\share\\...
        return "\\\\?\\UNC\\" + raw.lstrip("\\")
    abs_path = os.path.abspath(raw)
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def normalize_path(path: str | Path) -> str:
    """Windows/UNC ve POSIX yollarını OS'ye uygun kanonik hale getir."""
    raw = str(path).strip()
    if not raw:
        return ""
    # POSIX çalışma/test ortamında Windows slash dönüşümü yapılmaz.
    if os.name != "nt":
        p = Path(raw)
        try:
            return str(p.resolve())
        except (OSError, RuntimeError):
            return os.path.normpath(raw)
    raw = strip_extended_path_prefix(raw)
    if raw.startswith(("\\\\", "//")):
        return normalize_source_root(raw)
    p = Path(raw)
    try:
        resolved = p.resolve()
        return str(resolved)
    except (OSError, RuntimeError):
        return os.path.normpath(str(p))


def normalize_source_root(path: str | Path) -> str:
    """Canonical source root without resolving or touching an UNC share."""
    raw = strip_extended_path_prefix(str(path or "").strip()).replace("/", "\\")
    if not raw:
        return ""
    if raw.startswith("\\"):
        parts = [part for part in raw.split("\\") if part]
        return "\\\\" + "\\".join(parts)
    return os.path.normpath(raw).rstrip("\\/")


def source_root_key(path: str | Path) -> str:
    """Case-insensitive comparison key for Windows local and UNC roots."""
    return normalize_source_root(path).casefold()


def customer_from_path(file_path: str | Path, archive_root: str | Path) -> str:
    """Arşiv kökünden sonraki ilk klasör adını müşteri olarak al."""
    try:
        file_p = Path(normalize_path(file_path))
        root_p = Path(normalize_path(archive_root))
        rel = file_p.relative_to(root_p)
        parts = rel.parts
        return parts[0] if parts else ""
    except ValueError:
        return ""


def safe_stat(
    path: str | Path,
    timeout_sec: int = 30,
    retry_count: int = 3,
) -> os.stat_result | None:
    """Ağ dosyaları için timeout ve yeniden deneme ile stat."""
    # Önce wide-path, gerekirse kanonik yolla tekrar dene (Errno 22 / UNC+Unicode).
    candidates = []
    canonical = normalize_path(path)
    access = fs_access_path(canonical)
    candidates.append(access)
    if access != canonical:
        candidates.append(canonical)
    last_exc: OSError | None = None
    for attempt in range(retry_count):
        for path_str in candidates:
            try:
                if not os.path.exists(path_str):
                    continue
                return os.stat(path_str)
            except OSError as exc:
                last_exc = exc
                continue
        logger.warning(
            "stat hatası (deneme %d/%d): %s — %s",
            attempt + 1,
            retry_count,
            canonical,
            last_exc,
        )
        time.sleep(min(2**attempt, 5))
    return None


def partial_file_hash(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
    head_chunks: int = 2,
    tail_chunks: int = 1,
) -> str:
    """Büyük dosyalar için kısmi SHA256 (baş + son bloklar + boyut)."""
    path_str = fs_access_path(path)
    h = hashlib.sha256()
    stat = safe_stat(path)
    if stat is None:
        return ""
    h.update(str(stat.st_size).encode())
    h.update(str(int(stat.st_mtime)).encode())
    try:
        with open(path_str, "rb") as f:
            for _ in range(head_chunks):
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
            if stat.st_size > chunk_size * head_chunks:
                f.seek(max(0, stat.st_size - chunk_size * tail_chunks))
                for _ in range(tail_chunks):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    h.update(chunk)
    except OSError as exc:
        logger.warning("Kısmi hash okunamadı: %s — %s", path_str, exc)
        return ""
    return h.hexdigest()


def full_file_hash(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Tam dosya SHA256 — isteğe bağlı, yavaş."""
    path_str = fs_access_path(path)
    h = hashlib.sha256()
    try:
        with open(path_str, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("Tam hash okunamadı: %s — %s", path_str, exc)
        return ""


def file_id_from_path(path: str | Path) -> str:
    """Thumbnail/cache dosya adı için kısa kimlik."""
    norm = normalize_path(path).lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def iter_fs_path_candidates(path: str | Path) -> list[str]:
    """Açılış denemeleri: önce wide-path, sonra kanonik."""
    canonical = normalize_path(path)
    access = fs_access_path(canonical)
    out = [access]
    if access != canonical:
        out.append(canonical)
    return out


def format_file_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    if size < 1024**3:
        return f"{size / 1024**2:.1f} MB"
    return f"{size / 1024**3:.2f} GB"


def hamming_distance_hex(a: str, b: str) -> int:
    """Hex string perceptual hash'ler arası Hamming mesafesi."""
    if not a or not b or len(a) != len(b):
        return 64
    try:
        ia = int(a, 16)
        ib = int(b, 16)
        return (ia ^ ib).bit_count()
    except ValueError:
        return 64


def phash_similarity(a: str, b: str, max_bits: int = 64) -> float:
    dist = hamming_distance_hex(a, b)
    return max(0.0, 1.0 - dist / max_bits)


# Metin arama — dil / yazım varyantları (leopar ↔ leopard vb.)
_TEXT_VARIANTS: dict[str, list[str]] = {
    "leopar": ["leopard", "leopar"],
    "leopard": ["leopar", "leopard"],
    "cicek": ["çiçek", "cicek", "flower", "floral"],
    "çiçek": ["çiçek", "cicek", "flower", "floral"],
    "flower": ["flower", "çiçek", "cicek", "floral"],
    "floral": ["floral", "çiçek", "flower"],
    "geometrik": ["geometric", "geometrik"],
    "geometric": ["geometric", "geometrik"],
    "desen": ["pattern", "desen", "motif"],
    "pattern": ["pattern", "desen", "motif"],
}


def text_search_variants(text: str) -> list[str]:
    """Arama sorgusu için yazım/dil varyantları üret."""
    base = text.strip().lower()
    if not base:
        return []
    variants: set[str] = {base}
    for key, alts in _TEXT_VARIANTS.items():
        if key in base or base in key:
            variants.update(alts)
            for alt in alts:
                variants.add(base.replace(key, alt))
    # Türkçe karakter normalize
    tr_map = str.maketrans("çğıöşü", "cgiosu")
    normalized = base.translate(tr_map)
    variants.add(normalized)
    return list(variants)[:12]


def retry_operation(
    func: Callable,
    retry_count: int = 3,
    delay_base: float = 1.0,
    *args,
    **kwargs,
):
    last_exc: Exception | None = None
    for attempt in range(retry_count):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            time.sleep(delay_base * (2**attempt))
    if last_exc:
        raise last_exc
    return None
