"""Ayar yollarında geçici test dizinlerini tespit et."""

from __future__ import annotations

import re
from pathlib import Path

# Gerçek kullanımda kabul edilmeyen yol kalıpları
_TEMP_PATH_MARKERS = (
    "pytest-of-harun",
    "pytest-of-",
    "appdata\\local\\temp",
    "/appdata/local/temp",
    "\\temp\\",
    "/tmp/",
    "\\tmp\\",
)

_TEST_PATH_RE = re.compile(r"(^|[\\/])test_", re.IGNORECASE)

_NON_PRODUCTION_DB_NAMES = frozenset({"test.db", "temp.db", "pytest.db"})

_LEGACY_WORK_MARKERS = (
    "mnt/data/work_v12424",
    r"mnt\data\work_v12424",
)


def remap_legacy_storage_path(path: str | Path | None) -> str:
    """Eski WSL/C:\\mnt\\data\\work_v12424 yollarını proje köküne al."""
    raw = str(path or "").strip()
    if not raw:
        return raw
    from core.settings import PROJECT_ROOT

    norm = raw.replace("/", "\\")
    lower = norm.lower()
    marker = r"mnt\data\work_v12424"
    idx = lower.find(marker)
    if idx < 0:
        return raw
    rest = norm[idx + len(marker) :].lstrip("\\")
    return str((PROJECT_ROOT / rest).resolve()) if rest else str(PROJECT_ROOT.resolve())



def is_temporary_test_path(path: str | Path | None) -> bool:
    """Geçici test / pytest yolu mu?"""
    if not path:
        return False
    norm = str(path).replace("/", "\\").lower()
    if "pytest-of-" in norm:
        return True
    if "appdata\\local\\temp\\" in norm and ("pytest" in norm or "test_" in norm):
        return True
    parts = [p.lower() for p in Path(norm).parts]
    if any(p.startswith("test_") for p in parts):
        return True
    if "pytest-of-harun" in norm:
        return True
    return False


_BLOCKED_SOURCE_NAMES = {
    "cache",
    "data",
    "reports",
    "logs",
    "thumbnails",
    ".pytest_cache",
    ".venv",
    "__pycache__",
}


def internal_source_reason(path: str | Path | None, settings=None) -> str:
    """Return a user-facing reason when a source points at technical storage."""
    if not path:
        return "Kaynak yolu boş."
    from core.utils import normalize_source_root, source_root_key

    root = normalize_source_root(path)
    parts = [part.casefold() for part in Path(root).parts]
    blocked = next((part for part in parts if part in _BLOCKED_SOURCE_NAMES), "")
    if blocked:
        return f"Teknik klasör kaynak olamaz: {blocked}"
    if settings is not None:
        cache_key = source_root_key(getattr(settings, "cache_dir", ""))
        data_key = source_root_key(Path(getattr(settings, "db_path", "")).parent)
        key = source_root_key(root)
        if key == data_key or key == cache_key or key.startswith(cache_key + "\\"):
            return "Uygulamanın cache veya data klasörü indeks kaynağı olamaz."
    return ""


def _project_path(path: str | Path) -> Path:
    from core.settings import PROJECT_ROOT

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def quick_db_file_count(db_path: Path) -> int:
    """FAISS/şema yüklemeden hızlı dosya sayısı."""
    if not db_path.is_file():
        return 0
    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='files' LIMIT 1"
            ).fetchone():
                return 0
            row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
            return int(row[0] or 0) if row else 0
        finally:
            conn.close()
    except Exception:
        return 0


def resolve_effective_db_path(settings) -> str:
    """
    Ayarlardaki db_path ile workspace data/patterns.db arasında doğru DB'yi seç.
    Yalnızca proje data/ klasöründeki boş veya test DB yollarını düzeltir.
    """
    from core.settings import DEFAULT_DATA_DIR, PROJECT_ROOT

    configured_raw = getattr(settings, "db_path", "") or str(
        DEFAULT_DATA_DIR / "patterns.db"
    )
    configured = _project_path(configured_raw)
    default = (DEFAULT_DATA_DIR / "patterns.db").resolve()
    data_dir = DEFAULT_DATA_DIR.resolve()

    if configured.parent != data_dir:
        return configured_raw if not Path(configured_raw).is_absolute() else str(configured)

    configured_count = quick_db_file_count(configured)
    default_count = (
        quick_db_file_count(default) if configured != default else configured_count
    )

    use_default = False
    if (
        configured.name.lower() in _NON_PRODUCTION_DB_NAMES
        and default_count > configured_count
    ):
        use_default = True
    elif configured_count == 0 and default_count > 0:
        use_default = True

    if use_default:
        try:
            return str(default.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(default)
    return configured_raw


def sanitize_settings_paths(settings) -> bool:
    """
    Test yollarını proje varsayılanlarına çevir.
    True dönerse en az bir alan düzeltildi.
    """
    from core.logger import setup_logger
    from core.settings import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR

    logger = setup_logger(__name__)
    defaults = {
        "db_path": str(DEFAULT_DATA_DIR / "patterns.db"),
        "cache_dir": str(DEFAULT_CACHE_DIR),
        "faiss_dino_path": str(DEFAULT_DATA_DIR / "faiss_dino.index"),
        "faiss_clip_path": str(DEFAULT_DATA_DIR / "faiss_clip.index"),
    }
    changed = False
    for field, default in defaults.items():
        current = getattr(settings, field, "")
        remapped = remap_legacy_storage_path(current)
        if remapped != current:
            setattr(settings, field, remapped)
            changed = True
            current = remapped
        if is_temporary_test_path(current):
            setattr(settings, field, default)
            changed = True
    # archive_root test yolundaysa temizle
    if is_temporary_test_path(getattr(settings, "archive_root", "")):
        settings.archive_root = ""
        changed = True
    resolved_db = resolve_effective_db_path(settings)
    if resolved_db != getattr(settings, "db_path", ""):
        logger.warning(
            "DB yolu düzeltildi: %s -> %s",
            getattr(settings, "db_path", ""),
            resolved_db,
        )
        settings.db_path = resolved_db
        changed = True
    if changed:
        logger.warning(
            "Geçici test yolu tespit edildi, gerçek proje ayarlarına dönüldü."
        )
    return changed
