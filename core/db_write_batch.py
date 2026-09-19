"""SQLite toplu yazma — disk I/O baskısını azaltır."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator

from core.logger import setup_logger

logger = setup_logger(__name__)


class DbWriteBatch:
    """
    Tek bağlantıda N işlem biriktirip commit eder.
    Thread-safe: connection() kullanımı exclusive_lock altında olmalı.
    """

    def __init__(self, db_path: str | Path, *, batch_size: int = 200) -> None:
        self.db_path = str(db_path)
        self.batch_size = max(50, min(500, int(batch_size or 200)))
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._pending = 0
        self._total_commits = 0
        self._total_ops = 0

    def __enter__(self) -> "DbWriteBatch":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def open(self) -> None:
        with self._lock:
            if self._conn is not None:
                return
            conn = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Autocommit kapalı; BEGIN/SAVEPOINT ile yönetilir
            conn.isolation_level = None
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-64000")
            conn.execute("BEGIN")
            self._conn = conn
            self._pending = 0

    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    @contextmanager
    def exclusive(self) -> Generator[sqlite3.Connection, None, None]:
        """Tek writer — nested transaction yok (SAVEPOINT)."""
        with self._lock:
            conn = self.connection()
            conn.execute("SAVEPOINT vezir_op")
            try:
                yield conn
                conn.execute("RELEASE SAVEPOINT vezir_op")
                self._pending += 1
                self._total_ops += 1
                if self._pending >= self.batch_size:
                    self._commit_unlocked()
            except Exception:
                try:
                    conn.execute("ROLLBACK TO SAVEPOINT vezir_op")
                    conn.execute("RELEASE SAVEPOINT vezir_op")
                except Exception:
                    pass
                raise

    def mark_op(self, n: int = 1) -> None:
        """Geriye uyum — exclusive() tercih edilir."""
        with self._lock:
            self._pending += n
            self._total_ops += n
            if self._pending >= self.batch_size:
                self._commit_unlocked()

    def flush(self) -> None:
        with self._lock:
            self._commit_unlocked()

    def _commit_unlocked(self) -> None:
        if self._conn is None or self._pending <= 0:
            self._pending = 0
            return
        try:
            self._conn.execute("COMMIT")
            self._total_commits += 1
            self._conn.execute("BEGIN")
        except Exception as exc:
            logger.warning("DbWriteBatch commit failed: %s", exc)
            try:
                self._conn.execute("ROLLBACK")
            except Exception:
                pass
            try:
                self._conn.execute("BEGIN")
            except Exception:
                pass
        self._pending = 0

    def close(self) -> None:
        with self._lock:
            self._commit_unlocked()
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = None
            logger.info(
                "DbWriteBatch closed ops=%s commits=%s",
                self._total_ops,
                self._total_commits,
            )

    def stats(self) -> dict[str, int]:
        return {
            "ops": self._total_ops,
            "commits": self._total_commits,
            "batch_size": self.batch_size,
        }


@contextmanager
def optional_write_batch(
    db_path: str | Path,
    *,
    enabled: bool,
    batch_size: int = 200,
) -> Iterator[DbWriteBatch | None]:
    if not enabled:
        yield None
        return
    batch = DbWriteBatch(db_path, batch_size=batch_size)
    try:
        batch.open()
        yield batch
    finally:
        batch.close()
