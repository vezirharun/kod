"""Windows recursive delete notifications for watched source roots.

QFileSystemWatcher reports only the directories it was given, and not which
child disappeared. ReadDirectoryChangesW with a subtree watch reports the
relative path and the remove/rename-away action for nested files.
"""

from __future__ import annotations

import os
import threading
from typing import Callable

from core.logger import setup_logger

logger = setup_logger(__name__)

FILE_ACTION_REMOVED = 0x00000002
FILE_ACTION_RENAMED_OLD_NAME = 0x00000004
REMOVED_ACTIONS = frozenset({FILE_ACTION_REMOVED, FILE_ACTION_RENAMED_OLD_NAME})
_ERROR_OPERATION_ABORTED = 995
_ERROR_NOTIFY_ENUM_DIR = 1022


def parse_notify_buffer(data: bytes | memoryview) -> list[tuple[int, str]]:
    """Decode a ReadDirectoryChangesW buffer into ``(action, relative_name)``."""
    out: list[tuple[int, str]] = []
    buf = memoryview(data)
    offset = 0
    size = len(buf)
    while offset + 12 <= size:
        next_off = int.from_bytes(buf[offset : offset + 4], "little")
        action = int.from_bytes(buf[offset + 4 : offset + 8], "little")
        name_len = int.from_bytes(buf[offset + 8 : offset + 12], "little")
        start = offset + 12
        end = start + name_len
        if name_len < 0 or end > size:
            break
        name = bytes(buf[start:end]).decode("utf-16-le", errors="replace").rstrip("\x00")
        if name:
            out.append((action, name.replace("/", "\\")))
        if next_off <= 0:
            break
        offset += next_off
    return out


def removed_paths_from_buffer(root: str, data: bytes | memoryview) -> list[str]:
    """Absolute paths for remove and rename-away records."""
    out: list[str] = []
    for action, name in parse_notify_buffer(data):
        if action not in REMOVED_ACTIONS or not name:
            continue
        out.append(_notify_full_path(root, name))
    return out


def _notify_full_path(root: str, name: str) -> str:
    parts = [part for part in name.replace("/", "\\").split("\\") if part not in ("", ".")]
    if not parts:
        return os.path.normpath(root)
    return os.path.normpath(os.path.join(root, *parts))


class SubtreeDeleteWatch:
    """One blocking ReadDirectoryChangesW thread per source root. No-op off Windows."""

    def __init__(
        self,
        on_removed: Callable[[str], None],
        on_overflow: Callable[[str], None],
    ) -> None:
        self._on_removed = on_removed
        self._on_overflow = on_overflow
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._handles: list[int] = []
        self._lock = threading.Lock()

    def set_roots(self, roots: list[str]) -> bool:
        self.stop()
        cleaned = [str(path) for path in roots if str(path or "").strip()]
        if os.name != "nt" or not cleaned:
            return False
        self._stop = threading.Event()
        with self._lock:
            self._handles = []
            self._threads = []
        started = 0
        for root in cleaned:
            if not os.path.isdir(root):
                continue
            thread = threading.Thread(
                target=_watch_loop,
                args=(
                    root,
                    self._stop,
                    self._handles,
                    self._lock,
                    self._on_removed,
                    self._on_overflow,
                ),
                name="vezir-subtree-delete",
                daemon=True,
            )
            thread.start()
            with self._lock:
                self._threads.append(thread)
            started += 1
        return started > 0

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            handles = list(self._handles)
            threads = list(self._threads)
        _cancel_handles(handles)
        for thread in threads:
            thread.join(0.3)
        with self._lock:
            self._threads = []
            self._handles = []


def _watch_loop(
    root: str,
    stop: threading.Event,
    handles: list[int],
    lock: threading.Lock,
    on_removed: Callable[[str], None],
    on_overflow: Callable[[str], None],
) -> None:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        logger.warning("subtree watch unavailable", exc_info=True)
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.ReadDirectoryChangesW.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel.ReadDirectoryChangesW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    file_list_directory = 0x0001
    share = 0x0001 | 0x0002 | 0x0004
    open_existing = 3
    backup_semantics = 0x02000000
    notify_filter = 0x00000001 | 0x00000002  # file name, dir name
    invalid = ctypes.c_void_p(-1).value

    handle = kernel.CreateFileW(
        root,
        file_list_directory,
        share,
        None,
        open_existing,
        backup_semantics,
        None,
    )
    if not handle or int(handle) == int(invalid or 0):
        logger.warning("Subtree watch open failed: %s", root)
        return
    with lock:
        handles.append(int(handle))
    buffer = ctypes.create_string_buffer(64 * 1024)
    returned = wintypes.DWORD(0)
    try:
        while not stop.is_set():
            ok = kernel.ReadDirectoryChangesW(
                handle,
                ctypes.cast(buffer, ctypes.c_void_p),
                ctypes.sizeof(buffer),
                True,
                notify_filter,
                ctypes.byref(returned),
                None,
                None,
            )
            if stop.is_set():
                break
            if not ok:
                err = ctypes.get_last_error()
                if err == _ERROR_NOTIFY_ENUM_DIR:
                    try:
                        on_overflow(root)
                    except Exception:
                        logger.warning("subtree overflow callback failed", exc_info=True)
                    continue
                if err != _ERROR_OPERATION_ABORTED:
                    logger.warning("Subtree watch stopped for %s (winerror %s)", root, err)
                break
            try:
                payload = buffer.raw[: int(returned.value)]
                for path in removed_paths_from_buffer(root, payload):
                    on_removed(path)
            except Exception:
                logger.warning("subtree remove callback failed", exc_info=True)
    finally:
        kernel.CloseHandle(handle)


def _cancel_handles(handles: list[int]) -> None:
    if os.name != "nt" or not handles:
        return
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel.CancelIoEx.restype = wintypes.BOOL
    for handle in handles:
        try:
            kernel.CancelIoEx(ctypes.c_void_p(int(handle)), None)
        except Exception:
            continue
