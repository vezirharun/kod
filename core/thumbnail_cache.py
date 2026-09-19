"""Thumbnail LRU cache — RAM QImage + QPixmap, disk/decode tekrar yok."""

from __future__ import annotations

import threading
from collections import OrderedDict

from PySide6.QtGui import QImage, QPixmap


class ThumbnailLRUCache:
    """Thread-safe LRU; key = (file_id, size)."""

    def __init__(self, max_entries: int = 800) -> None:
        self._max = max(100, int(max_entries))
        self._lock = threading.Lock()
        # key -> (QImage, QPixmap|None)
        self._data: OrderedDict[tuple[int, int], tuple[QImage, QPixmap | None]] = OrderedDict()

    def get(self, file_id: int, size: int) -> tuple[QImage | None, QPixmap | None]:
        key = (int(file_id), int(size))
        with self._lock:
            if not isinstance(self._data, OrderedDict):
                self._data = OrderedDict(self._data.items())
            if key not in self._data:
                return None, None
            self._data.move_to_end(key)
            img, pix = self._data[key]
            return img, pix

    def put(self, file_id: int, size: int, image: QImage) -> QPixmap:
        key = (int(file_id), int(size))
        pix = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
        with self._lock:
            self._data[key] = (image, pix if not pix.isNull() else None)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)
        return pix

    def invalidate(self, file_id: int) -> None:
        fid = int(file_id)
        with self._lock:
            for k in list(self._data.keys()):
                if k[0] == fid:
                    del self._data[k]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._data), "max": self._max}


_global_cache: ThumbnailLRUCache | None = None


def get_thumbnail_cache(max_entries: int = 800) -> ThumbnailLRUCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = ThumbnailLRUCache(max_entries)
    return _global_cache
