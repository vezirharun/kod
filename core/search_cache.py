"""Thread-safe bounded cache for repeated visual searches."""

from __future__ import annotations

import copy
import os
import threading
from collections import OrderedDict
from typing import Any

from core.search_models import SearchQuery, SearchResponse


class SearchResponseCache:
    def __init__(self, max_entries: int = 12, max_results: int = 1000):
        self.max_entries = max(1, int(max_entries))
        self.max_results = max(1, int(max_results))
        self._items: OrderedDict[tuple[Any, ...], SearchResponse] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: tuple[Any, ...]) -> SearchResponse | None:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            result = copy.deepcopy(value)
            result.meta = {**(result.meta or {}), "search_cache_hit": True}
            return result

    def put(self, key: tuple[Any, ...], response: SearchResponse) -> None:
        stored = copy.deepcopy(response)
        stored.all_results = list(stored.all_results[: self.max_results])
        allowed = {int(r.file_id) for r in stored.all_results}
        stored.results = [r for r in stored.results if int(r.file_id) in allowed]
        stored.below_threshold_preview = list(stored.below_threshold_preview[:20])
        stored.meta = {**(stored.meta or {}), "search_cache_hit": False}
        with self._lock:
            self._items[key] = stored
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


SEARCH_RESPONSE_CACHE = SearchResponseCache()

# Bump when text ranking/evidence fusion changes. Search-memory only; not index.
TEXT_RETRIEVAL_SCHEMA = "text-retrieval-v12.4.concept-evidence-1"


def search_cache_key(settings, query: SearchQuery) -> tuple[Any, ...]:
    path = (
        os.path.normcase(os.path.abspath(query.image_path)) if query.image_path else ""
    )
    try:
        stat = os.stat(path) if path else None
    except OSError:
        stat = None
    identity = (
        path,
        int(stat.st_size) if stat else 0,
        round(float(stat.st_mtime), 3) if stat else 0.0,
    )
    crop = tuple(int(v) for v in (query.crop_rect or ())) if query.use_crop else ()
    return (
        identity,
        crop,
        query.mode,
        query.text.strip().casefold(),
        str(getattr(settings, "search_mode", "")),
        str(getattr(settings, "color_weight_mode", "")),
        query.customer,
        round(float(query.threshold or 0), 4),
        query.category_path_filter,
        str(getattr(settings, "search_scope", "")),
        tuple(sorted(getattr(settings, "selected_source_ids", []) or [])),
        bool(getattr(settings, "ai_embedding_enabled", False)),
        TEXT_RETRIEVAL_SCHEMA,
    )
