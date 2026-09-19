"""Customer Visual Memory — Phase 1 foundation.

Isolated from SearchEngine ranking, FAISS rebuild, and production patterns.db.
"""

from core.visual_memory.engine import DiscoverResult, VisualItem, VisualMemoryEngine
from core.visual_memory.knowledge import knowledge_pack_info, load_knowledge_pack

KNOWLEDGE_PACK = load_knowledge_pack
from core.visual_memory.schema import (
    STATUS_CONFIRMED,
    STATUS_DISCOVERED,
    STATUS_LIKELY,
    STATUS_SUPPORTED,
    STATUS_UNKNOWN,
)
from core.visual_memory.store import VisualMemoryStore, customer_key

__all__ = [
    "DiscoverResult",
    "KNOWLEDGE_PACK",
    "load_knowledge_pack",
    "STATUS_CONFIRMED",
    "STATUS_DISCOVERED",
    "STATUS_LIKELY",
    "STATUS_SUPPORTED",
    "STATUS_UNKNOWN",
    "VisualItem",
    "VisualMemoryEngine",
    "VisualMemoryStore",
    "customer_key",
    "knowledge_pack_info",
]
