"""Visual Memory Phase 1 — clusters are not class labels."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from core.visual_memory.engine import VisualItem, VisualMemoryEngine
from core.visual_memory.knowledge import is_global_knowledge_class, knowledge_pack_info
from core.visual_memory.schema import (
    AUTO_CONFIDENCE_CAP,
    STATUS_CONFIRMED,
    STATUS_DISCOVERED,
    STATUS_LIKELY,
    STATUS_SUPPORTED,
    STATUS_UNKNOWN,
)
from core.visual_memory.store import customer_key, memory_db_path


def _blob_cluster(center: np.ndarray, n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, scale, size=(n, center.size)).astype(np.float32)
    x = center.reshape(1, -1) + noise
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return x


def test_customer_key_isolation():
    assert customer_key("") == "_default"
    assert customer_key("Acme Textile") == "Acme_Textile"
    a = memory_db_path("mem", "A")
    b = memory_db_path("mem", "B")
    assert a != b


def test_synthetic_clusters_are_discovered_not_labeled(tmp_path: Path):
    rng = np.random.default_rng(7)
    dim = 32
    a = _blob_cluster(rng.standard_normal(dim).astype(np.float32), 12, 0.02, rng)
    b = _blob_cluster(rng.standard_normal(dim).astype(np.float32), 12, 0.02, rng)
    c = _blob_cluster(rng.standard_normal(dim).astype(np.float32), 12, 0.02, rng)
    items: list[VisualItem] = []
    fid = 1
    for vecs, obj in ((a, "snake"), (b, "leopard"), (c, "rose")):
        for v in vecs:
            items.append(
                VisualItem(
                    file_id=fid,
                    embedding=v,
                    object_label=obj,
                    family_label="animal_print" if obj != "rose" else "floral",
                    evidence_sources=["clip_embedding"],
                )
            )
            fid += 1
    engine = VisualMemoryEngine(tmp_path)
    result = engine.discover_clusters("custA", items, sim_threshold=0.75, min_size=3)
    discovered = [x for x in result.clusters if x["status"] == STATUS_DISCOVERED]
    assert len(discovered) >= 3
    assert all(x["claimed_class"] == "" for x in result.clusters)
    assert all(x["status"] in (STATUS_UNKNOWN, STATUS_DISCOVERED) for x in result.clusters)
    assert all(x["confidence"] <= AUTO_CONFIDENCE_CAP for x in result.clusters)
    snake_like = [
        x for x in discovered if x["object_distribution"].get("snake", 0) >= 8
    ]
    assert snake_like
    assert snake_like[0]["claimed_class"] == ""
    assert snake_like[0]["status"] == STATUS_DISCOVERED
    store = engine.store_for("custA")
    stats = store.stats()
    assert stats["claimed_class_n"] == 0
    loaded = store.list_clusters()
    assert loaded
    assert "embedding_centroid" in loaded[0]
    assert loaded[0]["member_count"] >= 3


def test_majority_snake_metadata_does_not_confirm(tmp_path: Path):
    rng = np.random.default_rng(1)
    center = rng.standard_normal(16).astype(np.float32)
    vecs = _blob_cluster(center, 20, 0.01, rng)
    items = [
        VisualItem(file_id=i + 1, embedding=vecs[i], object_label="snake")
        for i in range(20)
    ]
    result = VisualMemoryEngine(tmp_path).discover_clusters("x", items, min_size=3)
    big = max(result.clusters, key=lambda c: c["member_count"])
    assert big["status"] == STATUS_DISCOVERED
    assert big["claimed_class"] == ""
    assert big["object_distribution"].get("snake", 0) == 20
    assert big["status"] != STATUS_CONFIRMED
    assert big["status"] != STATUS_SUPPORTED
    assert big["status"] != STATUS_LIKELY


def test_singleton_is_unknown(tmp_path: Path):
    rng = np.random.default_rng(2)
    items = [
        VisualItem(file_id=1, embedding=_norm(rng.standard_normal(8))),
        VisualItem(file_id=2, embedding=_norm(rng.standard_normal(8))),
    ]
    result = VisualMemoryEngine(tmp_path).discover_clusters(
        "solo", items, sim_threshold=0.99, min_size=3
    )
    assert all(c["status"] == STATUS_UNKNOWN for c in result.clusters)


def _norm(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-8)


def test_customers_cannot_read_each_other(tmp_path: Path):
    rng = np.random.default_rng(3)
    v = _norm(rng.standard_normal(8))
    items = [
        VisualItem(file_id=i, embedding=_blob_cluster(v, 1, 0.01, rng)[0])
        for i in range(1, 6)
    ]
    engine = VisualMemoryEngine(tmp_path)
    engine.discover_clusters("alpha", items, min_size=3)
    engine.discover_clusters("beta", [], persist=True)
    a = engine.store_for("alpha").stats()
    b = engine.store_for("beta").stats()
    assert a["clusters"] >= 1
    assert b["clusters"] == 0
    assert Path(a["path"]) != Path(b["path"])


def test_feedback_is_recorded_but_not_applied(tmp_path: Path):
    rng = np.random.default_rng(4)
    vecs = _blob_cluster(rng.standard_normal(8).astype(np.float32), 6, 0.01, rng)
    items = [VisualItem(file_id=i + 1, embedding=vecs[i]) for i in range(6)]
    engine = VisualMemoryEngine(tmp_path)
    result = engine.discover_clusters("fb", items, min_size=3)
    store = engine.store_for("fb")
    cid = result.clusters[0]["cluster_id"]
    fid = store.record_feedback(file_id=1, action="confirm", label="snake", cluster_id=cid)
    assert fid > 0
    again = store.get_cluster(cid)
    assert again["claimed_class"] == ""
    assert again["status"] in (STATUS_UNKNOWN, STATUS_DISCOVERED)
    assert store.feedback_count() == 1


def test_knowledge_pack_is_separate():
    info = knowledge_pack_info()
    assert info["id"] == "vezir_core"
    assert info["version"] == 2
    assert is_global_knowledge_class("snake")
    assert not is_global_knowledge_class("")


def test_visual_memory_does_not_import_search_stack():
    root = Path(__file__).resolve().parents[1] / "core" / "visual_memory"
    banned = {
        "core.search_engine",
        "core.indexer",
        "core.faiss_store",
        "core.query_evidence",
        "core.semantic_pattern_intel",
    }
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in banned, path.name
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in banned
