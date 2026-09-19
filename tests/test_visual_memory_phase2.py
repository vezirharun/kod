"""Phase 2 DINO-gated family split — shadow only. Phase 1 engine unchanged."""

from __future__ import annotations

import numpy as np

from core.visual_memory.engine import VisualItem, VisualMemoryEngine
from core.visual_memory.phase2_analyze import (
    Phase2Item,
    apply_dino_gated_family_split,
    phase1_clusters_from_clip,
)
from core.visual_memory.schema import STATUS_DISCOVERED, STATUS_UNKNOWN


def _blob(center: np.ndarray, n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    x = center.reshape(1, -1) + rng.normal(0, scale, size=(n, center.size)).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return x


def _glued_family_items(rng: np.random.Generator) -> list[Phase2Item]:
    clip_c = rng.standard_normal(32).astype(np.float32)
    dino_a = rng.standard_normal(24).astype(np.float32)
    dino_b = rng.standard_normal(24).astype(np.float32)
    clips = _blob(clip_c, 16, 0.02, rng)
    dinos_a = _blob(dino_a, 8, 0.02, rng)
    dinos_b = _blob(dino_b, 8, 0.02, rng)
    items = []
    for i in range(8):
        items.append(
            Phase2Item(
                file_id=i + 1,
                clip=clips[i],
                dino=dinos_a[i],
                family="animal_print",
                obj="leopard",
            )
        )
    for i in range(8):
        items.append(
            Phase2Item(
                file_id=i + 9,
                clip=clips[i + 8],
                dino=dinos_b[i],
                family="floral",
                obj="rose",
            )
        )
    return items


def test_phase1_clip_still_glues_mixed_families(tmp_path):
    rng = np.random.default_rng(11)
    items = _glued_family_items(rng)
    vis = [
        VisualItem(
            file_id=it.file_id,
            embedding=it.clip,
            object_label=it.obj,
            family_label=it.family,
        )
        for it in items
    ]
    result = VisualMemoryEngine(tmp_path).discover_clusters(
        "p1", vis, sim_threshold=0.75, min_size=3
    )
    assert len(result.clusters) == 1
    assert result.clusters[0]["claimed_class"] == ""
    assert result.clusters[0]["status"] in (STATUS_UNKNOWN, STATUS_DISCOVERED)


def test_dino_gate_splits_mixed_family_not_class():
    rng = np.random.default_rng(11)
    items = _glued_family_items(rng)
    p1 = phase1_clusters_from_clip(items, sim_threshold=0.75, min_size=3)
    assert len(p1) == 1
    out = apply_dino_gated_family_split(items, p1)
    assert out["persisted"] is False
    assert out["claimed_class_n"] == 0
    assert out["after"]["mixed_family_clusters"] < out["before"]["mixed_family_clusters"]
    assert out["after"]["largest_cluster"]["member_count"] < out["before"]["largest_cluster"]["member_count"]
    families = {s["family"] for s in out["splits"]}
    assert families == {"animal_print", "floral"}
    assert all(s["claimed_class"] == "" for s in out["splits"])
    assert all(s["status"] == STATUS_DISCOVERED for s in out["splits"])
    assert all(s["split_gain"] >= 0.05 for s in out["splits"])


def test_color_variant_same_family_is_not_split():
    rng = np.random.default_rng(21)
    clip_c = rng.standard_normal(32).astype(np.float32)
    dino_c = rng.standard_normal(24).astype(np.float32)
    clips = _blob(clip_c, 8, 0.02, rng)
    dinos = _blob(dino_c, 8, 0.02, rng)
    items = []
    for i in range(8):
        items.append(
            Phase2Item(
                file_id=i + 1,
                clip=clips[i],
                dino=dinos[i],
                family="plaid_check",
                color="grayscale" if i < 4 else "black_white",
            )
        )
    p1 = phase1_clusters_from_clip(items, sim_threshold=0.75, min_size=3)
    out = apply_dino_gated_family_split(items, p1)
    assert out["splits"] == []
    assert out["after"]["cluster_count"] == out["before"]["cluster_count"]
    assert out["after"]["color_variant_clusters"] >= 1 or out["before"]["color_variant_clusters"] >= 1


def test_low_dino_gain_does_not_split():
    rng = np.random.default_rng(22)
    clip_c = rng.standard_normal(32).astype(np.float32)
    dino_c = rng.standard_normal(24).astype(np.float32)
    clips = _blob(clip_c, 12, 0.02, rng)
    dinos = _blob(dino_c, 12, 0.02, rng)
    items = []
    for i in range(12):
        items.append(
            Phase2Item(
                file_id=i + 1,
                clip=clips[i],
                dino=dinos[i],
                family="animal_print" if i < 6 else "floral",
            )
        )
    p1 = phase1_clusters_from_clip(items, sim_threshold=0.75, min_size=3)
    out = apply_dino_gated_family_split(items, p1)
    assert out["before"]["mixed_family_clusters"] >= 1
    assert out["splits"] == []
    assert out["after"]["mixed_family_clusters"] == out["before"]["mixed_family_clusters"]


def test_phase2_shadow_persist_idempotent_and_isolated(tmp_path):
    import ast
    from pathlib import Path

    from core.visual_memory.phase2_shadow import Phase2ShadowStore

    rng = np.random.default_rng(11)
    items = _glued_family_items(rng)
    p1 = phase1_clusters_from_clip(items, sim_threshold=0.75, min_size=3)
    out = apply_dino_gated_family_split(items, p1)
    prod = tmp_path / "patterns.db"
    faiss = tmp_path / "faiss_clip.index"
    p1_vm = tmp_path / "visual_memory" / "_shadow.db"
    prod.write_bytes(b"PROD")
    faiss.write_bytes(b"FAISS")
    p1_vm.parent.mkdir()
    p1_vm.write_bytes(b"PHASE1")
    prod_m, faiss_m, p1_m = prod.stat().st_mtime, faiss.stat().st_mtime, p1_vm.stat().st_mtime
    store = Phase2ShadowStore(tmp_path / "visual_memory_phase2" / "_shadow.db")
    a = store.replace_snapshot(out["after_clusters"], splits=out["splits"], before=out["before"], after=out["after"])
    b = store.replace_snapshot(out["after_clusters"], splits=out["splits"], before=out["before"], after=out["after"])
    assert a["clusters"] == b["clusters"]
    assert a["ids"] == b["ids"]
    assert b["claimed_class_n"] == 0
    rows = store.list_clusters()
    families = {r["family"] for r in rows if r["parent_cluster_id"]}
    assert "animal_print" in families and "floral" in families
    assert all(r["claimed_class"] == "" for r in rows)
    assert prod.read_bytes() == b"PROD" and prod.stat().st_mtime == prod_m
    assert faiss.read_bytes() == b"FAISS" and faiss.stat().st_mtime == faiss_m
    assert p1_vm.read_bytes() == b"PHASE1" and p1_vm.stat().st_mtime == p1_m
    banned = {"core.search_engine", "core.faiss_store", "core.indexer"}
    shadow_src = Path(__file__).resolve().parents[1] / "core" / "visual_memory" / "phase2_shadow.py"
    tree = ast.parse(shadow_src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module not in banned

