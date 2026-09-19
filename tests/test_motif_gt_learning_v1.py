"""Motif Learning V1: user feedback → GT. INDEX FROZEN. No training."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.motif_gt_store import (
    MIN_GATE,
    enough_to_train,
    ingest_ai_prediction,
    ingest_from_wrong_match,
    ingest_user_motif_label,
    load_user_gt,
    motif_class_from_any,
)
from core.settings import DEFAULT_DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
PROD_GT = ROOT / "motif-data" / "user_gt.json"


def _gt(tmp: Path) -> Path:
    p = tmp / "motif-data" / "user_gt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _snap() -> dict:
    return snapshot_index_artifacts(
        db_path=DEFAULT_DATA_DIR / "patterns.db",
        faiss_dino_path=DEFAULT_DATA_DIR / "faiss_dino.index",
        faiss_clip_path=DEFAULT_DATA_DIR / "faiss_clip.index",
        cache_dir="",
    )


def test_write_gt(tmp_path: Path) -> None:
    path = _gt(tmp_path)
    ex = ingest_user_motif_label(
        file_id=42,
        path=str(tmp_path / "a.jpg"),
        filename="a.jpg",
        user_class="paisley",
        source="inspector_dogru",
        gt_path=path,
    )
    assert ex and ex["is_gt"] is True
    assert ex["positive_classes"] == ["paisley"]
    assert ex["bbox"] is None
    data = load_user_gt(path)
    assert data["n_training_examples_merged_images"] == 1
    assert data["positive_images_by_class"]["paisley"] == 1
    assert data["ai_predictions_are_not_gt"] is True
    man = json.loads((path.parent / "learning_manifest.json").read_text(encoding="utf-8"))
    assert man["enough_to_train"] is False
    assert man["min_gate"]["leopard"] == MIN_GATE["leopard"]
    assert man["writes_faiss"] is False
    ok, why = enough_to_train(data)
    assert ok is False
    assert "min-gate" in why.lower() or "below" in why.lower()


def test_remap_snake_to_leopard(tmp_path: Path) -> None:
    path = _gt(tmp_path)
    ingest_user_motif_label(
        file_id=5557,
        path="snake_print.eps",
        filename="snake_print.eps",
        user_class="leopard",
        rejected_class="snake",
        source="apply_wrong_match",
        gt_path=path,
    )
    data = load_user_gt(path)
    ex = data["examples"][0]
    assert ex["positive_classes"] == ["leopard"]
    assert "snake" in ex["negative_classes"]
    assert 5557 in data["snake_to_leopard_remaps"]
    hook = ingest_from_wrong_match(
        file_id=9,
        path="b.jpg",
        filename="b.jpg",
        query_path="snake",
        category_path="Animal Print/Leopard",
        animal_print_type="leopard",
        reject_query_family="snake",
        gt_path=path,
    )
    assert hook["positive_classes"] == ["leopard"]
    assert "snake" in hook["negative_classes"]


def test_dedupe_latest_user_label_wins(tmp_path: Path) -> None:
    path = _gt(tmp_path)
    ingest_user_motif_label(
        file_id=81, path="x.jpg", filename="x.jpg",
        user_class="snake", gt_path=path,
    )
    ingest_user_motif_label(
        file_id=81, path="x.jpg", filename="x.jpg",
        user_class="leopard", rejected_class="snake", gt_path=path,
    )
    data = load_user_gt(path)
    assert data["n_training_examples_merged_images"] == 1
    ex = data["examples"][0]
    assert ex["positive_classes"] == ["leopard"]
    assert "snake" in ex["negative_classes"]
    assert "snake" not in ex["positive_classes"]


def test_user_outranks_ai_never_gt(tmp_path: Path) -> None:
    path = _gt(tmp_path)
    ingest_ai_prediction(file_id=1, ai_class="snake", gt_path=path)
    assert not path.exists()
    ingest_user_motif_label(
        file_id=1, path="y.jpg", filename="y.jpg",
        user_class="leopard", rejected_class="snake", gt_path=path,
    )
    data = load_user_gt(path)
    assert data["examples"][0]["positive_classes"] == ["leopard"]
    assert motif_class_from_any("Animal Print/Leopard") == "leopard"


def test_freeze_sentinel_no_index_faiss(tmp_path: Path) -> None:
    before = freeze_fingerprint(_snap())
    path = _gt(tmp_path)
    db_path = str(tmp_path / "patterns.db")
    ingest_user_motif_label(
        file_id=7,
        path="z.jpg",
        filename="z.jpg",
        user_class="flower",
        gt_path=path,
        db_path=db_path,
    )
    after = freeze_fingerprint(_snap())
    assert before == after
    faiss = {k: v for k, v in after.items() if "faiss" in k.replace("\\", "/").lower()}
    assert faiss == {
        k: v for k, v in before.items() if "faiss" in k.replace("\\", "/").lower()
    }
    assert not (tmp_path / "patterns.db").exists() or (
        Path(db_path).stat().st_size == 0
    )
    mem = tmp_path / "search_memory.db"
    if mem.exists():
        import sqlite3

        con = sqlite3.connect(str(mem))
        n = con.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='motif_user_gt'"
        ).fetchone()[0]
        con.close()
        assert n == 1


def test_existing_13_gt_preserved(tmp_path: Path) -> None:
    src = json.loads(PROD_GT.read_text(encoding="utf-8"))
    ids = sorted(int(e["file_id"]) for e in src["examples"])
    assert len(ids) == 13
    dest = _gt(tmp_path)
    shutil.copy(PROD_GT, dest)
    ingest_user_motif_label(
        file_id=999001,
        path="new_user.jpg",
        filename="new_user.jpg",
        user_class="rose",
        gt_path=dest,
    )
    data = load_user_gt(dest)
    kept = sorted(int(e["file_id"]) for e in data["examples"] if int(e["file_id"]) in set(ids))
    assert kept == ids
    assert all(e.get("is_gt") for e in data["examples"] if int(e["file_id"]) in set(ids))
    live = json.loads(PROD_GT.read_text(encoding="utf-8"))
    assert len(live["examples"]) == 13
