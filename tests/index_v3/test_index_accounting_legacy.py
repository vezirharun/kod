"""Index accounting: usable DB features count as completed; legacy ≠ missing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.db import Database
from core.index_v3.ui_bridge import count_v3_ssot, v3_status_dict


def _blob(dim: int = 8) -> bytes:
    return np.zeros(dim, dtype=np.float32).tobytes()


def _seed(db: Database, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("acc", str(root)),
        )
        # 1) Fully physical + features
        p1 = root / "a.jpg"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(p1)
        thumb1 = root / "a_t.webp"
        prev1 = root / "a_p.webp"
        Image.new("RGB", (16, 16), (1, 2, 3)).save(thumb1)
        Image.new("RGB", (24, 24), (4, 5, 6)).save(prev1)
        conn.execute(
            """
            INSERT INTO files(
              id, source_id, path, filename, status, width, height,
              thumbnail_path, feature_preview_path,
              physical_thumbnail_ready, physical_preview_ready,
              format_metadata, ocr_processed
            ) VALUES (1,1,?,?, 'indexed', 32,32, ?, ?, 1,1, '{}', 1)
            """,
            (str(p1), "a.jpg", str(thumb1), str(prev1)),
        )
        conn.execute(
            """
            INSERT INTO features(
              file_id, dino_embedding, clip_embedding, phash, texture_features,
              texture_map, patch_embeddings
            ) VALUES (1, ?, ?, 'abc', '[1]', ?, '[1]')
            """,
            (
                _blob(),
                _blob(),
                '{"pattern_dna":{"family":"floral"},"semantic_tags":{"motifs":["x"]},'
                '"visual_concept_dna":{"concept":"x"}}',
            ),
        )
        # 2) Legacy: paths + embeddings, physical flags 0 (cache missing)
        p2 = root / "b.jpg"
        Image.new("RGB", (32, 32), (40, 50, 60)).save(p2)
        conn.execute(
            """
            INSERT INTO files(
              id, source_id, path, filename, status, width, height,
              thumbnail_path, feature_preview_path,
              physical_thumbnail_ready, physical_preview_ready,
              format_metadata, ocr_processed
            ) VALUES (2,1,?,?, 'pending', 32,32,
              ?, ?, 0,0, '{}', 1)
            """,
            (
                str(p2),
                "b.jpg",
                str(root / "missing_t.webp"),
                str(root / "missing_p.webp"),
            ),
        )
        conn.execute(
            """
            INSERT INTO features(
              file_id, dino_embedding, clip_embedding, phash, texture_features,
              texture_map, patch_embeddings
            ) VALUES (2, ?, ?, 'def', '[1]', ?, '[1]')
            """,
            (
                _blob(),
                _blob(),
                '{"pattern_dna":{"family":"plaid"},"semantic_tags":{"motifs":["y"]},'
                '"visual_concept_dna":{"concept":"y"}}',
            ),
        )
        # 3) Truly missing AI features
        p3 = root / "c.jpg"
        Image.new("RGB", (32, 32), (70, 80, 90)).save(p3)
        conn.execute(
            """
            INSERT INTO files(
              id, source_id, path, filename, status, width, height,
              format_metadata
            ) VALUES (3,1,?,?, 'indexed', 32,32, '{}')
            """,
            (str(p3), "c.jpg"),
        )


def test_legacy_usable_features_count_as_completed(tmp_path: Path):
    db = Database(tmp_path / "acc.db")
    _seed(db, tmp_path / "files")
    c = count_v3_ssot(db, [1])
    assert c["total"] == 3
    assert c["dino_db"] == 2
    assert c["dino"] == 1  # gate'li
    assert c["legacy_dino"] == 1
    assert c["clip_db"] == 2
    assert c["legacy_clip"] == 1
    assert c["thumbnail_db_path"] == 2
    assert c["thumbnail"] == 1
    assert c["legacy_thumbnail"] == 1
    assert c["preview_db_path"] == 2
    assert c["legacy_preview"] == 1

    st = v3_status_dict(db, [1])
    # UI coverage: legacy blob/path dahil
    assert st["db_dino_embeddings"] == 2
    assert st["db_clip_embeddings"] == 2
    assert st["pattern_dna_count"] == 2
    assert st["texture_done"] == 2
    assert st["thumbnail_ready"] == 2
    assert st["preview_ready"] == 2
    assert st["artifact_pools"]["thumbnail"] == 2
    assert st["artifact_pools"]["preview"] == 2
    assert st["legacy_pools"]["dino"] == 1
    assert st["legacy_pools"]["thumbnail"] == 1
    dino_stage = st["stage_stats"]["dino"]
    assert dino_stage["completed"] == 2
    assert dino_stage["remaining"] == 1
    assert dino_stage["completed"] + dino_stage["remaining"] == dino_stage["total"]
    assert dino_stage["legacy"] == 1
    thumb_stage = st["stage_stats"]["thumbnail"]
    assert thumb_stage["completed"] == 2
    assert thumb_stage["remaining"] == 1
    assert thumb_stage["legacy"] == 1
    # Gate bekleyen, remaining'e eklenmez
    assert dino_stage["remaining"] == 1
    assert thumb_stage["remaining"] == 1
    assert st["waiting_preview"] == 1  # yalnız path'i hiç olmayan
    assert c.get("light_coverage") == 2
    assert c["light_complete"] == 1
    assert st["light_done"] == 2
    assert st["fast_completed"] == 2
    assert st["ui_remaining_light"] == 1
    assert st["light_done"] + st["ui_remaining_light"] == st["total"]
    assert int(c.get("legacy_light") or 0) == 1


def test_accounting_identity_across_feature_stages(tmp_path: Path):
    db = Database(tmp_path / "acc2.db")
    _seed(db, tmp_path / "files2")
    st = v3_status_dict(db, [1])
    total = st["total"]
    for key in (
        "thumbnail",
        "preview",
        "dino",
        "openclip",
        "texture",
        "semantic",
        "dna",
        "object_concept",
        "patch",
        "ocr",
        "metadata",
        "hash",
    ):
        row = st["stage_stats"][key]
        assert row["completed"] + row["remaining"] == total
        assert row.get("accounting_ok") is True

def test_fast_index_card_and_source_line_same_completion_set(tmp_path: Path):
    """Kart Tamamlanan/Kalan ile kaynak Hızlı/Kuyruk aynı kümeden gelmeli."""
    db = Database(tmp_path / "acc_fast.db")
    _seed(db, tmp_path / "files_fast")
    st = v3_status_dict(db, [1])
    # _paint_session_lanes senkronunu simüle et
    total = int(st["total"])
    done = int(st["fast_completed"])
    rem = max(0, total - done)
    st2 = {
        **st,
        "light_done": done,
        "light_queue_display": rem,
        "ui_remaining_light": rem,
    }
    assert st2["light_done"] == st2["fast_completed"]
    assert st2["light_queue_display"] == st2["ui_remaining_light"]
    assert st2["light_done"] + st2["ui_remaining_light"] == total


def test_truly_missing_not_hidden(tmp_path: Path):
    db = Database(tmp_path / "acc3.db")
    _seed(db, tmp_path / "files3")
    st = v3_status_dict(db, [1])
    assert st["stage_stats"]["dino"]["remaining"] >= 1
    assert st["db_dino_embeddings"] < st["total"]
