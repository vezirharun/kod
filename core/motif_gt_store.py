"""Motif Learning V1 — user feedback → image-level GT (INDEX FROZEN).

Writes motif-data/user_gt.json + learning_manifest.json, optionally
search_memory.db. Never patterns.db / FAISS / Pattern Index. Never trains.
AI predictions are not GT unless the user confirms (Doğru / Düzenle / remap).
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.textile_motif_v4 import MOTIF_CLASSES, MOTIF_DATA_DIR

_LOCK = threading.Lock()

VERSION = "motif_learning_v1"

# Suggested minima (V4). Placeholder only — not a train trigger.
MIN_GATE: dict[str, int] = {
    "flower": 40,
    "leaf": 40,
    "rose": 30,
    "leopard": 40,
    "zebra": 20,
    "snake": 20,
    "butterfly": 20,
    "paisley": 20,
    "geometric": 20,
}

_ALIAS = {
    "flower": "flower",
    "floral": "flower",
    "cicek": "flower",
    "çiçek": "flower",
    "leaf": "leaf",
    "yaprak": "leaf",
    "rose": "rose",
    "gul": "rose",
    "gül": "rose",
    "leopard": "leopard",
    "leopar": "leopard",
    "zebra": "zebra",
    "snake": "snake",
    "yilan": "snake",
    "yılan": "snake",
    "butterfly": "butterfly",
    "kelebek": "butterfly",
    "paisley": "paisley",
    "geometric": "geometric",
    "geometrik": "geometric",
}


def default_gt_path() -> Path:
    return MOTIF_DATA_DIR / "user_gt.json"


def default_manifest_path() -> Path:
    return MOTIF_DATA_DIR / "learning_manifest.json"


def motif_class_from_any(*parts: Any) -> str | None:
    chunks: list[str] = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if s:
            chunks.append(s)
    if not chunks:
        return None
    try:
        from core.textile_terms import normalize_turkish

        blob = normalize_turkish(" ".join(chunks))
    except Exception:
        blob = " ".join(chunks).lower()
    blob = blob.replace("_", " ").replace("-", " ").replace("/", " ")
    tokens = [t for t in re.split(r"[^a-z0-9]+", blob) if t]
    joined = " ".join(tokens)
    for key in sorted(_ALIAS, key=len, reverse=True):
        kn = key.replace("ü", "u").replace("ı", "i").replace("ç", "c").replace("ğ", "g")
        if kn in tokens or key in tokens:
            return _ALIAS[key]
        if f" {kn} " in f" {joined} ":
            return _ALIAS[key]
    return None


def ingest_ai_prediction(*_a: Any, **_k: Any) -> None:
    """Never persist AI-only labels as ground truth."""
    return None


def ingest_user_motif_label(
    *,
    file_id: int,
    path: str = "",
    filename: str = "",
    user_class: str | None = None,
    rejected_class: str | None = None,
    source: str = "user_feedback",
    gt_path: Path | None = None,
    db_path: str = "",
) -> dict[str, Any] | None:
    """Append/merge one user label. Latest user label wins per image."""
    user_class = _norm_cls(user_class)
    rejected_class = _norm_cls(rejected_class)
    if not user_class and not rejected_class:
        return None
    target = Path(gt_path) if gt_path else None
    if target is None:
        import os

        env = (os.environ.get("VEZIR_MOTIF_GT_PATH") or "").strip()
        target = Path(env) if env else default_gt_path()
    with _LOCK:
        data = _load(target)
        example = _merge_example(
            data,
            file_id=int(file_id or 0),
            path=path or "",
            filename=filename or (Path(path).name if path else ""),
            user_class=user_class,
            rejected_class=rejected_class,
            source=source,
        )
        _recount(data)
        _save(target, data)
        _write_manifest(target.parent / "learning_manifest.json", data)
    if db_path:
        _mirror_search_memory(db_path, example)
    return example


def ingest_from_store_record(
    db: Any,
    action: str,
    query_path: str,
    result_file_id: int,
    label: str,
    *,
    gt_path: Path | None = None,
) -> dict[str, Any] | None:
    if str(action or "") != "ai_category_correct":
        return None
    rec = _file_rec(db, result_file_id)
    user_class = motif_class_from_any(label)
    if not user_class:
        return None
    return ingest_user_motif_label(
        file_id=int(result_file_id),
        path=str(rec.get("path") or ""),
        filename=str(rec.get("filename") or ""),
        user_class=user_class,
        source="inspector_dogru",
        gt_path=gt_path,
        db_path=str(getattr(db, "db_path", "") or ""),
    )


def ingest_from_metadata_overlay(
    db: Any,
    file_id: int,
    overlay: dict[str, Any],
    *,
    query_path: str = "",
    gt_path: Path | None = None,
) -> dict[str, Any] | None:
    rec = _file_rec(db, file_id)
    user_class = motif_class_from_any(
        (overlay or {}).get("category_path"),
        (overlay or {}).get("animal_print_type"),
        (overlay or {}).get("pattern_family"),
        (overlay or {}).get("child"),
        (overlay or {}).get("parent"),
    )
    rejected = motif_class_from_any(query_path)
    if not user_class:
        return None
    return ingest_user_motif_label(
        file_id=int(file_id),
        path=str(rec.get("path") or ""),
        filename=str(rec.get("filename") or ""),
        user_class=user_class,
        rejected_class=rejected if rejected != user_class else None,
        source="metadata_edit",
        gt_path=gt_path,
        db_path=str(getattr(db, "db_path", "") or ""),
    )


def ingest_from_wrong_match(
    *,
    file_id: int,
    path: str = "",
    filename: str = "",
    query_path: str = "",
    category_path: str = "",
    pattern_family: str = "",
    animal_print_type: str = "",
    reject_query_family: str = "",
    tag: str = "",
    db_path: str = "",
    gt_path: Path | None = None,
) -> dict[str, Any] | None:
    # Motif class must be in the *correction*, not merely the query string.
    user_class = motif_class_from_any(
        category_path, animal_print_type, pattern_family, tag
    )
    if not user_class:
        return None
    rejected = motif_class_from_any(reject_query_family, query_path)
    return ingest_user_motif_label(
        file_id=int(file_id),
        path=path,
        filename=filename,
        user_class=user_class,
        rejected_class=rejected if rejected != user_class else None,
        source="apply_wrong_match",
        gt_path=gt_path,
        db_path=db_path,
    )


def load_user_gt(gt_path: Path | None = None) -> dict[str, Any]:
    return _load(Path(gt_path) if gt_path else default_gt_path())


def enough_to_train(data: dict[str, Any] | None = None) -> tuple[bool, str]:
    data = data or _load(default_gt_path())
    pos = data.get("positive_images_by_class") or {}
    n = int(data.get("n_training_examples_merged_images") or 0)
    gaps = []
    for cls, need in MIN_GATE.items():
        have = int(pos.get(cls) or 0)
        if have < need:
            gaps.append(f"{cls}:{have}/{need}")
    if n <= 0:
        return False, "No user GT images."
    if gaps:
        return False, (
            f"User GT only (no invented boxes). Below min-gate placeholder: "
            f"{', '.join(gaps)}. Candidates/AI preds do not count."
        )
    return False, (
        "Class counts meet min-gate placeholder but labels are image-level "
        "(no human boxes). Detector training still NO."
    )


def _norm_cls(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    v = _ALIAS.get(v, v)
    return v if v in MOTIF_CLASSES else motif_class_from_any(value)


def _file_rec(db: Any, file_id: int) -> dict[str, Any]:
    try:
        rec = db.get_file_by_id(int(file_id)) if db is not None else None
    except Exception:
        rec = None
    return rec if isinstance(rec, dict) else {}


def _empty(now: str) -> dict[str, Any]:
    return {
        "version": VERSION,
        "generated_at": now,
        "isolated_from_pattern_index": True,
        "note": (
            "Image-level user labels. Not Pattern Index. No invented bboxes. "
            "AI proposals are not GT."
        ),
        "classes": list(MOTIF_CLASSES),
        "mapping_rule": "AI: snake → User: leopard => leopard positive, snake negative",
        "ai_predictions_are_not_gt": True,
        "examples": [],
        "n_training_examples_merged_images": 0,
        "positive_images_by_class": {c: 0 for c in MOTIF_CLASSES},
        "negative_images_by_class": {c: 0 for c in MOTIF_CLASSES},
    }


def _load(path: Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    if not path.exists():
        return _empty(now)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty(now)
    if not isinstance(data, dict):
        return _empty(now)
    data.setdefault("examples", [])
    data["isolated_from_pattern_index"] = True
    data["ai_predictions_are_not_gt"] = True
    data["version"] = VERSION
    return data


def _example_key(ex: dict[str, Any]) -> str:
    fid = int(ex.get("file_id") or 0)
    if fid > 0:
        return f"id:{fid}"
    p = str(ex.get("path") or "").replace("\\", "/").lower()
    return f"path:{p}" if p else ""


def _merge_example(
    data: dict[str, Any],
    *,
    file_id: int,
    path: str,
    filename: str,
    user_class: str | None,
    rejected_class: str | None,
    source: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    incoming = {
        "file_id": int(file_id or 0),
        "path": path,
        "filename": filename,
        "positive_classes": [user_class] if user_class else [],
        "negative_classes": [rejected_class] if rejected_class else [],
        "sources": [source],
        "is_gt": True,
        "bbox": None,
        "bbox_invented": False,
        "updated_at": now,
    }
    key = _example_key(incoming)
    examples: list[dict[str, Any]] = list(data.get("examples") or [])
    idx = -1
    for i, ex in enumerate(examples):
        if _example_key(ex) == key and key:
            idx = i
            break
    if idx < 0:
        examples.append(incoming)
        data["examples"] = examples
        data["generated_at"] = now
        return incoming

    prev = dict(examples[idx])
    pos = {str(x) for x in (prev.get("positive_classes") or []) if x}
    neg = {str(x) for x in (prev.get("negative_classes") or []) if x}
    if user_class:
        for old in list(pos):
            if old != user_class:
                neg.add(old)
        pos = {user_class}
        neg.discard(user_class)
    if rejected_class and rejected_class != user_class:
        neg.add(rejected_class)
        pos.discard(rejected_class)
    src = list(prev.get("sources") or [])
    if source not in src:
        src.append(source)
    merged = {
        **prev,
        "file_id": int(file_id or prev.get("file_id") or 0),
        "path": path or prev.get("path") or "",
        "filename": filename or prev.get("filename") or "",
        "positive_classes": sorted(pos),
        "negative_classes": sorted(neg),
        "sources": src,
        "is_gt": True,
        "bbox": None,
        "bbox_invented": False,
        "updated_at": now,
    }
    examples[idx] = merged
    data["examples"] = examples
    data["generated_at"] = now
    return merged


def _recount(data: dict[str, Any]) -> None:
    pos_c = {c: 0 for c in MOTIF_CLASSES}
    neg_c = {c: 0 for c in MOTIF_CLASSES}
    remaps: list[Any] = []
    for ex in data.get("examples") or []:
        pos = [c for c in (ex.get("positive_classes") or []) if c in pos_c]
        neg = [c for c in (ex.get("negative_classes") or []) if c in neg_c]
        for c in pos:
            pos_c[c] += 1
        for c in neg:
            neg_c[c] += 1
        if "leopard" in pos and "snake" in neg:
            remaps.append(ex.get("file_id"))
    data["positive_images_by_class"] = pos_c
    data["negative_images_by_class"] = neg_c
    data["n_training_examples_merged_images"] = len(data.get("examples") or [])
    data["snake_to_leopard_remaps"] = remaps
    data["mapping_rule"] = (
        "AI: snake → User: leopard => leopard positive, snake negative"
    )


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    ok, why = enough_to_train(data)
    pos = data.get("positive_images_by_class") or {}
    manifest = {
        "version": VERSION,
        "schema": "vezir.motif_learning_v1.user_gt",
        "isolated_from_pattern_index": True,
        "ai_predictions_are_not_gt": True,
        "bbox_invented": False,
        "n_user_gt_images": int(data.get("n_training_examples_merged_images") or 0),
        "positive_images_by_class": pos,
        "negative_images_by_class": data.get("negative_images_by_class") or {},
        "min_gate": dict(MIN_GATE),
        "min_gate_is_placeholder": True,
        "enough_to_train": False,
        "enough_to_train_honest": ok,
        "enough_why": why,
        "index_frozen": True,
        "writes_patterns_db": False,
        "writes_faiss": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mirror_search_memory(db_path: str, example: dict[str, Any]) -> None:
    if not db_path:
        return
    try:
        from core.search_memory import memory_db_path

        mem = memory_db_path(db_path)
    except Exception:
        return
    if not mem or Path(mem).name.lower() == "patterns.db":
        return
    try:
        Path(mem).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(mem), timeout=10) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS motif_user_gt (
                    file_id INTEGER PRIMARY KEY,
                    path TEXT DEFAULT '',
                    positive_classes TEXT DEFAULT '[]',
                    negative_classes TEXT DEFAULT '[]',
                    source TEXT DEFAULT 'user',
                    updated_at TEXT DEFAULT ''
                )
                """
            )
            con.execute(
                """
                INSERT INTO motif_user_gt(
                    file_id, path, positive_classes, negative_classes,
                    source, updated_at
                ) VALUES (?,?,?,?,?,?)
                ON CONFLICT(file_id) DO UPDATE SET
                    path=excluded.path,
                    positive_classes=excluded.positive_classes,
                    negative_classes=excluded.negative_classes,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (
                    int(example.get("file_id") or 0),
                    str(example.get("path") or ""),
                    json.dumps(example.get("positive_classes") or []),
                    json.dumps(example.get("negative_classes") or []),
                    str((example.get("sources") or ["user"])[-1]),
                    str(example.get("updated_at") or ""),
                ),
            )
            con.commit()
    except Exception:
        return
