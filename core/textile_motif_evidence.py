"""Textile motif evidence (not Pattern Index, not segmentation).

Independent SQLite store. RT-DETR-L is measured for COCO-80 only.
YOLO-World supplies open-vocab textile boxes (CLIP encodes class names —
never CLIP-similarity heatmaps). Tests use tmp; optional data path is not
patterns.db. Does not walk the 116K archive. Does not write FAISS.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from core.global_object_intelligence import DetectedObject
from core.object_evidence import (
    CLIP_AS_DETECTOR,
    TEXTILE_OPEN,
    _TR_EXTRA,
    _nms_same_label,
    detect_open_vocab,
    extra_detector_available,
    extra_detector_backend,
)
from core.object_intelligence import DETECTOR_BACKEND, ObjectIntelligence, _box_iou

PRIORITY_CLASSES: tuple[str, ...] = (
    "flower",
    "leaf",
    "rose",
    "leopard",
    "butterfly",
    "snake",
    "zebra",
    "paisley",
    "geometric",
    "floral",
)
EVIDENCE_TYPE = "detection_box"
MATCH_IOU = 0.35
SMALL_AREA_RATIO = 0.06

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS files(
  file_id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS textile_motif_evidence(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id INTEGER NOT NULL,
  motif_class TEXT NOT NULL,
  bbox TEXT NOT NULL,
  confidence REAL NOT NULL,
  source TEXT NOT NULL,
  evidence_type TEXT NOT NULL,
  instance_id TEXT DEFAULT '',
  FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tme_class ON textile_motif_evidence(motif_class);
CREATE INDEX IF NOT EXISTS idx_tme_file ON textile_motif_evidence(file_id);
"""


class TextileMotifEvidenceStore:
    """Isolated motif boxes. SpatialEngine duck-types this like ObjectIndexStore."""

    def __init__(self, db_path: str | Path, *, readonly: bool = False):
        self.path = str(db_path)
        self.readonly = bool(readonly)
        self._missing = False
        if self.readonly:
            self._missing = not Path(self.path).is_file()
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        if self.readonly:
            if self._missing:
                raise sqlite3.OperationalError("textile_motif_evidence missing")
            uri = Path(self.path).resolve().as_posix()
            con = sqlite3.connect(f"file:{uri}?mode=ro", uri=True, timeout=60)
            con.row_factory = sqlite3.Row
            return con
        con = sqlite3.connect(self.path, timeout=60)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init(self) -> None:
        with self._connect() as con:
            con.executescript(SCHEMA)
            con.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('kind','textile_motif_evidence')"
            )
            con.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('clip_as_detector','0')")

    def replace_file_evidence(
        self,
        file_id: int,
        path: str,
        records: Iterable[dict[str, Any]],
    ) -> int:
        if self.readonly:
            raise sqlite3.OperationalError("textile_motif_evidence is read-only")
        rows = list(records)
        with self._connect() as con:
            con.execute("DELETE FROM textile_motif_evidence WHERE file_id=?", (int(file_id),))
            con.execute(
                """INSERT INTO files(file_id,path) VALUES(?,?)
                   ON CONFLICT(file_id) DO UPDATE SET path=excluded.path""",
                (int(file_id), str(path)),
            )
            for rec in rows:
                bbox = rec.get("bbox") or (0, 0, 0, 0)
                con.execute(
                    """INSERT INTO textile_motif_evidence(
                         file_id,motif_class,bbox,confidence,source,evidence_type,instance_id)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        int(file_id),
                        str(rec.get("motif_class") or rec.get("label") or ""),
                        json.dumps([int(x) for x in bbox]),
                        float(rec.get("confidence") or 0),
                        str(rec.get("source") or rec.get("model") or ""),
                        str(rec.get("evidence_type") or EVIDENCE_TYPE),
                        str(rec.get("instance_id") or ""),
                    ),
                )
        return len(rows)

    def _row_as_instance(self, r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["bbox"] = tuple(json.loads(d["bbox"]))
        d["label"] = str(d.get("motif_class") or "")
        d["label_tr"] = _TR_EXTRA.get(d["label"], d["label"])
        return d

    def instances_for_file(self, file_id: int) -> list[dict[str, Any]]:
        if self.readonly and self._missing:
            return []
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM textile_motif_evidence WHERE file_id=? ORDER BY id",
                (int(file_id),),
            ).fetchall()
        return [self._row_as_instance(r) for r in rows]

    def object_count(self, file_id: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for inst in self.instances_for_file(file_id):
            lab = str(inst.get("label") or "")
            if lab:
                counts[lab] = counts.get(lab, 0) + 1
        return counts

    def instances_for_files(self, file_ids: Iterable[int]) -> dict[int, list[dict[str, Any]]]:
        ids = [int(x) for x in file_ids]
        out: dict[int, list[dict[str, Any]]] = {i: [] for i in ids}
        if not ids or (self.readonly and self._missing):
            return out
        placeholders = ",".join("?" * len(ids))
        with self._connect() as con:
            rows = con.execute(
                f"""SELECT * FROM textile_motif_evidence
                    WHERE file_id IN ({placeholders}) ORDER BY file_id, id""",
                ids,
            ).fetchall()
        for r in rows:
            d = self._row_as_instance(r)
            out.setdefault(int(d["file_id"]), []).append(d)
        return out

    def file_paths(self, file_ids: Iterable[int]) -> dict[int, str]:
        ids = [int(x) for x in file_ids]
        if not ids or (self.readonly and self._missing):
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._connect() as con:
            rows = con.execute(
                f"SELECT file_id, path FROM files WHERE file_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {int(r["file_id"]): str(r["path"] or "") for r in rows}

    def search_label(self, labels: Iterable[str], *, min_count: int = 1, limit: int = 400) -> list[int]:
        wanted = {str(x) for x in labels if str(x)}
        if not wanted or (self.readonly and self._missing):
            return []
        placeholders = ",".join("?" * len(wanted))
        with self._connect() as con:
            rows = con.execute(
                f"""SELECT file_id, COUNT(*) AS n FROM textile_motif_evidence
                    WHERE motif_class IN ({placeholders})
                    GROUP BY file_id HAVING n >= ?
                    ORDER BY n DESC LIMIT ?""",
                (*wanted, int(min_count), int(limit)),
            ).fetchall()
        return [int(r["file_id"]) for r in rows]

    def search_labels_and(self, label_groups: list, *, limit: int = 400) -> list[int]:
        if not label_groups or (self.readonly and self._missing):
            return []
        sets: list[set[int]] = []
        for group in label_groups:
            if isinstance(group, dict):
                labels = group.get("labels") or {group.get("label")}
                min_count = int(group.get("min_count") or 1)
            else:
                labels, min_count = group, 1
            sets.append(set(self.search_label(labels, min_count=min_count, limit=max(limit, 4000))))
        if not sets:
            return []
        return list(set.intersection(*sets))[: int(limit)]


class TextileMotifEvidenceLayer:
    """Detect textile classes with YOLO-World; persist into an isolated store."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        min_confidence: float = 0.20,
        max_detections: int = 50,
    ):
        self.db_path = str(db_path)
        self.store = TextileMotifEvidenceStore(self.db_path)
        self.min_confidence = float(min_confidence)
        self.max_detections = max(1, min(200, int(max_detections)))
        self._coco = ObjectIntelligence(
            str(Path(self.db_path).with_name("rtdetr_measure_only.db")),
            min_confidence=0.45,
            max_detections=self.max_detections,
        )

    def detect_rtdetr(self, image_path: str) -> list[DetectedObject]:
        if not self._coco.detector_available():
            return []
        return self._coco.detect(image_path)

    def detect_yoloworld(self, image_path: str) -> list[DetectedObject]:
        if CLIP_AS_DETECTOR:
            return []
        dets = detect_open_vocab(
            image_path,
            min_confidence=self.min_confidence,
            max_detections=self.max_detections,
        )
        textile = [d for d in dets if d.label in TEXTILE_OPEN]
        return _nms_same_label(textile, iou_th=0.55)

    def detect(self, image_path: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        coco = self.detect_rtdetr(image_path)
        t1 = time.perf_counter()
        extra = self.detect_yoloworld(image_path) if extra_detector_available() else []
        t2 = time.perf_counter()
        return {
            "rtdetr": coco,
            "yoloworld": extra,
            "textile": extra,
            "clip_as_detection": False,
            "rtdetr_backend": self._coco.detector_backend() or DETECTOR_BACKEND,
            "yoloworld_backend": extra_detector_backend(),
            "rtdetr_ms": round((t1 - t0) * 1000, 1),
            "yoloworld_ms": round((t2 - t1) * 1000, 1),
        }

    def persist(self, file_id: int, image_path: str, objects: list[DetectedObject] | None = None) -> int:
        if objects is None:
            objects = self.detect_yoloworld(image_path)
        rows = []
        source = extra_detector_backend() if extra_detector_available() else "unavailable"
        objects = _nms_same_label([d for d in objects if d.label in TEXTILE_OPEN], iou_th=0.55)
        counters: dict[str, int] = {}
        for d in objects:
            counters[d.label] = counters.get(d.label, 0) + 1
            iid = d.instance_id or f"{d.label}_{counters[d.label]:02d}"
            rows.append(
                {
                    "motif_class": d.label,
                    "bbox": d.bbox,
                    "confidence": d.confidence,
                    "source": source,
                    "evidence_type": EVIDENCE_TYPE,
                    "instance_id": iid,
                }
            )
        return self.store.replace_file_evidence(int(file_id), str(image_path), rows)


def match_preds(
    gt_boxes: list[tuple[str, tuple[int, int, int, int]]],
    preds: list[DetectedObject],
    *,
    iou_th: float = MATCH_IOU,
) -> dict[str, Any]:
    """Greedy IoU match per class. GT is independent — never taken from preds."""
    used: set[int] = set()
    tp = 0
    pairs = []
    for lab, box in gt_boxes:
        best_i, best_iou = -1, 0.0
        for i, p in enumerate(preds):
            if i in used or p.label != lab:
                continue
            iou = _box_iou(box, p.bbox)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= iou_th:
            used.add(best_i)
            tp += 1
            pairs.append((lab, best_iou, preds[best_i].confidence))
    fp = len(preds) - len(used)
    fn = len(gt_boxes) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "pairs": pairs}


def class_metrics(
    per_class: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lab, s in per_class.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out[lab] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_conf": round(s["conf_sum"] / tp, 4) if tp else 0.0,
            "mean_iou": round(s.get("iou_sum", 0.0) / tp, 4) if tp else 0.0,
            "small_tp": s["small_tp"],
            "small_gt": s["small_gt"],
            "small_success": round(s["small_tp"] / s["small_gt"], 4) if s["small_gt"] else 0.0,
            "fp_rate": round(fp / (tp + fp), 4) if (tp + fp) else 0.0,
            "latency_ms_sum": round(s["lat_ms"], 1),
        }
    return out
