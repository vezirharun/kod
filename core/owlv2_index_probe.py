"""OWLv2 index-time production probe. Sidecar only. Never writes production indexes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from PIL import Image

from core.open_vocab_object import OVD_MAX_BOX_AREA, filter_box_area, nms_boxes, textile_blocks_ovd
from core.search_evidence_gate import print_blocks_zero_shot, score_zero_shot_gate

# Packed OWL texts only — EXACT COCO / jewelry / lips are not OWL work.
OWL_VOCAB: tuple[str, ...] = ("crow", "butterfly", "cherry", "eagle")
PROBE_VOCAB = OWL_VOCAB
EXACT_CONCEPTS = frozenset({"bird", "car", "cat", "dog", "apple"})
OPEN_CONCEPTS = frozenset({"crow", "eagle", "butterfly", "cherry", "necklace", "jewelry"})
PATTERN_ROLES = frozenset({"leopard", "snake", "snakeskin", "animal_print"})
LIPS_ROLES = frozenset({"lips", "face"})
JEWELRY_ROLES = frozenset({"jewelry"})
PATTERN_LABELS = frozenset({"leopard", "snake", "snakeskin", "floral"})
OWL_CONF_MIN = 0.10
OWL_MAX_SIDE = 768
CLIP_CROP_PAD = 0.04
MAX_PIXELS = 12_000_000
SCHEMA_VERSION = "owlv2-index-probe-v1"
_LABEL_VOCAB = OWL_VOCAB + ("necklace", "jewelry", "lips", "bird", "car", "cat", "dog", "apple")

DetectFn = Callable[[Image.Image, list[str], float], list[dict[str, Any]]]
ClipFn = Callable[[Image.Image, list[float], str], dict[str, Any]]


def normalize_bbox(xyxy: list[float], width: int, height: int) -> list[float]:
    w = max(1.0, float(width or 1))
    h = max(1.0, float(height or 1))
    x1, y1, x2, y2 = [float(x) for x in xyxy]
    return [
        max(0.0, min(1.0, x1 / w)),
        max(0.0, min(1.0, y1 / h)),
        max(0.0, min(1.0, x2 / w)),
        max(0.0, min(1.0, y2 / h)),
    ]


def canonicalize_label(raw: str) -> str:
    lab = str(raw or "").strip().lower().rstrip(".")
    if lab in _LABEL_VOCAB:
        return lab
    for v in _LABEL_VOCAB:
        if v in lab or lab in v:
            return v
    return lab


def skip_owl_reason(role: str) -> str:
    r = str(role or "")
    if r in PATTERN_ROLES:
        return "pattern_lane_skip_owl"
    if r in LIPS_ROLES:
        return "lips_deferred_face_crop"
    if r in EXACT_CONCEPTS:
        return "exact_lane_no_open_vocab"
    if r in JEWELRY_ROLES:
        return "jewelry_clip_family_skip_owl"
    return ""


def prepare_owl_image(im: Image.Image, max_side: int = OWL_MAX_SIDE) -> tuple[Image.Image, float, float]:
    w, h = im.size
    m = max(w, h)
    if m <= int(max_side):
        return im, 1.0, 1.0
    scale = float(max_side) / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = im.resize((nw, nh), Image.BILINEAR)
    return small, float(w) / float(nw), float(h) / float(nh)


def scale_xyxy(xyxy: list[float], sx: float, sy: float) -> list[float]:
    x1, y1, x2, y2 = [float(x) for x in xyxy]
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def bbox_key(norm: list[float]) -> str:
    return ",".join(f"{float(x):.4f}" for x in norm)


def assert_probe_safe(settings: Any) -> None:
    if bool(getattr(settings, "open_vocab_object_enabled", False)):
        raise RuntimeError("open_vocab_object_enabled must stay false")


def production_fingerprint(settings: Any) -> dict[str, Any]:
    paths = [
        Path(settings.object_db_path),
        Path(settings.faiss_dino_path),
        Path(settings.faiss_clip_path),
        Path(settings.db_path),
    ]
    out: dict[str, Any] = {
        "open_vocab_object_enabled": bool(settings.open_vocab_object_enabled),
    }
    for p in paths:
        if not p.is_file():
            out[str(p)] = None
            continue
        st = p.stat()
        h = hashlib.sha256()
        with p.open("rb") as fh:
            h.update(fh.read(65536))
            if st.st_size > 65536:
                fh.seek(max(0, st.st_size - 65536))
                h.update(fh.read(65536))
        out[str(p)] = {
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "edge_sha256": h.hexdigest(),
        }
    return out


@dataclass
class ProbeStore:
    path: Path
    conn: sqlite3.Connection = field(init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS files(
              file_id INTEGER PRIMARY KEY,
              path TEXT NOT NULL,
              filename TEXT,
              role TEXT,
              pattern_family TEXT,
              texture_map TEXT,
              status TEXT NOT NULL DEFAULT 'pending',
              attempt INTEGER NOT NULL DEFAULT 0,
              error TEXT,
              owl_ms REAL,
              clip_ms REAL,
              total_ms REAL,
              n_boxes INTEGER DEFAULT 0,
              n_accepted INTEGER DEFAULT 0,
              n_rejected INTEGER DEFAULT 0,
              n_open_vocab INTEGER DEFAULT 0,
              n_rejected_textile INTEGER DEFAULT 0,
              n_rejected_clip INTEGER DEFAULT 0,
              owl_ran INTEGER DEFAULT 0,
              processed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS detections(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              file_id INTEGER NOT NULL,
              concept TEXT NOT NULL,
              bbox TEXT NOT NULL,
              owl_confidence REAL,
              clip_target REAL,
              clip_rival REAL,
              clip_margin REAL,
              textile_score REAL,
              pattern_verdict TEXT,
              final_verdict TEXT NOT NULL,
              evidence_type TEXT NOT NULL,
              UNIQUE(file_id, concept, bbox)
            );
            """
        )
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
            (key, value),
        )

    def get_meta(self, key: str) -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return "" if row is None else str(row[0] or "")

    def enqueue(self, rec: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO files(file_id, path, filename, role, pattern_family, texture_map, status)
            VALUES (?,?,?,?,?,?,'pending')
            """,
            (
                int(rec["file_id"]),
                str(rec["path"]),
                str(rec.get("filename") or ""),
                str(rec.get("role") or ""),
                str(rec.get("pattern_family") or ""),
                json.dumps(rec.get("texture_map") or {}, ensure_ascii=False)
                if not isinstance(rec.get("texture_map"), str)
                else rec.get("texture_map"),
            ),
        )
        self.conn.commit()

    def pending(self, include_error: bool = True) -> list[dict[str, Any]]:
        q = "SELECT * FROM files WHERE status='pending'"
        if include_error:
            q = "SELECT * FROM files WHERE status IN ('pending','error')"
        return [dict(r) for r in self.conn.execute(q).fetchall()]

    def file_row(self, file_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM files WHERE file_id=?", (int(file_id),)).fetchone()
        return dict(row) if row else None

    def detections(self, file_id: int | None = None) -> list[dict[str, Any]]:
        if file_id is None:
            rows = self.conn.execute("SELECT * FROM detections ORDER BY id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM detections WHERE file_id=? ORDER BY id", (int(file_id),)
            ).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        n_files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_done = self.conn.execute("SELECT COUNT(*) FROM files WHERE status='done'").fetchone()[0]
        n_err = self.conn.execute("SELECT COUNT(*) FROM files WHERE status='error'").fetchone()[0]
        n_det = self.conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        n_ov = self.conn.execute(
            "SELECT COUNT(*) FROM detections WHERE evidence_type='OPEN_VOCAB_OBJECT'"
        ).fetchone()[0]
        n_tex = self.conn.execute(
            "SELECT COUNT(*) FROM detections WHERE final_verdict='rejected_textile'"
        ).fetchone()[0]
        n_clip = self.conn.execute(
            "SELECT COUNT(*) FROM detections WHERE final_verdict='rejected_clip'"
        ).fetchone()[0]
        n_boxes = self.conn.execute("SELECT COALESCE(SUM(n_boxes),0) FROM files").fetchone()[0]
        return {
            "files": int(n_files),
            "done": int(n_done),
            "error": int(n_err),
            "detections": int(n_det),
            "open_vocab": int(n_ov),
            "rejected_textile": int(n_tex),
            "rejected_clip": int(n_clip),
            "owl_boxes": int(n_boxes),
        }


def _dna_ns(rec: dict[str, Any]) -> SimpleNamespace:
    fam = str(rec.get("pattern_family") or "")
    tm = rec.get("texture_map") or {}
    if isinstance(tm, str):
        try:
            tm = json.loads(tm) if tm else {}
        except Exception:
            tm = {}
    if not isinstance(tm, dict):
        tm = {}
    return SimpleNamespace(pattern_family=fam, debug={"texture_map": tm})


def _crop(im: Image.Image, xyxy: list[float]) -> Image.Image:
    w, h = im.size
    x1, y1, x2, y2 = [float(x) for x in xyxy]
    nx1 = max(0.0, x1 / w - CLIP_CROP_PAD)
    ny1 = max(0.0, y1 / h - CLIP_CROP_PAD)
    nx2 = min(1.0, x2 / w + CLIP_CROP_PAD)
    ny2 = min(1.0, y2 / h + CLIP_CROP_PAD)
    box = (int(nx1 * w), int(ny1 * h), max(int(nx2 * w), 2), max(int(ny2 * h), 2))
    return im.crop(box)


def judge_box(
    rec: dict[str, Any],
    concept: str,
    box: dict[str, Any],
    clip_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    role = str(rec.get("role") or "")
    ns = _dna_ns(rec)
    clip_t = None if clip_meta is None else clip_meta.get("clip_target")
    clip_r = None if clip_meta is None else clip_meta.get("clip_rival")
    clip_m = None if clip_meta is None else clip_meta.get("clip_margin")
    clip_tex = None if clip_meta is None else clip_meta.get("clip_textile")
    clip_ok = bool(clip_meta and clip_meta.get("ok"))
    pattern_hit = print_blocks_zero_shot(ns, concept) or textile_blocks_ovd(ns, concept)
    textile_veto = bool(pattern_hit)
    if clip_tex is not None and clip_t is not None and clip_tex >= clip_t and concept in {
        "cherry", "crow", "butterfly",
    }:
        textile_veto = True

    if concept in PATTERN_LABELS:
        verdict = "pattern_lane_skip_owl"
        etype = "PATTERN_DNA"
    elif role in LIPS_ROLES or concept == "lips":
        verdict = "lips_deferred_face_crop"
        etype = "LIPS_DEFERRED"
    elif concept in EXACT_CONCEPTS:
        verdict = "exact_lane_no_open_vocab"
        etype = "EXACT_OBJECT"
    elif textile_veto:
        verdict = "rejected_textile"
        etype = "REJECT"
    elif concept in {"jewelry", "necklace"} and not clip_ok:
        verdict = "rejected_clip"
        etype = "REJECT"
    elif not clip_ok:
        verdict = "rejected_clip"
        etype = "REJECT"
    elif concept in OPEN_CONCEPTS:
        verdict = "OPEN_VOCAB_OBJECT"
        etype = "OPEN_VOCAB_OBJECT"
    else:
        verdict = "ignored_label"
        etype = "REJECT"

    return {
        "concept": concept,
        "bbox": box.get("xyxy_norm"),
        "owl_confidence": box.get("confidence"),
        "clip_target": clip_t,
        "clip_rival": clip_r,
        "clip_margin": clip_m,
        "textile_score": clip_tex,
        "pattern_verdict": "veto" if pattern_hit else "pass",
        "final_verdict": verdict,
        "evidence_type": etype,
    }


def process_file(
    store: ProbeStore,
    rec: dict[str, Any],
    detect_fn: DetectFn,
    clip_fn: ClipFn | None = None,
    *,
    threshold: float = OWL_CONF_MIN,
) -> dict[str, Any]:
    file_id = int(rec["file_id"])
    if store.file_row(file_id) is None:
        store.enqueue(rec)
    existing = store.file_row(file_id)
    if existing and existing.get("status") == "done":
        return {"file_id": file_id, "skipped": "already_done", "status": "done"}
    rec = dict(rec)
    if existing:
        for k in ("path", "filename", "role", "pattern_family", "texture_map"):
            if rec.get(k) in (None, "") and existing.get(k) not in (None, ""):
                rec[k] = existing.get(k)
    tm = rec.get("texture_map")
    if isinstance(tm, str):
        try:
            rec["texture_map"] = json.loads(tm) if tm else {}
        except Exception:
            rec["texture_map"] = {}

    store.conn.execute(
        "UPDATE files SET attempt=attempt+1, error=NULL WHERE file_id=?",
        (file_id,),
    )
    store.conn.commit()
    t0 = time.perf_counter()
    owl_ms = 0.0
    clip_ms = 0.0
    owl_ran = 0
    detections: list[dict[str, Any]] = []
    n_boxes = 0
    try:
        role = str(rec.get("role") or existing and existing.get("role") or "")
        skip = skip_owl_reason(role)
        if skip:
            summary = {"owl_ran": 0, "n_boxes": 0, "verdict": skip}
        else:
            path = Path(str(rec.get("path") or (existing or {}).get("path") or ""))
            if not path.is_file():
                raise FileNotFoundError(str(path))
            im = Image.open(path).convert("RGB")
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise RuntimeError(f"image_too_large:{w}x{h}")
            owl_im, sx, sy = prepare_owl_image(im)
            t_owl = time.perf_counter()
            raw = detect_fn(owl_im, list(OWL_VOCAB), float(threshold)) or []
            owl_ms = (time.perf_counter() - t_owl) * 1000.0
            owl_ran = 1
            scaled = []
            for b in raw:
                item = dict(b)
                xy = item.get("xyxy") or []
                if len(xy) == 4:
                    item["xyxy"] = scale_xyxy(xy, sx, sy)
                scaled.append(item)
            kept = filter_box_area(scaled, float(w * h), OVD_MAX_BOX_AREA)
            kept = [b for b in kept if float(b.get("confidence") or 0) >= float(threshold)]
            kept = nms_boxes(kept)
            n_boxes = len(kept)
            for b in kept:
                xy = [float(x) for x in (b.get("xyxy") or [0, 0, 0, 0])]
                if len(xy) != 4:
                    continue
                concept = canonicalize_label(str(b.get("label") or ""))
                if not concept:
                    continue
                norm = normalize_bbox(xy, w, h)
                box = {
                    "label": concept,
                    "confidence": round(float(b.get("confidence") or 0), 4),
                    "xyxy": xy,
                    "xyxy_norm": [round(v, 4) for v in norm],
                    "area_ratio": b.get("bbox_area_ratio"),
                }
                clip_meta = None
                if clip_fn is not None and concept in OPEN_CONCEPTS:
                    t_c = time.perf_counter()
                    clip_meta = clip_fn(im, xy, concept)
                    clip_ms += (time.perf_counter() - t_c) * 1000.0
                judged = judge_box(rec if rec.get("pattern_family") is not None else (existing or rec), concept, box, clip_meta)
                detections.append(judged)
            summary = {"owl_ran": owl_ran, "n_boxes": n_boxes}

        n_ov = sum(1 for d in detections if d["evidence_type"] == "OPEN_VOCAB_OBJECT")
        n_tex = sum(1 for d in detections if d["final_verdict"] == "rejected_textile")
        n_clip = sum(1 for d in detections if d["final_verdict"] == "rejected_clip")
        n_acc = n_ov
        n_rej = len(detections) - n_acc
        total_ms = (time.perf_counter() - t0) * 1000.0
        now = datetime.now(timezone.utc).isoformat()
        cur = store.conn
        cur.execute("BEGIN")
        cur.execute("DELETE FROM detections WHERE file_id=?", (file_id,))
        for d in detections:
            cur.execute(
                """
                INSERT INTO detections(
                  file_id, concept, bbox, owl_confidence, clip_target, clip_rival,
                  clip_margin, textile_score, pattern_verdict, final_verdict, evidence_type
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    file_id,
                    d["concept"],
                    bbox_key(d["bbox"] or [0, 0, 0, 0]),
                    d.get("owl_confidence"),
                    d.get("clip_target"),
                    d.get("clip_rival"),
                    d.get("clip_margin"),
                    d.get("textile_score"),
                    d.get("pattern_verdict"),
                    d["final_verdict"],
                    d["evidence_type"],
                ),
            )
        cur.execute(
            """
            UPDATE files SET
              status='done', error=NULL, owl_ms=?, clip_ms=?, total_ms=?,
              n_boxes=?, n_accepted=?, n_rejected=?, n_open_vocab=?,
              n_rejected_textile=?, n_rejected_clip=?, owl_ran=?, processed_at=?
            WHERE file_id=?
            """,
            (
                round(owl_ms, 1), round(clip_ms, 1), round(total_ms, 1),
                n_boxes, n_acc, n_rej, n_ov, n_tex, n_clip, owl_ran, now, file_id,
            ),
        )
        cur.commit()
        return {
            "file_id": file_id,
            "status": "done",
            "skipped": "",
            "owl_ms": owl_ms,
            "clip_ms": clip_ms,
            "total_ms": total_ms,
            "n_boxes": n_boxes,
            "n_open_vocab": n_ov,
            "detections": detections,
            **summary,
        }
    except Exception as exc:
        try:
            store.conn.rollback()
        except Exception:
            pass
        store.conn.execute(
            "UPDATE files SET status='error', error=? WHERE file_id=?",
            (str(exc)[:500], file_id),
        )
        store.conn.commit()
        return {"file_id": file_id, "status": "error", "error": str(exc)[:500]}


def process_queue(
    store: ProbeStore,
    records: list[dict[str, Any]],
    detect_fn: DetectFn,
    clip_fn: ClipFn | None = None,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = 0
    for rec in records:
        fid = int(rec["file_id"])
        store.enqueue(rec)
        row = store.file_row(fid) or rec
        if str(row.get("status")) == "done":
            out.append({"file_id": fid, "skipped": "already_done", "status": "done"})
            continue
        merged = {**row, **rec}
        result = process_file(store, merged, detect_fn, clip_fn)
        out.append(result)
        if result.get("status") != "already_done" and not result.get("skipped"):
            n += 1
        if limit is not None and n >= int(limit):
            break
    return out


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def timing_stats(store: ProbeStore) -> dict[str, Any]:
    rows = [dict(r) for r in store.conn.execute(
        "SELECT file_id, filename, total_ms FROM files WHERE status='done' AND total_ms IS NOT NULL"
    ).fetchall()]
    vals = [float(r["total_ms"]) for r in rows]
    slow = max(rows, key=lambda r: float(r["total_ms"] or 0), default=None)
    return {
        "n": len(vals),
        "total_ms": round(sum(vals), 1) if vals else 0.0,
        "mean_ms": round(sum(vals) / len(vals), 1) if vals else None,
        "median_ms": None if not vals else round(float(percentile(vals, 50) or 0), 1),
        "p95_ms": None if not vals else round(float(percentile(vals, 95) or 0), 1),
        "slowest": slow,
    }


def default_clip_fn(fe: Any, text_emb: dict[str, Any]) -> ClipFn:
    import numpy as np
    from core.search_evidence_gate import _ZS_PROMPT, zero_shot_verify_prompts

    def _cos(a, b) -> float:
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    extra_prompts = {
        **_ZS_PROMPT,
        "floral": "a repeating floral textile print, not a real fruit",
        "animal_print": "an animal print textile, not a real animal",
        "textile": "a repeating decorative fabric pattern, not a real object",
        "necklace": _ZS_PROMPT["jewelry"],
    }

    def _fn(im: Image.Image, xyxy: list[float], concept: str) -> dict[str, Any]:
        clip_key = "jewelry" if concept in {"jewelry", "necklace"} else concept
        crop = _crop(im, xyxy)
        cemb = fe._embed_clip_image(crop)
        scores = {k: round(_cos(cemb, text_emb.get(k)), 4) for k in extra_prompts}
        rivals = {k: scores.get(k, 0.0) for k in zero_shot_verify_prompts(clip_key)}
        tgt = float(scores.get(clip_key, 0.0))
        ok, meta = score_zero_shot_gate(clip_key, tgt, rivals)
        return {
            "ok": ok,
            "clip_target": tgt,
            "clip_rival": meta.get("clip_best_rival"),
            "clip_margin": meta.get("clip_margin"),
            "clip_textile": meta.get("clip_textile"),
        }

    return _fn


def select_probe_files(patterns_db: Any, object_db_path: Path, limit: int = 40) -> list[dict[str, Any]]:
    needles = [
        ("crow", "%pngtree-crow-image-black%"),
        ("butterfly", "%OSCAR-STONE-DECOR-KELEBEK%"),
        ("cherry_blossom", "%487-4877576_cherry-blossom%"),
        ("jewelry", "%jewelry608.png%"),
        ("lips", "%14999ec9e4edee1b851307148062.jpg%"),
        ("leopard_floral", "%stock-photo-flowers-and-leopard%"),
        ("leopard", "%leopar%"),
        ("snake", "%yilan%"),
        ("snake", "%snake%"),
        ("floral", "%gül%"),
        ("floral", "%gul%"),
        ("dog", "%bulldog%"),
    ]
    picked: list[dict[str, Any]] = []
    seen: set[int] = set()

    def _add(row: dict[str, Any], role: str) -> None:
        fid = int(row["id"])
        if fid in seen:
            return
        path = Path(str(row.get("path") or ""))
        if not path.is_file():
            return
        try:
            with Image.open(path) as im:
                w, h = im.size
            if w * h > MAX_PIXELS:
                return
        except Exception:
            return
        seen.add(fid)
        picked.append({
            "file_id": fid,
            "path": str(path),
            "filename": row.get("filename") or path.name,
            "role": role,
            "pattern_family": row.get("pattern_family") or "",
            "texture_map": row.get("texture_map"),
        })

    with patterns_db.connect() as conn:
        for role, needle in needles:
            if len(picked) >= limit:
                break
            row = conn.execute(
                """
                SELECT f.id, f.path, f.filename, f.pattern_family, fe.texture_map
                FROM files f JOIN features fe ON fe.file_id=f.id
                WHERE f.status NOT IN ('missing','excluded_internal')
                  AND (f.filename LIKE ? OR f.path LIKE ?)
                  AND f.path IS NOT NULL
                LIMIT 1
                """,
                (needle, needle),
            ).fetchone()
            if row:
                _add(dict(row), role if role != "leopard_floral" else "leopard")
        for fam, role, n in (
            ("floral", "floral", 6),
            ("animal_print", "leopard", 4),
            ("reptile_skin", "snake", 3),
            ("watercolor", "floral", 2),
        ):
            rows = conn.execute(
                """
                SELECT f.id, f.path, f.filename, f.pattern_family, fe.texture_map
                FROM files f JOIN features fe ON fe.file_id=f.id
                WHERE f.status NOT IN ('missing','excluded_internal')
                  AND lower(COALESCE(f.pattern_family,'')) = ?
                  AND f.path IS NOT NULL
                LIMIT ?
                """,
                (fam, n),
            ).fetchall()
            for row in rows:
                if len(picked) >= limit:
                    break
                _add(dict(row), role)
        extra_needles = [
            ("jewelry", "%jewelry%.png"),
            ("jewelry", "%jewelry%.jpg"),
            ("butterfly", "%kelebek%"),
            ("floral", "%çiçek%"),
            ("floral", "%cicek%"),
        ]
        for role, needle in extra_needles:
            rows = conn.execute(
                """
                SELECT f.id, f.path, f.filename, f.pattern_family, fe.texture_map
                FROM files f JOIN features fe ON fe.file_id=f.id
                WHERE f.status NOT IN ('missing','excluded_internal')
                  AND (f.filename LIKE ? OR f.path LIKE ?)
                  AND f.path IS NOT NULL
                LIMIT 3
                """,
                (needle, needle),
            ).fetchall()
            for row in rows:
                if len(picked) >= limit:
                    break
                _add(dict(row), role)

    if object_db_path.is_file():
        oconn = sqlite3.connect(f"file:{object_db_path.as_posix()}?mode=ro", uri=True)
        oconn.row_factory = sqlite3.Row
        try:
            for lab, role in (("car", "car"), ("cat", "cat"), ("dog", "dog"), ("bird", "bird"), ("apple", "apple")):
                rows = oconn.execute(
                    """
                    SELECT of.file_id, of.path
                    FROM object_instances oi
                    JOIN object_files of ON of.file_id = oi.file_id
                    WHERE lower(oi.label)=?
                      AND of.detector LIKE '%rtdetr%'
                    ORDER BY oi.confidence DESC
                    LIMIT 3
                    """,
                    (lab,),
                ).fetchall()
                for r in rows:
                    if len(picked) >= limit:
                        break
                    fid = int(r["file_id"])
                    if fid in seen:
                        continue
                    path = Path(str(r["path"] or ""))
                    if not path.is_file():
                        continue
                    with patterns_db.connect() as conn:
                        prow = conn.execute(
                            """
                            SELECT f.id, f.path, f.filename, f.pattern_family, fe.texture_map
                            FROM files f JOIN features fe ON fe.file_id=f.id
                            WHERE f.id=?
                            """,
                            (fid,),
                        ).fetchone()
                    if prow:
                        _add(dict(prow), role)
                    else:
                        seen.add(fid)
                        picked.append({
                            "file_id": fid,
                            "path": str(path),
                            "filename": path.name,
                            "role": role,
                            "pattern_family": "",
                            "texture_map": {},
                        })
        finally:
            oconn.close()
    return picked[:limit]
