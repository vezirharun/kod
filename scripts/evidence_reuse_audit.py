"""Evidence Reuse Audit — production read-only, no ranking/index changes.

Measures whether search recomputes CLIP/DINO/OCR/DNA/texture or reuses DB/FAISS.

python scripts/evidence_reuse_audit.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.search_engine import SearchEngine
from core.settings import AppSettings

QUERIES = ["leopar", "yılan", "gül", "gucci", "kırmızı gül", "leopar çiçek"]


def _fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size": 0, "mtime": 0}
    st = path.stat()
    return {"path": str(path), "exists": True, "size": int(st.st_size), "mtime": st.st_mtime}


class EvidenceProbe:
    def __init__(self) -> None:
        self.reset()
        self._orig: list[tuple[Any, str, Any]] = []

    def reset(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.ms: dict[str, float] = defaultdict(float)
        self.ttfr_s: float | None = None
        self._t0 = 0.0
        self.db_rows = 0
        self.db_clip = 0
        self.db_dino = 0
        self.db_ocr = 0
        self.db_texture = 0
        self.db_family = 0
        self.db_semantic = 0
        self.faiss_clip_hits = 0
        self.faiss_dino_hits = 0

    def _wrap(self, obj: Any, name: str, channel: str) -> None:
        orig = getattr(obj, name, None)
        if orig is None:
            return
        probe = self

        def wrapped(*args, **kwargs):
            t = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                probe.calls[channel] += 1
                probe.ms[channel] += (time.perf_counter() - t) * 1000.0

        self._orig.append((obj, name, orig))
        setattr(obj, name, wrapped)

    def attach(self, engine: SearchEngine) -> None:
        ex = engine.extractor
        self._wrap(ex, "embed_text", "clip_query_encode")
        self._wrap(ex, "_embed_clip_image", "clip_image_recompute")
        self._wrap(ex, "_embed_dino", "dino_image_recompute")
        self._wrap(ex, "extract_from_path", "extract_from_path")
        self._wrap(ex, "extract_from_array", "extract_from_array")
        try:
            from core.texture_profile import TextureProfile

            self._wrap(TextureProfile, "from_dict", "texture_dna_from_db")
        except Exception:
            pass
        try:
            from core import ocr_engine as oe

            if hasattr(oe, "OCREngine"):
                self._wrap(oe.OCREngine, "extract", "ocr_recompute")
                self._wrap(oe.OCREngine, "read", "ocr_recompute")
        except Exception:
            pass
        orig_to = engine._to_result
        probe = self

        def to_result(*args, **kwargs):
            out = orig_to(*args, **kwargs)
            if probe.ttfr_s is None and probe._t0:
                probe.ttfr_s = time.perf_counter() - probe._t0
            return out

        self._orig.append((engine, "_to_result", orig_to))
        engine._to_result = to_result

        orig_cands = engine.db.search_text_candidates
        orig_ids = engine.db.get_indexed_files_by_ids

        def tally_rows(rows: list) -> list:
            for rec in rows or []:
                if not isinstance(rec, dict):
                    continue
                probe.db_rows += 1
                if rec.get("clip_embedding"):
                    probe.db_clip += 1
                if rec.get("dino_embedding"):
                    probe.db_dino += 1
                if str(rec.get("ocr_text") or "").strip():
                    probe.db_ocr += 1
                tm = rec.get("texture_map") or {}
                if isinstance(tm, str):
                    tm = {}
                if tm and tm != {}:
                    probe.db_texture += 1
                    if tm.get("pattern_dna") or tm.get("pattern_family"):
                        probe.db_semantic += 1
                if rec.get("pattern_family") or (isinstance(tm, dict) and tm.get("pattern_family")):
                    probe.db_family += 1
            return rows

        def wrapped_cands(*a, **k):
            t = time.perf_counter()
            rows = orig_cands(*a, **k)
            probe.ms["db_candidates"] += (time.perf_counter() - t) * 1000.0
            probe.calls["db_candidates"] += 1
            return tally_rows(rows)

        def wrapped_ids(*a, **k):
            t = time.perf_counter()
            rows = orig_ids(*a, **k)
            probe.ms["db_by_ids"] += (time.perf_counter() - t) * 1000.0
            probe.calls["db_by_ids"] += 1
            return tally_rows(rows)

        self._orig.append((engine.db, "search_text_candidates", orig_cands))
        self._orig.append((engine.db, "get_indexed_files_by_ids", orig_ids))
        engine.db.search_text_candidates = wrapped_cands
        engine.db.get_indexed_files_by_ids = wrapped_ids

        orig_clip = engine.faiss.search_clip
        orig_dino = engine.faiss.search_dino

        def wrap_faiss(orig: Callable, key: str, hit_attr: str):
            def inner(*a, **k):
                t = time.perf_counter()
                hits = orig(*a, **k)
                probe.ms[key] += (time.perf_counter() - t) * 1000.0
                probe.calls[key] += 1
                setattr(probe, hit_attr, getattr(probe, hit_attr) + len(hits or []))
                return hits

            return inner

        self._orig.append((engine.faiss, "search_clip", orig_clip))
        self._orig.append((engine.faiss, "search_dino", orig_dino))
        engine.faiss.search_clip = wrap_faiss(orig_clip, "clip_faiss_reuse", "faiss_clip_hits")
        engine.faiss.search_dino = wrap_faiss(orig_dino, "dino_faiss_reuse", "faiss_dino_hits")

    def detach(self) -> None:
        for obj, name, orig in reversed(self._orig):
            try:
                setattr(obj, name, orig)
            except Exception:
                pass
        self._orig.clear()

    def snapshot(self, n_results: int, total_s: float) -> dict[str, Any]:
        def row(ready: int, recompute: int, ms: float) -> dict[str, Any]:
            tot = ready + recompute
            reuse = round(100.0 * ready / tot, 1) if tot else 100.0
            return {
                "hazir": ready,
                "recompute": recompute,
                "reuse_pct": reuse,
                "sure_ms": round(ms, 2),
            }

        clip_re = self.calls["clip_image_recompute"] + self.calls["extract_from_path"]
        dino_re = self.calls["dino_image_recompute"]
        ocr_re = self.calls["ocr_recompute"]
        dna_re = self.calls["extract_from_array"]  # pixel DNA only if extract
        # texture/dna from_dict is reuse
        return {
            "n_results": n_results,
            "ttfr_s": round(self.ttfr_s or total_s, 3),
            "total_s": round(total_s, 3),
            "candidates": self.db_rows,
            "deep_candidates": 0,
            "faiss_clip_hits": self.faiss_clip_hits,
            "faiss_dino_hits": self.faiss_dino_hits,
            "calls": dict(self.calls),
            "ms": {k: round(v, 2) for k, v in self.ms.items()},
            "table": {
                "CLIP": row(self.faiss_clip_hits or self.db_clip, clip_re, self.ms["clip_query_encode"] + self.ms["clip_faiss_reuse"] + self.ms["clip_image_recompute"]),
                "DINO": row(self.faiss_dino_hits or self.db_dino, dino_re, self.ms["dino_faiss_reuse"] + self.ms["dino_image_recompute"]),
                "OCR": row(self.db_ocr, ocr_re, self.ms["ocr_recompute"]),
                "DNA": row(self.db_texture, dna_re, self.ms["texture_dna_from_db"]),
                "Texture": row(self.db_texture, dna_re, self.ms["texture_dna_from_db"]),
                "Semantic": row(self.db_semantic or self.db_family, 0, 0.0),
                "Family": row(self.db_family, 0, 0.0),
            },
            "query_clip_encodes": self.calls["clip_query_encode"],
            "query_clip_encode_ms": round(self.ms["clip_query_encode"], 2),
            "recomputed_evidence": int(clip_re + dino_re + ocr_re + dna_re),
        }


def _inventory(engine: SearchEngine) -> dict[str, int]:
    with engine.db.connect() as con:
        row = con.execute(
            """
            SELECT
              COUNT(*) AS files,
              SUM(CASE WHEN fe.clip_embedding IS NOT NULL AND length(fe.clip_embedding)>64 THEN 1 ELSE 0 END) AS clip,
              SUM(CASE WHEN fe.dino_embedding IS NOT NULL AND length(fe.dino_embedding)>64 THEN 1 ELSE 0 END) AS dino,
              SUM(CASE WHEN TRIM(COALESCE(f.ocr_text,''))<>'' THEN 1 ELSE 0 END) AS ocr,
              SUM(CASE WHEN fe.texture_map IS NOT NULL AND length(fe.texture_map)>5 THEN 1 ELSE 0 END) AS texture
            FROM files f
            LEFT JOIN features fe ON fe.file_id=f.id
            WHERE f.status NOT IN ('missing','excluded_internal')
            """
        ).fetchone()
    return {k: int(row[k] or 0) for k in row.keys()}


def run_query(engine: SearchEngine, probe: EvidenceProbe, q: str, limit: int = 20) -> dict[str, Any]:
    probe.reset()
    probe._t0 = time.perf_counter()
    rows = engine.search_by_text(q, limit=limit, threshold=0.55)
    total = time.perf_counter() - probe._t0
    snap = probe.snapshot(len(rows), total)
    snap["query"] = q
    return snap


def main() -> None:
    settings = AppSettings.load()
    data = Path(settings.db_path).parent
    fp_before = {
        "patterns_db": _fingerprint(Path(settings.db_path)),
        "faiss_clip": _fingerprint(Path(settings.faiss_clip_path)),
        "faiss_dino": _fingerprint(Path(settings.faiss_dino_path)),
        "clip_map": _fingerprint(Path(settings.faiss_clip_path).with_suffix(".map.npy")),
        "vm1": _fingerprint(data / "visual_memory" / "_shadow.db"),
        "vm2": _fingerprint(data / "visual_memory_phase2" / "_shadow.db"),
    }
    engine = SearchEngine(settings, load_ai=True)
    engine.ensure_ai_loaded()
    inv = _inventory(engine)
    probe = EvidenceProbe()
    probe.attach(engine)
    queries: dict[str, Any] = {}
    try:
        for q in QUERIES:
            print("cold", q)
            cold = run_query(engine, probe, q)
            print("warm", q)
            warm = run_query(engine, probe, q)
            queries[q] = {
                "cold": cold,
                "warm": warm,
                "speedup": round(cold["total_s"] / warm["total_s"], 2) if warm["total_s"] else None,
                "warm_faster": warm["total_s"] < cold["total_s"],
            }
    finally:
        probe.detach()
    fp_after = {
        "patterns_db": _fingerprint(Path(settings.db_path)),
        "faiss_clip": _fingerprint(Path(settings.faiss_clip_path)),
        "faiss_dino": _fingerprint(Path(settings.faiss_dino_path)),
        "clip_map": _fingerprint(Path(settings.faiss_clip_path).with_suffix(".map.npy")),
        "vm1": _fingerprint(data / "visual_memory" / "_shadow.db"),
        "vm2": _fingerprint(data / "visual_memory_phase2" / "_shadow.db"),
    }
    # Aggregate cold table
    agg = {k: {"hazir": 0, "recompute": 0, "sure_ms": 0.0} for k in ("CLIP", "DINO", "OCR", "DNA", "Texture", "Semantic", "Family")}
    ttfr = []
    total = []
    cands = []
    recomputed = []
    for q, pair in queries.items():
        c = pair["cold"]
        ttfr.append(c["ttfr_s"])
        total.append(c["total_s"])
        cands.append(c["candidates"])
        recomputed.append(c["recomputed_evidence"])
        for k, v in c["table"].items():
            agg[k]["hazir"] += v["hazir"]
            agg[k]["recompute"] += v["recompute"]
            agg[k]["sure_ms"] += v["sure_ms"]
    table = {}
    for k, v in agg.items():
        tot = v["hazir"] + v["recompute"]
        table[k] = {
            "hazir": v["hazir"],
            "recompute": v["recompute"],
            "reuse_pct": round(100.0 * v["hazir"] / tot, 1) if tot else 100.0,
            "sure_ms": round(v["sure_ms"], 2),
        }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "text_search_existing_index_only",
        "inventory_ready": inv,
        "table": table,
        "ttfr_s_mean": round(sum(ttfr) / max(len(ttfr), 1), 3),
        "total_search_s_mean": round(sum(total) / max(len(total), 1), 3),
        "candidates_mean": round(sum(cands) / max(len(cands), 1), 1),
        "deep_candidates": 0,
        "recomputed_evidence_sum": int(sum(recomputed)),
        "queries": queries,
        "production_unchanged": fp_before == fp_after,
        "fingerprints_before": fp_before,
        "fingerprints_after": fp_after,
        "note": (
            "CLIP query encode is expected each search (text→vector). "
            "File CLIP/DINO/OCR/DNA should be FAISS/DB reuse. "
            "extract_from_path / encode_image / OCREngine = recompute."
        ),
    }
    dest = data / "reports" / "evidence_reuse_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nEvidence | Hazır | Recompute | Reuse % | Süre ms")
    for k, v in table.items():
        print(f"{k:10} {v['hazir']:7} {v['recompute']:10} {v['reuse_pct']:7} {v['sure_ms']:8}")
    print("TTFR mean", report["ttfr_s_mean"], "Total mean", report["total_search_s_mean"])
    print("Candidates mean", report["candidates_mean"], "Recomputed", report["recomputed_evidence_sum"])
    print("production_unchanged", report["production_unchanged"])
    print("wrote", dest)


if __name__ == "__main__":
    main()
