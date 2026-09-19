"""Görsel Anlam Testi — indeks üzerinde kavram araması, ölçüm ve rapor.

Kod / ağırlık / index DEĞİŞTİRMEZ. Sıra: TEST → ÖLÇ → RAPORLA → motor teşhisi.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from core.logger import setup_logger
from core.textile_terms import normalize_turkish

logger = setup_logger(__name__)

TOP_K = 20
WEAK_PCT = 80.0
STRONG_PCT = 90.0
CLIP_AGREE = 0.28

DEFAULT_QUERIES: tuple[str, ...] = (
    "araba",
    "oyuncak araba",
    "gül",
    "kiraz",
    "çiçek",
    "kuş",
    "yılan",
    "leopar",
    "kelebek",
    "köpek",
    "kedi",
    "kadın",
    "erkek",
    "kadın yüzü",
    "erkek yüzü",
    "insan yüzü",
    "kırmızı araba",
    "siyah araba",
    "kırmızı gül",
    "pembe gül",
    "kırmızı kiraz",
    "beyaz çiçek",
)

_FACE_HINTS = frozenset(
    {"yuz", "yüz", "face", "faces", "kadın yüzü", "erkek yüzü", "insan yüzü"}
)
_COLOR_TR = {
    "kirmizi": "red",
    "siyah": "black",
    "pembe": "pink",
    "beyaz": "white",
}


def _fold(text: str) -> str:
    return normalize_turkish(str(text or "")).strip()


def _tokens(text: str) -> list[str]:
    t = _fold(text).replace("-", " ")
    return [p for p in t.split() if len(p) > 1]


def compile_query_needles(query: str) -> dict[str, Any]:
    """Beklenen kavram iğneleri — arama skorunu değiştirmez."""
    from core.object_search import parse_object_query
    from core.query_intent_router import classify_query
    from core.visual_concept import compile_visual_query, detector_labels_for_lemma

    q = str(query or "").strip()
    folded = _fold(q)
    compiled = compile_visual_query(q)
    intent = classify_query(q)
    obj_spec = parse_object_query(q)
    needles: set[str] = {folded, *(_tokens(q))}
    for tok in list(needles):
        try:
            from core.visual_concept import english_lemma

            needles.add(english_lemma(tok))
        except Exception:
            pass
    for c in compiled.concepts:
        lemma = str(c.concept_id or "").split(":", 1)[-1].replace("_", " ")
        needles.add(_fold(lemma))
        needles.update(_fold(x) for x in (c.detector_labels or ()))
        needles.update(_fold(x) for x in (c.hypernyms or ()))
        needles.update(_fold(x) for x in detector_labels_for_lemma(lemma))
    colors = list(compiled.colors or [])
    for tok in _tokens(q):
        mapped = _COLOR_TR.get(tok)
        if mapped and mapped not in colors:
            colors.append(mapped)
    face = any(h in folded for h in _FACE_HINTS) or "yuz" in folded
    return {
        "query": q,
        "needles": {n for n in needles if n and n not in {"bir", "olan"}},
        "colors": colors,
        "face": face,
        "object_parse": obj_spec,
        "intent_kind": getattr(intent, "kind", "") or "",
        "intent_channel": getattr(intent, "channel", "") or "",
        "visual_lemmas": [str(c.concept_id) for c in compiled.concepts],
        "clip_prompts": list(compiled.clip_prompts or []),
    }


def _haystack(values: Any) -> str:
    parts: list[str] = []

    def _walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            parts.append(x)
            return
        if isinstance(x, (int, float, bool)):
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if k in {"bbox", "embedding", "clip_embedding", "dino_embedding"}:
                    continue
                _walk(k)
                _walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                _walk(v)

    _walk(values)
    return " " + _fold(" ".join(parts)) + " "


def _contains_any(hay: str, needles: set[str]) -> bool:
    if not hay or not needles:
        return False
    for n in needles:
        if len(n) < 2:
            continue
        if f" {n} " in hay or hay.startswith(n + " ") or hay.endswith(" " + n):
            return True
        if n in hay and len(n) >= 4:
            return True
    return False


def _result_evidence(result: Any) -> dict[str, Any]:
    dbg = dict(getattr(result, "debug", None) or {})
    br = dict(getattr(result, "breakdown", None) or {})
    tm = dict(dbg.get("texture_map") or {})
    goi = dict(tm.get("global_object_intelligence") or {})
    dna = dict(tm.get("visual_concept_dna") or goi.get("visual_concept_dna") or {})
    objects = list(goi.get("objects") or dna.get("objects") or [])
    obj_labels = []
    for o in objects:
        if isinstance(o, dict):
            obj_labels.extend(
                [
                    str(o.get("label") or ""),
                    str(o.get("canonical_name") or ""),
                    str(o.get("label_tr") or ""),
                ]
            )
        else:
            obj_labels.append(str(o))
    sem = tm.get("semantic_tags") or {}
    clip = float(
        br.get("clip")
        or dbg.get("clip")
        or dbg.get("ai_score")
        or 0.0
    )
    return {
        "file_id": int(getattr(result, "file_id", 0) or 0),
        "filename": str(getattr(result, "filename", "") or ""),
        "path": str(getattr(result, "path", "") or ""),
        "score": float(getattr(result, "score", 0) or 0),
        "clip": clip,
        "semantic_score": float(dbg.get("semantic_score") or br.get("semantic") or 0),
        "object_labels": [x for x in obj_labels if x],
        "semantic": sem,
        "color_family": str(
            getattr(result, "color_family", "") or dbg.get("color_family") or ""
        ),
        "face": bool(
            dbg.get("face_gender_match")
            or dbg.get("human_semantic_mode")
            or float(dbg.get("human_semantic_score") or 0) >= 0.18
        ),
        "objects_blob": _haystack(objects) + _haystack(obj_labels),
        "semantic_blob": _haystack(sem) + _haystack(dna.get("concepts")),
        "name_blob": _haystack(
            [getattr(result, "filename", ""), getattr(result, "path", "")]
        ),
    }


def _classify_one(spec: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
    needles = spec["needles"]
    colors = spec["colors"]
    obj_hit = _contains_any(ev["objects_blob"], needles)
    sem_hit = _contains_any(ev["semantic_blob"], needles)
    name_hit = _contains_any(ev["name_blob"], needles)
    clip_hit = float(ev["clip"] or 0) >= CLIP_AGREE
    color_hit = True
    if colors:
        color_hit = _contains_any(
            _haystack([ev.get("color_family"), ev["name_blob"], ev["semantic_blob"]]),
            set(colors) | {_fold(c) for c in colors},
        )
    face_hit = bool(ev.get("face")) if spec["face"] else None
    grounded = obj_hit or sem_hit or name_hit or (face_hit is True)
    competing = bool(ev["object_labels"]) and not obj_hit

    if grounded and color_hit:
        verdict = "doğru"
    elif (not grounded) and competing and not clip_hit:
        verdict = "yanlış"
    elif (not grounded) and competing and clip_hit:
        verdict = "yanlış"
    elif grounded and not color_hit:
        verdict = "yanlış"
    elif clip_hit and not grounded:
        verdict = "belirsiz"
    else:
        verdict = "belirsiz"

    motors = {
        "object": obj_hit,
        "semantic": sem_hit,
        "openclip": clip_hit,
        "filename": name_hit,
        "color": color_hit if colors else None,
        "face": face_hit,
    }
    return {
        "verdict": verdict,
        "motors": motors,
        "file_id": ev["file_id"],
        "filename": ev["filename"],
        "score": round(float(ev["score"] or 0), 4),
        "clip": round(float(ev["clip"] or 0), 4),
        "semantic_score": round(float(ev["semantic_score"] or 0), 4),
        "object_labels": ev["object_labels"][:8],
    }


def _motor_diagnosis(spec: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    n = max(1, len(rows))
    counts = Counter(r["verdict"] for r in rows)
    ok = counts["doğru"]
    pct = 100.0 * ok / n
    obj_n = sum(1 for r in rows if r["motors"].get("object"))
    sem_n = sum(1 for r in rows if r["motors"].get("semantic"))
    clip_n = sum(1 for r in rows if r["motors"].get("openclip"))
    face_n = sum(1 for r in rows if r["motors"].get("face") is True)
    color_miss = sum(1 for r in rows if r["motors"].get("color") is False)
    suspects: list[str] = []
    if spec["object_parse"] is None and any(
        x in spec["needles"] for x in ("car", "cat", "dog", "bird", "person")
    ):
        suspects.append("sorgu yorumlama (detector parse yok)")
    if spec["object_parse"] is None and obj_n == 0:
        suspects.append("Object Index / detector sınıfı (COCO dışı kavram olabilir)")
    if obj_n < n * 0.35 and sem_n >= n * 0.45:
        suspects.append("Object (Semantic tutuyor, kutu yok/yanlış)")
    if sem_n < n * 0.35 and obj_n >= n * 0.45:
        suspects.append("Semantic (nesne var, etiket zayıf)")
    if clip_n >= n * 0.5 and obj_n < n * 0.25 and sem_n < n * 0.25:
        suspects.append("OpenCLIP sıralıyor; Object/Semantic kanıtı zayıf")
    if obj_n >= n * 0.4 and sem_n >= n * 0.4 and pct < WEAK_PCT:
        suspects.append("skor ağırlığı / birleşik sıralama")
    if spec["face"] and face_n < n * 0.4:
        suspects.append("yüz / cinsiyet katmanı")
    if spec["colors"] and color_miss >= n * 0.4:
        suspects.append("renk / skor ağırlığı")
    if not spec["visual_lemmas"]:
        suspects.append("sorgu yorumlama (görsel kavram derlenmedi)")
    if counts["belirsiz"] >= n * 0.45:
        suspects.append("kanıt eksik (index'te Object/Semantic boş)")
    if not suspects and pct < WEAK_PCT:
        suspects.append("birden fazla katman; tek motor suçlanamaz")
    return suspects


def format_report(payload: dict[str, Any]) -> str:
    lines = ["GÖRSEL ANLAM TESTİ", "=" * 18, ""]
    for row in payload.get("concepts") or []:
        q = str(row.get("query") or "")
        ok = int(row.get("correct") or 0)
        total = int(row.get("total") or 0)
        pct = float(row.get("pct") or 0)
        lines.append(f"{q:<16} {ok}/{total} doğru   %{pct:.0f}")
    lines.append("")
    lines.append(f"GENEL BAŞARI: %{float(payload.get('overall_pct') or 0):.0f}")
    lines += ["", "ZAYIF KAVRAMLAR", "-" * 15]
    weak = payload.get("weak") or []
    if not weak:
        lines.append("(eşik altında kavram yok)")
    else:
        for w in weak:
            lines.append(f"{w['query']:<16} → %{w['pct']:.0f}")
    lines += ["", "GÜÇLÜ KAVRAMLAR", "-" * 15]
    strong = payload.get("strong") or []
    if not strong:
        lines.append("(eşik üstünde kavram yok)")
    else:
        for s in strong:
            lines.append(f"{s['query']:<16} → %{s['pct']:.0f}")
    lines += ["", "MOTOR TEŞHİSİ (kod değiştirilmedi)", "-" * 34]
    for row in payload.get("concepts") or []:
        why = row.get("suspects") or []
        if not why:
            continue
        if float(row.get("pct") or 0) >= WEAK_PCT and "Object" not in " ".join(why):
            continue
        lines.append(f"{row['query']}: " + "; ".join(why))
    path = payload.get("report_path") or ""
    if path:
        lines += ["", f"Kayıt: {path}"]
    return "\n".join(lines)


def run_visual_meaning_test(
    settings,
    *,
    queries: list[str] | None = None,
    top_k: int = TOP_K,
    search_fn: Callable[[str, int], list[Any]] | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """Üretim indeksinde kavram taraması. Index/ağırlık yazılmaz."""
    terms = list(queries or DEFAULT_QUERIES)
    k = max(1, int(top_k))
    if search_fn is None:
        from core.search_engine import SearchEngine

        engine = SearchEngine(settings)

        def search_fn(text: str, limit: int) -> list[Any]:
            return engine.search_by_text(text, limit=limit)

    concepts: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, query in enumerate(terms):
        spec = compile_query_needles(query)
        try:
            hits = list(search_fn(query, k) or [])[:k]
            err = ""
        except Exception as exc:
            logger.exception("visual meaning search failed q=%s", query)
            hits = []
            err = str(exc)
        rows = []
        for item in hits:
            rows.append(_classify_one(spec, _result_evidence(item)))
        counts = Counter(r["verdict"] for r in rows)
        total = len(rows)
        ok = int(counts.get("doğru") or 0)
        pct = (100.0 * ok / total) if total else 0.0
        rec = {
            "query": query,
            "total": total,
            "correct": ok,
            "wrong": int(counts.get("yanlış") or 0),
            "uncertain": int(counts.get("belirsiz") or 0),
            "pct": round(pct, 1),
            "intent_kind": spec["intent_kind"],
            "object_parse": bool(spec["object_parse"]),
            "visual_lemmas": spec["visual_lemmas"],
            "suspects": _motor_diagnosis(spec, rows) if rows else ["sonuç yok"],
            "error": err,
            "samples": rows[:8],
        }
        concepts.append(rec)
        if progress_callback:
            progress_callback({"done": i + 1, "total": len(terms), "query": query})

    scored = [c for c in concepts if int(c["total"]) > 0]
    overall = (
        round(sum(c["pct"] for c in scored) / len(scored), 1) if scored else 0.0
    )
    weak = [
        {"query": c["query"], "pct": c["pct"]}
        for c in concepts
        if c["total"] and c["pct"] < WEAK_PCT
    ]
    strong = [
        {"query": c["query"], "pct": c["pct"]}
        for c in concepts
        if c["total"] and c["pct"] >= STRONG_PCT
    ]
    payload = {
        "ok": True,
        "mutated_index": False,
        "mutated_weights": False,
        "top_k": k,
        "overall_pct": overall,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "concepts": concepts,
        "weak": weak,
        "strong": strong,
        "report_text": "",
        "report_path": "",
    }
    text = format_report(payload)
    payload["report_text"] = text
    try:
        out_dir = Path(getattr(settings, "db_path", "data/patterns.db")).parent / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"visual_meaning_{ts}.txt"
        path.write_text(text + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)[:120000], encoding="utf-8")
        payload["report_path"] = str(path)
        payload["report_text"] = format_report(payload)
    except Exception:
        logger.exception("visual meaning report save failed")
    logger.info(
        "Görsel Anlam Testi overall=%s weak=%s",
        overall,
        [w["query"] for w in weak],
    )
    return payload
