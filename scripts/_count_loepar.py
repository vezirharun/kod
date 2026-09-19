from core.fuzzy_correct import correct_query
from core.search_engine import SearchEngine
from core.settings import AppSettings
from core.textile_terms import normalize_turkish

q = "loepar"
cr = correct_query(q)
print("QUERY", q)
print("CORRECTED", cr.corrected, "was_corrected", cr.was_corrected)

settings = AppSettings.load()
engine = SearchEngine(settings, load_ai=True)
rows = engine.search_by_text(q, limit=40, threshold=0.55)
print("N", len(rows))

RIVAL = ("snake", "yilan", "tiger", "kaplan", "zebra", "cow", "giraffe")


def nt(s: str) -> str:
    return normalize_turkish(s or "")


ok = wrong = unk = 0
for i, r in enumerate(rows, 1):
    d = dict(r.debug or {})
    bd = dict(r.breakdown or {})
    qev = d.get("query_evidence_report") or bd.get("query_evidence_report") or {}
    se = d.get("semantic_evidence") or {}
    fn = str(r.filename or "")
    fnn = nt(fn)
    animal = nt(str(se.get("animal_type") or ""))
    vis = str(d.get("visual_grade") or qev.get("visual_grade") or "")
    clip_only = bool(d.get("clip_only"))
    rival = d.get("rival_clip") or {}
    own = float((rival or {}).get("leopard") or d.get("clip_score") or 0)
    best_r, best_n = 0.0, ""
    for k, v in (rival or {}).items():
        if k == "leopard":
            continue
        if float(v or 0) > best_r:
            best_r, best_n = float(v or 0), str(k)
    fn_want = ("leopard" in fnn) or ("leopar" in fnn)
    fn_rival = any(x in fnn for x in RIVAL)
    meta_ok = animal == "leopard"
    meta_bad = animal in ("snake", "tiger", "zebra", "cow")
    vis_ok = vis in ("visual_exact", "visual_strong")
    vis_bad = bool(qev.get("visual_conflict")) or (
        best_n in ("snake", "tiger", "zebra") and best_r > own
    )
    if fn_rival or meta_bad or vis_bad:
        label = "YANLIS"
        wrong += 1
    elif fn_want or meta_ok:
        label = "DOGRU"
        ok += 1
    elif vis_ok:
        label = "GORSEL-ADAY"
        unk += 1
    else:
        label = "BELIRSIZ"
        unk += 1
    print(
        f"{i:2} {label:12} {r.score:.3f} {vis or '-':16} "
        f"animal={animal or '-'} clip_only={int(clip_only)} "
        f"rival={best_n}:{best_r:.2f} {fn[:70]}"
    )
print("---")
print("DOGRU", ok, "YANLIS", wrong, "GORSEL/BELIRSIZ", unk, "TOPLAM", len(rows))
