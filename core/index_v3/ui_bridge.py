"""Index Engine V3 — UI SSOT: fiziksel artifact havuzu (event yok).

V3 READY (iş kuyruğu / light gate):
  Thumbnail/Preview → kendi fiziksel artifact'i + physical_*_ready=1
  Heavy iş kuyruğu → fiziksel Preview READY gerekir

UI KAPSAM (coverage) sayaçları:
  DINO/OpenCLIP/Texture/Semantic/DNA/Object/Patch/OCR → DB'de
  kullanılabilir artifact varsa TAMAMLANAN sayılır.
  physical_preview gate yokluğu bunları \"eksik\" yapmaz.

LEGACY (tanı):
  Feature/path DB'de var ama V3 physical preview gate yok
  (çoğunlukla cache dosyası silinmiş / bayrak 0).
  Legacy ≠ yeniden DINO üret. Legacy = gate/cache onarımı adayı.
"""

from __future__ import annotations

from typing import Any

from core.index_v3.types import Artifact

STAGE_POOL_KEYS: tuple[tuple[str, str], ...] = (
    ("thumbnail", "thumbnail"),
    ("preview", "preview"),
    ("hash", "hash"),
    ("metadata", "metadata"),
    ("dino", "dino"),
    ("openclip", "clip"),
    ("texture", "texture"),
    ("semantic", "semantic"),
    ("dna", "dna"),
    ("object_concept", "object_concept"),
    ("patch", "patch"),
    ("ocr", "ocr"),
    ("ai_final", "ai_final"),
)

_THUMB_PHYS = (
    "ifnull(f.thumbnail_path,'')!='' AND ifnull(f.physical_thumbnail_ready,0)=1"
)
_PREV_PHYS = (
    "ifnull(f.feature_preview_path,'')!='' AND ifnull(f.physical_preview_ready,0)=1"
)
_THUMB_PATH = "ifnull(f.thumbnail_path,'')!=''"
_PREV_PATH = "ifnull(f.feature_preview_path,'')!=''"
# Hash = features.phash. Hızlı İndeks hash yazmaz; Genel AI Preview'dan üretir.
# Sayaç yanlış değil: preview hazır olsa bile phash yoksa Hash eksik sayılır.
_HASH = "ifnull(fe.phash,'')!=''"
_META = "ifnull(f.format_metadata,'')!='' OR ifnull(f.width,0)>0"
_DINO = "fe.dino_embedding IS NOT NULL AND length(fe.dino_embedding)>0"
_CLIP = "fe.clip_embedding IS NOT NULL AND length(fe.clip_embedding)>0"
_TEX = "ifnull(fe.texture_features,'') NOT IN ('','[]') AND ifnull(fe.phash,'')!=''"
_SEM = (
    "ifnull(json_extract(fe.texture_map,'$.semantic_tags'),'') "
    "NOT IN ('','{}','null')"
)
_DNA = (
    "ifnull(json_extract(fe.texture_map,'$.pattern_dna'),'') "
    "NOT IN ('','{}','null')"
)
_OBJ = (
    "ifnull(json_extract(fe.texture_map,'$.visual_concept_dna'),'') "
    "NOT IN ('','{}','null') "
    "OR ifnull(json_extract(fe.texture_map,"
    "'$.global_object_intelligence.visual_concept_dna'),'') "
    "NOT IN ('','{}','null')"
)
_PATCH = "ifnull(fe.patch_embeddings,'') NOT IN ('','[]')"
_OCR = "ifnull(f.ocr_processed,0)=1"

_AI_GATE = f"({_PREV_PHYS})"  # AI için yalnızca gerçek Preview gerekir; Thumbnail UI artifactidir.

# İş kuyruğu gate'li sayımlar (yeniden işleme kararı — UI coverage değil)
_V3_DINO = f"({_AI_GATE}) AND ({_DINO})"
_V3_CLIP = f"({_AI_GATE}) AND ({_CLIP})"
_V3_TEX = f"({_AI_GATE}) AND ({_TEX})"
_V3_SEM = f"({_AI_GATE}) AND ({_SEM})"
_V3_DNA = f"({_AI_GATE}) AND ({_DNA})"
_V3_OBJ = f"({_AI_GATE}) AND ({_OBJ})"
_V3_PATCH = f"({_AI_GATE}) AND ({_PATCH})"
_V3_OCR = f"({_AI_GATE}) AND ({_OCR})"
# Disk-ready Hızlı (iş/görüntü gate): her iki physical bayrak.
_LIGHT = f"({_THUMB_PHYS}) AND ({_PREV_PHYS})"
# UI coverage Hızlı: daha önce index yazılmış path'ler (oturumdan bağımsız).
# Cache silinse bile geçerli hızlı-index kaydı sayılır; JobStore ayrı onarır.
_LIGHT_COVERAGE = f"({_THUMB_PATH}) AND ({_PREV_PATH})"
# Genel AI tamamlanması Hash + Metadata + HASH…DNA. Object Index alt havuz;
# PATCH/OCR ayrı. Object Preview üretmez.
_AI_FINAL = (
    f"({_AI_GATE}) AND ({_HASH}) AND ({_META}) AND ({_DINO}) AND ({_CLIP}) "
    f"AND ({_TEX}) AND ({_SEM}) AND ({_DNA})"
)
# UI coverage: kullanılabilir feature var mı? (physical gate şart değil)
_AI_FINAL_COVERAGE = (
    f"({_HASH}) AND ({_META}) AND ({_DINO}) AND ({_CLIP}) "
    f"AND ({_TEX}) AND ({_SEM}) AND ({_DNA})"
)
_PATCH_ERR = "ifnull(f.patch_error,'')!=''"
_OCR_ERR = "ifnull(f.ocr_error,'')!=''"


def _scope_clause(source_ids: list[int] | None) -> tuple[str, list[Any]]:
    """source_ids=[] → sıfır dosya (orphan/all-files sızıntısı YOK).

    source_ids=None → bilinçli unscoped (yalnız nadir diagnostic).
    """
    if source_ids is None:
        return "f.status NOT IN ('excluded_internal','missing')", []
    if not source_ids:
        return "1=0", []
    ph = ",".join("?" * len(source_ids))
    return (
        f"f.status NOT IN ('excluded_internal','missing') AND f.source_id IN ({ph})",
        [int(x) for x in source_ids],
    )


def _cnt(expr: str) -> str:
    return f"COUNT(DISTINCT CASE WHEN ({expr}) THEN f.id END)"


def count_v3_ssot(db: Any, source_ids: list[int] | None = None) -> dict[str, int]:
    """V3 physical pools + legacy/db_path diagnostics. Event yok."""
    where, params = _scope_clause(source_ids)
    sql = f"""
        SELECT
          COUNT(DISTINCT f.id) AS total,
          {_cnt(_THUMB_PHYS)} AS thumbnail,
          {_cnt(_PREV_PHYS)} AS preview,
          {_cnt(_THUMB_PATH)} AS thumbnail_db_path,
          {_cnt(_PREV_PATH)} AS preview_db_path,
          {_cnt(_HASH)} AS hash,
          {_cnt(_META)} AS metadata,
          {_cnt(_V3_DINO)} AS dino,
          {_cnt(_V3_CLIP)} AS clip,
          {_cnt(_V3_TEX)} AS texture,
          {_cnt(_V3_SEM)} AS semantic,
          {_cnt(_V3_DNA)} AS dna,
          {_cnt(_V3_OBJ)} AS object_concept,
          {_cnt(_V3_PATCH)} AS patch,
          {_cnt(_V3_OCR)} AS ocr,
          {_cnt(_DINO)} AS dino_db,
          {_cnt(_CLIP)} AS clip_db,
          {_cnt(_TEX)} AS texture_db,
          {_cnt(_SEM)} AS semantic_db,
          {_cnt(_DNA)} AS dna_db,
          {_cnt(_OBJ)} AS object_concept_db,
          {_cnt(_PATCH)} AS patch_db,
          {_cnt(_OCR)} AS ocr_db,
          {_cnt(_LIGHT)} AS light_complete,
          {_cnt(_LIGHT_COVERAGE)} AS light_coverage,
          {_cnt(_AI_FINAL)} AS ai_final,
          {_cnt(_AI_FINAL_COVERAGE)} AS ai_final_coverage,
          {_cnt(_PATCH_ERR)} AS patch_error,
          {_cnt(_OCR_ERR)} AS ocr_error,
          {_cnt(f"NOT ({_PREV_PHYS})")} AS waiting_preview,
          {_cnt("ifnull(f.light_status,'')='failed' OR ifnull(f.heavy_status,'')='failed'")}
            AS failed
        FROM files f
        LEFT JOIN features fe ON fe.file_id = f.id
        WHERE {where}
    """
    with db.connect() as conn:
        row = conn.execute(sql, params).fetchone()
    d = dict(row) if row is not None else {}
    keys = (
        "total",
        "thumbnail",
        "preview",
        "thumbnail_db_path",
        "preview_db_path",
        "hash",
        "metadata",
        "dino",
        "clip",
        "texture",
        "semantic",
        "dna",
        "object_concept",
        "patch",
        "ocr",
        "dino_db",
        "clip_db",
        "texture_db",
        "semantic_db",
        "dna_db",
        "object_concept_db",
        "patch_db",
        "ocr_db",
        "light_complete",
        "light_coverage",
        "ai_final",
        "ai_final_coverage",
        "patch_error",
        "ocr_error",
        "waiting_preview",
        "failed",
    )
    out = {k: int(d.get(k) or 0) for k in keys}
    total = out["total"]
    out["light_queue"] = max(0, total - out["light_complete"])
    out["light_queue_coverage"] = max(0, total - out["light_coverage"])
    out["heavy_queue"] = max(0, out["preview"] - out["ai_final"])
    out["processing"] = 0
    out["legacy_thumbnail"] = max(0, out["thumbnail_db_path"] - out["thumbnail"])
    out["legacy_preview"] = max(0, out["preview_db_path"] - out["preview"])
    out["legacy_light"] = max(0, out["light_coverage"] - out["light_complete"])
    out["legacy_dino"] = max(0, out["dino_db"] - out["dino"])
    out["legacy_clip"] = max(0, out["clip_db"] - out["clip"])
    out["legacy_texture"] = max(0, out["texture_db"] - out["texture"])
    out["legacy_semantic"] = max(0, out["semantic_db"] - out["semantic"])
    out["legacy_dna"] = max(0, out["dna_db"] - out["dna"])
    out["legacy_object_concept"] = max(
        0, int(out.get("object_concept_db") or 0) - int(out.get("object_concept") or 0)
    )
    out["legacy_patch"] = max(0, out["patch_db"] - out["patch"])
    out["legacy_ocr"] = max(0, out["ocr_db"] - out["ocr"])
    return out


def _lane_label(
    *,
    done: int,
    failed: int,
    processing: int,
    eligible: int,
) -> str:
    """PATCH/OCR: Hazır / Bekliyor / Çalışıyor / Tamamlandı / Hatalı."""
    if processing > 0:
        return "Çalışıyor"
    if failed > 0 and done == 0 and eligible <= 0:
        return "Hatalı"
    if failed > 0 and done + failed >= eligible + failed and eligible <= 0:
        return "Hatalı"
    if failed > 0 and done < (eligible + done):
        # bazıları hatalı, bazıları hazır/tamam — Hatalı görünür (bağımsız OCR/PATCH)
        if eligible <= 0 and processing <= 0:
            return "Hatalı"
    if eligible > 0:
        return "Hazır"
    if done > 0 and eligible <= 0:
        return "Tamamlandı"
    return "Bekliyor"


def _general_ai_label(*, total: int, ai_final: int, failed: int) -> str:
    if failed > 0 and ai_final < total:
        return "Hatalı"
    if total > 0 and ai_final >= total:
        return "Tamamlandı"
    return "Bekliyor"


def v3_status_dict(
    db: Any,
    source_ids: list[int] | None = None,
    *,
    processing: int = 0,
    claimed: int = 0,
    light_processing: int | None = None,
    heavy_processing: int | None = None,
    pending_jobs: int = 0,
    failed_permanent_files: int = 0,
    engine: str = "index_v3",
    scope: Any | None = None,
    current_file: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if scope is not None and source_ids is None:
        source_ids = list(scope.source_ids)
    c = count_v3_ssot(db, source_ids)
    total = c["total"]
    # UI Genel AI coverage: kullanılabilir feature set (gate'siz).
    # İş kuyruğu hâlâ ai_final (gate'li) kullanır — aşağıda ayrı tutulur.
    ai_final_gate = c["ai_final"]
    ai_final = int(c.get("ai_final_coverage") or ai_final_gate)
    terminal_error_files = max(0, int(failed_permanent_files or 0))
    proc = max(0, int(processing or claimed or 0))
    # Light/Heavy claimed ayrı — tek proc her iki karta yazılmaz.
    light_proc = (
        max(0, int(light_processing))
        if light_processing is not None
        else 0
    )
    heavy_proc = (
        max(0, int(heavy_processing))
        if heavy_processing is not None
        else 0
    )
    if light_processing is None and heavy_processing is None and proc > 0:
        # Geriye uyumluluk: ayrım yoksa current_file kuyruğuna göre böl
        cur0 = dict(current_file or {})
        q = str(cur0.get("queue") or "").lower()
        art = str(cur0.get("artifact") or cur0.get("stage") or "").lower()
        is_light = q in ("light", "preview") or art in ("thumbnail", "preview")
        if is_light:
            light_proc = proc
        else:
            heavy_proc = proc
    exec_jobs = int(pending_jobs or 0)

    def _stage(completed: int) -> dict[str, Any]:
        rem = max(0, total - completed)
        pct = (100.0 * completed / total) if total > 0 else 0.0
        return {
            "completed": completed,
            "remaining": rem,
            "percent": pct,
            "total": total,
            "v3_ready": completed,
            "v3_missing": rem,
        }

    # UI coverage pools: DB'de path/blob varsa tamamlanmış say.
    # physical_* yalnızca iş kuyruğu / Hızlı Index (light_complete) için.
    # Gate bekleyen = path var ama physical yok → coverage'a DAHİL, kalan'a DEĞİL.
    pools = {
        "total": total,
        "thumbnail": c["thumbnail_db_path"],
        "preview": c["preview_db_path"],
        "hash": c["hash"],
        "metadata": c["metadata"],
        "dino": c["dino_db"],
        "clip": c["clip_db"],
        "openclip": c["clip_db"],
        "texture": c["texture_db"],
        "semantic": c["semantic_db"],
        "dna": c["dna_db"],
        "object_concept": c["object_concept_db"],
        "patch": c["patch_db"],
        "ocr": c["ocr_db"],
        "ai_final": ai_final,
        "light_complete": c["light_complete"],
        "light_coverage": int(c.get("light_coverage") or 0),
        # Job-gate / disk-ready (UI satır coverage değil)
        "thumbnail_gate": c["thumbnail"],
        "preview_gate": c["preview"],
        "dino_gate": c["dino"],
        "clip_gate": c["clip"],
        "ai_final_gate": ai_final_gate,
    }
    legacy = {
        "thumbnail": c["legacy_thumbnail"],
        "preview": c["legacy_preview"],
        "light": int(c.get("legacy_light") or 0),
        "dino": c["legacy_dino"],
        "clip": c["legacy_clip"],
        "texture": c["legacy_texture"],
        "semantic": c["legacy_semantic"],
        "dna": c["legacy_dna"],
        "object_concept": c.get("legacy_object_concept", 0),
        "patch": c["legacy_patch"],
        "ocr": c["legacy_ocr"],
    }
    stage_stats: dict[str, dict[str, Any]] = {
        "total": {"completed": total, "remaining": 0, "percent": 100.0, "total": total}
    }
    for stage_name, pool_key in STAGE_POOL_KEYS:
        n = int(pools.get(pool_key, 0) or 0)
        st = _stage(n)
        leg_key = "clip" if pool_key == "openclip" else pool_key
        st["legacy"] = int(legacy.get(leg_key, 0) or 0)
        # Kimlik: CURRENT(coverage) + MISSING = TOTAL; legacy ⊆ current
        st["missing"] = st["remaining"]
        st["accounting_ok"] = (n + st["remaining"]) == total
        stage_stats[stage_name] = st

    # Hızlı Index UI = path coverage (geçmiş oturumlar dahil).
    # Disk-ready physical ayrı tutulur (gate / JobStore onarımı).
    light_phys = int(c["light_complete"])
    light_done = int(c.get("light_coverage") or light_phys)
    preview_done = int(c["preview"])  # physical — AI iş kuyruğu gate
    preview_coverage = int(c["preview_db_path"])
    preview_remaining = max(0, total - light_done)
    light_phys_remaining = max(0, total - light_phys)
    waiting = max(0, total - preview_coverage)  # UI: gerçek preview-path eksiği
    ai_remaining = max(0, total - ai_final)

    source_count = 0
    orphan_source_count = 0
    orphan_file_total = 0
    scope_mode = "legacy"
    orphan_source_ids: list[int] = []
    if scope is not None:
        source_count = int(scope.source_count_display)
        orphan_source_count = len(scope.orphan_source_ids)
        orphan_file_total = int(scope.orphan_file_total)
        scope_mode = str(scope.mode)
        orphan_source_ids = list(scope.orphan_source_ids)

    exec_hint = ""
    if light_phys_remaining > 0 and exec_jobs == 0:
        if waiting > 0 or int(c.get("legacy_light") or 0) > 0:
            exec_hint = "Preview/cache bekliyor"
        elif c["failed"] > 0:
            exec_hint = "Hatalı / yeniden deneme"
        else:
            exec_hint = "Çalıştırılabilir iş yok (kuyruk boş veya bağımlılık bekleniyor)"
    cur = dict(current_file or {})
    patch_done = int(c["patch"])
    ocr_done = int(c["ocr"])
    patch_err = int(c.get("patch_error") or 0)
    ocr_err = int(c.get("ocr_error") or 0)
    patch_eligible = max(0, ai_final - patch_done - patch_err)
    ocr_eligible = max(0, ai_final - ocr_done - ocr_err)
    general_ai_status = _general_ai_label(
        total=total, ai_final=ai_final, failed=int(c["failed"] or 0)
    )
    patch_status = _lane_label(
        done=patch_done,
        failed=patch_err,
        processing=0,
        eligible=patch_eligible,
    )
    ocr_status = _lane_label(
        done=ocr_done,
        failed=ocr_err,
        processing=0,
        eligible=ocr_eligible,
    )
    # Manuel PATCH/OCR AI_FINAL beklemez; Preview havuzu yeter.
    post_ga_buttons_enabled = int(c.get("preview") or 0) > 0

    owlv2 = {
        "total": total, "completed": 0, "pending": total, "failed": 0, "retry": 0,
        "percent": 0.0, "last_file": "",
    }
    try:
        from core.object_index import ObjectIndexStore
        from core.settings import AppSettings

        obj_path = str(getattr(AppSettings.load(), "object_db_path", "") or "")
        if obj_path:
            owlv2 = ObjectIndexStore(obj_path, readonly=True).owlv2_scan_progress(
                total=int(total)
            )
            try:
                from pathlib import Path
                from core.index_v3.queues import JobStore

                job_db = Path(str(getattr(AppSettings.load(), "db_path", "") or ""))
                if job_db.suffix:
                    job_path = job_db.with_name(job_db.stem + ".v3jobs.db")
                    if job_path.is_file():
                        bands = JobStore(str(job_path)).owlv2_queue_bands(
                            source_ids=list(source_ids or []) or None
                        )
                        owlv2 = {**owlv2, **bands}
                        from core.ovd_index import owl_progress_tone

                        owlv2["bar_tone"] = owl_progress_tone(
                            completed=int(owlv2.get("completed") or 0),
                            total=int(owlv2.get("total") or total),
                            priority_pending=int(bands.get("priority_pending") or 0),
                            weak_pending=int(bands.get("weak_pending") or 0),
                            unranked_pending=int(bands.get("unranked_pending") or 0),
                        )
            except Exception:
                pass
    except Exception:
        pass

    return {
        "engine": engine,
        "ssot": "index_v3_physical_pool",
        "total": total,
        "archive_total": total,
        "total_searchable": total,
        "ssot_total": total,
        "status_scope_source_ids": list(source_ids or []),
        "source_count": source_count,
        "orphan_source_count": orphan_source_count,
        "orphan_file_total": orphan_file_total,
        "orphan_source_ids": orphan_source_ids,
        "scope_mode": scope_mode,
        "fast_completed": light_done,
        "fast_remaining": preview_remaining,
        "light_done": light_done,
        "light_complete_physical": light_phys,
        "light_pending": preview_remaining,
        "light_processing": light_proc,
        "light_failed": c["failed"],
        "light_queue_display": max(0, preview_remaining),
        "ui_remaining_light": max(0, preview_remaining),
        "light_disk_remaining": light_phys_remaining,
        "general_completed": ai_final,
        "general_remaining": ai_remaining,
        "heavy_done": ai_final,
        "heavy_pending": ai_remaining,
        # Gate uyumu: physical preview vs gate'li AI final
        "heavy_ready_pending": max(0, preview_done - ai_final_gate),
        "heavy_pending_all": ai_remaining,
        "heavy_processing": heavy_proc,
        "heavy_failed": 0,
        "heavy_queue_display": max(0, ai_remaining - terminal_error_files),
        "ui_remaining_heavy": max(0, ai_remaining - terminal_error_files),
        "waiting_preview": waiting,
        "terminal_error_files": terminal_error_files,
        "queue_pending": max(0, ai_remaining - terminal_error_files),
        "processing": max(light_proc, heavy_proc, proc),
        "pending_jobs": exec_jobs,
        "v3_executable_jobs": exec_jobs,
        "remaining_vs_executable_hint": exec_hint,
        "preview_ready": c["preview_db_path"],
        "verified_preview_ready": c["preview"],
        "thumbnail_ready": c["thumbnail_db_path"],
        "thumbnail_physical_ready": c["thumbnail"],
        "preview_physical_ready": c["preview"],
        "thumbnail_db_path_ready": c["thumbnail_db_path"],
        "preview_db_path_ready": c["preview_db_path"],
        # UI coverage = kullanılabilir DB artifact (legacy gate yüzünden düşürme)
        "db_dino_embeddings": c["dino_db"],
        "db_clip_embeddings": c["clip_db"],
        "patch_embedding_ready": c["patch_db"],
        "texture_done": c["texture_db"],
        "semantic_tag_count": c["semantic_db"],
        "pattern_dna_count": c["dna_db"],
        "object_concept_ready": c["object_concept_db"],
        "owlv2": owlv2,
        "owlv2_ready": int(owlv2.get("completed") or 0),
        "ocr_done": c["ocr_db"],
        "ai_final_ready": ai_final,
        "ai_final_gate_ready": ai_final_gate,
        "general_ai_status_label": general_ai_status,
        "patch_status_label": patch_status,
        "ocr_status_label": ocr_status,
        "post_ga_buttons_enabled": post_ga_buttons_enabled,
        "patch_error_count": patch_err,
        "ocr_error_count": ocr_err,
        "pipeline_counts_exact": True,
        "artifact_pools": pools,
        "legacy_pools": legacy,
        "stage_stats": stage_stats,
        "v3_stages": {
            a.value: _stage(c[a.value])
            for a in list(Artifact)
            if a.value in c
        },
        "v3_ai_final": _stage(ai_final),
        "current_file_info": cur,
        "lanes_only": True,
        "from_ram_cache": False,
    }


def map_ui_mode_to_v3(index_mode: str) -> str:
    m = str(index_mode or "complete").strip().lower()
    if m in ("fast_archive", "fast"):
        return "fast"
    if m in ("night_complete", "general_ai"):
        return "general_ai"
    if m in ("backfill", "repair"):
        return "repair"
    if m == "patch":
        return "patch"
    if m == "ocr":
        return "ocr"
    if m in ("patch_ocr", "post_ga"):
        return "post_ga"
    return "complete"


def index_lane_for_mode(index_mode: str) -> str:
    """UI index_mode → lane id (fast | general_ai | complete | repair | patch | ocr)."""
    return map_ui_mode_to_v3(index_mode)


_POST_GA_LANES = frozenset({"patch", "ocr", "post_ga"})


def lanes_are_parallel(running_lane: str, new_lane: str) -> bool:
    """Hızlı Index ve Genel AI aynı anda ayrı worker ile çalışabilir.

    Manuel PATCH/OCR, Genel AI ve Hızlı Index ile paraleldir; birbirleriyle
    de paraleldir. Complete/Repair her şeyi durdurur.
    """
    a = str(running_lane or "")
    b = str(new_lane or "")
    if a == b:
        return False
    s = {a, b}
    if "complete" in s or "repair" in s:
        return False
    if s == {"fast", "general_ai"}:
        return True
    if "general_ai" in s and (s & _POST_GA_LANES):
        return True
    if "fast" in s and (s & _POST_GA_LANES):
        return True
    if s <= _POST_GA_LANES:
        return True
    return False
