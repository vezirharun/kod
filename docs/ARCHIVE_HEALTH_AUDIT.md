# Arşiv Sağlığı (Archive Health) — Denetim Raporu

**Kapsam:** AUDIT ONLY — kod değişikliği yok.  
**Tarih:** 2026-09-19 (Europe/Istanbul)  
**Kaynak ağaç:** `/workspace/vezir_audit/ah_audit/` + `/workspace/vezir_audit/index_v3/` (JobStore / Mode.REPAIR bağlamı)  
**Ürün:** Vezir Pattern Search  

**Hedef:** Mevcut Archive Health altyapısını haritalamak; **paralel onarım motoru önermeden** mevcut sistemleri (Preview Health, Preview Self-Healing, Eksikleri Tamamla / Mode.REPAIR+COMPLETE, health scan, JobStore, artifact checks, AI/learning/index coverage) yeniden kullanan **birleşik sağlık görünümü** önermek. Indexer çekirdek mantığına dokunulmaz.

---

## 1. Mevcut health altyapısı

### 1.1 Katmanlar (özet)

| Katman | Modül / sembol | Rol |
|--------|----------------|-----|
| **Artifact SSOT (V3)** | `core/index_v3/artifact_state.py` → `assess_rows` / `assess_file` / `plan_missing` | Dosya başına `ArtifactStatus.READY\|MISSING\|INVALID` |
| **Artifact tipleri** | `core/index_v3/types.py` → `Artifact`, `ArtifactStatus`, `JobState`, `QueueKind`, `Mode`, `AI_FINAL_REQUIRED` | Sözlük: thumbnail…ocr; AI_FINAL = HASH…DNA |
| **UI pool SSOT** | `core/index_v3/ui_bridge.py` → `count_v3_ssot`, `v3_status_dict` | Fiziksel gate’li + DB coverage sayaçları; `legacy_*` tutarsızlıkları |
| **JobStore** | `core/index_v3/queues.py` → `JobStore` | `index_v3_jobs` (patterns.v3jobs.db): `pending\|claimed\|done\|failed\|failed_permanent` |
| **Planner / Eksikleri Tamamla** | `core/index_v3/planner.py` → `plan_jobs_for_file`; `discovery.py` Mode.REPAIR = “Eksikleri Tamamla” | Sparse job; Preview gate |
| **Background health scan** | `core/index_v3/background_scan.py` → `BackgroundIndexScan` | physical reconcile → preview self-heal → Mode.COMPLETE gap enqueue → disk walk |
| **Physical reconcile** | `core/index_v3/physical_reconcile.py` → `reconcile_stale_physical_flags` | Path/bayrak ↔ disk |
| **Preview Self-Healing** | `core/preview_self_heal/{validate,repair,hooks}.py` | VALID/INVALID/SUSPICIOUS; mono kaynak ≠ bozuk |
| **Cache reconciliation** | `core/cache_reconciliation.py` → `CacheReconciliationEngine` | Batch physical verify + light repair + AI missing sayımı |
| **Legacy RC2 assess** | `core/artifact_status.py` → `assess_row` / `reconcile_source_artifacts` | Boolean light/heavy; V3 ile örtüşür, ayrı API |
| **Index integrity / doctor** | `core/index_integrity.py` → `evaluate_row`, `scan_integrity`, `run_doctor` | Search-ready bayrakları; force-reindex |
| **Production health** | `core/production/{health_monitor,index_verify,self_healing,recovery}.py` | CPU/RAM/GPU + `verify_index` / `repair_index` / startup recovery |
| **Pipeline % UI** | `core/index_pipeline_status.py` → `build_pipeline_status` | pending_* = total − stage_* |
| **App status** | `core/app_status.py` | JobStore sayıları + physical reconcile throttle + kalan nedenleri |
| **Quarantine** | `core/quarantine.py` | Hard/soft reason; failed report |
| **UI** | `ui/health_panel.py` (“Sistem Sağlığı”), `ui/format_status_panel.py` | Canlı metrik + “Eksikleri Onar” / Self-Heal; format eksikleri |
| **Live contract** | `core/index_v3/live_contract.py` | Tamamlanan=READY pool; Executable=JobStore pending (ayrı) |

### 1.2 JobStore onarım durumları (önceden denetlenmiş)

DB: `patterns.v3jobs.db` → tablo `index_v3_jobs`  
Alanlar: `file_id`, `artifact`, `queue` (`light|preview|heavy|repair`), `state`, `attempts`, `error_msg`, …

**JobState** (`types.JobState`): `pending`, `claimed`, `done`, `failed`, `failed_permanent`

Örnek canlı dağılım (audit snapshot): ~288k `done`, ~111k `pending`, 81 `failed_permanent` (çoğunlukla preview). Pending hata örneği: `physical_artifact_missing` (~15k).

**Onarım kancaları:** `enqueue(..., reopen_done=, reopen_permanent=)`, `reopen_failed_permanent`, `requeue_stale_claims`, `fail(..., permanent=)`. OCR/PATCH permanent fail dosya sayımından hariç (`count_failed_permanent_files`).

### 1.3 Preview Self-Healing politikası (kritik)

- Solid siyah/beyaz/mono **tek başına BROKEN değil** (`validate.py`: mono → `SUSPICIOUS`).
- Kaynak da mono ise → `VALID` (`source_matches_mono`).
- Renkli kaynak + mono preview → `INVALID` (`misrepresenting`).
- Mod A: `invalidate_if_bad_preview` (Eksikleri Tamamla / REPAIR).
- Mod B: `heal_candidates_for_sources` (background scan).
- Mod C: `gate_new_preview` / `apply_gate_or_repair` (`real_processor` preview üretimi).

### 1.4 “Eksikleri Tamamla”

Kod yorumu: `discovery.enqueue_existing_gaps` içinde **Mode A (Eksikleri Tamamla / REPAIR)** → `invalidate_if_bad_preview` + `plan_jobs_for_file(..., Mode.REPAIR, repair=True)` + JobStore enqueue (`reopen_permanent=True`, light için `reopen_done=True`).  
UI’da `HealthPanel` düğmesi **“Eksikleri Onar”** ayrı bir yol: `production.index_verify.repair_index` (eski maintenance API).

---

## 2. On altı sağlık kategorisi — detaylı harita

Her satır: **nerede kontrol**, **DB/alan**, **üretici servis**, **mevcut onarım?**, **UI?**, **duplicate?**

### 1) File / path status (kaynak dosya)

| | |
|--|--|
| **Kontrol** | `files.status` (`missing`, `excluded_internal`, …); background walk `BackgroundIndexScan`; `format_audit` / FormatStatusPanel “Eksik dosya” |
| **DB** | `files.path`, `files.status`, `files.source_id` |
| **Üretici** | Discovery / disk walk; indexer upsert |
| **Onarım** | Missing işaretleme / purge (`purge_missing`); JobStore job iptali `cancel_jobs_for_file_ids` |
| **UI** | FormatStatusPanel; progress kalan nedenleri (`app_status._classify_remaining_reason`) |
| **Duplicate** | Status string’leri ile JobStore path ayrı; SSOT path JobStore’da kopya tutulur |

**Unified:** kaynak yok → dosya düzeyi **FAILED/MISSING**; AI varsa **STALE** (orphan features).

### 2) Thumbnail status

| | |
|--|--|
| **Kontrol** | `artifact_state.path_status(thumbnail_path, physical_thumbnail_ready)`; `ui_bridge._THUMB_PHYS`; `CacheReconciliationEngine._verify_row` |
| **DB** | `files.thumbnail_path`, `physical_thumbnail_ready`, `thumbnail_status` |
| **Üretici** | Light/Preview sonrası thumbnail (`Thumbnailer`); `FeaturePreviewCache` zinciri |
| **Onarım** | Job `Artifact.THUMBNAIL` / `QueueKind.LIGHT`; cache reconcile `_repair_row`; `heal_candidates`; `requeue_missing_thumbnails` (legacy verify) |
| **UI** | `count_v3_ssot.thumbnail`, `legacy_thumbnail` (path var, physical=0) |
| **Duplicate** | `artifact_status` (path-only opsiyonel), `index_integrity.flags.thumbnail`, `index_verify` issues.thumbnail |

### 3) Feature Preview status

| | |
|--|--|
| **Kontrol** | `Artifact.PREVIEW` + `physical_preview_ready`; `preview_status` (`preview_ok`, `preview_suspicious`, `preview_heal_failed`, `preview_invalid:*`, `preview_missing_physical`) |
| **DB** | `files.feature_preview_path`, `physical_preview_ready`, `preview_status` |
| **Üretici** | `FeaturePreviewCache.create` + `preview_self_heal` gate; `global_preview` renderers |
| **Onarım** | JobStore `preview` kuyruğu; Mode.REPAIR; `repair_preview_artifact`; cache reconcile |
| **UI** | preview pool, `waiting_preview`, HealthPanel physical sync |
| **Duplicate** | Production `verify_index` path `isfile` (physical bayrak yok); integrity preview opsiyonel |

### 4) DINO / embedding status

| | |
|--|--|
| **Kontrol** | `fe.dino_embedding` blob (`_blob_ok` / `_DINO`); gate’li `_V3_DINO` |
| **DB** | `features.dino_embedding`; FAISS `faiss_dino_path` (health_monitor) |
| **Üretici** | V3 heavy worker / real_processor DINO aşaması |
| **Onarım** | Job `Artifact.DINO` `QueueKind.HEAVY|REPAIR`; gap scan COMPLETE; `queue_missing_ai_embeddings` (legacy) |
| **UI** | stage dino / `db_dino_embeddings` / `legacy_dino` |
| **Duplicate** | `index_verify` “embedding” = dino **veya** clip; integrity ayrı flags |

### 5) CLIP / semantic feature status

| | |
|--|--|
| **Kontrol** | CLIP: `fe.clip_embedding`. Semantic tags: `texture_map.semantic_tags` (`texture_map_has_semantic_tags` / `_SEM`) — **CLIP ≠ semantic** |
| **DB** | `features.clip_embedding`; `features.texture_map` JSON |
| **Üretici** | Heavy CLIP; semantic TEXTURE sonrası (`ARTIFACT_DEPENDENCIES`) |
| **Onarım** | Jobs `clip`, `semantic`; `backfill_missing_semantic_tags` (legacy) |
| **UI** | openclip/clip + semantic pools |
| **Duplicate** | “semantic feature” adı UI’da CLIP ile karışabilir; SSOT ayrılmış |

### 6) Texture status

| | |
|--|--|
| **Kontrol** | `texture_features` liste + `phash` (`_TEX`); integrity’de `texture_map` genişliği farklı tanım |
| **DB** | `features.texture_features`, `phash`; `texture_version` (integrity) |
| **Üretici** | Heavy texture extract |
| **Onarım** | Job `Artifact.TEXTURE` |
| **UI** | texture / texture_db / legacy_texture |
| **Duplicate** | `index_integrity.evaluate_row` texture = texture_map içeriği; V3 = texture_features+phash → **tanım drift** |

### 7) Object / Owlv2 status

| | |
|--|--|
| **Kontrol** | Object concept: `texture_map.visual_concept_dna` / `global_object_intelligence…`. OWLV2: preview physical + `ObjectIndexStore.ovd_has_done_scan(fid)` (`_owlv2_artifact_ready`) |
| **DB** | `features.texture_map`; `object_index.db` (patterns.db sibling); JobStore `artifact=owlv2` |
| **Üretici** | Side lane (AI_FINAL’e girmez); `apply_owl_queue_policy` |
| **Onarım** | Jobs `object_concept`, `owlv2`; `owlv2_queue_bands` |
| **UI** | `v3_status_dict["owlv2"]` progress |
| **Duplicate** | Pipeline dashboard eski alanlarda OWL yok; yalnız V3 ui_bridge |

### 8) Pattern DNA status

| | |
|--|--|
| **Kontrol** | `texture_map.pattern_dna` (`texture_map_has_pattern_dna` / `_DNA`); SEMANTIC bağımlılığı |
| **DB** | `features.texture_map` |
| **Üretici** | Heavy DNA aşaması |
| **Onarım** | Job `dna`; legacy semantic+dna backfill |
| **UI** | dna / pattern_dna_count / pending_pattern_dna |
| **Duplicate** | `index_verify._has_dna` vs guards — benzer |

### 9) OCR status

| | |
|--|--|
| **Kontrol** | `files.ocr_processed=1` → READY; `ocr_error` → INVALID; boş text ≠ fail (`artifact_state` yorumu) |
| **DB** | `ocr_processed`, `ocr_text`, `ocr_error` |
| **Üretici** | Post-GA OCR job (`POST_GA_ARTIFACTS`) |
| **Onarım** | Job OCR; `db.queue_files_missing_ocr`; settings.ocr_enabled |
| **UI** | ocr pool; HealthPanel `ocr_pending` |
| **Duplicate** | `verify_index` OCR’yi **boş ocr_text** ile sayar → V3 `ocr_processed` ile **çelişki** |

### 10) AI Final status

| | |
|--|--|
| **Kontrol** | `FileArtifactReport.ai_final` = tüm `AI_FINAL_REQUIRED` (HASH, METADATA, DINO, CLIP, TEXTURE, SEMANTIC, DNA). UI coverage: `_AI_FINAL_COVERAGE` (physical preview şart değil); gate’li: `_AI_FINAL` |
| **DB** | features + files genişlik/meta; fiziksel preview bayrağı gate için |
| **Üretici** | Genel AI lane tamamlanınca |
| **Onarım** | Mode.COMPLETE/REPAIR sparse heavy gaps; **Object/OWL/PATCH/OCR AI_FINAL’i düşürmez** |
| **UI** | `ai_final_ready` / `ai_final_gate_ready` / pct_ai_final |
| **Duplicate** | `artifact_status.ai_final` preview+heavy (patch dahil eski set) — V3 patch’i dışarıda bırakır |

### 11) Preview artifact physical status

| | |
|--|--|
| **Kontrol** | `local_artifact_exists(feature_preview_path)`; `physical_preview_ready`; cache adayları `physical_reconcile._cache_candidates` (`feature_previews/{fid}_fp.webp`) |
| **DB** | path + `physical_preview_ready` + `last_verified` (count_physical_readiness) |
| **Üretici** | Preview create / reconcile |
| **Onarım** | Bayrak hizala; path rediscover; JobStore preview reopen |
| **UI** | physical_ready / missing / unverified (HealthPanel `set_reconciliation_status`) |
| **Duplicate** | CacheReconciliationEngine + BackgroundIndexScan physical_reconcile + app_status periyodik reconcile |

### 12) Stale / invalid artifact

| | |
|--|--|
| **Kontrol** | `ArtifactStatus.INVALID` (disk yok `require_disk`, `patch_error`, `ocr_error`, bad preview); `preview_status` invalid/suspicious; JobStore `error_msg=physical_artifact_missing`; `legacy_*` = DB feature var, physical preview gate yok |
| **DB** | yukarıdakiler + JobStore state/error |
| **Üretici** | assess + self-heal invalidate + reconcile |
| **Onarım** | Preview: unlink cache + requeue; Heavy: sparse enqueue (reopen_done=False); stale claims `requeue_stale_claims` |
| **UI** | legacy sayaçları; repair stats |
| **Duplicate** | “stale” kelimesi: physical flags, JobStore claimed, light_status processing (`reset_stale_processing`) |

### 13) Failed / Permanent failed

| | |
|--|--|
| **Kontrol** | JobStore `failed_permanent`; `files.light_status/heavy_status='failed'`; quarantine_reason; `repair_retryable=0` |
| **DB** | `index_v3_jobs.state`; `files.quarantine_reason`, `repair_failure_reason`, `error_msg` |
| **Üretici** | `JobStore.fail(permanent=True)`; cache `classify_light_repair_failure` non-retryable; quarantine classify |
| **Onarım** | `reopen_failed_permanent` yalnız REPAIR/eksik tamamla; hard quarantine tekrar denenmez |
| **UI** | failed_permanent_files; quarantine_summary; FormatStatusPanel “Hatalı” |
| **Duplicate** | Production Self-Heal “LIGHT_FAILED” mapping ≈ failed_permanent_files |

### 14) Repair pending

| | |
|--|--|
| **Kontrol** | JobStore `state=pending` (+ `queue=repair` veya reopen edilen light/preview/heavy); `count_pending_retry`; heal enqueue |
| **DB** | `index_v3_jobs`; files `preview_status` heal işaretleri |
| **Üretici** | planner REPAIR/COMPLETE; background heal; cache reconcile queue |
| **Onarım** | Worker drain — **yeni motor değil** |
| **UI** | LIGHT_PENDING, GENERAL_AI_PENDING, executable vs remaining (`live_contract.remaining_vs_executable`) |
| **Duplicate** | “pending” hem JobStore hem `files.light_status=pending` (legacy SSOT `index_ssot`) |

### 15) Missing metadata

| | |
|--|--|
| **Kontrol** | `Artifact.METADATA`: `format_metadata` veya `width>0`; Hash: `phash` |
| **DB** | `files.format_metadata`, `width`, `height`; `features.phash` |
| **Üretici** | Genel AI (Hızlı indeks hash yazmaz — ui_bridge yorumu) |
| **Onarım** | Jobs hash/metadata |
| **UI** | hash/metadata pools |
| **Duplicate** | Eski LIGHT_ARTIFACTS (`artifact_status`) hash+metadata’yı light sayar; V3 light yalnız thumb+preview |

### 16) Index/DB vs real file inconsistencies

| | |
|--|--|
| **Kontrol** | `legacy_preview` / `legacy_thumbnail`; path var physical=0; physical=1 path boş; DB embedding var kaynak `status=missing`; FAISS dosya boyutu vs DB count (yalnız health snapshot); JobStore file_id orphan → `purge_unknown_file_ids` |
| **DB** | files ↔ features ↔ v3jobs ↔ object_index ↔ cache FS |
| **Üretici** | Cache silme / NAS kesintisi / freeze write (`INDEX_FROZEN_WRITE_BLOCKED` job hataları) |
| **Onarım** | `reconcile_stale_physical_flags`; Mode.REPAIR preview; **yeniden DINO üretme (legacy ≠ recompute)** |
| **UI** | legacy_* diagnostikleri; physical sync % |
| **Duplicate** | Üç reconcile yolu (physical_reconcile, CacheReconciliationEngine, artifact_status.reconcile_source_artifacts) |

---

## 3. Birleşik sağlık modeli (mevcut durumlara bağlı)

**Önerilen enum (yeni paralel motor değil — mevcut state’lerin projeksiyonu):**

| Unified | Bağlandığı gerçek durumlar |
|---------|----------------------------|
| **HEALTHY** | Artifact READY + (preview için physical ready veya self-heal VALID); AI_FINAL coverage hedefe ulaşmışsa dosya “arama hazır” |
| **MISSING** | `ArtifactStatus.MISSING` veya path yok / stage henüz üretilmemiş; JobStore’da job yok veya henüz planlanmadı |
| **STALE** | DB path/feature var ama physical bayrak 0 / disk yok (`legacy_*`, `physical_artifact_missing`); AI var kaynak missing |
| **BROKEN** | Self-heal `Verdict.INVALID` (misrepresenting/corrupt/unusable); **kaynak-uyumlu solid mono BROKEN değil**; `patch_error`/`ocr_error` ilgili artifact’te |
| **REPAIR_PENDING** | JobStore `pending`/`claimed` (özellikle reopen/REPAIR); `preview_status` suspicious queued; cache repair aktif |
| **FAILED** | `failed_permanent`; hard quarantine; `repair_retryable=0`; light/heavy `failed` |
| **UNKNOWN** | Assess edilmedi; scope dışı; disk stat yapılamadı; SUSPICIOUS kaynak karşılaştırılamadı |

### Ayırt edilmesi gereken senaryolar

| Senaryo | Unified | Mevcut sinyal |
|---------|---------|---------------|
| Dosya var, preview yok | MISSING (+ REPAIR_PENDING if queued) | PREVIEW MISSING; planner preview job |
| Preview DB’de, fiziksel artifact yok | STALE / BROKEN→invalidate | physical=0 veya `preview_missing_physical`; legacy_preview |
| Preview unused/white/broken | BROKEN yalnızca INVALID; mono+mono kaynak = HEALTHY | `preview_self_heal.validate` |
| Preview OK, AI feature eksik | MISSING (feature) | assess heavy MISSING; heavy_queue |
| AI OK, kaynak dosya gitmiş | STALE (orphan) + file MISSING | status missing + features row |
| Executable ≠ kalan | (bilgi) | `live_contract.remaining_vs_executable` — UI’da karıştırma |

**Dosya özeti rollup (UI kovası):** worst-of önceliği:  
`FAILED > BROKEN > REPAIR_PENDING > STALE > MISSING > UNKNOWN > HEALTHY`  
(AI eksikliği preview sağlıklıyken “Eksik”; bozuk preview “Bozuk”.)

---

## 4. Önerilen basit UI özeti

**Başlık:** `Arşiv Sağlığı`

| Kova (TR) | Unified | Kaynak sayaç (reuse) |
|-----------|---------|----------------------|
| **Sağlıklı** | HEALTHY | `preview` physical + ilgili feature READY; opsiyonel `ai_final_coverage` |
| **Eksik** | MISSING | `waiting_preview`, `heavy_queue`, `pending_*` from `build_pipeline_status` / `count_v3_ssot` |
| **Onarım gerekli** | REPAIR_PENDING + STALE | JobStore pending (repair/reopen) + `legacy_*` + heal candidates |
| **Bozuk** | BROKEN | `preview_invalid%`, self-heal INVALID, corrupt quarantine soft-set |
| **Başarısız** | FAILED | `failed_permanent_files`, hard quarantine, non-retryable repair |

Detay drill-down: mevcut stage satırları (Thumbnail, Preview, DINO, CLIP, Texture, Semantic, DNA, Object, OWL, OCR, AI Final) — `v3_status_dict` / `STAGE_POOL_KEYS`.  
Aksiyon butonları **yeni motor açmasın:** “Eksikleri Tamamla” → mevcut `Mode.REPAIR`; physical → mevcut reconcile; bozuk preview → mevcut self-heal candidate enqueue.

---

## 5. Eksik parçalar

1. **Tek bir “ArchiveHealthReport” aggregator yok** — sayılar `ui_bridge`, `app_status`, `cache_reconciliation`, `index_verify`, `index_integrity` arasında dağınık.
2. **Unified enum / dosya kartı etiketi yok** — UI “Sistem Sağlığı” çoğunlukla host metrik + physical batch.
3. **OWL/Object** production `verify_index.CHECKS` ve eski integrity TASK listesinde yok/eksik.
4. **OCR doğruluk drift:** verify boş text sayar; V3 `ocr_processed` kullanır.
5. **Texture tanım drift:** integrity vs V3 assess.
6. **`artifact_status` (RC2) vs `artifact_state` (V3)** çift API; AI_FINAL/patch farkı.
7. **Preview content health** (VALID/INVALID) pool sayılarına yansımıyor — yalnız `preview_status` + heal aday sorgusu.
8. **Kaynak-gone + AI-ok** orphan için tek birleşik sayaç yok.
9. **HealthPanel “Eksikleri Onar”** V3 JobStore REPAIR yerine legacy `index_maintenance` çağırıyor → kullanıcı beklentisi “Eksikleri Tamamla” ile ayrışabilir.

---

## 6. Duplicate sistemler

| Konu | Sistem A | Sistem B | Sistem C | Not |
|------|----------|----------|----------|-----|
| Artifact ready | `artifact_state.assess_*` | `artifact_status.assess_*` | `index_integrity.evaluate_row` | Üç tanım |
| Physical cache | `physical_reconcile` | `CacheReconciliationEngine` | `update_physical_readiness` callers | Aynı hedef |
| Eksik onarım | Mode.REPAIR + JobStore | `repair_index` / Self-Heal production | cache `_repair_row` / `_repair_ai_row` | Üç giriş |
| Preview bozuk | preview_self_heal | cache reconcile recreate | index_verify isfile missing | Content vs existence |
| Kuyruk pending | JobStore | `index_ssot` light/heavy_status | IndexQueueManager (legacy) | V3 geçiş artığı |
| Health UI | HealthPanel | FormatStatusPanel | Index progress (v3_status_dict) | Üç yüzey |
| Embedding eksik | assess DINO/CLIP | verify “embedding” OR | integrity flags | OR vs AND |

**Öneri:** Yeni onarım motoru yok; **tek okuma modeli** `artifact_state` + `count_v3_ssot` + JobStore + preview_self_heal Verdict üzerine projeksiyon. Legacy verify/integrity yalnız “doctor” modunda kalsın veya V3 sinyallerine map edilsin.

---

## 7. Önerilen minimal entegrasyon noktası

**Tek ince aggregator (gelecek patch):** örn. `core/archive_health.py` (yeni, küçük):

1. Girdi: `db`, `JobStore`, `source_ids`, isteğe bağlı `assess_file` örneklemesi.
2. Bulk: `count_v3_ssot` + `JobStore.count_pending*` / `count_failed_permanent*` + `db.count_physical_readiness` + quarantine report.
3. Preview content: **yalnız** mevcut aday sorgusu (`heal_candidates` SQL koşulları) — full-archive decode yok.
4. Çıktı: kovalar `{healthy, missing, repair_needed, broken, failed}` + per-stage MISSING/READY.
5. UI: `HealthPanel` altına “Arşiv Sağlığı” satırı **veya** index progress yan paneli; butonlar mevcut `Mode.REPAIR` / background scan / self-heal hooks.

**Dokunulmayacak:** `Indexer` / ranking / embedding extract algoritmaları; JobStore şema; assess READY kuralları (projeksiyon katmanı).

---

## 8. Değişecek dosyalar (future patch list — şimdi uygulanmaz)

| Dosya | Değişiklik türü |
|-------|-----------------|
| `core/archive_health.py` | **Yeni** — unified projeksiyon + TR kova sayıları |
| `ui/health_panel.py` | Arşiv Sağlığı özet etiketleri; “Eksikleri Onar” → V3 REPAIR köprüsü (isteğe bağlı) |
| `core/index_v3/ui_bridge.py` | İsteğe bağlı: `v3_status_dict` içine `archive_health` alt dict |
| `core/app_status.py` | İsteğe bağlı: cached status’a kova alanları |
| `core/production/index_verify.py` | Align: OCR=`ocr_processed`; embedding=dino∧clip; OWL opsiyonel — veya deprecated uyarı |
| `core/index_pipeline_status.py` | Kova özet alanları (ince) |
| `ui/format_status_panel.py` | İsteğe bağlı link: format eksik ↔ archive health |
| Testler | `tests/test_archive_health_*.py` (projeksiyon birim testleri) |

**Değişmeyecek (bilinçli):** `core/indexer.py` çekirdek, FAISS search, ranking, `real_processor` feature extract mantığı (yalnız mevcut heal hook’ları zaten var), JobStore state makinesi.

---

## 9. Indexer’a neden dokunulmayacağı

1. Sağlık görünümü **okuma/projeksiyon** problemidir; üretim hattı zaten `artifact_state` + JobStore + self-heal ile tanımlı.
2. Indexer / heavy extract değişikliği AI skorları, timing, freeze-write ve race riski taşır.
3. Eksik tamamlamayı JobStore + Mode.REPAIR zaten yapıyor; paralel “repair engine” drift üretir (mevcut duplicate’ların kök nedeni).
4. Solid mono false-positive politikası preview_self_heal’de çözülmüş; indexer’a “beyaz=bozuk” eklemek regresyon olur.
5. Live contract: Tamamlanan = READY pool; Executable = pending — indexer’ı değiştirmeden UI bu ayrımı gösterebilir.

---

## 10. Test planı

1. **Unit — projeksiyon:** Sentetik `FileArtifactReport` + sahte JobStore state → beklenen Unified enum (16 senaryo tablosu).
2. **Mono koruması:** Solid white preview + solid white source → HEALTHY/VALID; colorful source + white preview → BROKEN/INVALID (`validate.compare_source_vs_preview`).
3. **Physical stale:** path set, file deleted, physical=1 → reconcile sonrası STALE/MISSING + legacy_preview artışı.
4. **DB preview / FS missing:** `feature_preview_path` dolu, `isfile=False` → INVALID/STALE; heal enqueue preview job.
5. **AI missing / preview OK:** PREVIEW READY, DINO MISSING → kova Eksik; AI_FINAL false.
6. **Orphan AI:** features var, `files.status=missing` → STALE/FAILED kova politikası.
7. **JobStore:** `failed_permanent` → Başarısız; `pending` repair → Onarım gerekli; OCR permanent dosya sayımında hariç.
8. **Regression:** `count_v3_ssot` sayıları aggregator ile birebir (thumbnail/preview/ai_final).
9. **UI smoke:** HealthPanel özet güncellenir; host CPU satırları bozulmaz.
10. **No-write freeze:** INDEX_FROZEN ortamında aggregator yalnız okur; enqueue çağırmaz (buton ayrı).

---

## 11. Kaynak dosya envanteri (ah_audit)

```
ah_audit/core/
  artifact_status.py
  app_status.py
  cache_reconciliation.py
  global_preview.py
  index_file_status.py
  index_integrity.py
  index_pipeline_status.py
  index_ssot.py
  preview_cache.py
  quarantine.py
  index_v3/{artifact_state,live_contract,physical_reconcile,queues,ui_bridge}.py
  preview_self_heal/{__init__,hooks,repair,validate}.py
  production/{health_monitor,index_verify,recovery,self_healing}.py
ah_audit/ui/{health_panel,format_status_panel}.py
(+ çapraz: index_v3/{background_scan,discovery,planner,engine,worker,types,real_processor}.py)
```

---

## 12. Sonuç

Vezir’de Archive Health için **yeterli SSOT ve onarım yolları zaten var**. Asıl boşluk: bunları tek TR özetinde birleştiren **ince bir projeksiyon katmanı** ve Health UI’da “Arşiv Sağlığı — Sağlıklı / Eksik / Onarım gerekli / Bozuk / Başarısız” kovaları.  
Paralel repair engine **önerilmez**; Eksikleri Tamamla (`Mode.REPAIR`), Preview Self-Healing, JobStore, `count_v3_ssot` ve physical reconcile **reuse** edilmelidir. Indexer çekirdeğine dokunulmamalıdır.

---

*Rapor sonu — AUDIT ONLY, 2026-09-19 Europe/Istanbul.*
