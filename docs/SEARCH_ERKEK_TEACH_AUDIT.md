# ERKEK öğretme → aramada çıkmama — AUDIT (no patch)

**Tarih:** 2026-09-20 Europe/Istanbul  
**Kapsam:** salt okuma; Indexer/JobStore dokunulmadı.

## ROOT CAUSE (kesin cümle)

**Inspector’da görünüyor çünkü** Düzenle/öğret sonucu `files.manual_category_path="insan/erkek"` (+ `category_source=manual_user`, texture_map `user_labeled`) yazılıyor ve inspector `category_predictions_from_metadata` ile bunu **admin_label → "erkek" %100** olarak gösteriyor — bu yol arama FTS’inden bağımsız.

**Arama sonucunda güvenilir şekilde çıkmıyor / kullanıcı göremeyebiliyor çünkü** sorgu `"erkek"` `query_intent_router` tarafından **gender/person** kanalına alınıyor (pattern/learned değil); klasik metin indeksi (`text_search_blob` / `files_fts`) neredeyse boş ve öğretilen id’leri içermiyor; `_label_text_search` `manual_category_path` üzerinde `erkek` aramıyor. Learned enjeksiyon kodda mevcut ve motor testinde exact id’leri %96 ile üretebiliyor, ama gender/face+UVI yolu + boş FTS, kategori öğretisini normal metin aramasıyla birleştirmiyor (dudak/leopar’dan farklı).

---

## 1) Öğretme nereye yazılıyor?

| Yazım | Dosya | Ne olur |
|-------|-------|---------|
| Inspector **Düzenle** | `ui/main_window.py` `_on_edit_result_metadata` → `teach_me.apply_metadata_edit_to_files` / category learning | Kategori path |
| **Doğru** (AI tahmini) | `_on_ai_prediction_action` → `UserFeedbackStore.record(..., ai_category_correct)` | Feedback; kategori path değil |
| Manuel kategori | `core/category_learning.py` `apply_manual_category` | `files.category_path`, `manual_category_path`, `text_search_blob`, FTS, texture_map |

Canlı DB (`patterns.db`) örnekleri:

- file_id **528, 562, 8470, 8474, 19847** → `manual_category_path=insan/erkek`, `category_source=manual_user`
- Path’ler çoğunlukla `F:\karşıdan yüklemeler\...` veya `\\server\...\erkek\...`
- `status=pending` (indexed değil); bazıları `physical_preview_ready=1`

---

## 2) Hangi katmanlara ulaşıyor?

| Katman | Ulaşıyor mu? | Kanıt |
|--------|--------------|-------|
| `files` kategori kolonları | **EVET** | 5 satır `insan/erkek` |
| `texture_map` / user_labeled | **EVET** (öğreti yolu) | category_learning |
| `text_search_blob` | **HAYIR (boş)** | blob_len=0; global 48948/49039 boş |
| `files_fts` | **HAYIR** | FTS count=91; taught id’ler yok; LIKE erkek=0 |
| `concept_registry` | **EVET** | id=69 canonical=`erkek` parent=`insan` |
| `concept_examples` tablo | **HAYIR (0 satır)** | SELECT boş |
| positive example vectors | **EVET** | file_id 528, 8470 (+ example_file_ids→19847) |
| search_memory overlay | olası (INDEX_FROZEN) | category_learning memory_only dalı |
| customer memory | hayır (bu bug için) | — |
| Search Intelligence / gender | **EVET (yanlış kanal)** | intent kind=`gender` |
| embedding/prototype (CLIP neighbor) | kısmi | learned pack neighbor_scores={} |

---

## 3) Inspector "Erkek %100" kaynağı

`ui/inspector_panel.py` → `category_predictions_from_result` → `core/category_predictions.py`:

- `user_labeled` / `manual_category_path` → `source=admin_label`, confidence ≥0.95
- Canlı simülasyon: `[{label: erkek, confidence: 1.0, category_path: insan/erkek, source: admin_label}]`
- Başlık simple mode: **"Desen tahmini"**

Bu, FTS/learned search skorundan **bağımsız metadata projeksiyonu**.

---

## 4) "erkek" Metinle Ara ne sorgular?

`SearchEngine.search_by_text` (`search_engine.py` ~2670+):

1. Intent: `classify_query("erkek")` → `{kind: gender, value: MALE, channel: person}`
2. Face index (açık) + adaylar: FTS / filename LIKE / OCR / `_label_text_search`
3. `collect_learned_hits` → exact_ids
4. Gender ise `human_semantic_mode` + UVI ranking

Karşılaştırma:

| Sorgu | Intent kanalı |
|-------|---------------|
| erkek / kadın | **gender / person** |
| leopar | **pattern** |
| dudak | **learned** |

`query_family_hints("erkek")` → `{}`  
`query_family_hints("leopar")` → `{pattern_family: animal_print, animal_print_type: leopard}`

`_label_text_search`: hints boşsa yalnızca `_custom_tag_label_search` (user_feedback custom_tag exact) — **manual_category_path’e bakmaz** → canlıda 0 hit.

---

## 5–6) Öğretilmiş Erkek ↔ text search bağlantısı

Bağlantı **kısmi**:

- Learned: `resolve_learned_concept` + `collect_learned_hits` exact_ids=`[528,8470,19847]` skor 0.96 üretir.
- Motor testi (`search_by_text`, load_ai=False, threshold=0.30): **3 sonuç, üçü de taught, %96, learned_concept_exact=True**.
- Ama FTS/blob yolu taught id’leri **aday olarak üretmez**; text_search_score(528)=0.0 (path/filename’de erkek yok); 19847≈0.71 (klasör adında erkek).

Neden kullanıcı yine de görmeyebilir:

1. Gender kanalı face/UVI odaklı; kategori öğretisi birincil metin kanıtı değil.
2. Boş FTS → aday havuzu path’te "erkek" geçen rastgele klasörlerle dolabilir.
3. `status=pending` + karşıdan yüklemeler — UI/kapsam algısı.
4. `similarity_threshold=0.6` textile için; learned özel geçiş var ama gender_only+UVI sonrası debug işaretleri karışabilir.
5. Yeni öğreti henüz vektör/exact_ids’e yazılmamışsa yalnızca inspector metadata kalır.

---

## 7–9) Senaryo / karşılaştırma

| Kavram | Kategori ağacı | Intent | FTS/label | Learned exact |
|--------|----------------|--------|-----------|---------------|
| Erkek | `insan/erkek` (resolve_category_query("erkek") **boş**) | gender | blob/FTS yok; label 0 | registry+vektör var |
| Kadın | `insan/kadın` | gender | blob boş | (benzer) |
| Leopar | Animal Print/Leopard | pattern | family hints var | concept Leopard |
| Dudak | — | **learned** | — | learned kanalı |

---

## 10) Dosya / satır özeti

- `core/category_predictions.py` ~84–97 — inspector %100
- `core/category_learning.py` ~65–234 — DB yazım + blob
- `core/text_index.py` `build_text_search_blob` — category_path kwarg; learning çağırması eksik/boş blob
- `core/db.py` `search_text_candidates` / `_label_text_search` ~2168–2414 — FTS+label
- `core/db.py` `_upsert_fts` — boş blob yazmaz
- `core/query_intent_router` — erkek→gender
- `core/learned_concept_search.py` `collect_learned_hits` / `apply_learned_to_results`
- `core/search_engine.py` ~2737–2810 face; ~3850–4069 human_semantic; ~1810–1813 learned threshold pass

---

## En küçük düzeltme yönü (PATCH YOK — sadece öneri)

1. Gender sorgusunda learned exact / manual_category_path eşleşmelerini face yokken de aday+eşik koruması.
2. Öğretide `text_search_blob` + FTS’i zorunlu doldur (`insan erkek` alias).
3. `_label_text_search`: `manual_category_path LIKE %term%`.
4. İsteğe bağlı: `erkek` için learned kanalını gender ile OR’la (dudak gibi).

