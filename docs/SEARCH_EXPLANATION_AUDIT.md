# SEARCH EXPLANATION AUDIT

**Kapsam:** `/workspace/vezir_audit/sx_audit/` — yalnızca okuma; ürün dosyasına yazılmadı.  
**Rapor:** bu dosya.  
**Tarih:** 2026-09-20 (Europe/Istanbul)  
**Örnek sorgu:** `Ünal Tekstil kırmızı leopar`

---

## Zorunlu kısa cevaplar

### SearchResult’ta reason / evidence / score breakdown / query meaning / matched attributes var mı?

| İhtiyaç | Durum | Konum |
|--------|-------|-------|
| **Reason** | **EVET** | `match_explanations`, `cluster_reason`, `text_match_reason` (`search_engine.py:188–190`); ayrıca `debug["search_reason"]` |
| **Evidence** | **EVET (çoğu `debug`)** | `debug["concept_evidence"]`, `debug["color_evidence_score"]`, `texture_map["color_evidence"]`, learned bayrakları; ayrı top-level `evidence` alanı yok |
| **Score breakdown** | **EVET** | `breakdown`, `text_score_breakdown` (`:174`, `:191`) |
| **Query meaning** | **EVET (`debug`)** | `debug["query_meaning"]` — top-level alan değil (`search_intelligence_chain.py:281–297`) |
| **Matched attributes** | **EVET (`debug` + bayraklar)** | `debug["query_attribute_intel"]`, `debug["visual_variant_intel"]`; `same_pattern_family` / `animal_print_type` / `color_family` top-level |

### Search reason’lar debug UI’da gösteriliyor mu?

**Kısmen evet.**

- **Evet:** Inspector `QGroupBox("Neden bu sonuç?")` + `build_result_explanation` / `match_explanations` chips (`inspector_panel.py:206–211`, `:615+`, `:911–937`); Doku sekmesi `cluster_reason` (`:864`); Skor Detayı teknik bileşenler (`:732+`); kartta ≤3 `reason_chips` (`result_card.py:265–268`).
- **Hayır / zayıf:** `debug.search_reason` ve `debug.query_meaning` ayrı kullanıcı satırı olarak **yok**; `concept_evidence` / `color_evidence_score` “Neden”e **bağlı değil**; `filter_explanations_for_query` tanımlı ama card/inspector **çağırmıyor**.

---

## 1. MEVCUT EXPLANATION KABİLİYETİ

“Why this result?” için **yeni motor yok ve gerekmiyor**. İki katman zaten çalışıyor:

1. **Backend evidence + reason üretimi**
   - `SearchResult` reason/breakdown/family/color alanları.
   - `search_intelligence_chain.apply_search_intelligence_chain` — query meaning → soft katmanlar → `debug.search_reason` + `debug.query_meaning` (skor yeniden yazmadan reason attach).
   - `concept_evidence`, `color_evidence`, `learned_concept_search`, `customer_discovery`.
   - Metin yolu: score reasons → `match_explanations` / `text_match_reason` / `text_score_breakdown` (`search_engine.py:3759–3769`).
   - Görsel yolu: `similarity_explanation.build_match_explanations` → `match_explanations` + `cluster_reason` (`:4840–4857`) — *modül sx_audit kopyasında yok, import ürün ağacına bağlı*.
   - `pattern_explanation.build_result_explanation` — şablon UI paketi (`title`, `reasons`, `search_reason`, `category_reason`, `confidence`, `dna`).

2. **UI explanation yüzeyi**
   - Result card: badge + tier + % + Güven + reason chips.
   - Inspector: “Neden bu sonuç?” / simple’da “Neden benzer?”, Pattern DNA, Skor Detayı, “Neden bu gruba girdi?”.
   - `designer_labels`: `reason_chips`, `filter_explanations_for_query`, `resolve_display_query_text` ← `debug.query_meaning`.

**Özet:** Kanıt ve reason **üretiliyor**. Asıl boşluk UI’ya **tam bağlama / sadeleştirme**.

---

## 2. BACKEND EVIDENCE

### 2.1 SearchResult alanları — `core/search_engine.py:161–193`

```text
# search_engine.py:161–193
file_id, path, filename, customer, thumbnail_path
score, score_percent
feature_preview_path, width, height, file_size, mtime
breakdown: dict[str, float]
source_name, source_type, category, cluster_group
color_family, pattern_family, animal_print_type
hierarchy_score, palette_similarity
same_color_family, same_pattern_family, same_animal_family
texture_family_score
cluster_reason: str
text_match_reason: str
match_explanations: list[str]
text_score_breakdown: dict[str, float]
is_self_match
debug: dict[str, Any]
```

`customer` = **dosya** müşteri etiketi. Sorgu müşterisi → `debug.query_meaning["customer"]`.

### 2.2 Senaryo `Ünal Tekstil kırmızı leopar` — ne hesaplanıyor?

| Sinyal | Hesap? | Kaynak | Örnek / anahtar |
|--------|--------|--------|-----------------|
| **Customer** | EVET | `customer_discovery.extract_customer_from_query` → `analyze_query_intelligence` | `customer`, `customer_high_confidence`, `scopes`, `search_text≈"kırmızı leopar"` |
| **Concept** | EVET | `concept_core`, attributes.motif; Leopard alias (`search_intelligence_chain.py` ~131–137) | `concept_core≈leopar`, `concept_relation_hint` |
| **Color** | EVET | `colors` + `apply_color_evidence_scoring` | `want`/`hits` in `debug.color_evidence_score` |
| **Visual similarity** | EVET | `breakdown` dino/clip/texture/… | debug `dino_score` / `clip_score` |
| **Family** | EVET | `pattern_family`, `animal_print_type`, `same_pattern_family` | animal_print / leopard |
| **Variant** | EVET | `debug.visual_variant_intel` → reason `VARIANT` | `build_result_reason` |
| **Learned evidence** | EVET* | `learned_concept_search` `LEARNED_REASON="Öğrenilmiş kavram"` (:18, :869–873) | `match_explanations`, `learned_concept_exact` |
| **Customer evidence** | EVET* | `concept_evidence` customer soft bonus (~+0.03) | `debug.concept_evidence["customer_soft_bonus"]` |
| **Confidence** | EVET | analysis `confidence`; DNA conf; UI `compute_confidence` | 0–1 / “Yüksek” |
| **Score components** | EVET | `breakdown`, `text_score_breakdown` | phash, clip, learned_concept, … |
| **Reason** | EVET | `match_explanations`, `cluster_reason`, `text_match_reason`, `debug.search_reason` | chip + compact label |

\*öğreti / müşteri örnekleri varsa.

### 2.3 Alan → file:line + örnek değerler

**Metin yolu reason/breakdown** — `search_engine.py:3759–3769`

```text
text_match_reason = " | ".join(reasons[:3])  # sonra ✓ birleşimi
match_explanations = reasons[:4]
text_score_breakdown = breakdown
breakdown = {**breakdown, **breakdown}
```

Örnek reason stringleri (aynı akış ~3627–3678): `"Öğrenilmiş kavram"`, `"Marka kanıtı"`.

**Görsel yolu** — `search_engine.py:4840–4866`

```text
match_explanations = build_match_explanations(...)
cluster_reason = cluster["cluster_reason"]
color_family / pattern_family / animal_print_type / palette_similarity / same_*_family
```

Cluster debug örneği (~6044–6062): `cluster_reason`, `palette_similarity`, `same_pattern_family`, …

Breakdown varsayılan anahtarları (~5385–5402): `phash`, `dhash`, `whash`, `color`, `texture`, `patch`, `dna`, `semantic`, `dino`, `clip`, …

**Intelligence chain** — `search_intelligence_chain.py`

| Fonksiyon | Satır | Yazdığı şey |
|-----------|-------|-------------|
| `analyze_query_intelligence` | 19–187 | `concept_core`, `colors`, `attributes`, `customer*`, `scopes`, `confidence`, `nl_parse` |
| `build_result_reason` | 190–270 | compact: `EXACT`, `LEARNED …`, `CONCEPT + RED`, `VARIANT`, `SAME FAMILY`, `VISUAL SIMILAR`, … |
| `attach_result_reasons` | 273–299 | `debug.search_reason`, `debug.query_meaning` |
| `apply_search_intelligence_chain` | 302–415 | attribute → concept_evidence → discriminative → object_gate → variant → color → reasons |

`debug.query_meaning` şekli (:281–297):

```python
{
  "concept_core": "leopar",
  "colors": ["kırmızı"],
  "attributes": {...},
  "intent_type": "...",
  "context": "...",
  "visual_type": "...",
  "normalized": "...",
  "search_text": "kırmızı leopar",
  "customer": "Ünal Tekstil",
  "customer_high_confidence": True,
  "scopes": [...],
  "confidence": 0.75,
  "nl_parse": {...},
}
```

Çağrı: `search_engine.py:4071–4089` (`apply_search_intelligence_chain`).

**Concept evidence** — `concept_evidence.py:577+`

- Profile: motif, scale/density confidence, colors, `attribute_coverage`, …
- Sonuca: `debug["concept_evidence"] = {applied/aligned/delta/parts, canonical, dna_*, customer_key, customer_soft_bonus?}`.
- Soft tavan `_MAX_EVIDENCE_BONUS` (0.06 civarı); customer soft ~0.03 (:684).

**Color evidence** — `color_evidence.py`

- Palette dict defaults (:249–311): `detected_colors`, `color_ratios`, `color_family`, `confidence`, `ratio_confidence="unknown"|"cluster_weight"`, `source="ai"`.
- Match meta (:488–496): `applied`, `delta`, `hits`, `misses`, `have`, `want`.
- Yazım: `debug["color_evidence_score"]` (:543–560); `apply_color_evidence_scoring` (:499).

**Customer discovery** — `customer_discovery.py:551–680`

Dönüş: `customer`, `score`, `high_confidence`, `ambiguous`, `matched_span`, `search_text`, `scope_prefixes`, `candidates`.  
(Chain bunu `customer_high_confidence` / `scopes` olarak map’ler — `:76–79`.)

**Pattern explanation** — `pattern_explanation.py:8–103`

```python
{"title", "reasons"[:8], "group_label", "confidence", "dna",
 "search_reason", "category_reason"}
```

### 2.4 Bu audit kopyasında eksik import’lar (gap notu)

| Modül | sx_audit’te |
|-------|-------------|
| `core.similarity_explanation` | yok |
| `core.confidence_engine` | yok |
| `core.query_attribute_intel` | yok |
| `core.visual_variant_intel` | yok |
| `core.object_pattern_gate` | yok |
| `core.query_evidence` | yalnızca `cm/core/query_evidence.py` |

Davranış ürün ağacında varsayılır; audit kopyası kısmi.

---

## 3. UI — ne görünüyor / ne görünmüyor

### Result card — `ui/result_card.py`

| Görünen | Kaynak |
|---------|--------|
| Family / renk / marka chip | `designer_labels` + `color_family` |
| Tier + % | confidence_engine / `similarity_tier_label` |
| `Güven: …` + tooltip | `compute_confidence` (~203–249) |
| ≤3 reason chip | `match_explanations` veya `cluster_reason` → `reason_chips` (:265–268) |

**Görünmeyen:** `query_meaning`, `search_reason`, concept/color evidence meta, sorgu rengi (dosya `color_family` gösterilir), text breakdown.

Compact view: badge + chips yok.

### Inspector — `ui/inspector_panel.py`

| Yüzey | İçerik | Simple mode |
|-------|--------|-------------|
| **"Neden bu sonuç?"** (simple: **"Neden benzer?"**) `:206`, `:295–296` | chips / `format_explanation_text`; DNA checklist | **Açık** |
| Pattern DNA | DNA alanları | Gizli |
| Skor Detayı | breakdown, text scores, UVI, query_evidence, family % | Tab gizli (`:298–299`) |
| Doku Haritası | family/animal/color + **"Neden bu gruba girdi?"** = `cluster_reason` `:259`, `:864` | Tab gizli |

`search_reason` / `query_meaning` / `concept_evidence` / `color_evidence_score` için **özel etiket yok**.

### designer_labels — `ui/designer_labels.py`

- `resolve_display_query_text` (:108–122): `debug.query_meaning`.
- `filter_explanations_for_query` (:388): tanımlı, **UI’da unused**.
- `reason_chips` (:511), `build_result_display_labels` (:413): provenance’lı etiketler.

### results_panel

Grup/bucket etiketleri (`same_pattern_family` vb.) — explanation metni değil. Seçim → inspector.

---

## 4. EKSİK OLAN

1. `debug.query_meaning` kullanıcı özeti (müşteri / kavram / renk parse).
2. `debug.search_reason` satırı (chain yazıyor, UI okumuyor).
3. `concept_evidence` / `color_evidence_score` → “Neden” chip’leri.
4. `filter_explanations_for_query` wire edilmemiş.
5. Card chips + inspector pattern_explanation + cluster_reason + search_reason = **dağınık kanallar**, tek designer cümlesi yok.
6. Audit kopyası bağımlılık gap’leri (yukarı §2.4).
7. **Yeni skor/index/DINO motoru eksik değil** — önerme.

---

## 5. YENİ MOTOR GEREKLİ Mİ?

### **HAYIR**

Backend zaten query meaning, customer/color/concept/learned evidence, score breakdown ve çoklu reason üretiyor. `search_intelligence_chain` bilinçli orchestration (“no new engine”). `pattern_explanation` şablon katmanı. Eksik olan **kanıt üretimi değil**, mevcut kanıtın **UI’ya bağlanması**.

Indexer / JobStore / Preview / DINO / AI / ranking’e dokunulmamalı.

---

## 6. EN KÜÇÜK GELİŞTİRME (wire-only öneri — bu auditte uygulanmadı)

1. Inspector “Neden …” bloğuna: `debug.search_reason` + kısa `query_meaning` özeti (`müşteri · kavram · renk`).
2. Card chips öncesi: `filter_explanations_for_query(...)`.
3. Color hit → chip: örn. “Kırmızı renk kanıtı” (`color_evidence_score.want/hits`).
4. `concept_evidence.customer_soft_bonus` varsa tek chip; learned zaten `"Öğrenilmiş kavram"`.
5. Simple mode’da Skor Detayı kapalı kalsın; “Neden” yeterli.

**Yapma:** yeni ExplanationEngine, çift skor, DINO/CLIP/index değişikliği.

---

## 7. TEST PLANI

### Mevcut testler

| Dosya | Konu | Explanation ilişkisi |
|-------|------|----------------------|
| `cm_tests/test_learned_concept_search.py` → `test_learned_reason_survives_generic_score` | `LEARNED_REASON` ∈ `match_explanations` | **Doğrudan** |
| `cm_tests/test_customer_discovery.py` | Ünal keşif, scope; `test_14_search_intelligence_untouched_import` | Customer + chain dumanı |
| `cm_tests/test_customer_memory.py` | customer_key prior/evidence | Customer evidence |
| `cm_tests/test_search_memory.py` | kalıcı search memory | Memory |
| `cm_tests/test_user_teach_authority.py`, `test_teach_me.py` | teach authority; boş `match_explanations` fixture | Learned path |
| `sx_audit/tests/` | **boş** | — |

### Wire sonrası önerilen testler

1. `analyze_query_intelligence("Ünal Tekstil kırmızı leopar")` → customer + colors + concept_core; `search_text` müşterisiz.
2. `attach_result_reasons` sonrası her satırda `debug.search_reason` + `debug.query_meaning`.
3. Mock texture_map ile kırmızı want → `color_evidence_score.hits >= 1`.
4. `filter_explanations_for_query` hizasız learned satırını düşürür.
5. UI: chip limiti; `search_reason` label’a yansır.

### Manuel checklist

1. Sorgu: `Ünal Tekstil kırmızı leopar`.
2. Inspector “Neden bu sonuç?” dolu mu?
3. Simple: neden açık, Skor Detayı gizli mi?
4. Kartta family/color + ≥1 chip.
5. Öğretilmiş leopar → “Öğrenilmiş kavram”.
6. İleri mod Skor Detayı: breakdown görünür mü?

---

## Senaryo × katman özeti

| | Backend | UI bugün |
|--|---------|----------|
| Customer | Evet (`query_meaning`) | Doğrudan hayır; dosya `customer` kaynakta |
| Concept | Evet | Kısmen (badge / learned chip) |
| Color | Evet | Kısmen (dosya `color_family` badge) |
| Visual similarity | Evet | Kısmen (% / tier / skor sekmesi) |
| Family | Evet | Evet |
| Variant | Evet (`search_reason`) | Hayır |
| Learned evidence | Evet | Evet (chip) |
| Customer evidence | Evet (soft) | Hayır |
| Confidence | Evet | Evet (kart) |
| Score components | Evet | Evet (Skor Detayı; simple gizli) |
| Reason | Evet | Evet (chips + Neden); `search_reason` eksik |

**Karar:** Explanation kabiliyeti **mevcut**. En küçük iş = **evidence → UI wire**. Yeni motor **gerekli değil**.
