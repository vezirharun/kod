# Pattern-First / Representation-Invariant Search Audit — 20260923_151639

## ROOT CAUSE (kesin)

Full-frame manken aramasında alakasız tekstillerin üste çıkması ve crop’ta leopard’ın düzelmesi **tek bir rastgele ağırlık hatası değil**; üç katmanlı bir davranış:

1. **Global CLIP/DINO embedding kişi/poz/siluet baskın**  
   Full `2.jpg` embedding’i person+garment kompozisyonunu taşır. FAISS adayları ve `effective_weights` (dino≈0.39, clip≈0.28, texture≈0.11) bu sinyali öne alır. Desen bölgesi (patch/texture/DNA) birincil ağırlıkta zayıf.

2. **`crop_search=True` bilinçli pattern-first reweight yapıyor**  
   ```text
   score = score*0.75 + patch*0.35 + texture*0.15
   ```  
   Bu yüzden gömlek/desen crop’u doğru leopard sonuçlarını yükseltir. Full image bu yolu kullanmaz → **farklı sonuç evreni**.

3. **Family mismatch + `family_score_cap=0.40` aynı deseni gömüyor**  
   Örnek: `8.jpg` indexed `animal_print_type=leopard` ama `pattern_family=floral`.  
   `animal_print` ↔ `floral` HARD unrelated → skor **0.40 tavan**.  
   Aynı leopard tipi olsa bile model↔kumaş/detay gömülüyordu.  
   `4.jpg` ise `marble_abstract` + boş animal → ayrı **classification gap**.

DNA/patch/motif `pattern_family_score` içinde var ama primary `effective_weights()` toplamına **girmiyor**; sonra structural/low-hash gate ve family cap onları ezebiliyor.

---

## FEATURE CONTRIBUTION (mevcut pipeline)

| Katman | Rol | Full-image sorun |
|--------|-----|------------------|
| OpenCLIP | Global semantik / FAISS | Kişi/moda fotoğrafı benzerliği |
| DINO | Global yapı / FAISS | Açı/parça değişince sert düşüş (2→6: 0.35) |
| FAISS | Aday daraltma | CLIP/DINO boost ile moda görselleri |
| Patch / multiscale | Desen tile | crop’ta güçlenir; full’da ikincil |
| Texture | Doku vektörü | Ağırlık düşük; probe’da 0 görülebilir |
| Pattern DNA | Aile/motif | Primary blend’de yok; family uyumsuzluğunda zayıf |
| Color | Palette | `color_weight_mode=ignore` |
| Semantic / Knowledge | Tag/KB | Yardımcı |
| Object / Person fusion | Metin person query | Image path’te asıl değil |
| Family / Gate | Cluster + cap | **Mislabeled family → 0.40 cap** |
| Re-rank (textile_v2) | Text/hybrid sonrası | Image skorundan sonra |

### Pairwise probe (indexed, klasör `leopard , zebra , yılan`)

Query **2.jpg** (id 5497, animal=leopard):

| Other | CLIP | DINO | DNA | Family (cand) |
|-------|------|------|-----|---------------|
| 3.jpg | 0.91 | 0.80 | 1.00 | animal_print |
| 4.jpg | 0.81 | 0.32 | 0.29 | **marble_abstract** (mislabel) |
| 6.jpg | 0.83 | 0.35 | 1.00 | animal_print |
| 8.jpg | 0.81 | 0.28 | 0.29 | **floral** (mislabel, animal=leopard) |

Eksik indexed isimler bu klasörde: `5.jpg`, `7.jpg`, `9.jpg`, `10.jpg`, `Untitled-1.jpg` → **PARTIAL SET**.

---

## FULL VS CROP

| | Full | Crop |
|--|------|------|
| Embedding | Tüm sahne (insan+poz) | Desen bölgesi |
| Score path | CLIP/DINO ağır | patch+texture boost |
| Beklenen | Pattern identity korunmalı | Daha temiz (OK) |
| Önce | Farklı evren | Leopard yukarı |
| Sonra (patch) | Aynı animal_type cap gevşer | Değişmedi |

---

## SAME PATTERN / DIFFERENT VIEW / GARMENT→FABRIC

- **2→3 (yan):** CLIP+DINO+DNA güçlü → OK sinyali var.  
- **2→6 (ürün/parça):** DNA=1, DINO düşük → gate/family kritik.  
- **2→8 (detay, floral mislabel):** animal_type aynı ama cap 0.40 → **kök neden**.  
- **2→9 / Untitled-1:** bu klasörde index yok → **UNTESTED** (dosya yolu kullanıcı setinde ayrı olabilir).

---

## FALSE POSITIVES

`family_score_cap` unrelated→0.40 zebra/floral’ı bastırmak için vardı. Patch yalnızca `same_subtype` (aynı `animal_print_type`) iken 0.88’e çıkar; zebra≠leopard ise cap 0.40 kalır.

---

## RANKING DEĞİŞİKLİĞİ (minimal patch)

**YAPILMADI:** kör CLIP/DINO % değişimi.

**YAPILDI:**

1. `group_gates.family_score_cap`: `same_subtype` iken unrelated family mislabel → cap **0.88** (önce 0.40).  
2. `search_engine._score_record`:  
   - `same_animal_print` → family_evidence / family_relief / DNA bonus  
   - family mismatch penalty skip  
   - structural/low-hash sonrası pattern-first soft floor (≥0.62 same animal)

Indexer / CLIP / DINO model / Teach / JobStore dokunulmadı.

---

## TEST / REGRESSION

| Suite | Result |
|-------|--------|
| `test_pattern_first_animal_relief` | PASS |
| `test_search_intelligence_unification` | PASS |
| `test_search_evidence_gate` | PASS |
| Live full search 2.jpg top-N | **UNTESTED** (NAS/path + süre) |
| Complete chain 5/7/9/10/Untitled | **BLOCKED** (klasörde index yok) |
| TeachMe / QThread / StatusWorker | koşuluyor / rapor commit anında |

---

## SKORLAR

| | |
|--|--|
| AUDIT | **PASS** |
| ROOT CAUSE | **PASS** (kesin) |
| MINIMAL PATCH | **PASS** (evidence-based) |
| FULL CHAIN LIVE | **UNTESTED / PARTIAL** |
| FAIL | 0 (unit) |
| WARNING | family mislabels (4.jpg marble); texture_sim=0 probe |
| BLOCKED | incomplete indexed test set names |
| PUSH | no |
