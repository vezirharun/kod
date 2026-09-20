# SEARCH EXPLANATION WIRE

**Kapsam:** `/workspace/vezir_audit/sx_wire/` — UI wire-only.  
**Tarih:** 2026-09-20 (Europe/Istanbul)

## Ne değişti

1. **Result cards** (`ui/result_card.py`): reason chips UI kaldırıldı (`reason_chips` / match_explanations chip satırları). Backend alanları duruyor; kartta Kavram/Renk/Müşteri/Neden chip yok.

2. **Inspector “Neden”** (`ui/inspector_panel.py`):
   - Mevcut `grp_ai_explanation` kullanılıyor.
   - Collapsible (`setCheckable(True)`), **varsayılan kapalı** (`setChecked(False)`).
   - Başlık: kapalı `Neden bu sonuç ▸`, açık `Neden bu sonuç ▾` (simple: `Neden benzer`).
   - İçerik önizlemenin **altında** (overlay yok).
   - İçerik yalnızca `format_why_lines` / `format_why_html`; boşsa grup gizlenir.
   - Simple mode’da teknik DNA checklist yok; advanced’de `_append_technical_why` ile kalabilir.
   - Skor Detayı sekmesi aynen; simple zaten gizli.

3. **Yeni helper** (`core/search_explanation_ui.py`): motor değil, ince formatter.
   - `format_why_lines(result) -> list[str]` — yalnızca gerçek alanlar.
   - Kaynaklar: `debug.query_meaning` (customer / concept_core / colors), `debug.search_reason`, `concept_evidence`, `color_evidence_score` (hits>0), `customer_soft_bonus`, `same_pattern_family` / animal, `visual_verdict` / `visual_grade`.
   - **Uydurma yok:** `pattern_explanation` fallback’i (“Benzerlik skoruyla sonuç geldi”) Neden kutusuna girmez.
   - Skor / ranking alanlarını mutate etmez.

4. **Dokunulmayan:** ranking, search_engine scoring, indexer, JobStore, DINO, CLIP, Preview, customer memory, concept identity, `pattern_explanation.py`.

## Test

```bash
cd /workspace/vezir_audit/sx_wire
PYTHONPATH=. python3 -m unittest tests.test_search_explanation_ui -v
```
