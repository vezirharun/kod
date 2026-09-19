# Müşteri Hafızası (Customer Memory) — Minimal Patch Report

**Date:** 2026-09-19 (Europe/Istanbul)  
**Scope:** `/workspace/vezir_audit/cm_impl/core/` + `/workspace/vezir_audit/cm_tests/`  
**Authority:** USER LEARNING > GLOBAL LEARNING > CUSTOMER CONTEXT > AI INFERENCE

## Files changed

| File | Change |
|------|--------|
| `core/concept_registry.py` | Optional `customer_key` on `concept_examples`; migration-safe UNIQUE rebuild; `add_example` / `learn` kwargs; `example_rows_for_concept`, `customer_example_file_ids` |
| `core/teach_me.py` | `resolve_active_customer_key`; stamp `customer_key` in `record_verified_concept_examples` / `learn_from_metadata_edit` / `teach_files` |
| `core/learned_concept_search.py` | `example_file_ids(..., customer_key=)`; `collect_learned_hits(..., customer_key=)` prefers scoped then global |
| `core/search_engine.py` | Passes customer into learned hits + intelligence chain; scopes overlay/memory lookups; stashes `_active_search_customer` |
| `core/search_intelligence_chain.py` | Forwards `customer` to concept/color evidence soft layers |
| `core/concept_evidence.py` | Customer-aware example selection + tiny soft bonus for customer-stamped file ids (skips user-exact) |
| `core/color_evidence.py` | Optional `customer_key` kwarg (chain parity; color evidence stays global) |
| `core/search_memory.py` | `scoped_query_key`; overlay/rewrite keys include customer when present |
| `cm_tests/test_customer_memory.py` | New coverage suite |
| `cm_tests/conftest.py` | Lightweight stubs for missing deps in this audit sandbox |

**Not touched:** indexer, DINO, Preview Pool, JobStore, index_v3 queues/engine, parallel CustomerMemoryEngine / new memory DB product.

## Behavior

- Customer memory is **CONTEXT / PRIOR / EVIDENCE only**.
- Empty `customer_key` = **global** (unchanged path).
- When customer scope is active:
  - Teaching stamps examples with normalized `customer_key`.
  - Search prefers matching customer examples, then falls back to global examples.
  - Another customer's examples are never used.
- Soft layers may apply a small bonus to customer-stamped file ids; they do not demote or rename concepts.

## Global identity

- `concept_registry.canonical` remains a **single global identity** (e.g. `Leopar`).
- Customer never merges into the canonical name (no `Ünal Tekstil Leopar` fork).
- `upsert` / `learn` still write one registry row; only `concept_examples` rows are scoped.

## User authority

- User-taught global examples remain visible under customer scope (fallback).
- Without customer scope, only global (`customer_key=''`) examples participate — customer rows do not leak.
- Concept-evidence soft bonus **skips** rows already marked `learned_concept_exact` / `user_taught_positive`.
- Hierarchy held: user teaching > global learning > customer context > AI inference.

## Search hook

1. `search_engine.search_by_text` → `collect_learned_hits(..., customer_key=customer)`
2. Later → `apply_search_intelligence_chain(..., customer=customer)`
3. Chain → `apply_concept_evidence_scoring(..., customer_key=cust)` (+ color kwarg)
4. Overlay / rewrite keys via `scoped_query_key(text, customer)` so caches do not cross customers
5. Indexed pool cache already keyed by `cust` (pre-existing)

## How to run tests

```bash
cd /workspace/vezir_audit
PYTHONPATH=cm_impl:cm_tests python3 -m unittest cm_tests.test_customer_memory -v
# or, if pytest is installed:
PYTHONPATH=cm_impl:cm_tests python3 -m pytest cm_tests/test_customer_memory.py -v
```

**Result (this sandbox):** 14/14 passed via unittest.  
`pytest` was not installed system-wide; unittest was used instead.  
`cm_tests/conftest.py` stubs `core.logger`, `textile_terms`, `utils`, `manual_label_guard`, `search_models`, etc., so the suite runs without the full app tree.

## Risks

1. **Legacy DB UNIQUE rebuild:** Opening an old `concept_examples` table without `customer_key` in UNIQUE triggers a one-time table rebuild. Low risk (dedupe prefers user source) but writes the concept DB once.
2. **API kwargs:** All new parameters are optional with defaults — existing callers stay compatible.
3. **Global path excludes foreign customer rows:** Intentional; if any pre-existing caller expected “all examples including scoped”, behavior differs once customer rows exist.
4. **`demote_positive`:** Still demotes by `(concept, file, role)` across scopes (not customer-filtered). Corrections remain strong; fine-grained per-customer demote was out of scope.
5. **Soft bonus magnitude:** Fixed `+0.03` — evidence only; may need tuning in production.
6. **Active customer in teach_me:** Resolved from explicit arg or `AppSettings.customer_filter` when available; UI must pass/filter correctly for stamps to apply.
