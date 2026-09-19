# V3 Discovery / DB Fast Reconcile Update

- Source discovery is now batch-oriented: filesystem scan -> batch DB registration -> one joined artifact-state read -> one queue transaction.
- New files become visible in the DB without waiting for Preview/Thumbnail/AI processing.
- Existing indexed artifacts are preserved; unchanged files are not reset.
- Changed files invalidate only their dependent pipeline state.
- Manual IndexEngineV3.run() now scans sources by default when walk_disk is omitted; callers can pass walk_disk=False for DB-only gap processing.
- Preview remains the hard gate for Thumbnail and General AI.
- General AI does not fall back to Thumbnail/source processing.
- Source file_count is reconciled after discovery.
- Missing files retain the existing safe purge/reconciliation behavior.
