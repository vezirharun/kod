"""Visual Memory schema — customer-isolated, not production search SSOT.

Clusters are similarity groups, not class labels.
Auto-discovery never writes LIKELY / SUPPORTED / CONFIRMED.
"""

from __future__ import annotations

STATUS_UNKNOWN = "UNKNOWN"
STATUS_DISCOVERED = "DISCOVERED"
STATUS_LIKELY = "LIKELY"
STATUS_SUPPORTED = "SUPPORTED"
STATUS_CONFIRMED = "CONFIRMED"

AUTO_STATUSES = frozenset({STATUS_UNKNOWN, STATUS_DISCOVERED})
FEEDBACK_STATUSES = frozenset({STATUS_LIKELY, STATUS_SUPPORTED, STATUS_CONFIRMED})
ALL_STATUSES = AUTO_STATUSES | FEEDBACK_STATUSES

# Auto path must never reach this ceiling (reserved for feedback / confirm).
AUTO_CONFIDENCE_CAP = 0.49

DDL = """
CREATE TABLE IF NOT EXISTS vm_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vm_clusters (
    cluster_id TEXT PRIMARY KEY,
    embedding_centroid BLOB,
    embedding_dim INTEGER DEFAULT 0,
    embedding_kind TEXT DEFAULT '',
    member_count INTEGER DEFAULT 0,
    representative_ids TEXT DEFAULT '[]',
    visual_dna_summary TEXT DEFAULT '{}',
    object_distribution TEXT DEFAULT '{}',
    color_distribution TEXT DEFAULT '{}',
    texture_distribution TEXT DEFAULT '{}',
    pattern_family_distribution TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0,
    status TEXT DEFAULT 'UNKNOWN',
    claimed_class TEXT DEFAULT '',
    evidence_sources TEXT DEFAULT '[]',
    cohesion REAL DEFAULT 0,
    created_at TEXT DEFAULT '',
    updated_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS vm_members (
    file_id INTEGER NOT NULL,
    cluster_id TEXT NOT NULL,
    sim_to_centroid REAL DEFAULT 0,
    source_id INTEGER DEFAULT 0,
    evidence TEXT DEFAULT '{}',
    created_at TEXT DEFAULT '',
    PRIMARY KEY (file_id),
    FOREIGN KEY (cluster_id) REFERENCES vm_clusters(cluster_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_vm_members_cluster ON vm_members(cluster_id);

CREATE TABLE IF NOT EXISTS vm_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    cluster_id TEXT DEFAULT '',
    label TEXT DEFAULT '',
    action TEXT NOT NULL,
    applied INTEGER DEFAULT 0,
    note TEXT DEFAULT '',
    created_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_vm_feedback_cluster ON vm_feedback(cluster_id);
"""
